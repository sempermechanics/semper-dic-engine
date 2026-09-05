#include "gpu/strain_dispatch.hpp"

#include "gpu/cl_api.hpp"
#include "gpu/cl_internal.hpp"
#include "gpu/cl_runtime.hpp"
#include "util/log.hpp"

#include <semper/gpu/embedded_kernels.hpp>
#include <semper/tuning.hpp>

#include <cmath>
#include <vector>

// The scalars computed below (radius_sq, grid_rad, d_radius_sq, the
// expected-window point count) are handed to the kernel and must match what
// the CPU reference computes in semper_math bit for bit. semper_pipeline is
// built -ffast-math, which licenses reassociation and contraction and would
// silently break that; cmake/SemperOpenCL.cmake overrides the flag for this
// file. This is the check that the override is actually in effect --
// without it, a build-system change would degrade the GPU path quietly
// instead of failing here. Same guard canonical_math.h uses.
#ifdef __FAST_MATH__
#error "strain_dispatch.cpp must not be compiled with -ffast-math; see cmake/SemperOpenCL.cmake"
#endif

#undef LOG_TAG
#define LOG_TAG "SemperCLStrain"

namespace Semper {
namespace gpu {
namespace {

// Small RAII wrapper so the many early-outs below cannot leak a device
// allocation. Deliberately not a general-purpose type.
struct Mem {
    const ClApi &api;
    cl_mem h = nullptr;
    explicit Mem(const ClApi &a) : api(a) {}
    ~Mem() { if (h) api.ReleaseMemObject(h); }
    Mem(const Mem &) = delete;
    Mem &operator=(const Mem &) = delete;

    bool alloc(cl_context ctx, cl_mem_flags flags, size_t bytes) {
        cl_int err = SEMPER_CL_SUCCESS;
        // A zero-byte buffer is invalid even when the logical count is zero.
        h = api.CreateBuffer(ctx, flags, bytes ? bytes : 1, nullptr, &err);
        return h && err == SEMPER_CL_SUCCESS;
    }
};

} // namespace

bool compute_vsg_strain_gpu(const DisplacementField &disp, int window_pixels,
                            StrainField *out) {
    if (!out) return false;

    // Per-stage gate, checked before anything else touches the device: the
    // fit is fp64 from end to end, so a device without cl_khr_fp64 is not
    // refused outright, it simply does not run THIS stage.
    const ClCaps &c = caps();
    if (!c.available || !c.fp64) return false;

    const int total_pts = disp.width * disp.height;
    if (total_pts <= 0 || disp.step <= 0) return false;
    if ((int) disp.u.size() < total_pts || (int) disp.v.size() < total_pts ||
        (int) disp.valid.size() < total_pts)
        return false;

    // Scalars are computed HERE, in the same types and the same order as the
    // CPU path, and passed down. Recomputing them inside the kernel would put
    // a float/int rounding decision on the far side of the boundary the whole
    // phase exists to keep identical.
    const float radius = window_pixels / 2.0f;
    const float radius_sq = radius * radius;
    const int grid_rad = (int) std::ceil(radius / disp.step);

    const double tiny = tuning::kVsgRadiusTiny;
    const double d_radius_sq = static_cast<double>(radius_sq) + tiny;

    int expected_full_window_pts = 0;
    for (int dy = -grid_rad; dy <= grid_rad; ++dy) {
        for (int dx = -grid_rad; dx <= grid_rad; ++dx) {
            double d_phys_dx = static_cast<double>(dx * disp.step);
            double d_phys_dy = static_cast<double>(dy * disp.step);
            if ((d_phys_dx * d_phys_dx + d_phys_dy * d_phys_dy) <= d_radius_sq)
                expected_full_window_pts++;
        }
    }
    if (expected_full_window_pts <= 0) return false;

    // std::vector<bool> is a bitfield with no contiguous storage, so it
    // cannot be uploaded; unpack to one byte per point.
    std::vector<unsigned char> valid_bytes((size_t) total_pts);
    for (int i = 0; i < total_pts; ++i)
        valid_bytes[(size_t) i] = disp.valid[(size_t) i] ? 1u : 0u;

    detail::ProgramScope scope("strain_vsg", kernels::STRAIN_VSG);
    if (!scope.ok()) return false;

    const ClApi &api = scope.api();
    cl_int err = SEMPER_CL_SUCCESS;

    const size_t f_bytes = sizeof(float) * (size_t) total_pts;
    Mem d_u(api), d_v(api), d_valid(api), d_exx(api), d_eyy(api), d_exy(api);
    if (!d_u.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, f_bytes) ||
        !d_v.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, f_bytes) ||
        !d_valid.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, (size_t) total_pts) ||
        !d_exx.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, f_bytes) ||
        !d_eyy.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, f_bytes) ||
        !d_exy.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, f_bytes))
        return false;

    // Non-blocking writes into the in-order queue: three blocking uploads
    // cost three host/device round-trips, and at this stage's data sizes the
    // round-trips dominate the transfer itself. The queue is in-order, so the
    // kernel still cannot start before they land, and the single Finish below
    // is what actually waits. The host buffers must stay alive until then --
    // disp outlives this call and valid_bytes is scoped to it.
    if (api.EnqueueWriteBuffer(scope.queue(), d_u.h, SEMPER_CL_FALSE, 0, f_bytes,
                               disp.u.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_v.h, SEMPER_CL_FALSE, 0, f_bytes,
                               disp.v.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_valid.h, SEMPER_CL_FALSE, 0,
                               (size_t) total_pts, valid_bytes.data(),
                               0, nullptr, nullptr) != SEMPER_CL_SUCCESS) {
        api.Finish(scope.queue());   // drain before valid_bytes goes away
        return false;
    }

    cl_kernel k = api.CreateKernel(scope.program(), "semper_strain_vsg", &err);
    if (!k || err != SEMPER_CL_SUCCESS) {
        if (k) api.ReleaseKernel(k);
        api.Finish(scope.queue());   // drain before valid_bytes goes away
        return false;
    }

    const float sentinel = tuning::kStrainUninitSentinel;
    const int width = disp.width, height = disp.height, step = disp.step;
    bool ok = api.SetKernelArg(k,  0, sizeof(cl_mem), &d_u.h)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  1, sizeof(cl_mem), &d_v.h)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  2, sizeof(cl_mem), &d_valid.h)  == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  3, sizeof(int),    &width)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  4, sizeof(int),    &height)     == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  5, sizeof(int),    &step)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  6, sizeof(int),    &grid_rad)   == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  7, sizeof(double), &d_radius_sq) == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  8, sizeof(int),    &expected_full_window_pts) == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  9, sizeof(float),  &sentinel)   == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 10, sizeof(cl_mem), &d_exx.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 11, sizeof(cl_mem), &d_eyy.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 12, sizeof(cl_mem), &d_exy.h)    == SEMPER_CL_SUCCESS;

    if (ok) {
        // No local size: each work-item is independent and carries its own
        // private normal equations, so there is nothing for a work-group to
        // share and no reason to constrain the driver's choice.
        const size_t global = (size_t) total_pts;
        ok = api.EnqueueNDRangeKernel(scope.queue(), k, 1, nullptr, &global, nullptr,
                                      0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
    }

    if (ok) {
        // Only resize on the success path: a caller that gets false must find
        // *out exactly as it left it.
        out->exx.assign((size_t) total_pts, 0.0f);
        out->eyy.assign((size_t) total_pts, 0.0f);
        out->exy.assign((size_t) total_pts, 0.0f);
        // Queued non-blocking and drained once, for the same reason as the
        // uploads. Nothing here reads the destinations before the Finish.
        ok = api.EnqueueReadBuffer(scope.queue(), d_exx.h, SEMPER_CL_FALSE, 0, f_bytes,
                                   out->exx.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueReadBuffer(scope.queue(), d_eyy.h, SEMPER_CL_FALSE, 0, f_bytes,
                                   out->eyy.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueReadBuffer(scope.queue(), d_exy.h, SEMPER_CL_FALSE, 0, f_bytes,
                                   out->exy.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
    }

    // One synchronisation point for the whole stage. Unconditional: even on
    // the failure path the queue must be drained before the buffers this
    // function owns are released and valid_bytes leaves scope.
    if (api.Finish(scope.queue()) != SEMPER_CL_SUCCESS) ok = false;

    api.ReleaseKernel(k);
    if (!ok) LOGE("strain VSG dispatch failed on device; falling back to CPU");
    return ok;
}

} // namespace gpu
} // namespace Semper
