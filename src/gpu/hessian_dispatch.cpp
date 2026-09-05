#include "gpu/hessian_dispatch.hpp"

#include "gpu/cl_api.hpp"
#include "gpu/cl_internal.hpp"
#include "gpu/cl_runtime.hpp"
#include "util/log.hpp"

#include <semper/gpu/embedded_kernels.hpp>

#include <vector>

// Same reason as strain_dispatch.cpp: this file is attached to
// semper_pipeline, which is built -ffast-math, and cmake/SemperOpenCL.cmake
// overrides the flag for every GPU source. This is the check that the
// override is actually in effect rather than a comment claiming it is.
#ifdef __FAST_MATH__
#error "hessian_dispatch.cpp must not be compiled with -ffast-math; see cmake/SemperOpenCL.cmake"
#endif

#undef LOG_TAG
#define LOG_TAG "SemperCLHessian"

namespace Semper {
namespace gpu {
namespace {

// Same small RAII wrapper strain_dispatch.cpp uses, for the same reason:
// the early-outs below must not leak a device allocation.
struct Mem {
    const ClApi &api;
    cl_mem h = nullptr;
    explicit Mem(const ClApi &a) : api(a) {}
    ~Mem() { if (h) api.ReleaseMemObject(h); }
    Mem(const Mem &) = delete;
    Mem &operator=(const Mem &) = delete;

    bool alloc(cl_context ctx, cl_mem_flags flags, size_t bytes) {
        cl_int err = SEMPER_CL_SUCCESS;
        h = api.CreateBuffer(ctx, flags, bytes ? bytes : 1, nullptr, &err);
        return h && err == SEMPER_CL_SUCCESS;
    }
};

} // namespace

bool compute_hessian_pool_gpu(const Image &ref_img,
                              int rect_x, int rect_y, int step,
                              int grid_w, int grid_h, int dim,
                              const unsigned char *wanted,
                              CachedHessianData *out, int out_count) {
    if (!wanted || !out) return false;

    // Per-stage gate. This kernel is fp32 end to end, so it asks for
    // correctly-rounded divide and sqrt -- NOT for cl_khr_fp64. A device
    // with one and not the other runs whichever stage it can.
    const ClCaps &c = caps();
    if (!c.available || !c.exact_fp32) return false;

    const int n_pts = grid_w * grid_h;
    if (n_pts <= 0 || dim <= 0 || out_count < n_pts) return false;
    if (ref_img.width <= 0 || ref_img.height <= 0) return false;

    const size_t img_px = (size_t) ref_img.width * (size_t) ref_img.height;
    if (ref_img.intensities.size() < img_px || ref_img.grad_x.size() < img_px ||
        ref_img.grad_y.size() < img_px)
        return false;

    detail::ProgramScope scope("hessian_prepass", kernels::HESSIAN_PREPASS);
    if (!scope.ok()) return false;

    const ClApi &api = scope.api();
    cl_int err = SEMPER_CL_SUCCESS;

    const size_t img_bytes = sizeof(float) * img_px;
    const size_t pt_bytes = sizeof(float) * (size_t) n_pts;
    const size_t mat_bytes = pt_bytes * 36;

    Mem d_int(api), d_gx(api), d_gy(api), d_want(api);
    Mem d_mean(api), d_std(api), d_H(api), d_Hinv(api), d_valid(api);
    if (!d_int.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, img_bytes) ||
        !d_gx.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, img_bytes) ||
        !d_gy.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, img_bytes) ||
        !d_want.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, (size_t) n_pts) ||
        !d_mean.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, pt_bytes) ||
        !d_std.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, pt_bytes) ||
        !d_H.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, mat_bytes) ||
        !d_Hinv.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, mat_bytes) ||
        !d_valid.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, (size_t) n_pts))
        return false;

    // Non-blocking into the in-order queue, drained by the single Finish
    // below -- see the same comment in strain_dispatch.cpp. Every host
    // buffer named here outlives that Finish.
    if (api.EnqueueWriteBuffer(scope.queue(), d_int.h, SEMPER_CL_FALSE, 0, img_bytes,
                               ref_img.intensities.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_gx.h, SEMPER_CL_FALSE, 0, img_bytes,
                               ref_img.grad_x.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_gy.h, SEMPER_CL_FALSE, 0, img_bytes,
                               ref_img.grad_y.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_want.h, SEMPER_CL_FALSE, 0, (size_t) n_pts,
                               wanted, 0, nullptr, nullptr) != SEMPER_CL_SUCCESS) {
        api.Finish(scope.queue());
        return false;
    }

    cl_kernel k = api.CreateKernel(scope.program(), "semper_hessian_prepass", &err);
    if (!k || err != SEMPER_CL_SUCCESS) {
        if (k) api.ReleaseKernel(k);
        api.Finish(scope.queue());
        return false;
    }

    const int img_w = (int) ref_img.width, img_h = (int) ref_img.height;
    bool ok = api.SetKernelArg(k,  0, sizeof(cl_mem), &d_int.h)   == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  1, sizeof(cl_mem), &d_gx.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  2, sizeof(cl_mem), &d_gy.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  3, sizeof(int),    &img_w)     == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  4, sizeof(int),    &img_h)     == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  5, sizeof(cl_mem), &d_want.h)  == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  6, sizeof(int),    &grid_w)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  7, sizeof(int),    &grid_h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  8, sizeof(int),    &rect_x)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  9, sizeof(int),    &rect_y)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 10, sizeof(int),    &step)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 11, sizeof(int),    &dim)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 12, sizeof(cl_mem), &d_mean.h)  == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 13, sizeof(cl_mem), &d_std.h)   == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 14, sizeof(cl_mem), &d_H.h)     == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 15, sizeof(cl_mem), &d_Hinv.h)  == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 16, sizeof(cl_mem), &d_valid.h) == SEMPER_CL_SUCCESS;

    if (ok) {
        // No local size: each work-item owns a private float[36] Hessian and
        // its inverse and shares nothing, so there is no work-group
        // structure to impose and no reason to fight the driver's choice.
        const size_t global = (size_t) n_pts;
        ok = api.EnqueueNDRangeKernel(scope.queue(), k, 1, nullptr, &global, nullptr,
                                      0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
    }

    // Read into flat host staging rather than straight into CachedHessianData:
    // that struct is Eigen-backed and column-major, and the kernel emits
    // row-major, so the transpose has to happen on arrival (the same
    // transpose subset_precomputer.cpp does, and the same reason it is
    // written out element-wise -- a symmetric H hides a reversed transpose
    // but its Gauss-Jordan inverse does not).
    std::vector<float> h_mean, h_std, h_H, h_Hinv;
    std::vector<unsigned char> h_valid;
    if (ok) {
        h_mean.resize((size_t) n_pts);
        h_std.resize((size_t) n_pts);
        h_H.resize((size_t) n_pts * 36);
        h_Hinv.resize((size_t) n_pts * 36);
        h_valid.resize((size_t) n_pts);
        ok = api.EnqueueReadBuffer(scope.queue(), d_mean.h, SEMPER_CL_FALSE, 0, pt_bytes,
                                   h_mean.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueReadBuffer(scope.queue(), d_std.h, SEMPER_CL_FALSE, 0, pt_bytes,
                                   h_std.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueReadBuffer(scope.queue(), d_H.h, SEMPER_CL_FALSE, 0, mat_bytes,
                                   h_H.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueReadBuffer(scope.queue(), d_Hinv.h, SEMPER_CL_FALSE, 0, mat_bytes,
                                   h_Hinv.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueReadBuffer(scope.queue(), d_valid.h, SEMPER_CL_FALSE, 0, (size_t) n_pts,
                                   h_valid.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
    }

    // One synchronisation point for the stage, unconditional so the failure
    // paths also drain before the staging vectors and the buffers go away.
    if (api.Finish(scope.queue()) != SEMPER_CL_SUCCESS) ok = false;
    api.ReleaseKernel(k);

    if (!ok) {
        LOGE("Hessian pre-pass dispatch failed on device; falling back to CPU");
        return false;
    }

    // Only now is anything written to the caller's pool, and only the slots
    // it asked for: a point the CPU pre-pass would have skipped keeps its
    // default-constructed entry.
    for (int i = 0; i < n_pts; ++i) {
        if (!wanted[i]) continue;
        CachedHessianData &d = out[i];
        d.mean_intensity = h_mean[(size_t) i];
        d.std_dev = h_std[(size_t) i];
        const float *H = &h_H[(size_t) i * 36];
        const float *Hi = &h_Hinv[(size_t) i * 36];
        for (int r = 0; r < 6; ++r)
            for (int cc = 0; cc < 6; ++cc) {
                d.H(r, cc) = H[r * 6 + cc];
                d.H_inv(r, cc) = Hi[r * 6 + cc];
            }
        d.valid = h_valid[(size_t) i] != 0;
    }
    return true;
}

} // namespace gpu
} // namespace Semper
