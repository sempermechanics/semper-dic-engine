#include "gpu/image_prep_dispatch.hpp"

#include "gpu/cl_api.hpp"
#include "gpu/cl_internal.hpp"
#include "gpu/cl_runtime.hpp"
#include "util/log.hpp"

#include <semper/gpu/embedded_kernels.hpp>

#include <vector>

// Same reason as strain_dispatch.cpp and hessian_dispatch.cpp: this file is
// attached to semper_pipeline, which is built -ffast-math, and
// cmake/SemperOpenCL.cmake overrides the flag for every GPU source. This is
// the check that the override is actually in effect rather than a comment
// claiming it is.
#ifdef __FAST_MATH__
#error "image_prep_dispatch.cpp must not be compiled with -ffast-math; see cmake/SemperOpenCL.cmake"
#endif

#undef LOG_TAG
#define LOG_TAG "SemperCLImagePrep"

namespace Semper {
namespace gpu {
namespace {

// Same small RAII wrapper the other dispatch files use, for the same reason:
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

bool compute_image_gradients_gpu(Image &img) {
    // Per-stage gate. The stencil is fp32 end to end and its only rounding
    // operations are three adds and a divide by 12, so it asks for
    // correctly-rounded fp32 divide -- NOT for cl_khr_fp64.
    const ClCaps &c = caps();
    if (!c.available || !c.exact_fp32) return false;

    const int w = (int) img.width, h = (int) img.height;
    if (w <= 0 || h <= 0) return false;

    const size_t px = (size_t) w * (size_t) h;
    if (img.intensities.size() < px) return false;

    detail::ProgramScope scope("image_grad", kernels::IMAGE_GRAD);
    if (!scope.ok()) return false;

    const ClApi &api = scope.api();
    cl_int err = SEMPER_CL_SUCCESS;
    const size_t bytes = sizeof(float) * px;

    Mem d_img(api), d_gx(api), d_gy(api);
    if (!d_img.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, bytes) ||
        !d_gx.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, bytes) ||
        !d_gy.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, bytes))
        return false;

    // Non-blocking into the in-order queue, drained by the single Finish
    // below. img.intensities outlives that Finish -- the caller owns it.
    if (api.EnqueueWriteBuffer(scope.queue(), d_img.h, SEMPER_CL_FALSE, 0, bytes,
                               img.intensities.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS) {
        api.Finish(scope.queue());
        return false;
    }

    cl_kernel k = api.CreateKernel(scope.program(), "semper_image_gradients", &err);
    if (!k || err != SEMPER_CL_SUCCESS) {
        if (k) api.ReleaseKernel(k);
        api.Finish(scope.queue());
        return false;
    }

    bool ok = api.SetKernelArg(k, 0, sizeof(cl_mem), &d_img.h) == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 1, sizeof(int),    &w)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 2, sizeof(int),    &h)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 3, sizeof(cl_mem), &d_gx.h)  == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 4, sizeof(cl_mem), &d_gy.h)  == SEMPER_CL_SUCCESS;

    if (ok) {
        // No local size: every work-item reads five pixels and writes two,
        // shares nothing, and the reads of neighbouring items overlap in a
        // way the cache handles better than any hand-picked group size did
        // in the sweep. Leave it to the driver.
        const size_t global = px;
        ok = api.EnqueueNDRangeKernel(scope.queue(), k, 1, nullptr, &global, nullptr,
                                      0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
    }

    // Read into staging, not straight into img.grad_x/grad_y. The kernel may
    // still fail at the Finish below, and a half-written gradient plane on a
    // caller that has been told "false, use the CPU" is worse than an extra
    // copy of an 8 MB buffer -- the caller is entitled to assume its Image is
    // untouched when this returns false.
    std::vector<float> h_gx, h_gy;
    if (ok) {
        h_gx.resize(px);
        h_gy.resize(px);
        ok = api.EnqueueReadBuffer(scope.queue(), d_gx.h, SEMPER_CL_FALSE, 0, bytes,
                                   h_gx.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueReadBuffer(scope.queue(), d_gy.h, SEMPER_CL_FALSE, 0, bytes,
                                   h_gy.data(), 0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
    }

    // One synchronisation point for the stage, unconditional so the failure
    // paths also drain before the staging vectors and the buffers go away.
    if (api.Finish(scope.queue()) != SEMPER_CL_SUCCESS) ok = false;
    api.ReleaseKernel(k);

    if (!ok) {
        LOGE("Image gradient dispatch failed on device; falling back to CPU");
        return false;
    }

    img.grad_x.swap(h_gx);
    img.grad_y.swap(h_gy);
    return true;
}

} // namespace gpu
} // namespace Semper
