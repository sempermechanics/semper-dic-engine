#include "gpu/icgn_dispatch.hpp"

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
#error "icgn_dispatch.cpp must not be compiled with -ffast-math; see cmake/SemperOpenCL.cmake"
#endif

#undef LOG_TAG
#define LOG_TAG "SemperCLIcgn"

namespace Semper {
namespace gpu {
namespace {

// Same small RAII wrapper the other two dispatch files use, for the same
// reason: the early-outs below must not leak a device allocation.
struct Mem {
    const ClApi &api;
    cl_mem h = nullptr;
    explicit Mem(const ClApi &a) : api(a) {}
    ~Mem() { if (h) api.ReleaseMemObject(h); }
    Mem(const Mem &) = delete;
    Mem &operator=(const Mem &) = delete;

    bool alloc(cl_context ctx, cl_mem_flags flags, size_t bytes) {
        if (h) { api.ReleaseMemObject(h); h = nullptr; }
        cl_int err = SEMPER_CL_SUCCESS;
        h = api.CreateBuffer(ctx, flags, bytes ? bytes : 1, nullptr, &err);
        if (h && err == SEMPER_CL_SUCCESS) return true;
        if (h) { api.ReleaseMemObject(h); h = nullptr; }
        return false;
    }
};

// Scratch is what bounds the launch, not the point count.
//
// Each work-item owns def_vals[n], norm_ref[n], sdi[6n] and ref_valid[n],
// which is 33n bytes -- 14.5 KB for a 21 px subset. Launching all of Path A
// at once would therefore ask for ~50 MB of scratch on a 512x512 field at
// step 7, and far more on a real image, so the work is tiled.
//
// This is exactly the remedy the roadmap names for the throughput gate:
// smaller launches over compacted point lists, NOT a different reduction
// shape. Tiling changes nothing about the arithmetic, because every subset
// is solved entirely inside one work-item either way.
constexpr size_t kScratchBudgetBytes = 64u * 1024u * 1024u;
// Floor for the halve-on-failure retry below. A tile this small is already
// pathological; giving up and letting the caller use the CPU beats thrashing.
constexpr int kMinTilePoints = 64;

} // namespace

bool solve_icgn_batch_gpu(const Image &ref_img, const Image &def_img,
                          int dim, bool use_keys6, int max_iter, float lm_alpha,
                          const IcgnGpuPoint *pts, int n_pts,
                          IcgnGpuResult *out) {
    if (!pts || !out || n_pts <= 0) return false;

    // Per-stage gate. This kernel is fp32 end to end, so it asks for
    // correctly-rounded divide and sqrt -- NOT for cl_khr_fp64. A device
    // with one and not the other runs whichever stage it can.
    const ClCaps &c = caps();
    if (!c.available || !c.exact_fp32) return false;

    if (dim <= 0 || max_iter <= 0) return false;
    if (ref_img.width <= 0 || ref_img.height <= 0) return false;
    if (def_img.width <= 0 || def_img.height <= 0) return false;

    const size_t ref_px = (size_t) ref_img.width * (size_t) ref_img.height;
    const size_t def_px = (size_t) def_img.width * (size_t) def_img.height;
    if (ref_img.intensities.size() < ref_px || ref_img.grad_x.size() < ref_px ||
        ref_img.grad_y.size() < ref_px || def_img.intensities.size() < def_px)
        return false;

    const int n = dim * dim;

    detail::ProgramScope scope("icgn_solve", kernels::ICGN_SOLVE);
    if (!scope.ok()) return false;

    const ClApi &api = scope.api();
    cl_int err = SEMPER_CL_SUCCESS;

    // --- Flatten the per-point inputs once, for every point ---
    //
    // CachedHessianData holds Eigen storage, which is COLUMN-major, while
    // the canonical helpers the kernel calls are row-major. The transpose
    // happens here, on the way out, and is the mirror of the one
    // solve_icgn does at optimization_engine.cpp:128-130. Getting it
    // backwards would be silent for H, which is symmetric, but not for its
    // Gauss-Jordan inverse.
    std::vector<int> h_cx((size_t) n_pts), h_cy((size_t) n_pts);
    std::vector<float> h_mean((size_t) n_pts), h_std((size_t) n_pts);
    std::vector<float> h_H((size_t) n_pts * 36, 0.0f);
    std::vector<float> h_Hinv((size_t) n_pts * 36, 0.0f);
    std::vector<float> h_guess((size_t) n_pts * 6, 0.0f);
    std::vector<unsigned char> h_valid((size_t) n_pts, 0);

    for (int i = 0; i < n_pts; ++i) {
        const IcgnGpuPoint &p = pts[i];
        h_cx[(size_t) i] = p.cx;
        h_cy[(size_t) i] = p.cy;
        for (int k = 0; k < 6; ++k) h_guess[(size_t) i * 6 + k] = p.guess[k];

        // A null or invalid pool entry is the precompute_subset_fast
        // fallback case; the kernel hands the point straight back.
        if (!p.cached || !p.cached->valid) {
            h_mean[(size_t) i] = 0.0f;
            h_std[(size_t) i] = 1.0f;
            continue;
        }
        h_mean[(size_t) i] = p.cached->mean_intensity;
        h_std[(size_t) i] = p.cached->std_dev;
        h_valid[(size_t) i] = 1;
        for (int r = 0; r < 6; ++r)
            for (int cc = 0; cc < 6; ++cc) {
                h_H[(size_t) i * 36 + r * 6 + cc] = p.cached->H(r, cc);
                h_Hinv[(size_t) i * 36 + r * 6 + cc] = p.cached->H_inv(r, cc);
            }
    }

    // --- Images, uploaded once and reused by every tile ---
    const size_t ref_bytes = sizeof(float) * ref_px;
    const size_t def_bytes = sizeof(float) * def_px;

    Mem d_ref(api), d_rgx(api), d_rgy(api), d_def(api);
    if (!d_ref.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, ref_bytes) ||
        !d_rgx.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, ref_bytes) ||
        !d_rgy.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, ref_bytes) ||
        !d_def.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, def_bytes))
        return false;

    // Non-blocking into the in-order queue, drained by the Finish at the end
    // of the first tile. Every host buffer named here outlives the call.
    if (api.EnqueueWriteBuffer(scope.queue(), d_ref.h, SEMPER_CL_FALSE, 0, ref_bytes,
                               ref_img.intensities.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_rgx.h, SEMPER_CL_FALSE, 0, ref_bytes,
                               ref_img.grad_x.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_rgy.h, SEMPER_CL_FALSE, 0, ref_bytes,
                               ref_img.grad_y.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS ||
        api.EnqueueWriteBuffer(scope.queue(), d_def.h, SEMPER_CL_FALSE, 0, def_bytes,
                               def_img.intensities.data(), 0, nullptr, nullptr) != SEMPER_CL_SUCCESS) {
        api.Finish(scope.queue());
        return false;
    }

    // --- Tile size, then scratch ---
    //
    // Start from the budget and halve on allocation failure. Querying
    // CL_DEVICE_MAX_MEM_ALLOC_SIZE would give a first guess but not a
    // guarantee -- the device may be sharing memory with a display -- so the
    // allocation itself is the test.
    // def_vals[n] + norm_ref[n] + sdi[6n] floats, plus ref_valid[n] uchar.
    const size_t scratch_bytes_per_pt = (size_t) n * 8 * sizeof(float) + (size_t) n;

    int tile = n_pts;
    {
        const size_t fits = kScratchBudgetBytes / scratch_bytes_per_pt;
        if (fits == 0) return false;   // one subset alone blows the budget
        if ((size_t) tile > fits) tile = (int) fits;
    }

    Mem d_scr_def(api), d_scr_nref(api), d_scr_sdi(api), d_scr_rv(api);
    bool scratch_ok = false;
    for (;;) {
        const size_t plane = (size_t) tile * (size_t) n * sizeof(float);
        if (d_scr_def.alloc(scope.context(), SEMPER_CL_MEM_READ_WRITE, plane) &&
            d_scr_nref.alloc(scope.context(), SEMPER_CL_MEM_READ_WRITE, plane) &&
            d_scr_sdi.alloc(scope.context(), SEMPER_CL_MEM_READ_WRITE, plane * 6) &&
            d_scr_rv.alloc(scope.context(), SEMPER_CL_MEM_READ_WRITE,
                           (size_t) tile * (size_t) n)) {
            scratch_ok = true;
            break;
        }
        if (tile <= kMinTilePoints) break;
        tile /= 2;
        if (tile < kMinTilePoints) tile = kMinTilePoints;
    }
    if (!scratch_ok) {
        api.Finish(scope.queue());
        LOGE("ICGN scratch allocation failed at every tile size; falling back to CPU");
        return false;
    }

    // --- Per-tile point inputs and outputs ---
    const size_t t_pts = (size_t) tile;
    Mem d_cx(api), d_cy(api), d_mean(api), d_std(api), d_pvalid(api);
    Mem d_H(api), d_Hinv(api), d_guess(api);
    Mem d_op(api), d_oscore(api), d_ostatus(api), d_oiters(api), d_oinv(api);
    if (!d_cx.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts * sizeof(int)) ||
        !d_cy.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts * sizeof(int)) ||
        !d_mean.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts * sizeof(float)) ||
        !d_std.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts * sizeof(float)) ||
        !d_pvalid.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts) ||
        !d_H.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts * 36 * sizeof(float)) ||
        !d_Hinv.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts * 36 * sizeof(float)) ||
        !d_guess.alloc(scope.context(), SEMPER_CL_MEM_READ_ONLY, t_pts * 6 * sizeof(float)) ||
        !d_op.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, t_pts * 6 * sizeof(float)) ||
        !d_oscore.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, t_pts * sizeof(float)) ||
        !d_ostatus.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, t_pts * sizeof(int)) ||
        !d_oiters.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, t_pts * sizeof(int)) ||
        !d_oinv.alloc(scope.context(), SEMPER_CL_MEM_WRITE_ONLY, t_pts * sizeof(int))) {
        api.Finish(scope.queue());
        return false;
    }

    cl_kernel k = api.CreateKernel(scope.program(), "semper_icgn_solve", &err);
    if (!k || err != SEMPER_CL_SUCCESS) {
        if (k) api.ReleaseKernel(k);
        api.Finish(scope.queue());
        return false;
    }

    const int ref_w = ref_img.width, ref_h = ref_img.height;
    const int def_w = def_img.width, def_h = def_img.height;
    const int keys6 = use_keys6 ? 1 : 0;

    // Arguments 0..12 and 21..24 never change between tiles; only the
    // per-point slices and the count do.
    bool ok = api.SetKernelArg(k,  0, sizeof(cl_mem), &d_ref.h)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  1, sizeof(cl_mem), &d_rgx.h)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  2, sizeof(cl_mem), &d_rgy.h)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  3, sizeof(int),    &ref_w)         == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  4, sizeof(int),    &ref_h)         == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  5, sizeof(cl_mem), &d_def.h)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  6, sizeof(int),    &def_w)         == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  7, sizeof(int),    &def_h)         == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  8, sizeof(int),    &dim)           == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k,  9, sizeof(int),    &keys6)         == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 10, sizeof(int),    &max_iter)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 11, sizeof(float),  &lm_alpha)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 12, sizeof(cl_mem), &d_cx.h)        == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 13, sizeof(cl_mem), &d_cy.h)        == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 14, sizeof(cl_mem), &d_mean.h)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 15, sizeof(cl_mem), &d_std.h)       == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 16, sizeof(cl_mem), &d_pvalid.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 17, sizeof(cl_mem), &d_H.h)         == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 18, sizeof(cl_mem), &d_Hinv.h)      == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 19, sizeof(cl_mem), &d_guess.h)     == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 20, sizeof(cl_mem), &d_scr_def.h)   == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 21, sizeof(cl_mem), &d_scr_nref.h)  == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 22, sizeof(cl_mem), &d_scr_sdi.h)   == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 23, sizeof(cl_mem), &d_scr_rv.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 24, sizeof(cl_mem), &d_op.h)        == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 25, sizeof(cl_mem), &d_oscore.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 26, sizeof(cl_mem), &d_ostatus.h)   == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 27, sizeof(cl_mem), &d_oiters.h)    == SEMPER_CL_SUCCESS &&
              api.SetKernelArg(k, 28, sizeof(cl_mem), &d_oinv.h)      == SEMPER_CL_SUCCESS;

    std::vector<float> r_p(t_pts * 6), r_score(t_pts);
    std::vector<int> r_status(t_pts), r_iters(t_pts), r_inv(t_pts);

    // Results are staged here and copied to the caller only once every tile
    // has succeeded. A tile failing halfway through would otherwise leave
    // the first half of `out` overwritten while this function reports
    // failure, and the caller -- which then re-solves everything on the CPU
    // -- is entitled to find its buffer as it left it.
    std::vector<IcgnGpuResult> staged((size_t) n_pts);

    for (int base = 0; ok && base < n_pts; base += tile) {
        const int cnt = (n_pts - base < tile) ? (n_pts - base) : tile;
        const size_t cn = (size_t) cnt;

        ok = api.EnqueueWriteBuffer(scope.queue(), d_cx.h, SEMPER_CL_FALSE, 0,
                                    cn * sizeof(int), &h_cx[(size_t) base],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueWriteBuffer(scope.queue(), d_cy.h, SEMPER_CL_FALSE, 0,
                                    cn * sizeof(int), &h_cy[(size_t) base],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueWriteBuffer(scope.queue(), d_mean.h, SEMPER_CL_FALSE, 0,
                                    cn * sizeof(float), &h_mean[(size_t) base],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueWriteBuffer(scope.queue(), d_std.h, SEMPER_CL_FALSE, 0,
                                    cn * sizeof(float), &h_std[(size_t) base],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueWriteBuffer(scope.queue(), d_pvalid.h, SEMPER_CL_FALSE, 0,
                                    cn, &h_valid[(size_t) base],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueWriteBuffer(scope.queue(), d_H.h, SEMPER_CL_FALSE, 0,
                                    cn * 36 * sizeof(float), &h_H[(size_t) base * 36],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueWriteBuffer(scope.queue(), d_Hinv.h, SEMPER_CL_FALSE, 0,
                                    cn * 36 * sizeof(float), &h_Hinv[(size_t) base * 36],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.EnqueueWriteBuffer(scope.queue(), d_guess.h, SEMPER_CL_FALSE, 0,
                                    cn * 6 * sizeof(float), &h_guess[(size_t) base * 6],
                                    0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
             api.SetKernelArg(k, 29, sizeof(int), &cnt) == SEMPER_CL_SUCCESS;

        if (ok) {
            // No local size. Each work-item owns its subset outright and
            // shares nothing, so there is no work-group structure to impose
            // and no reason to fight the driver's choice.
            const size_t global = cn;
            ok = api.EnqueueNDRangeKernel(scope.queue(), k, 1, nullptr, &global, nullptr,
                                          0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
        }

        if (ok) {
            ok = api.EnqueueReadBuffer(scope.queue(), d_op.h, SEMPER_CL_FALSE, 0,
                                       cn * 6 * sizeof(float), r_p.data(),
                                       0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
                 api.EnqueueReadBuffer(scope.queue(), d_oscore.h, SEMPER_CL_FALSE, 0,
                                       cn * sizeof(float), r_score.data(),
                                       0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
                 api.EnqueueReadBuffer(scope.queue(), d_ostatus.h, SEMPER_CL_FALSE, 0,
                                       cn * sizeof(int), r_status.data(),
                                       0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
                 api.EnqueueReadBuffer(scope.queue(), d_oiters.h, SEMPER_CL_FALSE, 0,
                                       cn * sizeof(int), r_iters.data(),
                                       0, nullptr, nullptr) == SEMPER_CL_SUCCESS &&
                 api.EnqueueReadBuffer(scope.queue(), d_oinv.h, SEMPER_CL_FALSE, 0,
                                       cn * sizeof(int), r_inv.data(),
                                       0, nullptr, nullptr) == SEMPER_CL_SUCCESS;
        }

        // One synchronisation point per tile, unconditional so the failure
        // paths also drain before the next iteration reuses the buffers.
        if (api.Finish(scope.queue()) != SEMPER_CL_SUCCESS) ok = false;
        if (!ok) break;

        for (int i = 0; i < cnt; ++i) {
            IcgnGpuResult &r = staged[(size_t) base + (size_t) i];
            for (int kk = 0; kk < 6; ++kk) r.p[kk] = r_p[(size_t) i * 6 + kk];
            r.score = r_score[(size_t) i];
            r.status = r_status[(size_t) i];
            r.iters = r_iters[(size_t) i];
            r.invalid_ref_pixels = r_inv[(size_t) i];
        }
    }

    api.ReleaseKernel(k);

    if (!ok) {
        LOGE("ICGN dispatch failed on device; falling back to CPU");
        return false;
    }

    for (int i = 0; i < n_pts; ++i) out[(size_t) i] = staged[(size_t) i];
    return true;
}

} // namespace gpu
} // namespace Semper
