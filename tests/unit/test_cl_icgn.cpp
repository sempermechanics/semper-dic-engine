// =====================================================================
// SUITE: ClParity — src/gpu/icgn_dispatch.cpp + src/gpu/kernels/icgn_solve.cl
//
// GPU Phase 4's parity gate, continuing the ClParity suite the strain and
// Hessian kernels opened. The claim: solve_icgn_batch_gpu returns exactly
// what OptimizationEngine::calculate_deformation(..., INIT_NO_SIMPLEX)
// returns — the converged six-parameter vector, the final correlation
// score, the status, the iteration count and the invalid-reference-pixel
// count — for every point, bit for bit. Everything compares with ==. A
// failure is a divergence to find, never a tolerance to widen
// (docs/DETERMINISM.md, docs/GPU_ACCELERATION.md §1).
//
// INIT_NO_SIMPLEX is the right reference precisely because it runs one ICGN
// and nothing else, which is what the kernel computes. Path A's own mode
// (INIT_NO_SEARCH) adds the Nelder-Mead rescue on top for the few percent of
// points that need it; the dispatch hands those back to the caller instead.
//
// THE ITERATION COUNT IS PART OF THE COMPARISON, not a diagnostic. An equal
// answer reached in a different number of iterations means the convergence
// test semper_norm6(delta_p) < 0.001f diverged somewhere, and the agreement
// on this input would be luck.
//
// What is deliberately covered:
//
//   * interior points converging on the fast path, where every pixel
//     survives and both reductions run in the canonical 4-accumulator form;
//   * points behind a stamped -5.0f ghost wall, which take the PARTIAL path
//     — scattered survivors, sequential sums, a rebuilt dynamic Hessian —
//     and which also exercise invalid_ref_pixels;
//   * points whose subset hangs off the reference image, and points with no
//     valid pooled Hessian: both must come back kIcgnHostRequired rather
//     than guessed at;
//   * a launch big enough to be tiled, so the dispatch's tile bookkeeping is
//     executed rather than assumed;
//   * run-to-run reproducibility on one device.
//
// Like the rest of ClParity, every test must pass on a machine with no
// OpenCL: the device-dependent ones report themselves skipped rather than
// passing vacuously.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(SEMPER_OPENCL)

#include "gpu/cl_runtime.hpp"
#include "gpu/icgn_dispatch.hpp"

#include <semper/gpu/embedded_kernels.hpp>
#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>

using Semper::AnalysisResult;
using Semper::CachedHessianData;
using Semper::Image;
using Semper::OptimizationEngine;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;
using Semper::gpu::caps;
using Semper::gpu::IcgnGpuPoint;
using Semper::gpu::IcgnGpuResult;
using Semper::gpu::kIcgnHostRequired;
using Semper::gpu::reset_for_testing;
using Semper::gpu::solve_icgn_batch_gpu;

namespace {

// This stage is fp32 end to end, so it needs correctly-rounded divide and
// sqrt — NOT cl_khr_fp64. A device with one and not the other still runs it.
bool stage_available() { return caps().available && caps().exact_fp32; }

void report_skip(const char *what) {
    const auto &c = caps();
    if (!c.available)
        std::printf("  ClParity: skipped %s (no OpenCL device) — %s\n", what,
                    c.unavailable_reason.c_str());
    else
        std::printf("  ClParity: skipped %s (device '%s' lacks correctly-rounded fp32)\n",
                    what, c.device_name.c_str());
}

// Path A's settings, copied from full_field_path_a.cpp:46-49. The kernel is
// only ever asked to reproduce this configuration, so testing it under any
// other would be testing something the engine does not run.
constexpr float kLmAlpha = Semper::tuning::kLmAlpha;
constexpr int kMaxIter = Semper::tuning::kIcgnMaxIter;

struct Scene {
    Image ref;
    Image def;
    dictest::AffineDeformation truth;
};

Scene make_scene(int w, int h, unsigned seed, int blobs) {
    dictest::SpeckleField field(seed, w, h, blobs);
    dictest::AffineDeformation d;
    d.u = 1.75f;
    d.v = -1.25f;
    d.ux = 0.006f;
    d.vy = -0.004f;
    d.uy = 0.002f;
    d.cx = w / 2.0f;
    d.cy = h / 2.0f;
    Scene s{dictest::make_reference_image(field, w, h),
            dictest::make_deformed_image(field, w, h, d), d};
    return s;
}

// A guess deliberately offset from the truth, so ICGN actually iterates
// instead of converging on the first step. A solver that converges in one
// iteration everywhere would compare equal while exercising almost none of
// the kernel.
void guess_for(const dictest::AffineDeformation &t, int cx, int cy, float g[6]) {
    const float dx = (float) cx - t.cx, dy = (float) cy - t.cy;
    g[0] = t.u + t.ux * dx + t.uy * dy - 0.35f;
    g[1] = t.v + t.vx * dx + t.vy * dy + 0.28f;
    g[2] = 0.0f;
    g[3] = 0.0f;
    g[4] = 0.0f;
    g[5] = 0.0f;
}

struct Grid {
    int x0, y0, step, nx, ny, dim;
    // Which interpolator both sides use. FullFieldParams::use_6x6_interpolator
    // selects it in the pipeline, and it changes the sampler on BOTH sides --
    // Image::interpolate_keys_fourth on the CPU, semper_canon_sample_keys6 in
    // the kernel -- so parity has to be shown for each, not assumed from one.
    bool keys6 = false;
    int count() const { return nx * ny; }
};

// Build the point list and, alongside it, the CPU answer for each point
// through the untouched engine. Points Path A itself would skip (an
// uninitialised subset) are marked so the comparison can require the device
// to refuse them too.
struct Reference {
    std::vector<IcgnGpuPoint> pts;
    std::vector<CachedHessianData> pool;   // owns what pts[].cached points at
    std::vector<AnalysisResult> cpu;
    std::vector<unsigned char> host_only;  // 1 = Path A could not solve it here
};

Reference build_reference(const Scene &s, const Grid &g) {
    Reference r;
    const int n = g.count();
    r.pool.resize((size_t) n);
    r.pts.resize((size_t) n);
    r.cpu.resize((size_t) n);
    r.host_only.assign((size_t) n, 0u);

    OptimizationEngine eng;
    eng.lm_enabled = true;
    eng.lm_alpha = kLmAlpha;
    eng.use_6x6_interpolator = g.keys6;
    SubsetData subset;

    for (int i = 0; i < n; ++i) {
        const int gx = i % g.nx, gy = i / g.nx;
        const int cx = g.x0 + gx * g.step, cy = g.y0 + gy * g.step;

        r.pool[(size_t) i] =
                SubsetPrecomputer::compute_hessian_only(s.ref, cx, cy, g.dim);

        IcgnGpuPoint &p = r.pts[(size_t) i];
        p.cx = cx;
        p.cy = cy;
        p.cached = &r.pool[(size_t) i];
        guess_for(s.truth, cx, cy, p.guess);

        // Exactly Path A's sequence: fast precompute from the pool, then
        // solve only if the subset came back initialised.
        SubsetPrecomputer::precompute_subset_fast(subset, s.ref, cx, cy, g.dim,
                                                  r.pool[(size_t) i]);
        if (!subset.is_initialized || !r.pool[(size_t) i].valid) {
            r.host_only[(size_t) i] = 1u;
            continue;
        }
        r.cpu[(size_t) i] = eng.calculate_deformation(
                subset, s.def, p.guess[0], p.guess[1], p.guess[2], p.guess[3],
                p.guess[4], p.guess[5], Semper::INIT_NO_SIMPLEX);
    }
    return r;
}

struct Tally {
    int compared = 0;
    int mismatched = 0;
    int refused = 0;       // device handed the point back
    int converged = 0;     // status == 0 on the CPU
    int partial_path = 0;  // CPU saw at least one dead reference pixel
    int total_iters = 0;
};

Tally compare(const Reference &r, const std::vector<IcgnGpuResult> &gpu,
              const char *label) {
    Tally t;
    for (size_t i = 0; i < gpu.size(); ++i) {
        const IcgnGpuResult &b = gpu[i];

        if (r.host_only[i]) {
            // The device must refuse precisely what the host would have had
            // to redo anyway. A device that "solved" one of these would be
            // solving a subset the CPU never built.
            if (b.status != kIcgnHostRequired) {
                if (t.mismatched < 6)
                    std::printf("  [%s] idx %zu: device solved a point Path A skips"
                                " (status %d)\n", label, i, b.status);
                ++t.mismatched;
            } else {
                ++t.refused;
            }
            continue;
        }

        const AnalysisResult &a = r.cpu[i];
        ++t.compared;
        if (a.status == 0) ++t.converged;
        if (a.invalid_ref_pixels > 0) ++t.partial_path;
        t.total_iters += a.iters;

        const float ap[6] = {a.u, a.v, a.ux, a.uy, a.vx, a.vy};
        const char *why = nullptr;
        for (int k = 0; k < 6 && !why; ++k)
            if (ap[k] != b.p[k]) why = "parameter";
        if (!why && a.correlation_score != b.score) why = "score";
        if (!why && a.status != b.status) why = "status";
        // Equal answers reached in different iteration counts mean the
        // convergence test diverged; the agreement would be coincidence.
        if (!why && a.iters != b.iters) why = "iters";
        if (!why && a.invalid_ref_pixels != b.invalid_ref_pixels) why = "invalid_ref";
        if (!why) continue;

        if (t.mismatched < 6)
            std::printf("  [%s] idx %zu (%d,%d): %s differs\n"
                        "        cpu u=%.9g v=%.9g ux=%.9g uy=%.9g vx=%.9g vy=%.9g"
                        " score=%.9g st=%d it=%d ir=%d\n"
                        "        gpu u=%.9g v=%.9g ux=%.9g uy=%.9g vx=%.9g vy=%.9g"
                        " score=%.9g st=%d it=%d ir=%d\n",
                        label, i, r.pts[i].cx, r.pts[i].cy, why,
                        (double) a.u, (double) a.v, (double) a.ux, (double) a.uy,
                        (double) a.vx, (double) a.vy, (double) a.correlation_score,
                        a.status, a.iters, a.invalid_ref_pixels,
                        (double) b.p[0], (double) b.p[1], (double) b.p[2],
                        (double) b.p[3], (double) b.p[4], (double) b.p[5],
                        (double) b.score, b.status, b.iters, b.invalid_ref_pixels);
        ++t.mismatched;
    }
    return t;
}

bool run_gpu(const Scene &s, const Grid &g, const Reference &r,
             std::vector<IcgnGpuResult> *out) {
    out->assign((size_t) g.count(), IcgnGpuResult());
    return solve_icgn_batch_gpu(s.ref, s.def, g.dim, g.keys6,
                                kMaxIter, kLmAlpha, r.pts.data(), g.count(),
                                out->data());
}

} // namespace

// The gate itself, and the one parity test that runs everywhere. A machine
// with no device must get a clean false and an untouched output buffer — a
// dispatch that half-filled it before failing would corrupt the points the
// caller then solves on the CPU.
TEST_CASE(ClParity, IcgnDispatchDeclinesCleanlyWhenStageUnavailable) {
    const Scene s = make_scene(160, 140, 8801, 700);
    const Grid g{30, 30, 9, 6, 5, 21};
    const Reference r = build_reference(s, g);

    std::vector<IcgnGpuResult> out((size_t) g.count());
    for (IcgnGpuResult &o : out) {
        o.score = 7.5f;          // recognisable prior contents
        o.status = 99;
        o.iters = 42;
        o.p[0] = -3.25f;
    }

    const bool ran = solve_icgn_batch_gpu(s.ref, s.def, g.dim, false, kMaxIter,
                                          kLmAlpha, r.pts.data(), g.count(),
                                          out.data());
    if (!stage_available()) {
        CHECK(!ran);
        // Untouched, not merely "not garbage".
        CHECK(out[0].score == 7.5f);
        CHECK(out[0].status == 99);
        CHECK(out[out.size() - 1].iters == 42);
        CHECK(out[out.size() - 1].p[0] == -3.25f);
        report_skip("icgn parity (gate test still ran)");
    } else {
        CHECK(ran);
    }
}

// Null arguments and nonsense geometry must be refused rather than
// dereferenced or launched.
TEST_CASE(ClParity, IcgnDispatchRejectsBadArguments) {
    const Scene s = make_scene(120, 110, 991, 400);
    std::vector<IcgnGpuPoint> pts(8);
    std::vector<IcgnGpuResult> out(8);
    for (IcgnGpuPoint &p : pts) { p.cx = 40; p.cy = 40; }

    CHECK(!solve_icgn_batch_gpu(s.ref, s.def, 21, false, kMaxIter, kLmAlpha,
                                nullptr, 8, out.data()));
    CHECK(!solve_icgn_batch_gpu(s.ref, s.def, 21, false, kMaxIter, kLmAlpha,
                                pts.data(), 8, nullptr));
    CHECK(!solve_icgn_batch_gpu(s.ref, s.def, 21, false, kMaxIter, kLmAlpha,
                                pts.data(), 0, out.data()));
    CHECK(!solve_icgn_batch_gpu(s.ref, s.def, 0, false, kMaxIter, kLmAlpha,
                                pts.data(), 8, out.data()));
    CHECK(!solve_icgn_batch_gpu(s.ref, s.def, 21, false, 0, kLmAlpha,
                                pts.data(), 8, out.data()));
}

// The env override is the documented way to force the CPU path (there is no
// FullFieldParams field for it — see docs/CONTRACT.md). It must shut this
// stage down too, not just the probe and the earlier kernels.
TEST_CASE(ClParity, IcgnDispatchHonoursEnvDisable) {
    reset_for_testing();
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "1");
#else
    setenv("SEMPER_OPENCL_DISABLE", "1", 1);
#endif
    const Scene s = make_scene(160, 140, 8801, 700);
    const Grid g{30, 30, 9, 6, 5, 21};
    const Reference r = build_reference(s, g);
    std::vector<IcgnGpuResult> out;
    const bool ran = run_gpu(s, g, r, &out);
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "");
#else
    unsetenv("SEMPER_OPENCL_DISABLE");
#endif
    reset_for_testing();

    CHECK(!ran);
}

// The embedded source must be self-contained, must keep contraction off,
// and must be free of any double: this stage is gated on exact_fp32 and has
// to build on a device with no cl_khr_fp64 at all. A stray double would make
// it fail to compile there, and since a program build failure must not flip
// caps().available, the symptom would be a silent CPU fallback.
TEST_CASE(ClParity, EmbeddedIcgnKernelIsSelfContained) {
    const std::string src = Semper::gpu::kernels::ICGN_SOLVE;
    REQUIRE(!src.empty());
    const size_t entry = src.find("__kernel void semper_icgn_solve");
    REQUIRE(entry != std::string::npos);
    CHECK(src.find("#include <semper/") == std::string::npos);
    CHECK(src.find("FP_CONTRACT OFF") != std::string::npos);
    // The shared routines the CPU reference also calls. The _g sampler and
    // the _g reductions are the reason canonical_math.h expands its includes
    // once per address space at all.
    CHECK(src.find("semper_canon_sample_g") != std::string::npos);
    CHECK(src.find("semper_canon_znssd_error_and_gradient_g") != std::string::npos);
    CHECK(src.find("semper_canon_sum_sq_diff_g") != std::string::npos);
    CHECK(src.find("semper_inv6x6") != std::string::npos);
    CHECK(src.find("semper_inv3x3f") != std::string::npos);
    CHECK(src.find("semper_norm6") != std::string::npos);
    // No cross-lane reduction may creep in: that would reassociate the sums
    // this whole kernel exists to keep in order.
    CHECK(src.find("barrier(") == std::string::npos);
    CHECK(src.find("__local") == std::string::npos);
    // The header's fp64 block is conditional...
    CHECK(src.find("SEMPER_HAS_FP64") != std::string::npos);
    // ...and this kernel is outside it.
    CHECK(src.find("double", entry) == std::string::npos);
}

// The main event: several geometries, every point solved both ways.
TEST_CASE(ClParity, IcgnMatchesCpuBitExactly) {
    if (!stage_available()) { report_skip("IcgnMatchesCpuBitExactly"); return; }

    struct Case { Grid g; unsigned seed; int blobs; int w, h; const char *name; };
    const Case cases[] = {
        {{30, 30,  9, 12, 10, 21}, 8801, 900,  220, 200, "12x10 step9 dim21"},
        {{26, 24,  7, 14, 13, 31}, 4127, 1100, 240, 220, "14x13 step7 dim31"},
        {{20, 20, 11,  9,  8, 15}, 6553, 800,  200, 180, "9x8 step11 dim15"},
        // Runs off the right and bottom edges, so a band of subsets never
        // initialises and must be handed back.
        {{60, 55, 13, 12, 12, 25}, 3313, 900,  220, 200, "12x12 step13 dim25 (edge overrun)"},
    };

    for (const Case &c : cases) {
        const Scene s = make_scene(c.w, c.h, c.seed, c.blobs);
        const Reference r = build_reference(s, c.g);
        std::vector<IcgnGpuResult> gpu;
        REQUIRE(run_gpu(s, c.g, r, &gpu));

        const Tally t = compare(r, gpu, c.name);
        std::printf("  ClParity[%s]: %d compared (%d converged, %d iters total),"
                    " %d refused, %d mismatch\n",
                    c.name, t.compared, t.converged, t.total_iters, t.refused,
                    t.mismatched);
        // A case where nothing solved, or where every point converged on its
        // first step, would compare equal while exercising almost nothing.
        CHECK(t.compared > 0);
        CHECK(t.converged > 0);
        CHECK(t.total_iters > t.compared);
        CHECK(t.mismatched == 0);
    }
}

// The other interpolator. FullFieldParams::use_6x6_interpolator swaps the
// deformed-image sampler on both sides, and the Keys 6x6 kernel is a
// different polynomial with a different pixel footprint and a different
// summation order -- so it needs its own parity evidence rather than
// inheriting the bicubic result. Path A passes this flag straight through,
// which is why the pipeline may hand it to the device at all.
TEST_CASE(ClParity, IcgnKeysInterpolatorMatchesCpuBitExactly) {
    if (!stage_available()) { report_skip("IcgnKeysInterpolatorMatchesCpuBitExactly"); return; }

    const Scene s = make_scene(220, 200, 8801, 900);
    Grid g{30, 30, 9, 12, 10, 21};
    g.keys6 = true;
    const Reference r = build_reference(s, g);
    std::vector<IcgnGpuResult> gpu;
    REQUIRE(run_gpu(s, g, r, &gpu));

    const Tally t = compare(r, gpu, "keys6");
    std::printf("  ClParity[keys6]: %d compared (%d converged, %d iters total),"
                " %d refused, %d mismatch\n",
                t.compared, t.converged, t.total_iters, t.refused, t.mismatched);
    CHECK(t.compared > 0);
    CHECK(t.converged > 0);
    CHECK(t.total_iters > t.compared);
    CHECK(t.mismatched == 0);
}

// The partial path, which is where the two implementations are most likely
// to disagree: with pixels missing, neither side can use the canonical
// 4-accumulator reductions, both fall back to a sequential scan, and the
// Hessian is rebuilt per iteration from the survivors. This also exercises
// invalid_ref_pixels, which is 0 everywhere in the test above.
TEST_CASE(ClParity, IcgnPartialPathMatchesCpuBitExactly) {
    if (!stage_available()) { report_skip("IcgnPartialPathMatchesCpuBitExactly"); return; }

    Scene s = make_scene(220, 200, 5150, 950);
    // Stamp ghost walls straight into the prepared reference buffer. Both
    // paths read the same array, so this is a legitimate input, and it is
    // the only way to reach the -5.0f branch from a speckle field that
    // clamps to [5, 250]. Several thin strips rather than one block, so
    // subsets are clipped by varying amounts and land on both sides of the
    // 90%-survival abort.
    for (int y = 40; y < 170; ++y)
        for (int x = 40; x < 170; ++x)
            if (((x / 7) % 5) == 0) s.ref.intensities[(size_t) y * 220 + x] = -10.0f;

    const Grid g{45, 45, 6, 20, 20, 21};
    const Reference r = build_reference(s, g);
    std::vector<IcgnGpuResult> gpu;
    REQUIRE(run_gpu(s, g, r, &gpu));

    const Tally t = compare(r, gpu, "ghost wall");
    std::printf("  ClParity[ghost wall]: %d compared (%d converged, %d on the"
                " partial path), %d refused, %d mismatch\n",
                t.compared, t.converged, t.partial_path, t.refused, t.mismatched);
    CHECK(t.compared > 0);
    // If no point lost a reference pixel, this test is the previous one.
    CHECK(t.partial_path > 0);
    CHECK(t.mismatched == 0);
}

// A launch large enough that the dispatch has to split it. Tiling is pure
// bookkeeping — every subset is still solved entirely inside one work-item —
// but a wrong offset into the staged results would be invisible in the
// smaller tests above, where everything fits in a single tile.
TEST_CASE(ClParity, IcgnTiledLaunchMatchesCpuBitExactly) {
    if (!stage_available()) { report_skip("IcgnTiledLaunchMatchesCpuBitExactly"); return; }

    // dim 31 is 961 px, so scratch is ~31.7 KB per point and the dispatch's
    // 64 MB budget holds a little over 2000 of them. 2209 points therefore
    // crosses into a second tile.
    const Scene s = make_scene(400, 400, 7717, 2600);
    const Grid g{16, 16, 8, 47, 47, 31};
    REQUIRE(g.count() == 2209);

    const Reference r = build_reference(s, g);
    std::vector<IcgnGpuResult> gpu;
    REQUIRE(run_gpu(s, g, r, &gpu));

    const Tally t = compare(r, gpu, "tiled");
    std::printf("  ClParity[tiled]: %d points, %d compared, %d refused, %d mismatch\n",
                g.count(), t.compared, t.refused, t.mismatched);
    CHECK(t.compared > 2048);   // more than one tile's worth actually solved
    CHECK(t.mismatched == 0);
}

// Same input twice must give the same bytes. A kernel whose result depended
// on work-group scheduling would still pass the parity tests intermittently.
TEST_CASE(ClParity, IcgnIsRunToRunReproducible) {
    if (!stage_available()) { report_skip("IcgnIsRunToRunReproducible"); return; }

    const Scene s = make_scene(240, 220, 4127, 1100);
    const Grid g{26, 24, 7, 14, 13, 31};
    const Reference r = build_reference(s, g);

    std::vector<IcgnGpuResult> a, b;
    REQUIRE(run_gpu(s, g, r, &a));
    REQUIRE(run_gpu(s, g, r, &b));

    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        bool same = a[i].score == b[i].score && a[i].status == b[i].status &&
                    a[i].iters == b[i].iters &&
                    a[i].invalid_ref_pixels == b[i].invalid_ref_pixels;
        for (int k = 0; k < 6 && same; ++k)
            if (a[i].p[k] != b[i].p[k]) same = false;
        if (!same) ++bad;
    }
    CHECK(bad == 0);
}

#else // !SEMPER_OPENCL

TEST_CASE(ClParity, IcgnKernelAbsentWithoutOpenCLBuild) {
    std::printf("  ClParity: icgn solve compiled out (SEMPER_OPENCL=OFF)\n");
    CHECK(true);
}

#endif // SEMPER_OPENCL
