// Solver throughput — precompute and solve rates, printed for tracking.
//
// NOT a benchmark gate. CI runners are noisy and this same binary is built
// under ASan/UBSan/TSan, where everything is several times slower, so a
// subsets/sec assertion would be flaky and meaningless. The measured rate is
// printed for humans to compare across commits; the only assertion is a
// generous wall-clock ceiling that catches a hang or catastrophic regression.
//
// For real benchmarking, run this locally on a quiet machine.
#include "framework/test_framework.h"
#include "framework/synthetic.h"
#include <semper/solver.hpp>
#include <semper/subset.hpp>

#include <chrono>
#include <cstdio>

using Semper::Image;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;
using Semper::OptimizationEngine;
using Semper::AnalysisResult;
using Semper::INIT_NO_SIMPLEX;

namespace {
    constexpr int W = 512, H = 512;
    constexpr int SUBSET_SIZE = 27;
    constexpr int GRID_LO = 60, GRID_HI = 452, GRID_STEP = 28; // 15x15 = 225 subsets

    // Generous: ~225 subsets should take well under a second natively, and a
    // few seconds even under TSan. 120 s only trips on a hang or a pathological
    // regression.
    constexpr double WALL_CLOCK_CEILING_S = 120.0;
} // namespace

TEST_CASE(Perf, SubsetSolveThroughput) {
    dictest::SpeckleField field(/*seed=*/7, W, H);
    dictest::AffineDeformation def;
    def.u = 0.35f;
    def.v = -0.20f;
    def.cx = W / 2.0f;
    def.cy = H / 2.0f;

    const Image ref = dictest::make_reference_image(field, W, H);
    const Image deformed = dictest::make_deformed_image(field, W, H, def);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    int precomputed = 0, solved = 0;
    double precompute_s = 0.0, solve_s = 0.0;
    for (int y = GRID_LO; y <= GRID_HI; y += GRID_STEP) {
        for (int x = GRID_LO; x <= GRID_HI; x += GRID_STEP) {
            const auto tp = clock::now();
            SubsetData subset;
            SubsetPrecomputer::precompute_subset(subset, ref, x, y, SUBSET_SIZE);
            precompute_s += std::chrono::duration<double>(clock::now() - tp).count();
            if (!subset.is_initialized) continue;
            ++precomputed;

            const auto ts = clock::now();
            OptimizationEngine engine;
            AnalysisResult res = engine.calculate_deformation(
                subset, deformed, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);
            solve_s += std::chrono::duration<double>(clock::now() - ts).count();
            if (res.status == 0) ++solved;
        }
    }

    const double total_s = std::chrono::duration<double>(clock::now() - t0).count();
    REQUIRE(solved > 0);

    std::printf("  Perf: %d subsets precomputed, %d solved in %.3f s\n"
                "     precompute %.3f s (%.0f subsets/s) | solve %.3f s (%.0f solves/s)\n"
                "     %.3f ms per solve\n",
                precomputed, solved, total_s,
                precompute_s, precomputed / (precompute_s > 0 ? precompute_s : 1.0),
                solve_s, solved / (solve_s > 0 ? solve_s : 1.0),
                1000.0 * solve_s / solved);

    // Smoke-level guard only — see the header comment.
    CHECK(total_s < WALL_CLOCK_CEILING_S);
}

// ---------------------------------------------------------------------------
// GPU Phase 2: VSG strain, CPU vs device, swept across grid sizes.
//
// Same caveats as above -- printed, not asserted. Two things make this one
// worth keeping. The paths are bit-identical (tests/unit/test_cl_strain.cpp
// checks that with ==), so the ratio is a clean measure of moving to the
// device rather than of two algorithms trading accuracy for speed. And it is
// swept rather than measured at one size, because the answer is not a single
// number: dispatch costs a fixed amount in buffer traffic and launch latency,
// so the device loses on small grids and wins on large ones. The break-even
// printed here is what the size threshold in
// src/pipeline/full_field_solver_stats.cpp is set from -- re-run this after
// touching the kernel or the transfers, and move the constant if it shifts.
//
// Timing includes the host-side buffer traffic and the std::vector<bool>
// unpack, because that is what the caller actually pays.
// ---------------------------------------------------------------------------
#if defined(SEMPER_OPENCL)
#include "gpu/cl_runtime.hpp"
#include "gpu/strain_dispatch.hpp"
#include <semper/strain.hpp>

namespace {

Semper::DisplacementField strain_bench_field(int n, int step) {
    Semper::DisplacementField f;
    f.width = n; f.height = n; f.step = step;
    f.u.resize((size_t) n * n); f.v.resize((size_t) n * n);
    f.valid.assign((size_t) n * n, true);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const float X = (float) (x * step), Y = (float) (y * step);
            const size_t i = (size_t) y * n + x;
            f.u[i] = 0.011f * X + 0.0037f * Y + 0.00004f * X * X;
            f.v[i] = -0.0061f * X + 0.0092f * Y - 0.00003f * Y * Y;
        }
    }
    return f;
}

} // namespace

TEST_CASE(Perf, StrainVsgThroughputCpuVsGpu) {
    const int step = 11, window = 44;
    const int sizes[] = {47, 60, 78, 96, 192};   // 2.2k .. 37k grid points
    const int reps = 20;

    using clock = std::chrono::steady_clock;
    const auto &c = Semper::gpu::caps();
    const bool have_device = c.available && c.fp64;

    if (have_device)
        std::printf("  Perf: strain VSG on '%s' (step %d, window %d)\n",
                    c.device_name.c_str(), step, window);
    else
        std::printf("  Perf: strain VSG, CPU only -- %s\n",
                    c.available ? "device has no cl_khr_fp64"
                                : c.unavailable_reason.c_str());

    double worst_case_s = 0.0;
    for (int n : sizes) {
        const Semper::DisplacementField f = strain_bench_field(n, step);

        auto t0 = clock::now();
        for (int i = 0; i < reps; ++i)
            (void) Semper::StrainCalculator::compute_vsg_strain(f, window);
        const double cpu_s =
                std::chrono::duration<double>(clock::now() - t0).count() / reps;
        if (cpu_s > worst_case_s) worst_case_s = cpu_s;

        if (!have_device) {
            std::printf("     %5d pts: CPU %7.3f ms\n", n * n, 1000.0 * cpu_s);
            continue;
        }

        // One untimed call first: the device program is compiled lazily and
        // cached, and charging this stage for a one-time LLVM run would say
        // nothing about steady-state throughput.
        Semper::StrainField warm;
        REQUIRE(Semper::gpu::compute_vsg_strain_gpu(f, window, &warm));

        t0 = clock::now();
        for (int i = 0; i < reps; ++i) {
            Semper::StrainField g;
            REQUIRE(Semper::gpu::compute_vsg_strain_gpu(f, window, &g));
        }
        const double gpu_s =
                std::chrono::duration<double>(clock::now() - t0).count() / reps;
        if (gpu_s > worst_case_s) worst_case_s = gpu_s;

        std::printf("     %5d pts: CPU %7.3f ms | GPU %7.3f ms | %5.2fx %s\n",
                    n * n, 1000.0 * cpu_s, 1000.0 * gpu_s,
                    cpu_s / (gpu_s > 0 ? gpu_s : 1.0),
                    cpu_s > gpu_s ? "GPU" : "CPU");
    }

    // Smoke-level guard only -- see the header comment.
    CHECK(worst_case_s < WALL_CLOCK_CEILING_S);
}
#endif // SEMPER_OPENCL

// ---------------------------------------------------------------------------
// GPU Phase 3: the static Hessian pre-pass, CPU vs device, swept across grid
// sizes at a fixed image size, then the fixed dispatch cost swept across
// image sizes.
//
// Printed, not asserted, for the same reasons as the sweep above. Two things
// to read carefully before trusting the ratio:
//
//   * The CPU column here is SERIAL. In the pipeline the pre-pass is an
//     `#pragma omp parallel for` over safe_cores (full_field_solver.cpp), and
//     this binary does not link OpenMP. Divide it by the core count to get
//     what the caller actually races against -- which is why the threshold in
//     full_field_solver.cpp sits well above the naive crossover below.
//   * The GPU column carries the whole reference image every call:
//     intensities, grad_x, grad_y, three float planes. That cost scales with
//     the IMAGE, not the grid, so a big image with a coarse grid is the shape
//     that loses on the device. The second sweep isolates it, and the
//     caller's threshold is expressed per megapixel because of it.
//
// Re-run after touching the kernel or the transfers, and move the constant if
// the crossover shifts.
// ---------------------------------------------------------------------------
#if defined(SEMPER_OPENCL)
#include "gpu/hessian_dispatch.hpp"

TEST_CASE(Perf, HessianPrepassThroughputCpuVsGpu) {
    constexpr int IW = 1024, IH = 1024;
    const int dim = 21, margin = 32;
    const int steps[] = {19, 13, 9, 6, 4};     // ~2.5k .. ~58k grid points
    const int reps = 10;

    dictest::SpeckleField field(/*seed=*/91, IW, IH);
    const Image img = dictest::make_reference_image(field, IW, IH);

    using clock = std::chrono::steady_clock;
    const auto &c = Semper::gpu::caps();
    // fp32 end to end: this stage wants correctly-rounded divide/sqrt, not fp64.
    const bool have_device = c.available && c.exact_fp32;

    if (have_device)
        std::printf("  Perf: Hessian pre-pass on '%s' (%dx%d image, subset %d)\n",
                    c.device_name.c_str(), IW, IH, dim);
    else
        std::printf("  Perf: Hessian pre-pass, CPU only -- %s\n",
                    c.available ? "device lacks correctly-rounded fp32"
                                : c.unavailable_reason.c_str());

    double worst_case_s = 0.0;
    for (int step : steps) {
        const int gw = (IW - 2 * margin) / step, gh = (IH - 2 * margin) / step;
        const int n = gw * gh;
        const std::vector<unsigned char> wanted((size_t) n, 1u);
        std::vector<Semper::CachedHessianData> pool((size_t) n);

        auto t0 = clock::now();
        for (int r = 0; r < reps; ++r)
            for (int i = 0; i < n; ++i)
                pool[(size_t) i] = SubsetPrecomputer::compute_hessian_only(
                        img, margin + (i % gw) * step, margin + (i / gw) * step, dim);
        const double cpu_s =
                std::chrono::duration<double>(clock::now() - t0).count() / reps;
        if (cpu_s > worst_case_s) worst_case_s = cpu_s;

        if (!have_device) {
            std::printf("     %6d pts (step %2d): CPU-serial %8.3f ms\n",
                        n, step, 1000.0 * cpu_s);
            continue;
        }

        // One untimed call first: the device program is built lazily and
        // cached, and charging this stage for a one-time compile would say
        // nothing about steady-state throughput.
        REQUIRE(Semper::gpu::compute_hessian_pool_gpu(img, margin, margin, step, gw, gh,
                                                      dim, wanted.data(), pool.data(), n));
        t0 = clock::now();
        for (int r = 0; r < reps; ++r)
            REQUIRE(Semper::gpu::compute_hessian_pool_gpu(img, margin, margin, step, gw, gh,
                                                          dim, wanted.data(), pool.data(), n));
        const double gpu_s =
                std::chrono::duration<double>(clock::now() - t0).count() / reps;
        if (gpu_s > worst_case_s) worst_case_s = gpu_s;

        std::printf("     %6d pts (step %2d): CPU-serial %8.3f ms | GPU %7.3f ms | %6.2fx\n",
                    n, step, 1000.0 * cpu_s, 1000.0 * gpu_s,
                    cpu_s / (gpu_s > 0 ? gpu_s : 1.0));
    }

    // The fixed half of the cost, isolated: a one-point grid does no
    // meaningful kernel work, so what remains is launch latency and the three
    // image planes crossing the bus.
    if (have_device) {
        const int img_sizes[] = {512, 1024, 2048};
        const std::vector<unsigned char> one(1, 1u);
        for (int s : img_sizes) {
            dictest::SpeckleField f2(/*seed=*/91, s, s);
            const Image im = dictest::make_reference_image(f2, s, s);
            std::vector<Semper::CachedHessianData> p1(1);
            REQUIRE(Semper::gpu::compute_hessian_pool_gpu(im, s / 2, s / 2, 1, 1, 1, dim,
                                                          one.data(), p1.data(), 1));
            const auto tf = clock::now();
            for (int r = 0; r < reps; ++r)
                REQUIRE(Semper::gpu::compute_hessian_pool_gpu(im, s / 2, s / 2, 1, 1, 1, dim,
                                                              one.data(), p1.data(), 1));
            const double fixed_s =
                    std::chrono::duration<double>(clock::now() - tf).count() / reps;
            const double mp = (double) s * s / 1.0e6;
            std::printf("     dispatch fixed cost, %4dx%4d image: %6.3f ms (%6.3f ms/MP)\n",
                        s, s, 1000.0 * fixed_s, 1000.0 * fixed_s / mp);
        }
    }

    // Smoke-level guard only -- see the header comment.
    CHECK(worst_case_s < WALL_CLOCK_CEILING_S);
}
#endif // SEMPER_OPENCL


// ---------------------------------------------------------------------------
// Phase 4 throughput: Path A ICGN, CPU vs device.
//
// Same shape as the pre-pass case above, and the same caveats -- the CPU side
// is SERIAL because this binary does not link OpenMP, so the printed ratio is
// an upper bound on what the pipeline sees. The number that sets the caller's
// threshold in full_field_path_a.cpp is the per-point device cost against a
// SIX-CORE CPU pre-pass, which ClPipelineParity measures end to end.
//
// The CPU reference is precompute_subset_fast + calculate_deformation under
// INIT_NO_SIMPLEX: exactly the work the kernel replaces, with the Nelder-Mead
// rescue excluded on both sides because the kernel never runs it.
// ---------------------------------------------------------------------------
#if defined(SEMPER_OPENCL)
#include "gpu/icgn_dispatch.hpp"
#include <semper/tuning.hpp>

TEST_CASE(Perf, IcgnPathAThroughputCpuVsGpu) {
    constexpr int IW = 1024, IH = 1024;
    const int dim = 21, margin = 32;
    const int steps[] = {19, 13, 9, 6, 4};  // ~2.5k .. ~58k grid points
    // The serial CPU rate is flat in n (see the printout: it varies by under
    // 2% across the whole sweep), so it is measured on the two smallest
    // grids and the rest are timed on the device only. Timing 58k serial
    // solves would add half a minute to the run and tell us nothing new.
    const int cpu_steps_measured = 2;
    const int reps = 3;

    dictest::SpeckleField field(/*seed=*/57, IW, IH);
    dictest::AffineDeformation d;
    d.u = 1.75f; d.v = -1.25f;
    d.ux = 0.006f; d.vy = -0.004f; d.uy = 0.002f;
    d.cx = IW / 2.0f; d.cy = IH / 2.0f;
    const Image ref = dictest::make_reference_image(field, IW, IH);
    const Image def = dictest::make_deformed_image(field, IW, IH, d);

    using clock = std::chrono::steady_clock;
    const auto &c = Semper::gpu::caps();
    const bool have_device = c.available && c.exact_fp32;

    if (have_device)
        std::printf("  Perf: Path A ICGN on '%s' (%dx%d image, subset %d)\n",
                    c.device_name.c_str(), IW, IH, dim);
    else
        std::printf("  Perf: Path A ICGN, CPU only -- %s\n",
                    c.available ? "device lacks correctly-rounded fp32"
                                : c.unavailable_reason.c_str());

    double worst_case_s = 0.0;
    double cpu_us_per_pt = 0.0;
    int step_index = 0;
    for (int step : steps) {
        const bool time_cpu = (step_index++ < cpu_steps_measured);
        const int gw = (IW - 2 * margin) / step, gh = (IH - 2 * margin) / step;
        const int n = gw * gh;

        std::vector<Semper::CachedHessianData> pool((size_t) n);
        std::vector<Semper::gpu::IcgnGpuPoint> pts((size_t) n);
        for (int i = 0; i < n; ++i) {
            const int cx = margin + (i % gw) * step, cy = margin + (i / gw) * step;
            pool[(size_t) i] = SubsetPrecomputer::compute_hessian_only(ref, cx, cy, dim);
            Semper::gpu::IcgnGpuPoint &pt = pts[(size_t) i];
            pt.cx = cx;
            pt.cy = cy;
            pt.cached = &pool[(size_t) i];
            // Offset from the truth so ICGN actually iterates; a one-step
            // convergence everywhere would time almost nothing.
            const float ddx = (float) cx - d.cx, ddy = (float) cy - d.cy;
            pt.guess[0] = d.u + d.ux * ddx + d.uy * ddy - 0.35f;
            pt.guess[1] = d.v + d.vx * ddx + d.vy * ddy + 0.28f;
        }

        OptimizationEngine eng;
        eng.lm_enabled = true;
        eng.lm_alpha = Semper::tuning::kLmAlpha;
        eng.use_6x6_interpolator = false;
        SubsetData subset;

        auto t0 = clock::now();
        double cpu_s;
        if (time_cpu) {
            for (int r = 0; r < reps; ++r)
                for (int i = 0; i < n; ++i) {
                    SubsetPrecomputer::precompute_subset_fast(
                            subset, ref, pts[(size_t) i].cx, pts[(size_t) i].cy, dim,
                            pool[(size_t) i]);
                    if (!subset.is_initialized) continue;
                    AnalysisResult a = eng.calculate_deformation(
                            subset, def, pts[(size_t) i].guess[0], pts[(size_t) i].guess[1],
                            0.0f, 0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);
                    (void) a;
                }
            cpu_s = std::chrono::duration<double>(clock::now() - t0).count() / reps;
            cpu_us_per_pt = 1.0e6 * cpu_s / n;
            if (cpu_s > worst_case_s) worst_case_s = cpu_s;
        } else {
            cpu_s = cpu_us_per_pt * n / 1.0e6;   // extrapolated at the flat rate
        }

        if (!have_device) {
            std::printf("     %6d pts (step %2d): CPU-serial %8.3f ms%s\n",
                        n, step, 1000.0 * cpu_s, time_cpu ? "" : " (extrapolated)");
            continue;
        }

        std::vector<Semper::gpu::IcgnGpuResult> out((size_t) n);
        // Untimed first call: the program is built lazily and cached once.
        REQUIRE(Semper::gpu::solve_icgn_batch_gpu(
                ref, def, dim, false, Semper::tuning::kIcgnMaxIter,
                Semper::tuning::kLmAlpha, pts.data(), n, out.data()));
        t0 = clock::now();
        for (int r = 0; r < reps; ++r)
            REQUIRE(Semper::gpu::solve_icgn_batch_gpu(
                    ref, def, dim, false, Semper::tuning::kIcgnMaxIter,
                    Semper::tuning::kLmAlpha, pts.data(), n, out.data()));
        const double gpu_s =
                std::chrono::duration<double>(clock::now() - t0).count() / reps;
        if (gpu_s > worst_case_s) worst_case_s = gpu_s;

        std::printf("     %6d pts (step %2d): CPU-serial %8.3f ms%s | GPU %7.3f ms | %6.2fx"
                    " (%.3f us/pt CPU, %.3f us/pt GPU)\n",
                    n, step, 1000.0 * cpu_s, time_cpu ? "  " : "* ", 1000.0 * gpu_s,
                    cpu_s / (gpu_s > 0 ? gpu_s : 1.0),
                    1.0e6 * cpu_s / n, 1.0e6 * gpu_s / n);
    }

    // Where the per-point cost actually goes. Every ICGN iteration re-reads
    // this point's 8n floats of __global scratch, so capping the iteration
    // count separates the one-time per-point setup -- the normalised
    // reference, the steepest-descent planes, the damped 6x6 inversion --
    // from the per-iteration warp+interpolate+reduce. The slope is the
    // per-iteration cost and the intercept is the setup; which one dominates
    // decides what an optimisation pass would have to attack.
    if (have_device) {
        const int step = 9;
        const int gw = (IW - 2 * margin) / step, gh = (IH - 2 * margin) / step;
        const int n = gw * gh;
        std::vector<Semper::CachedHessianData> pool((size_t) n);
        std::vector<Semper::gpu::IcgnGpuPoint> pts((size_t) n);
        for (int i = 0; i < n; ++i) {
            const int cx = margin + (i % gw) * step, cy = margin + (i / gw) * step;
            pool[(size_t) i] = SubsetPrecomputer::compute_hessian_only(ref, cx, cy, dim);
            pts[(size_t) i].cx = cx;
            pts[(size_t) i].cy = cy;
            pts[(size_t) i].cached = &pool[(size_t) i];
            const float ddx = (float) cx - d.cx, ddy = (float) cy - d.cy;
            pts[(size_t) i].guess[0] = d.u + d.ux * ddx + d.uy * ddy - 0.35f;
            pts[(size_t) i].guess[1] = d.v + d.vx * ddx + d.vy * ddy + 0.28f;
        }
        std::vector<Semper::gpu::IcgnGpuResult> out((size_t) n);
        const int caps_iter[] = {1, 2, 4, 8, Semper::tuning::kIcgnMaxIter};
        for (int mi : caps_iter) {
            REQUIRE(Semper::gpu::solve_icgn_batch_gpu(
                    ref, def, dim, false, mi, Semper::tuning::kLmAlpha,
                    pts.data(), n, out.data()));
            const auto tc = clock::now();
            for (int r = 0; r < reps; ++r)
                REQUIRE(Semper::gpu::solve_icgn_batch_gpu(
                        ref, def, dim, false, mi, Semper::tuning::kLmAlpha,
                        pts.data(), n, out.data()));
            const double s_ = std::chrono::duration<double>(clock::now() - tc).count() / reps;
            std::printf("     iteration cap %2d, %d pts: %8.3f ms (%.3f us/pt)\n",
                        mi, n, 1000.0 * s_, 1.0e6 * s_ / n);
        }
    }

    // The fixed half of the cost, isolated: a one-point launch does no
    // meaningful kernel work, so what remains is launch latency plus the
    // three reference planes and the deformed image crossing the bus. This
    // is what the per-megapixel term in the caller's threshold pays for.
    if (have_device) {
        const int img_sizes[] = {512, 1024, 2048};
        for (int sz : img_sizes) {
            dictest::SpeckleField f2(/*seed=*/57, sz, sz);
            const Image r2 = dictest::make_reference_image(f2, sz, sz);
            const Image d2 = dictest::make_deformed_image(f2, sz, sz, d);
            std::vector<Semper::CachedHessianData> p1(1);
            p1[0] = SubsetPrecomputer::compute_hessian_only(r2, sz / 2, sz / 2, dim);
            std::vector<Semper::gpu::IcgnGpuPoint> one(1);
            one[0].cx = sz / 2;
            one[0].cy = sz / 2;
            one[0].cached = &p1[0];
            std::vector<Semper::gpu::IcgnGpuResult> o1(1);
            REQUIRE(Semper::gpu::solve_icgn_batch_gpu(
                    r2, d2, dim, false, Semper::tuning::kIcgnMaxIter,
                    Semper::tuning::kLmAlpha, one.data(), 1, o1.data()));
            const auto tf = clock::now();
            for (int r = 0; r < reps; ++r)
                REQUIRE(Semper::gpu::solve_icgn_batch_gpu(
                        r2, d2, dim, false, Semper::tuning::kIcgnMaxIter,
                        Semper::tuning::kLmAlpha, one.data(), 1, o1.data()));
            const double fixed_s =
                    std::chrono::duration<double>(clock::now() - tf).count() / reps;
            const double mp = (double) sz * sz / 1.0e6;
            std::printf("     dispatch fixed cost, %4dx%4d image: %6.3f ms (%6.3f ms/MP)\n",
                        sz, sz, 1000.0 * fixed_s, 1000.0 * fixed_s / mp);
        }
    }

    // Smoke-level guard only -- see the header comment.
    CHECK(worst_case_s < WALL_CLOCK_CEILING_S);
}
#endif // SEMPER_OPENCL


// ---------------------------------------------------------------------------
// Phase 4 throughput through the REAL caller.
//
// The case above times solve_icgn_batch_gpu against a SERIAL CPU loop, which
// answers "is the kernel faster than one core" -- a question nobody has.
// Path A runs on hardware_concurrency() threads, so the number that decides
// whether the device is worth engaging is the whole-solve wall clock with and
// without it, on this machine, at several grid sizes. That is what sets
// kGpuIcgnPointsPerCore and kGpuIcgnPointsPerMegapixel in
// src/pipeline/full_field_path_a.cpp.
//
// Everything outside Path A -- image prep, AKAZE, RANSAC, Delaunay, Path B,
// strain -- runs identically in both, so the difference in total wall clock
// is Path A's difference plus noise.
//
// AS SHIPPED this measures the stages the pipeline actually engages, which
// after the Phase 4 verdict means the Hessian pre-pass and the strain fit;
// Path A ICGN is off. To reproduce the Phase 4 table quoted in
// src/pipeline/full_field_path_a.cpp, flip kGpuIcgnPathAEnabled there to true
// and re-run this case. Restricting the process affinity mask before the run
// is how the 4- and 8-thread rows in that table were taken.
// ---------------------------------------------------------------------------
#if defined(SEMPER_OPENCL) && defined(DIC_HAVE_OPENCV)
#include <semper/pipeline.hpp>
#include <opencv2/core.hpp>
#include <thread>

namespace {

cv::Mat perf_gray8(const Image &img) {
    cv::Mat m(img.height, img.width, CV_8UC1);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            float v = img.intensities[(size_t) y * img.width + x];
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    return m;
}

void perf_set_disable(bool on) {
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", on ? "1" : "");
#else
    if (on) setenv("SEMPER_OPENCL_DISABLE", "1", 1);
    else unsetenv("SEMPER_OPENCL_DISABLE");
#endif
    Semper::gpu::reset_for_testing();
}

} // namespace

TEST_CASE(Perf, PathAPipelineThroughputCpuVsGpu) {
    using clock = std::chrono::steady_clock;
    const auto &c = Semper::gpu::caps();
    if (!(c.available && c.exact_fp32)) {
        std::printf("  Perf: pipeline Path A, CPU only -- %s\n",
                    c.available ? "device lacks correctly-rounded fp32"
                                : c.unavailable_reason.c_str());
        CHECK(true);
        return;
    }
    std::printf("  Perf: pipeline Path A on '%s', %u hardware threads\n",
                c.device_name.c_str(), std::thread::hardware_concurrency());

    struct Geom { int side, step; };
    const Geom geoms[] = {{512, 7}, {1024, 9}, {1024, 7}, {1024, 5}};
    const int reps = 3;
    double worst_case_s = 0.0;

    for (const Geom &gm : geoms) {
        dictest::SpeckleField field(/*seed=*/23, gm.side, gm.side,
                                    /*blob_count=*/gm.side * gm.side / 187);
        dictest::AffineDeformation dd;
        dd.u = 2.5f; dd.v = -1.5f; dd.ux = 0.004f; dd.vy = -0.003f;
        dd.cx = gm.side / 2.0f; dd.cy = gm.side / 2.0f;
        const cv::Mat ref_gray = perf_gray8(dictest::make_reference_image(field, gm.side, gm.side));
        const cv::Mat def_gray = perf_gray8(dictest::make_deformed_image(field, gm.side, gm.side, dd));

        const int grid_pts = (gm.side / gm.step) * (gm.side / gm.step);
        std::vector<float> out((size_t) grid_pts * 8, 0.0f);

        double best[2] = {1.0e9, 1.0e9};
        float pathA_pts = 0.0f, valid[2] = {0.0f, 0.0f};
        for (int mode = 0; mode < 2; ++mode) {           // 0 = CPU, 1 = device
            perf_set_disable(mode == 0);
            for (int r = 0; r < reps + 1; ++r) {         // first run is warm-up
                Semper::pipeline::ReferenceCache cache;
                cache.set_from_gray(ref_gray, cv::Mat());
                Semper::pipeline::FullFieldParams p;
                p.rect_x = 0; p.rect_y = 0;
                p.rect_w = gm.side; p.rect_h = gm.side;
                p.step = gm.step;
                p.subset_size = 21;
                p.strain_window = 21;
                p.use_6x6_interpolator = false;
                float metrics[19] = {};
                metrics[16] = -1.0f;
                const auto t0 = clock::now();
                const int rc = Semper::pipeline::run_full_field(
                        cache, def_gray, cv::Mat(), p, out.data(), (int) out.size(),
                        metrics, 19, nullptr);
                const double s_ = std::chrono::duration<double>(clock::now() - t0).count();
                REQUIRE(rc >= 0);
                if (r == 0) continue;
                if (s_ < best[mode]) best[mode] = s_;
                if (s_ > worst_case_s) worst_case_s = s_;
                pathA_pts = metrics[3];
                valid[mode] = metrics[1];
            }
        }
        perf_set_disable(false);

        // Same answers, or the comparison is meaningless. Byte-level parity is
        // ClPipelineParity's job; this only guards against timing a solve that
        // quietly found a different number of points.
        CHECK(valid[0] == valid[1]);
        std::printf("     %4dx%4d step %d: %5.0f Path A pts | CPU %7.2f ms |"
                    " GPU %7.2f ms | %5.2fx %s\n",
                    gm.side, gm.side, gm.step, (double) pathA_pts,
                    1000.0 * best[0], 1000.0 * best[1],
                    best[0] > best[1] ? best[0] / best[1] : best[1] / best[0],
                    best[1] < best[0] ? "GPU" : "CPU");
    }

    // Smoke-level guard only -- see the header comment.
    CHECK(worst_case_s < WALL_CLOCK_CEILING_S);
}
#endif // SEMPER_OPENCL && DIC_HAVE_OPENCV
