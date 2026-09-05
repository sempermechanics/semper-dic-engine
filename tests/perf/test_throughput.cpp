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
