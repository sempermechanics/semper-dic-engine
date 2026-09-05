// Path A — mesh-guided OpenMP solve.
// Extract-only split from full_field_path_a.cpp; algorithms unchanged.
// schedule(dynamic, 32), thread count, and arithmetic are unchanged.

#include "full_field_internal.hpp"

#include <semper/seeding.hpp>
#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#if defined(SEMPER_OPENCL)
#include "gpu/icgn_dispatch.hpp"
#endif

#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

void run_path_a(
        const SolveContext& ctx,
        const MeshGuessField& guess,
        ResultGrid& resultGrid,
        std::vector<ThreadStats>& stats_pathA) {

    const FullFieldParams& params = ctx.params;
    const int gridW = ctx.gridW;
    const int gridH = ctx.gridH;
    const int safe_cores = ctx.safe_cores;

#if defined(SEMPER_OPENCL)
    // ---- GPU Phase 4: one ICGN solve per point, ahead of the loop --------
    // ---- HOW THE DEVICE PATH WORKS ---------------------------------------
    //
    // Not a fast path with different answers. semper_icgn_solve is a
    // transcription of solve_icgn fused with the part of
    // precompute_subset_fast it consumes, and tests/unit/test_cl_icgn.cpp
    // pins the two together with == on all six parameters, the score, the
    // status, the iteration count and invalid_ref_pixels. Nothing below
    // depends on which one ran.
    //
    // WHAT THE DEVICE IS ALLOWED TO ANSWER FOR. calculate_deformation runs
    // ICGN and then, under INIT_NO_SEARCH, runs Nelder-Mead plus a second
    // ICGN whenever status != 0 || score > kCorrSimplexTrigger
    // (optimization_engine.cpp:54). The kernel does not implement the
    // rescue -- a sequential search with a data-dependent trip count would
    // make every lane wait on the few percent of points that need it -- so
    // a device result is accepted only when it does NOT trip that test.
    // Everything else, and every point the dispatch hands back as
    // kIcgnHostRequired, falls through to the untouched CPU path below,
    // rescue and all. That is what keeps this equivalent rather than merely
    // close.
    //
    // ---- AND WHY IT IS OFF -----------------------------------------------
    //
    // Phase 4 clears two of its three gates and fails the third. The kernel
    // is bit-exact -- ClParity compares nine geometries point by point, and
    // with the flag below flipped to true ClPipelineParity reported 0 float
    // mismatches across 5329 grid points with an identical Path A/Path B
    // split, rescue tally and mean iteration count. It is simply not faster
    // here. Whole-solve wall clock on an RTX 3060 Laptop, one 1024x1024
    // frame, subset 21, best of three (Perf.PathAPipelineThroughputCpuVsGpu):
    //
    //   Path A pts    20 threads       8 threads        4 threads
    //     3470      1.27x CPU        1.09x CPU        1.07x CPU
    //    10562      1.44x CPU        1.05x CPU        1.01x CPU
    //    17110      1.21x CPU        1.00x            1.06x GPU
    //    32645      1.12x CPU        1.02x GPU        1.04x GPU
    //
    // The cause is measured, not guessed. Capping the iteration count
    // (Perf.IcgnPathAThroughputCpuVsGpu) puts 11236 points at 10.2 ms for one
    // iteration and 20.4 ms for eight, but 68.0 ms for the real cap of 50 --
    // against a mean of ~9.4 iterations per point. One work-item per subset
    // means a warp runs until its slowest lane converges, and a full-field
    // solve puts ~137 points that time out at 50 iterations among 3470, so
    // nearly every warp contains one. The launch effectively costs 50
    // iterations per point instead of 9.4.
    //
    // The remedy is the one the roadmap already names: round-based launches
    // over a compacted active-point list, NOT a different reduction shape.
    // Stage the reference planes once, run a bounded number of iterations per
    // launch, read back which points are still running, and relaunch over
    // just those. Resuming from the stored warp matrix is bit-identical --
    // every per-iteration input is either recomputed or carried in full
    // precision -- so it costs no accuracy. It is a real restructure of the
    // kernel and it has not been done, so this stays false rather than
    // shipping a 1.1x-to-1.4x regression.
    //
    // The two thresholds below are where the 4- and 8-thread measurements put
    // the crossover TODAY. They are provisional: whoever lands the compaction
    // work must re-measure them, because the shape of the curve changes.
    //
    // The policy lives here, in the caller. solve_icgn_batch_gpu stays free
    // of it so the parity tests can drive tiny grids without defeating a
    // heuristic.
    constexpr bool kGpuIcgnPathAEnabled = false;
    constexpr double kGpuIcgnPointsPerMegapixel = 16000.0;
    constexpr int kGpuIcgnMinPoints = 16000;

    std::vector<int> gpu_slot;                    // grid index -> slot, -1 = none
    std::vector<gpu::IcgnGpuResult> gpu_res;
    {
        int wanted = 0;
        for (int idx = 0; idx < gridW * gridH; ++idx)
            if (guess.in_mesh[idx] && !resultGrid[idx / gridW][idx % gridW].solved)
                ++wanted;

        const double ref_megapixels =
                static_cast<double>(ctx.cache.ref_img->width) *
                ctx.cache.ref_img->height / 1.0e6;

        if (kGpuIcgnPathAEnabled && !cancel_requested() && wanted >= kGpuIcgnMinPoints &&
            wanted >= kGpuIcgnPointsPerMegapixel * ref_megapixels) {
            std::vector<gpu::IcgnGpuPoint> pts;
            pts.reserve(static_cast<size_t>(wanted));
            gpu_slot.assign(static_cast<size_t>(gridW) * gridH, -1);

            for (int idx = 0; idx < gridW * gridH; ++idx) {
                if (!guess.in_mesh[idx]) continue;
                const int x = idx % gridW, y = idx / gridW;
                if (resultGrid[y][x].solved) continue;

                gpu::IcgnGpuPoint p;
                p.cx = params.rect_x + x * params.step;
                p.cy = params.rect_y + y * params.step;
                p.guess[0] = guess.u[idx];
                p.guess[1] = guess.v[idx];
                p.guess[2] = guess.ux[idx];
                p.guess[3] = guess.uy[idx];
                p.guess[4] = guess.vx[idx];
                p.guess[5] = guess.vy[idx];
                // The pooled entry this point precompute_subset_fast would
                // have been handed. A slot that is not valid comes back as
                // kIcgnHostRequired rather than guessed at.
                p.cached = &ctx.hessian_pool[idx];

                gpu_slot[static_cast<size_t>(idx)] = static_cast<int>(pts.size());
                pts.push_back(p);
            }

            gpu_res.assign(pts.size(), gpu::IcgnGpuResult());
            if (!gpu::solve_icgn_batch_gpu(
                        *ctx.cache.ref_img, ctx.def_img, params.subset_size,
                        params.use_6x6_interpolator, tuning::kIcgnMaxIter,
                        tuning::kLmAlpha, pts.data(), static_cast<int>(pts.size()),
                        gpu_res.data())) {
                // Declined or failed. It touched nothing, so drop the map and
                // let every point take the CPU path.
                gpu_slot.clear();
                gpu_res.clear();
            }
        }
    }
#endif

    std::atomic<bool> omp_region_threw(false);
#pragma omp parallel num_threads(safe_cores)
    {
        int tid = omp_get_thread_num();
        OptimizationEngine local_engine; Semper::SubsetData local_subset;
        // 🚀 ENABLE LEVENBERG-MARQUARDT (TIKHONOV REGULARIZATION) - PATH A
        local_engine.lm_enabled = true;
        local_engine.lm_alpha = tuning::kLmAlpha; // <--- TUNE THIS VALUE
        local_engine.use_6x6_interpolator = params.use_6x6_interpolator;
        double local_hessian = 0.0, local_wait = 0.0; int local_pts = 0;
        EngineStatFlusher flusher(local_engine, stats_pathA[tid], local_pts, local_hessian, local_wait);

#pragma omp for schedule(dynamic, 32)
        for (int idx = 0; idx < gridW * gridH; ++idx) {
            if (omp_region_threw.load(std::memory_order_relaxed)) continue;
            if (cancel_requested()) continue;
            if (!guess.in_mesh[idx]) continue;
            int x = idx % gridW, y = idx / gridW;
            if (resultGrid[y][x].solved) continue;

            int realX = params.rect_x + x * params.step, realY = params.rect_y + y * params.step;

            // A point the device already converged on skips the CPU
            // precompute as well as the solve, which is where most of the
            // saving is: precompute_subset_fast still rebuilds the
            // steepest-descent images for every point it touches.
            bool solved_on_device = false;
            Semper::AnalysisResult res;
            int simplex_count_before = local_engine.count_simplex;
#if defined(SEMPER_OPENCL)
            if (!gpu_slot.empty() && gpu_slot[static_cast<size_t>(idx)] >= 0) {
                const gpu::IcgnGpuResult &d =
                        gpu_res[static_cast<size_t>(gpu_slot[static_cast<size_t>(idx)])];
                // The rescue trigger, mirrored from optimization_engine.cpp:54.
                // Anything that would have gone on to Nelder-Mead is left to
                // the CPU branch below, which runs it exactly as before.
                if (d.status == 0 && d.score <= tuning::kCorrSimplexTrigger) {
                    res.u = d.p[0];
                    res.v = d.p[1];
                    res.ux = d.p[2];
                    res.uy = d.p[3];
                    res.vx = d.p[4];
                    res.vy = d.p[5];
                    res.correlation_score = d.score;
                    res.status = d.status;
                    res.iters = d.iters;
                    res.invalid_ref_pixels = d.invalid_ref_pixels;
                    solved_on_device = true;
                }
            }
#endif

            if (!solved_on_device) {
                auto th1 = std::chrono::high_resolution_clock::now();
                SubsetPrecomputer::precompute_subset_fast(local_subset, *ctx.cache.ref_img, realX, realY, params.subset_size, ctx.hessian_pool[idx]);
                local_hessian += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - th1).count();
                if (!local_subset.is_initialized) continue;
            }

            {
                // Save the exact guess before we run
                resultGrid[y][x].guess_u = guess.u[idx];
                resultGrid[y][x].guess_v = guess.v[idx];
                resultGrid[y][x].guess_ux = guess.ux[idx];
                resultGrid[y][x].guess_uy = guess.uy[idx];
                resultGrid[y][x].guess_vx = guess.vx[idx];
                resultGrid[y][x].guess_vy = guess.vy[idx];

                if (!solved_on_device) {
                    auto search_flag = ALLOW_SIMPLEX_RESCUE ? INIT_NO_SEARCH : INIT_NO_SIMPLEX;
                    res = local_engine.calculate_deformation(
                            local_subset, ctx.def_img, guess.u[idx], guess.v[idx], guess.ux[idx], guess.uy[idx], guess.vx[idx], guess.vy[idx], search_flag);
                }
                reject_if_ghosted(res, params.subset_size);
                stats_pathA[tid].icgn_iters += res.iters;
                bool needed_rescue = (local_engine.count_simplex > simplex_count_before);

                if (!ALLOW_SIMPLEX_RESCUE && res.status != 0) { res.correlation_score = 1.0f; }

                if (needed_rescue) {
                    record_simplex_outcome(stats_pathA[tid], res);
                }

                if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept) {
                    int order = ctx.compute_order_counter.fetch_add(1, std::memory_order_relaxed);
                    // 🚀 FIX: Passed res.iters at the end instead of icgn_iters_used
                    resultGrid[y][x] = {(float)realX, (float)realY, res.u, res.v, res.ux, res.uy, res.vx, res.vy,
                                        res.correlation_score, true, tid, order, resultGrid[y][x].mesh_assignment_type,
                                        needed_rescue, res.iters};
                    ctx.global_points_solved.fetch_add(1, std::memory_order_relaxed); local_pts++;
                } else {
                    resultGrid[y][x].solved = false;
                    resultGrid[y][x].corr = CORR_INVALID;
                    resultGrid[y][x].used_simplex = needed_rescue;
                    // 🚀 FIX: Assign the real iterations on failure
                    resultGrid[y][x].icgn_iters = res.iters;
                }
            }
        }
    }
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
