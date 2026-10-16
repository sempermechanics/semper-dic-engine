// Path A — mesh-guided OpenMP solve.
// Extract-only split from full_field_path_a.cpp; algorithms unchanged.
// schedule(dynamic, 32), thread count, and arithmetic are unchanged.

#include "full_field_internal.hpp"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

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
            auto th1 = std::chrono::high_resolution_clock::now();
            SubsetPrecomputer::precompute_subset_fast(local_subset, *ctx.cache.ref_img, realX, realY, params.subset_size, ctx.hessian_pool[idx]);
            local_hessian += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - th1).count();

            if (local_subset.is_initialized) {
                // Save the exact guess before we run
                resultGrid[y][x].guess_u = guess.u[idx];
                resultGrid[y][x].guess_v = guess.v[idx];
                resultGrid[y][x].guess_ux = guess.ux[idx];
                resultGrid[y][x].guess_uy = guess.uy[idx];
                resultGrid[y][x].guess_vx = guess.vx[idx];
                resultGrid[y][x].guess_vy = guess.vy[idx];

                int simplex_count_before = local_engine.count_simplex;

                auto search_flag = ALLOW_SIMPLEX_RESCUE ? INIT_NO_SEARCH : INIT_NO_SIMPLEX;
                Semper::AnalysisResult res = local_engine.calculate_deformation(
                        local_subset, ctx.def_img, guess.u[idx], guess.v[idx], guess.ux[idx], guess.uy[idx], guess.vx[idx], guess.vy[idx], search_flag);
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
