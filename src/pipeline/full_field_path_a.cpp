// Path A — mesh-guided OpenMP solve of every grid node the Delaunay guess
// field (full_field_mesh.cpp) covers, schedule(dynamic, 32).

#include "full_field_internal.hpp"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"
#include "util/omp.hpp"

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

    RegionErrors errors;
#pragma omp parallel num_threads(safe_cores)
    {
        int tid = util::omp_thread();
        OptimizationEngine local_engine = make_engine(params.use_6x6_interpolator);
        Semper::SubsetData local_subset;
        double local_hessian = 0.0, local_wait = 0.0; int local_pts = 0;
        EngineStatFlusher flusher(local_engine, stats_pathA[tid], local_pts, local_hessian, local_wait);

#pragma omp for schedule(dynamic, 32)
        for (int idx = 0; idx < gridW * gridH; ++idx) {
            if (errors.failed()) continue;
            if (ctx.cancel.requested()) continue;
            try {
                if (!guess.in_mesh[idx]) continue;
                int x = idx % gridW, y = idx / gridW;
                if (resultGrid[y][x].solved) continue;

                int realX = params.rect_x + x * params.step, realY = params.rect_y + y * params.step;
                const float g[6] = {guess.u[idx], guess.v[idx], guess.ux[idx],
                                    guess.uy[idx], guess.vx[idx], guess.vy[idx]};
                // A guess from outside the mesh (extrapolated, or from the
                // homography) is good wherever there is material to match, so
                // when IC-GN fails from it the node is usually off the specimen
                // and a Simplex rescue fails too. Leave such a node to Path B,
                // which retries it from solved neighbours with its own rescue,
                // rather than paying for the rescue twice.
                const int tag = resultGrid[y][x].mesh_assignment_type;
                const bool outside_mesh = tag == kMeshExtrapolated || tag == kMeshHomography;
                const NodeSolve node = solve_node(local_engine, local_subset, *ctx.cache.ref_img,
                                                  ctx.def_img, realX, realY, params.subset_size,
                                                  ctx.hessian_pool[idx], g,
                                                  outside_mesh ? INIT_NO_SIMPLEX : INIT_NO_SEARCH);
                local_hessian += node.hessian_ms;

                if (node.initialized) {
                    // Save the exact guess before we run
                    resultGrid[y][x].guess_u = g[0];
                    resultGrid[y][x].guess_v = g[1];
                    resultGrid[y][x].guess_ux = g[2];
                    resultGrid[y][x].guess_uy = g[3];
                    resultGrid[y][x].guess_vx = g[4];
                    resultGrid[y][x].guess_vy = g[5];

                    const Semper::AnalysisResult& res = node.res;
                    const bool needed_rescue = node.needed_rescue;
                    stats_pathA[tid].icgn_iters += res.iters;

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
            } catch (...) {
                errors.capture();
            }
        }
    }
    errors.rethrow_if_any();
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
