// Path B — deterministic level-synchronous flood fill (RGDIC).
//
// This used to be a global std::priority_queue drained by N racing worker
// threads. Each point's initial guess is extrapolated from whichever parent
// won the compare_exchange for that cell, so the propagation order -- and
// with it the answer at hard points -- varied between runs. Measured on the
// host suite before this change: 0 of 15 runs reproduced, with 10-230 of
// ~3100 output floats differing between two consecutive solves of the same
// binary on identical input.
//
// That made "bit-exact with the CPU" unsatisfiable for the OpenCL backend:
// there was no fixed CPU answer to be exact against.
//
// The queue is replaced by rounds. Each round takes the whole current
// frontier, resolves every child's parent BEFORE any solving happens (best
// parent correlation score, ties to the lowest parent flat index), and only
// then solves the round's children in parallel. Because each guess is fully
// determined before the parallel section starts, thread interleaving can no
// longer affect the result. The same structure is what maps onto a GPU:
// one kernel launch per round.
//
// See docs/DETERMINISM.md.

#include "full_field_internal.hpp"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

// ==========================================
// 🚀 PATH B (GLOBAL QUEUE EXECUTION)
// ==========================================
void run_path_b(
        const SolveContext& ctx,
        const std::vector<cv::Point2f>& akaze_ref_pts,
        const std::vector<cv::Point2f>& akaze_def_pts,
        float globalU,
        float globalV,
        int path_c_seed_x,
        int path_c_seed_y,
        ResultGrid& resultGrid,
        std::vector<ThreadStats>& stats_pathB) {

    const FullFieldParams& params = ctx.params;
    const int gridW = ctx.gridW;
    const int gridH = ctx.gridH;
    const int safe_cores = ctx.safe_cores;

    std::unique_ptr<std::atomic<bool>[]> cell_claimed(new std::atomic<bool>[gridW * gridH]);
    for (int i = 0; i < gridW * gridH; ++i) {
        int gx = i % gridW, gy = i / gridW;
        cell_claimed[i].store(resultGrid[gy][gx].solved, std::memory_order_relaxed);
    }

    std::vector<Semper::SeedNode> boundary_seeds;
    std::vector<Semper::SeedNode> global_seeds;

    const int dx4[] = {1, -1, 0, 0}, dy4[] = {0, 0, 1, -1};
    for (int y = 0; y < gridH; ++y) {
        for (int x = 0; x < gridW; ++x) {
            if (!resultGrid[y][x].solved || resultGrid[y][x].corr < 0.f) continue;
            bool touching = false;
            for (int k = 0; k < 4; ++k) {
                int nx = x + dx4[k], ny = y + dy4[k];
                if (nx >= 0 && nx < gridW && ny >= 0 && ny < gridH && !resultGrid[ny][nx].solved) { touching = true; break; }
            }
            if (touching) boundary_seeds.push_back(Semper::SeedNode(x, y, resultGrid[y][x].u, resultGrid[y][x].v, resultGrid[y][x].ux, resultGrid[y][x].uy, resultGrid[y][x].vx, resultGrid[y][x].vy, resultGrid[y][x].corr));
        }
    }

    if (boundary_seeds.empty()) {
        struct SeedCandidate {
            int ix, iy;
            float u_init, v_init;
            float dist_from_center;
            float displacement_mag;
        };
        std::vector<SeedCandidate> candidates;
        float grid_cx = params.rect_x + (gridW / 2.f) * params.step, grid_cy = params.rect_y + (gridH / 2.f) * params.step;

        for (size_t fi = 0; fi < akaze_ref_pts.size(); ++fi) {
            float fx = akaze_ref_pts[fi].x, fy = akaze_ref_pts[fi].y;
            int ix = (int)std::round((fx - params.rect_x) / (float)params.step);
            int iy = (int)std::round((fy - params.rect_y) / (float)params.step);
            if (ix < 0 || ix >= gridW || iy < 0 || iy >= gridH || resultGrid[iy][ix].solved) continue;
            float du = akaze_def_pts[fi].x - fx, dv = akaze_def_pts[fi].y - fy;
            float world_x = params.rect_x + ix * params.step, world_y = params.rect_y + iy * params.step;
            float dist_c = std::sqrt((world_x - grid_cx)*(world_x - grid_cx) + (world_y - grid_cy)*(world_y - grid_cy));
            candidates.push_back({ix, iy, du, dv, dist_c, std::sqrt(du*du + dv*dv)});
        }

        if (candidates.empty()) {
            // 🚀 PRIORITY 4: Use the intelligently found Smart Seed from Path C
            global_seeds.push_back(Semper::SeedNode(path_c_seed_x, path_c_seed_y, globalU, globalV, 0.f, 0.f, 0.f, 0.f, 0.f));
        } else {
            std::sort(candidates.begin(), candidates.end(), [](const SeedCandidate& a, const SeedCandidate& b) {
                if (std::abs(a.displacement_mag - b.displacement_mag) > 1.f) return a.displacement_mag < b.displacement_mag;
                return a.dist_from_center < b.dist_from_center;
            });
            int seeds_pushed = 0;
            for (const auto& c : candidates) {
                if (seeds_pushed >= 5) break;
                global_seeds.push_back(Semper::SeedNode(c.ix, c.iy, c.u_init, c.v_init, 0.f, 0.f, 0.f, 0.f, 0.f));
                seeds_pushed++;
            }
        }
    }

    const int cores_to_use = safe_cores;
    const int cell_count = gridW * gridH;

    // Per-thread solver state, reused across every round so the engines are
    // not reconstructed gridW*gridH times.
    struct Worker {
        OptimizationEngine engine;
        Semper::SubsetData subset;
        double hessian_ms = 0.0;
        int points = 0;
    };
    std::vector<Worker> workers((size_t) cores_to_use);
    for (auto &w : workers) {
        // 🚀 ENABLE LEVENBERG-MARQUARDT - PATH B
        w.engine.lm_enabled = true;
        w.engine.lm_alpha = tuning::kLmAlpha;
        w.engine.use_6x6_interpolator = params.use_6x6_interpolator;
    }

    // Solve one grid cell from a fully-determined guess. Shared by the serial
    // seed pass and the parallel round body so both take exactly the same
    // arithmetic path.
    auto solve_cell = [&](Worker &w, ThreadStats &st, int gx, int gy,
                          float guess_u, float guess_v, float guess_ux,
                          float guess_uy, float guess_vx, float guess_vy,
                          Semper::AnalysisResult &out) -> bool {
        const int flat = gy * gridW + gx;
        const int realX = params.rect_x + gx * params.step;
        const int realY = params.rect_y + gy * params.step;

        auto th1 = std::chrono::high_resolution_clock::now();
        SubsetPrecomputer::precompute_subset_fast(w.subset, *ctx.cache.ref_img,
                                                  realX, realY,
                                                  params.subset_size,
                                                  ctx.hessian_pool[flat]);
        w.hessian_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - th1).count();
        if (!w.subset.is_initialized) return false;

        const int simplex_before = w.engine.count_simplex;
        const auto search_flag = ALLOW_SIMPLEX_RESCUE ? INIT_NO_SEARCH : INIT_NO_SIMPLEX;
        out = w.engine.calculate_deformation(w.subset, ctx.def_img, guess_u, guess_v,
                                             guess_ux, guess_uy, guess_vx, guess_vy,
                                             search_flag);
        reject_if_ghosted(out, params.subset_size);
        st.icgn_iters += out.iters;

        const bool needed_rescue = (w.engine.count_simplex > simplex_before);
        if (!ALLOW_SIMPLEX_RESCUE && out.status != 0) out.correlation_score = 1.0f;
        if (needed_rescue) record_simplex_outcome(st, out);

        const bool accepted = (out.status == 0 &&
                               out.correlation_score <= tuning::kCorrAccept);
        if (accepted) {
            resultGrid[gy][gx] = {(float) realX, (float) realY, out.u, out.v,
                                  out.ux, out.uy, out.vx, out.vy,
                                  out.correlation_score, true, 0, 0,
                                  resultGrid[gy][gx].mesh_assignment_type,
                                  needed_rescue, out.iters};
            ctx.global_points_solved.fetch_add(1, std::memory_order_relaxed);
            w.points++;
        } else {
            resultGrid[gy][gx].corr = CORR_INVALID;
            resultGrid[gy][gx].used_simplex = needed_rescue;
            resultGrid[gy][gx].icgn_iters = out.iters;
        }
        return accepted;
    };

    // ── Establish the initial frontier ───────────────────────────────────
    // Path A's boundary is already solved and needs no seeding pass. Only
    // when it is empty do we fall back to the AKAZE / Path C seeds, solved
    // serially in list order so the starting frontier is reproducible.
    std::vector<Semper::SeedNode> frontier = boundary_seeds;

    if (frontier.empty()) {
        for (const auto &seed : global_seeds) {
            if (cancel_requested()) return;
            const int flat = seed.y_idx * gridW + seed.x_idx;
            bool unclaimed = false;
            if (!cell_claimed[flat].compare_exchange_strong(
                    unclaimed, true, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) continue;

            Semper::AnalysisResult res;
            if (solve_cell(workers[0], stats_pathB[0], seed.x_idx, seed.y_idx,
                           seed.u, seed.v, 0.f, 0.f, 0.f, 0.f, res)) {
                frontier.push_back(Semper::SeedNode(seed.x_idx, seed.y_idx,
                                                    res.u, res.v, res.ux, res.uy,
                                                    res.vx, res.vy,
                                                    res.correlation_score));
            }
        }
    }

    // Frontier order is part of the tie-break, so keep it sorted by flat
    // index rather than by whatever order the seeds happened to arrive in.
    auto by_flat = [gridW](const Semper::SeedNode &a, const Semper::SeedNode &b) {
        return (a.y_idx * gridW + a.x_idx) < (b.y_idx * gridW + b.x_idx);
    };
    std::sort(frontier.begin(), frontier.end(), by_flat);

    // ── Round loop ───────────────────────────────────────────────────────
    // Scratch reused across rounds; sized once.
    std::vector<int> best_parent((size_t) cell_count, -1);
    std::vector<float> best_corr((size_t) cell_count, 0.0f);
    std::vector<int> touched;      // cells written this round, for cheap reset
    std::vector<int> winners;      // child flat indices to solve this round
    const int DX[4] = {1, -1, 0, 0}, DY[4] = {0, 0, 1, -1};

    std::atomic<bool> round_threw(false);

    while (!frontier.empty()) {
        if (cancel_requested()) return;

        // 1. Propose. Serial and in sorted frontier order, so the winner for
        //    every contested cell is decided identically on every run. This
        //    is 4 comparisons per frontier node — negligible beside the
        //    ICGN solves in step 3.
        touched.clear();
        winners.clear();
        for (size_t pi = 0; pi < frontier.size(); ++pi) {
            const Semper::SeedNode &cur = frontier[pi];
            for (int k = 0; k < 4; ++k) {
                const int nx = cur.x_idx + DX[k], ny = cur.y_idx + DY[k];
                if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) continue;
                const int flat = ny * gridW + nx;
                if (cell_claimed[flat].load(std::memory_order_relaxed)) continue;

                if (best_parent[(size_t) flat] < 0) {
                    best_parent[(size_t) flat] = (int) pi;
                    best_corr[(size_t) flat] = cur.correlation_score;
                    touched.push_back(flat);
                } else if (cur.correlation_score < best_corr[(size_t) flat]) {
                    // Strictly-better only: on a tie the earlier (lower flat
                    // index) parent keeps the cell.
                    best_parent[(size_t) flat] = (int) pi;
                    best_corr[(size_t) flat] = cur.correlation_score;
                }
            }
        }

        // 2. Claim. Sorted so the round's work list — and therefore the
        //    compute_order assigned below — does not depend on push order.
        std::sort(touched.begin(), touched.end());
        for (int flat : touched) {
            bool unclaimed = false;
            if (cell_claimed[flat].compare_exchange_strong(
                    unclaimed, true, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                resultGrid[flat / gridW][flat % gridW].solved = true;
                winners.push_back(flat);
            }
        }

        // 3. Solve. Every guess was fixed in step 1, so this is order-free.
        std::vector<Semper::AnalysisResult> results((size_t) winners.size());
        std::vector<unsigned char> accepted((size_t) winners.size(), 0);

#pragma omp parallel num_threads(cores_to_use)
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 8)
            for (int wi = 0; wi < (int) winners.size(); ++wi) {
                if (round_threw.load(std::memory_order_relaxed)) continue;
                if (cancel_requested()) continue;
                try {
                    const int flat = winners[(size_t) wi];
                    const int nx = flat % gridW, ny = flat / gridW;
                    const Semper::SeedNode &cur =
                            frontier[(size_t) best_parent[(size_t) flat]];

                    // 🚀 FIRST-ORDER KINEMATIC EXPANSION
                    const float dx = (float) (nx - cur.x_idx) * (float) params.step;
                    const float dy = (float) (ny - cur.y_idx) * (float) params.step;
                    const float guess_u = cur.u + cur.ux * dx + cur.uy * dy;
                    const float guess_v = cur.v + cur.vx * dx + cur.vy * dy;

                    accepted[(size_t) wi] = solve_cell(
                            workers[(size_t) tid], stats_pathB[tid], nx, ny,
                            guess_u, guess_v, cur.ux, cur.uy, cur.vx, cur.vy,
                            results[(size_t) wi]) ? 1 : 0;
                } catch (const std::exception &e) {
                    LOGE("Path B round worker %d aborted: %s", tid, e.what());
                    round_threw.store(true, std::memory_order_relaxed);
                } catch (...) {
                    LOGE("Path B round worker %d aborted: unknown exception", tid);
                    round_threw.store(true, std::memory_order_relaxed);
                }
            }
        }

        if (round_threw.load(std::memory_order_relaxed)) return;

        // 4. Commit in winner order, so compute_order is a deterministic
        //    function of the round and the flat index rather than of which
        //    thread happened to finish first.
        std::vector<Semper::SeedNode> next;
        next.reserve(winners.size());
        for (size_t wi = 0; wi < winners.size(); ++wi) {
            if (!accepted[wi]) continue;
            const int flat = winners[wi];
            const int nx = flat % gridW, ny = flat / gridW;
            resultGrid[ny][nx].compute_order =
                    ctx.compute_order_counter.fetch_add(1, std::memory_order_relaxed);
            const auto &r = results[wi];
            next.push_back(Semper::SeedNode(nx, ny, r.u, r.v, r.ux, r.uy, r.vx,
                                            r.vy, r.correlation_score));
        }

        // Reset only the cells this round touched.
        for (int flat : touched) best_parent[(size_t) flat] = -1;

        // `next` is built from `winners`, which was sorted by flat index, so
        // it is already in the order by_flat wants.
        frontier.swap(next);
    }

    // Flush the per-thread accounting the RAII flusher used to handle.
    for (int t = 0; t < cores_to_use; ++t) {
        stats_pathB[t].icgn_time_ms += workers[(size_t) t].engine.time_icgn_ms;
        stats_pathB[t].simplex_time_ms += workers[(size_t) t].engine.time_simplex_ms;
        stats_pathB[t].simplex_iters += workers[(size_t) t].engine.count_simplex;
        stats_pathB[t].points_solved += workers[(size_t) t].points;
        stats_pathB[t].hessian_time_ms += workers[(size_t) t].hessian_ms;
    }
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
