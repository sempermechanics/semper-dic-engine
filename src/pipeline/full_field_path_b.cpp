// Path B — global priority-queue flood fill (RGDIC). Lifted verbatim out of
// full_field_solver.cpp: same worker count, same queue discipline, same
// bounded cancel-poll wait, same locking.

#include "full_field_internal.hpp"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
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
        const std::vector<cv::Point2f>& seed_ref_pts,
        const std::vector<cv::Point2f>& seed_def_pts,
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

        for (size_t fi = 0; fi < seed_ref_pts.size(); ++fi) {
            float fx = seed_ref_pts[fi].x, fy = seed_ref_pts[fi].y;
            int ix = (int)std::round((fx - params.rect_x) / (float)params.step);
            int iy = (int)std::round((fy - params.rect_y) / (float)params.step);
            if (ix < 0 || ix >= gridW || iy < 0 || iy >= gridH || resultGrid[iy][ix].solved) continue;
            float du = seed_def_pts[fi].x - fx, dv = seed_def_pts[fi].y - fy;
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

    struct GlobalQueue {
        std::priority_queue<Semper::SeedNode> q;
        std::mutex mtx; std::condition_variable cv;
        int active = 0; bool done = false;
    } gq;
    std::atomic<int> seed_idx(0);
    // No grid_mutex: every resultGrid[y][x] write below is preceded by that
    // cell's compare_exchange_strong on cell_claimed, so exactly one thread
    // ever owns a given cell — a different memory location than any other
    // thread is touching. ResultGrid is a pre-sized (never resized during
    // this parallel phase) std::vector<std::vector<GridPoint>> of a plain POD
    // struct, so concurrent writes to distinct elements are race-free by the
    // standard's element-independence guarantee — no lock was ever needed
    // here for correctness, only for the now-removed shared stat accumulation
    // this loop used to also do.
    const int cores_to_use = safe_cores;

    for (const auto &s : boundary_seeds) {
        gq.q.push(s);
    }

    OptimizationEngine prewarm_engine;
    // 🚀 ENABLE LEVENBERG-MARQUARDT - PATH B (PREWARM)
    prewarm_engine.lm_enabled = true;
    prewarm_engine.lm_alpha = tuning::kLmAlpha; // <--- TUNE THIS VALUE
    prewarm_engine.use_6x6_interpolator = params.use_6x6_interpolator;

    Semper::SubsetData prewarm_subset;

    while ((int)gq.q.size() < cores_to_use && seed_idx.load() < (int)global_seeds.size()) {
        int si = seed_idx.fetch_add(1, std::memory_order_relaxed);
        if (si >= (int)global_seeds.size()) break;
        const auto &seed = global_seeds[si];
        int flat = seed.y_idx * gridW + seed.x_idx;

        bool unclaimed = false;
        if (!cell_claimed[flat].compare_exchange_strong(unclaimed, true, std::memory_order_acq_rel, std::memory_order_relaxed)) continue;

        int realX = params.rect_x + seed.x_idx * params.step, realY = params.rect_y + seed.y_idx * params.step;
        SubsetPrecomputer::precompute_subset_fast(prewarm_subset, *ctx.cache.ref_img, realX, realY, params.subset_size, ctx.hessian_pool[flat]);
        if (!prewarm_subset.is_initialized) continue;

        int simplex_count_before = prewarm_engine.count_simplex;

        auto search_flag = ALLOW_SIMPLEX_RESCUE ? INIT_NO_SEARCH : INIT_NO_SIMPLEX;
        Semper::AnalysisResult res = prewarm_engine.calculate_deformation(
                prewarm_subset, ctx.def_img, seed.u, seed.v, 0.f, 0.f, 0.f, 0.f, search_flag);
        // Matters more here than on the other paths: an accepted prewarm seed is
        // pushed onto the propagation queue below, so a subset sitting in the mask
        // seeds every point that floods out from it.
        reject_if_ghosted(res, params.subset_size);
        stats_pathB[0].icgn_iters += res.iters;
        bool needed_rescue = (prewarm_engine.count_simplex > simplex_count_before);

        if (!ALLOW_SIMPLEX_RESCUE && res.status != 0) { res.correlation_score = 1.0f; }

        if (needed_rescue) {
            record_simplex_outcome(stats_pathB[0], res);
        }

        if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept) {
            int order = ctx.compute_order_counter.fetch_add(1, std::memory_order_relaxed);
            resultGrid[seed.y_idx][seed.x_idx] = {(float)realX, (float)realY, res.u, res.v, res.ux, res.uy, res.vx, res.vy, res.correlation_score, true, -1, order, resultGrid[seed.y_idx][seed.x_idx].mesh_assignment_type, needed_rescue, res.iters};
            ctx.global_points_solved.fetch_add(1, std::memory_order_relaxed);
            gq.q.push(Semper::SeedNode(seed.x_idx, seed.y_idx, res.u, res.v, res.ux, res.uy, res.vx, res.vy, res.correlation_score));
        } else {
            resultGrid[seed.y_idx][seed.x_idx].corr = CORR_INVALID;
            resultGrid[seed.y_idx][seed.x_idx].used_simplex = needed_rescue;
            resultGrid[seed.y_idx][seed.x_idx].icgn_iters = res.iters;
        }
    }

    if (gq.q.empty() && seed_idx.load() >= (int)global_seeds.size()) {
        gq.done = true;
    }

    std::vector<std::thread> workers;
    struct WorkerGuard { std::vector<std::thread> &ws; ~WorkerGuard() { for (auto &w : ws) if (w.joinable()) w.join(); } } wg{workers};

    for (int t = 0; t < cores_to_use; ++t) {
        workers.emplace_back([&, t]() {
            try {
                const int tid = t;
                const int DX[] = {1, -1, 0, 0}, DY[] = {0, 0, 1, -1};
                OptimizationEngine local_engine;
                // 🚀 ENABLE LEVENBERG-MARQUARDT - PATH B (WORKERS)
                local_engine.lm_enabled = true;
                local_engine.lm_alpha = tuning::kLmAlpha; // <--- TUNE THIS VALUE
                local_engine.use_6x6_interpolator = params.use_6x6_interpolator;
                Semper::SubsetData local_subset;
                double local_hessian_ms = 0.0, local_wait_ms = 0.0;
                int local_points_solved = 0;
                EngineStatFlusher flusher(local_engine, stats_pathB[tid], local_points_solved, local_hessian_ms, local_wait_ms);

                while (true) {
                    if (cancel_requested()) return;
                    Semper::SeedNode cur;
                    bool has_node = false;
                    {
                        std::unique_lock<std::mutex> lk(gq.mtx);
                        auto wait_start = std::chrono::high_resolution_clock::now();
                        // Bounded wait: a worker parked on the queue has no
                        // one to notify it of a cancel, so it re-checks on a
                        // timer instead. A timeout is not an error — it just
                        // sends the worker round the loop again.
                        bool ready = gq.cv.wait_for(lk, std::chrono::milliseconds(tuning::kCancelPollMs), [&] { return !gq.q.empty() || gq.done || cancel_requested() || (gq.active == 0 && seed_idx.load(std::memory_order_relaxed) < (int)global_seeds.size()); });
                        local_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - wait_start).count();
                        if (gq.done || cancel_requested()) return;
                        if (!ready) continue;
                        if (!gq.q.empty()) { cur = gq.q.top(); gq.q.pop(); gq.active++; has_node = true; } else { gq.active++; }
                    }

                    if (!has_node) {
                        int si = seed_idx.fetch_add(1, std::memory_order_relaxed);
                        if (si < (int)global_seeds.size()) {
                            const auto &seed = global_seeds[si];
                            int flat = seed.y_idx * gridW + seed.x_idx;
                            bool unclaimed = false;
                            if (cell_claimed[flat].compare_exchange_strong(unclaimed, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                                int realX = params.rect_x + seed.x_idx * params.step, realY = params.rect_y + seed.y_idx * params.step;
                                auto th1 = std::chrono::high_resolution_clock::now();
                                SubsetPrecomputer::precompute_subset_fast(local_subset, *ctx.cache.ref_img, realX, realY, params.subset_size, ctx.hessian_pool[flat]);
                                local_hessian_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - th1).count();
                                if (local_subset.is_initialized) {

                                    int simplex_count_before = local_engine.count_simplex;

                                    auto search_flag = ALLOW_SIMPLEX_RESCUE ? INIT_NO_SEARCH : INIT_NO_SIMPLEX;
                                    Semper::AnalysisResult res = local_engine.calculate_deformation(
                                            local_subset, ctx.def_img, seed.u, seed.v, 0.f, 0.f, 0.f, 0.f, search_flag);

                                    reject_if_ghosted(res, params.subset_size);
                                    stats_pathB[tid].icgn_iters += res.iters;
                                    bool needed_rescue = (local_engine.count_simplex > simplex_count_before);

                                    if (!ALLOW_SIMPLEX_RESCUE && res.status != 0) { res.correlation_score = 1.0f; }

                                    if (needed_rescue) {
                                        record_simplex_outcome(stats_pathB[tid], res);
                                    }

                                    if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept) {
                                        int order = ctx.compute_order_counter.fetch_add(1, std::memory_order_relaxed);
                                        // 🚀 FIX: Use res.iters at the end
                                        resultGrid[seed.y_idx][seed.x_idx] = {(float)realX, (float)realY, res.u, res.v, res.ux, res.uy, res.vx, res.vy,
                                                                              res.correlation_score, true, tid, order, 0, needed_rescue, res.iters};
                                        ctx.global_points_solved.fetch_add(1, std::memory_order_relaxed); local_points_solved++;
                                        { std::lock_guard<std::mutex> lq(gq.mtx); gq.q.push(Semper::SeedNode(seed.x_idx, seed.y_idx, res.u, res.v, res.ux, res.uy, res.vx, res.vy, res.correlation_score)); gq.cv.notify_one(); }
                                    } else {
                                        resultGrid[seed.y_idx][seed.x_idx].corr = CORR_INVALID;
                                        resultGrid[seed.y_idx][seed.x_idx].used_simplex = needed_rescue;
                                        // 🚀 FIX: Assign the real iterations
                                        resultGrid[seed.y_idx][seed.x_idx].icgn_iters = res.iters;
                                    }
                                }
                            }
                        }
                        { std::lock_guard<std::mutex> lq(gq.mtx); gq.active--; bool seeds_exhausted = seed_idx.load(std::memory_order_relaxed) >= (int)global_seeds.size(); if (gq.q.empty() && gq.active == 0 && seeds_exhausted) { gq.done = true; } gq.cv.notify_all(); }
                        continue;
                    }

                    std::vector<Semper::SeedNode> pending_pushes;
                    pending_pushes.reserve(4);
                    for (int k = 0; k < 4; ++k) {
                        const int nx = cur.x_idx + DX[k], ny = cur.y_idx + DY[k];
                        if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) continue;
                        const int flat = ny * gridW + nx;
                        bool was_unclaimed = false;
                        if (!cell_claimed[flat].compare_exchange_strong(was_unclaimed, true, std::memory_order_acq_rel, std::memory_order_relaxed)) continue;
                        resultGrid[ny][nx].solved = true;
                        const int realX = params.rect_x + nx * params.step, realY = params.rect_y + ny * params.step;
                        auto th1 = std::chrono::high_resolution_clock::now();
                        SubsetPrecomputer::precompute_subset_fast(local_subset, *ctx.cache.ref_img, realX, realY, params.subset_size, ctx.hessian_pool[flat]);
                        local_hessian_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - th1).count();
                        if (!local_subset.is_initialized) continue;

                        int simplex_count_before = local_engine.count_simplex;

                        // 🚀 FIRST-ORDER KINEMATIC EXPANSION (The Path B Fix)
                        // Calculate the physical distance from the solved point to the new neighbor
                        float dx = (nx - cur.x_idx) * params.step;
                        float dy = (ny - cur.y_idx) * params.step;

                        // Project the initial guess using the solved point's strain gradients
                        float guess_u = cur.u + cur.ux * dx + cur.uy * dy;
                        float guess_v = cur.v + cur.vx * dx + cur.vy * dy;

                        auto search_flag = ALLOW_SIMPLEX_RESCUE ? INIT_NO_SEARCH : INIT_NO_SIMPLEX;
                        Semper::AnalysisResult res = local_engine.calculate_deformation(
                                local_subset, ctx.def_img, guess_u, guess_v, cur.ux, cur.uy, cur.vx, cur.vy, search_flag);
                        reject_if_ghosted(res, params.subset_size);
                        stats_pathB[tid].icgn_iters += res.iters; // 🚀 ADD THIS
                        bool needed_rescue = (local_engine.count_simplex > simplex_count_before);

                        if (!ALLOW_SIMPLEX_RESCUE && res.status != 0) { res.correlation_score = 1.0f; }

                        if (needed_rescue) {
                            record_simplex_outcome(stats_pathB[tid], res);
                        }

                        if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept) {
                            const int order = ctx.compute_order_counter.fetch_add(1, std::memory_order_relaxed);
                            // 🚀 FIX: Use res.iters at the end
                            resultGrid[ny][nx] = {(float)realX, (float)realY, res.u, res.v, res.ux, res.uy, res.vx, res.vy,
                                                  res.correlation_score, true, tid, order, resultGrid[ny][nx].mesh_assignment_type,
                                                  needed_rescue, res.iters};
                            ctx.global_points_solved.fetch_add(1, std::memory_order_relaxed); local_points_solved++;
                            pending_pushes.push_back(Semper::SeedNode(nx, ny, res.u, res.v, res.ux, res.uy, res.vx, res.vy, res.correlation_score));
                        } else {
                            resultGrid[ny][nx].corr = CORR_INVALID;
                            resultGrid[ny][nx].used_simplex = needed_rescue;
                            // 🚀 FIX: Assign the real iterations
                            resultGrid[ny][nx].icgn_iters = res.iters;
                        }
                    }
                    if (!pending_pushes.empty()) { std::lock_guard<std::mutex> lq(gq.mtx); for (auto &node : pending_pushes) { gq.q.push(std::move(node)); gq.cv.notify_one(); } }
                    { std::lock_guard<std::mutex> lq(gq.mtx); gq.active--; if (gq.q.empty() && gq.active == 0) { bool seeds_exhausted = seed_idx.load(std::memory_order_relaxed) >= (int)global_seeds.size(); if (seeds_exhausted) { gq.done = true; } gq.cv.notify_all(); } }
                }
            } catch (const std::exception &e) {
                // A worker that throws mid-item never runs its gq.active--,
                // so the queue's (active == 0) termination becomes
                // unreachable and the remaining workers park until cancel.
                // Log it (it used to vanish silently) and signal done so the
                // solve finishes with a partial field instead of hanging.
                LOGE("Path B worker %d aborted: %s", t, e.what());
                std::lock_guard<std::mutex> lq(gq.mtx); gq.done = true; gq.cv.notify_all();
            } catch (...) {
                LOGE("Path B worker %d aborted: unknown exception", t);
                std::lock_guard<std::mutex> lq(gq.mtx); gq.done = true; gq.cv.notify_all();
            }
        });
    }
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
