// Path B — reliability-guided flood fill (RGDIC), deterministic in parallel.
//
// The flood fill grows the solved region outwards from Path A's boundary (or
// from AKAZE / Path C seeds when Path A solved nothing). Each unsolved cell is
// solved once, from the first solved neighbour that reaches it: the
// neighbour's displacement, projected by its gradients, is the cell's initial
// guess. Which neighbour that is decides the answer — a different guess
// converges to a slightly different optimum, and near the ZNSSD gate it can
// flip a cell between accepted and failed.
//
// This used to run free-running workers on one shared priority queue, and
// each claim was a compare-exchange race, so the neighbour that won a cell
// depended on thread timing and the field changed from run to run (app
// TD-65). The flood now advances in numbered rounds, and only the solves run
// concurrently:
//
//   claim  (serial)   pop the kPathBBatchNodes most reliable nodes (lowest
//                     ZNSSD; ties by row, then column — a total order) and
//                     hand each unclaimed 4-neighbour to the first of them
//                     that touches it. The cell and its guess are now fixed.
//   solve  (parallel) any worker solves any claimed cell; a task writes only
//                     its own slot.
//   commit (serial)   once every cell of round r is solved and round r-1 is
//                     committed, write round r into the grid and push its
//                     accepted cells, in claim order; then claim new rounds
//                     until two are in flight.
//
// A claim therefore sees exactly the rounds committed before it, whichever
// thread runs it and however the solves interleave, so the output does not
// depend on timing or on the thread count. Keeping two rounds in flight lets
// workers carry on with round r+1 while one slow cell (an ICGN timeout plus a
// Simplex rescue) finishes round r, so the determinism costs little
// throughput. The batch size is a fixed constant, not the core count, so a
// 4-core and an 8-core device give the same field.

#include "full_field_internal.hpp"

#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

namespace {

// Strict weak order with no ties between distinct cells: lower ZNSSD first,
// then lower row, then lower column. std::priority_queue keeps the "largest"
// on top, so "a < b" here means "b is popped before a".
struct ReliabilityOrder {
    bool operator()(const SeedNode& a, const SeedNode& b) const {
        if (a.correlation_score != b.correlation_score) return a.correlation_score > b.correlation_score;
        if (a.y_idx != b.y_idx) return a.y_idx > b.y_idx;
        return a.x_idx > b.x_idx;
    }
};

using ReliabilityQueue = std::priority_queue<SeedNode, std::vector<SeedNode>, ReliabilityOrder>;

// One cell to solve: where, from which guess, and (filled by the solve) what
// came out.
struct CellTask {
    int x = 0, y = 0;
    float gu = 0, gv = 0, gux = 0, guy = 0, gvx = 0, gvy = 0;
    bool counts_as_path_b = true; // AKAZE / Path C seeds are not counted

    bool initialized = false;     // subset fit inside the image
    bool needed_rescue = false;   // Simplex ran
    int tid = 0;
    double hessian_ms = 0.0;
    AnalysisResult res;
};

// One claim's worth of cells. `tasks` is never resized after the claim, so a
// worker can hold a pointer into it while solving without the lock.
struct Round {
    std::vector<CellTask> tasks;
    size_t next = 0;      // first task no worker has taken yet
    size_t remaining = 0; // tasks not yet solved
};

using SubsetPool = std::vector<SubsetData, Eigen::aligned_allocator<SubsetData>>;

void solve_cell(const SolveContext& ctx, OptimizationEngine& engine, SubsetData& subset, CellTask& t) {
    const FullFieldParams& params = ctx.params;
    const int flat = t.y * ctx.gridW + t.x;
    const int realX = params.rect_x + t.x * params.step, realY = params.rect_y + t.y * params.step;

    auto th1 = std::chrono::high_resolution_clock::now();
    SubsetPrecomputer::precompute_subset_fast(subset, *ctx.cache.ref_img, realX, realY, params.subset_size, ctx.hessian_pool[flat]);
    t.hessian_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - th1).count();
    t.initialized = subset.is_initialized;
    if (!t.initialized) return;

    const int simplex_count_before = engine.count_simplex;
    auto search_flag = ALLOW_SIMPLEX_RESCUE ? INIT_NO_SEARCH : INIT_NO_SIMPLEX;
    t.res = engine.calculate_deformation(subset, ctx.def_img, t.gu, t.gv, t.gux, t.guy, t.gvx, t.gvy, search_flag);
    // Matters more here than on Path A: an accepted cell seeds every point
    // that floods out from it, so a subset sitting in the mask must not pass.
    reject_if_ghosted(t.res, params.subset_size);
    t.needed_rescue = (engine.count_simplex > simplex_count_before);
    if (!ALLOW_SIMPLEX_RESCUE && t.res.status != 0) { t.res.correlation_score = 1.0f; }
}

// Serial: record one solved cell in the grid, the stats and the queue.
void commit_cell(const SolveContext& ctx, const CellTask& t, ResultGrid& resultGrid,
                 std::vector<ThreadStats>& stats_pathB, ReliabilityQueue& q) {
    GridPoint& gp = resultGrid[t.y][t.x];
    ThreadStats& bucket = stats_pathB[t.tid];
    bucket.hessian_time_ms += t.hessian_ms;
    if (!t.initialized) return;

    bucket.icgn_iters += t.res.iters;
    if (t.needed_rescue) record_simplex_outcome(bucket, t.res);

    const AnalysisResult& res = t.res;
    if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept) {
        const FullFieldParams& params = ctx.params;
        const int realX = params.rect_x + t.x * params.step, realY = params.rect_y + t.y * params.step;
        const int order = ctx.compute_order_counter.fetch_add(1, std::memory_order_relaxed);
        gp = {(float)realX, (float)realY, res.u, res.v, res.ux, res.uy, res.vx, res.vy,
              res.correlation_score, true, t.tid, order, gp.mesh_assignment_type,
              t.needed_rescue, res.iters};
        ctx.global_points_solved.fetch_add(1, std::memory_order_relaxed);
        if (t.counts_as_path_b) bucket.points_solved++;
        q.push(SeedNode(t.x, t.y, res.u, res.v, res.ux, res.uy, res.vx, res.vy, res.correlation_score));
    } else {
        gp.corr = CORR_INVALID;
        gp.used_simplex = t.needed_rescue;
        gp.icgn_iters = res.iters;
    }
}

} // namespace

// ==========================================
// 🚀 PATH B (DETERMINISTIC ROUND-BASED FLOOD FILL)
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

    // A cell is claimed once (skipped, solved by Path A, or handed to a
    // neighbour) and never attempted again. Only claims touch it, and claims
    // are serial.
    std::vector<unsigned char> claimed((size_t)gridW * gridH);
    for (int i = 0; i < gridW * gridH; ++i) {
        claimed[i] = resultGrid[i / gridW][i % gridW].solved ? 1 : 0;
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

    ReliabilityQueue q;
    for (const auto &s : boundary_seeds) q.push(s);

    // One engine and one subset buffer per worker, reused for every cell it
    // solves. calculate_deformation and precompute_subset_fast overwrite all
    // they read, so which worker solves a cell cannot change its result (Path
    // A relies on the same property under schedule(dynamic)).
    std::vector<OptimizationEngine> engines(safe_cores);
    SubsetPool subsets(safe_cores);
    for (auto& e : engines) {
        // 🚀 ENABLE LEVENBERG-MARQUARDT - PATH B
        e.lm_enabled = true;
        e.lm_alpha = tuning::kLmAlpha; // <--- TUNE THIS VALUE
        e.use_6x6_interpolator = params.use_6x6_interpolator;
    }
    struct StatsFlush {
        std::vector<OptimizationEngine>& engines; std::vector<ThreadStats>& stats;
        ~StatsFlush() {
            for (size_t t = 0; t < engines.size(); ++t) {
                stats[t].icgn_time_ms += engines[t].time_icgn_ms;
                stats[t].simplex_time_ms += engines[t].time_simplex_ms;
                stats[t].simplex_iters += engines[t].count_simplex;
            }
        }
    } flush{engines, stats_pathB};

    // Seeds (only when Path A left no boundary): solved serially, in their
    // sorted order, before the flood starts — all of them, as the threaded
    // version did on any device with at least as many cores as seeds.
    for (const auto& seed : global_seeds) {
        if (cancel_requested()) return;
        const int flat = seed.y_idx * gridW + seed.x_idx;
        if (claimed[flat]) continue;
        claimed[flat] = 1;
        CellTask t;
        t.x = seed.x_idx; t.y = seed.y_idx;
        t.gu = seed.u; t.gv = seed.v;
        t.counts_as_path_b = false;
        solve_cell(ctx, engines[0], subsets[0], t);
        commit_cell(ctx, t, resultGrid, stats_pathB, q);
    }

    // ---- Round bookkeeping. Everything below is guarded by `mtx`. ----------
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<Round> rounds;   // in-flight rounds, oldest first
    bool done = false;

    // Claim one round from the queue as it stands. Returns false (and adds
    // nothing) when the queue is empty.
    auto claim_round = [&]() -> bool {
        if (q.empty()) return false;
        Round round;
        int popped = 0;
        while (!q.empty() && popped < tuning::kPathBBatchNodes) {
            const SeedNode cur = q.top();
            q.pop();
            ++popped;
            for (int k = 0; k < 4; ++k) {
                const int nx = cur.x_idx + dx4[k], ny = cur.y_idx + dy4[k];
                if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) continue;
                const int flat = ny * gridW + nx;
                if (claimed[flat]) continue;
                claimed[flat] = 1;
                resultGrid[ny][nx].solved = true;

                // 🚀 FIRST-ORDER KINEMATIC EXPANSION (The Path B Fix)
                // Project the parent's solution to the neighbour through its
                // displacement gradients.
                const float dx = (nx - cur.x_idx) * params.step;
                const float dy = (ny - cur.y_idx) * params.step;
                CellTask t;
                t.x = nx; t.y = ny;
                t.gu = cur.u + cur.ux * dx + cur.uy * dy;
                t.gv = cur.v + cur.vx * dx + cur.vy * dy;
                t.gux = cur.ux; t.guy = cur.uy; t.gvx = cur.vx; t.gvy = cur.vy;
                round.tasks.push_back(t);
            }
        }
        round.remaining = round.tasks.size();
        rounds.push_back(std::move(round));
        return true;
    };

    // Commit every finished round at the head, in order; after each commit,
    // refill to two rounds in flight. Sets `done` when nothing is left.
    auto advance = [&]() {
        while (!rounds.empty() && rounds.front().remaining == 0) {
            for (const CellTask& t : rounds.front().tasks) commit_cell(ctx, t, resultGrid, stats_pathB, q);
            rounds.pop_front();
            while (rounds.size() < 2 && claim_round()) {}
        }
        if (rounds.empty()) done = true;
    };

    while (rounds.size() < 2 && claim_round()) {}
    advance(); // commits any round that claimed nothing; may finish outright

    std::vector<std::thread> workers;
    struct WorkerGuard { std::vector<std::thread> &ws; ~WorkerGuard() { for (auto &w : ws) if (w.joinable()) w.join(); } } wg{workers};

    for (int w = 0; w < safe_cores && !done; ++w) {
        workers.emplace_back([&, w]() {
            try {
                std::unique_lock<std::mutex> lk(mtx);
                while (true) {
                    // Bounded wait: a worker parked here has no one to notify
                    // it of a cancel, so it re-checks on a timer. A timeout is
                    // not an error — it just goes round the loop again.
                    Round* round = nullptr;
                    auto has_work = [&] {
                        round = nullptr;
                        for (Round& r : rounds) {
                            if (r.next < r.tasks.size()) { round = &r; return true; }
                        }
                        return false;
                    };
                    cv.wait_for(lk, std::chrono::milliseconds(tuning::kCancelPollMs),
                                [&] { return done || cancel_requested() || has_work(); });
                    if (done || cancel_requested()) return;
                    if (round == nullptr) continue;

                    // Oldest round first: that is the one blocking the next commit.
                    CellTask& task = round->tasks[round->next++];
                    lk.unlock();
                    task.tid = w;
                    solve_cell(ctx, engines[w], subsets[w], task);
                    lk.lock();
                    // `round` is still valid: a round leaves the deque only
                    // once all its tasks are solved, and this one was not.
                    if (--round->remaining == 0) {
                        advance();
                        cv.notify_all();
                    }
                }
            } catch (const std::exception &e) {
                // A worker that throws mid-task leaves its round unfinished,
                // so the flood can never commit past it. Log it and stop the
                // flood so the solve ends with a partial field, not a hang.
                LOGE("Path B worker %d aborted: %s", w, e.what());
                std::lock_guard<std::mutex> lg(mtx); done = true; cv.notify_all();
            } catch (...) {
                LOGE("Path B worker %d aborted: unknown exception", w);
                std::lock_guard<std::mutex> lg(mtx); done = true; cv.notify_all();
            }
        });
    }
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
