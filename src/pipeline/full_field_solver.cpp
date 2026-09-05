// Orchestration for the hybrid full-field DIC solve.
//
// The individual phases live in sibling translation units (see
// full_field_internal.hpp): anchor seeding + Delaunay mesh + mesh-guided solve
// in full_field_path_a.cpp, the flood fill in full_field_path_b.cpp, fallback
// seeding in full_field_path_c.cpp, packing/telemetry in
// full_field_solver_stats.cpp and the debug maps in
// full_field_debug_export.cpp. This file only sequences them.

#include <semper/pipeline.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include <semper/assert.hpp>
#include "full_field_internal.hpp"
#include "util/log.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {

// Cooperative cancel (the flag itself lives in cancel.cpp). Polled in the hot
// point loops: the cost is one relaxed atomic load per point against a full ICGN
// solve, and a late poll only means one more point, never a wrong result.

using namespace Semper::pipeline::internal;

// Re-entrant overload: bind the caller-owned token as the active cancel target for
// the duration of the solve, then run the shared implementation below unchanged —
// its clear_cancel()/cancel_requested() calls route to `cancel` while bound.
int run_full_field(
    ReferenceCache& cache,
    const cv::Mat& def_gray,
    const cv::Mat& roi_mask,
    const FullFieldParams& params,
    float* output_ptr,
    int output_capacity,
    float* metrics,
    int metrics_len,
    CancelToken& cancel,
    ProgressCallback on_progress) {
    detail::ScopedCancelToken bind(cancel);
    return run_full_field(cache, def_gray, roi_mask, params, output_ptr,
                          output_capacity, metrics, metrics_len, on_progress);
}

int run_full_field(
    ReferenceCache& cache,
    const cv::Mat& def_gray,
    const cv::Mat& roi_mask,
    const FullFieldParams& params,
    float* output_ptr,
    int output_capacity,
    float* metrics,
    int metrics_len,
    ProgressCallback on_progress) {

        std::string local_debug_dir = cache.debug_dir;
        cache.debug_dir.clear();

        auto t_total_start = std::chrono::high_resolution_clock::now();
        PhaseTimings timings;

        // 🚀 PRIORITY 3: Return -3 for memory/init errors
        if (output_ptr == nullptr || cache.ref_img == nullptr || def_gray.empty()) return -3;
        // step <= 0 would divide-by-zero at `rect_w / step` below (a SIGFPE in
        // release builds, where the assert is compiled out). Guard it with the
        // same -2 ROI code the degenerate-grid check already returns.
        if (params.step <= 0) return -2;
        SEMPER_ASSERT(params.subset_size > 0);
        SEMPER_ASSERT(params.step > 0);
        SEMPER_ASSERT(output_capacity >= 0);
        // Clear any leftover cancel from a previous solve so this run starts fresh.
        // The flag is process-global; without this, a caller that cancelled and
        // then started a new solve without resetting would get an instant -99.
        clear_cancel();
        if (cancel_requested()) return kCancelled;

        static int s_frame_count = 0; s_frame_count++;
        LOGD("=== FRAME %d computeFullFieldDirect (HYBRID CORE) START ===", s_frame_count);

        auto t_prep_start = std::chrono::high_resolution_clock::now();
        cv::Mat defMat = def_gray;
        cv::Mat roiMask;
        if (!roi_mask.empty()) {
            roiMask = roi_mask;
            if (roiMask.cols != cache.width || roiMask.rows != cache.height) {
                cv::resize(roiMask, roiMask, cv::Size(cache.width, cache.height), 0, 0, cv::INTER_NEAREST);
            }
        }

        Image defImg(defMat.cols, defMat.rows, defMat.data);
        defImg.prepare_data(false);
        timings.img_prep = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_prep_start).count();

        int safe_cores = std::max(1, (int)std::thread::hardware_concurrency());

        int gridW = params.rect_w / params.step;
        int gridH = params.rect_h / params.step;
        if (gridW <= 0 || gridH <= 0) return -2; // 🚀 PRIORITY 3: Return -2 for ROI Errors

        // Now that gridW and gridH exist, we can set their default fallbacks
        int path_c_seed_x = gridW / 2;
        int path_c_seed_y = gridH / 2;

        ResultGrid resultGrid(gridH, std::vector<GridPoint>(gridW));
        int total_valid_points = 0;

        // === 🚀 100% STRICT RULE: PURE SUBSETS ONLY ===
        // We scan the ENTIRE subset bounding box before allowing a grid point to exist.
        // If a single pixel of the subset touches the void, the image boundary, or the mask,
        // the point is discarded. This guarantees pure tracking and prevents Ghost Displacements.

        int half_subset = params.subset_size / 2;
        // DICe's required 4-pixel interpolation buffer, plus a ~15-pixel deformation buffer
        int absolute_boundary_buffer = half_subset + 4 + 15;

        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                int realX = params.rect_x + x * params.step;
                int realY = params.rect_y + y * params.step;
                bool shouldSkip = false;

                // 1. DICe Absolute Boundary Force Field (Maps exactly to DICe's ~40px edge buffer)
                if (realX - absolute_boundary_buffer < 0 || realX + absolute_boundary_buffer >= cache.width ||
                    realY - absolute_boundary_buffer < 0 || realY + absolute_boundary_buffer >= cache.height) {
                    shouldSkip = true;
                }

                // 2. The 100% ROI Strict Scan
                if (!shouldSkip) {
                    // Scan EVERY SINGLE PIXEL the subset bounding box will touch.
                    for (int dy = -half_subset; dy <= half_subset; dy += 1) {
                        for (int dx = -half_subset; dx <= half_subset; dx += 1) {
                            int checkY = realY + dy;
                            int checkX = realX + dx;

                            // Failsafe bounds check (should be caught by the absolute buffer above)
                            if (checkX < 0 || checkX >= cache.width || checkY < 0 || checkY >= cache.height) {
                                shouldSkip = true;
                                break;
                            }

                            // If ANY pixel in the subset hits the mask, kill the whole point
                            if (!roiMask.empty() && roiMask.at<uchar>(checkY, checkX) < 128) {
                                shouldSkip = true;
                                break;
                            }
                        }
                        if (shouldSkip) break;
                    }
                }

                if (!shouldSkip) total_valid_points++;

                // Note: If shouldSkip is true, the point is marked as 'solved' so the OpenMP
                // workers will ignore it entirely, just like DICe's kd-tree ignores it.
                resultGrid[y][x] = {(float)realX, (float)realY, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, CORR_INVALID, shouldSkip, -1, -1, 0, false, 0};
            }
        }

        if (total_valid_points == 0) return -2; // 🚀 PRIORITY 3: Return -2 for Empty Mask

        std::atomic<int> global_points_solved(0);
        std::atomic<int> compute_order_counter(1);

        std::vector<ThreadStats> stats_pathA(safe_cores);
        std::vector<ThreadStats> stats_pathB(safe_cores);

        std::mutex progress_cv_mutex;
        std::condition_variable progress_cv;
        std::atomic<bool> progress_thread_should_stop(false);
        bool progress_thread_detached = false;

        std::thread progress_thread([&]() {
            if (!on_progress) {
                std::lock_guard<std::mutex> lk(progress_cv_mutex);
                progress_thread_detached = true;
                progress_cv.notify_one();
                return;
            }
            while (!progress_thread_should_stop.load(std::memory_order_acquire)) {
                int solved = global_points_solved.load(std::memory_order_relaxed);
                int percentage = (int)((((float)solved / total_valid_points) * 80.0f) + 10.0f);
                percentage = std::max(0, std::min(100, percentage));
                try { on_progress(percentage); } catch (...) {
                    LOGE("on_progress callback threw; ignoring");
                }
                for (int i = 0; i < 10; ++i) {
                    if (progress_thread_should_stop.load(std::memory_order_acquire)) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            {
                std::lock_guard<std::mutex> lk(progress_cv_mutex);
                progress_thread_detached = true;
            }
            progress_cv.notify_one();
        });

        struct ThreadJoinGuard {
            std::thread &t; std::atomic<bool> &stop_flag; std::condition_variable &cv;
            std::mutex &cv_mutex; bool &detached_flag;
            ~ThreadJoinGuard() {
                stop_flag.store(true, std::memory_order_release);
                {
                    std::unique_lock<std::mutex> lk(cv_mutex);
                    cv.wait_for(lk, std::chrono::seconds(2), [&]{ return detached_flag; });
                }
                if (t.joinable()) t.join();
            }
        };
        ThreadJoinGuard progressGuard{progress_thread, progress_thread_should_stop, progress_cv, progress_cv_mutex, progress_thread_detached};

        // =========================================================
        // 🚀 GLOBAL HESSIAN PRE-PASS & CONTRAST THRESHOLDING
        // =========================================================
        // Runs before seeding: the anchor lattice sits on grid nodes and reuses
        // this pool, so its solves cost little more than the field costs anyway.
        auto t_prepass_start = std::chrono::high_resolution_clock::now();
        HessianPool hessian_pool(gridW * gridH);

    #pragma omp parallel for schedule(static) num_threads(safe_cores)
        for (int pool_idx = 0; pool_idx < gridW * gridH; ++pool_idx) {
            // OpenMP forbids breaking out of a parallel for, so a cancel skips
            // the remaining iterations instead — the same shape the existing
            // exception guard in Path A uses.
            if (cancel_requested()) continue;
            int gx = pool_idx % gridW, gy = pool_idx / gridW;
            if (!resultGrid[gy][gx].solved) {
                int realX = params.rect_x + gx * params.step, realY = params.rect_y + gy * params.step;
                hessian_pool[pool_idx] = SubsetPrecomputer::compute_hessian_only(*cache.ref_img, realX, realY, params.subset_size);
            }
        }

        timings.prepass = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_prepass_start).count();

        // Anchor-lattice seeding: phase correlation for the global rigid shift,
        // then IC-GN on a lattice of grid nodes. Nodes that clear the result
        // gate are marked solved here, so Path A skips them and Path B takes
        // them as boundary seeds.
        std::vector<ThreadStats> stats_anchors(safe_cores);
        MeshSeedResult seeds = solve_anchor_seeds(
                cache, defMat, defImg, params, gridW, gridH, safe_cores,
                hessian_pool, global_points_solved, compute_order_counter,
                resultGrid, stats_anchors, timings);

        const std::vector<cv::Point2f>& anchor_ref_pts = seeds.ref_pts;
        const std::vector<cv::Point2f>& anchor_def_pts = seeds.def_pts;
        MeshQuality mesh_quality = seeds.quality;
        float globalU = seeds.globalU, globalV = seeds.globalV;

        // Path C is the "we have no seed at all" fallback, and its failure aborts
        // the solve. That was right when the front-end was AKAZE, which either
        // produced a mesh or produced nothing. The anchor lattice also yields a
        // phase-correlation lock and often some solved nodes, so a mesh too
        // sparse to guide Path A is not the same as having no seed: Path B can
        // flood-fill from what the lattice already found.
        const bool have_seed = seeds.phase_locked || seeds.solved > 0 || seeds.accepted >= 3;
        bool execute_path_c = (mesh_quality == MeshQuality::NONE) && !have_seed;

        if (cancel_requested()) return kCancelled;

        std::vector<AffineTriangle> affTriangles;
        MeshGuessField guess = build_mesh_guess_field(
                cache, params, anchor_ref_pts, anchor_def_pts, mesh_quality,
                globalU, globalV, gridW, gridH, local_debug_dir,
                resultGrid, affTriangles, timings);

        const SolveContext ctx{cache, params, defImg, gridW, gridH, safe_cores,
                               hessian_pool, global_points_solved, compute_order_counter};

        if (execute_path_c) {
            int rc = run_path_c(ctx, anchor_ref_pts, resultGrid, path_c_seed_x, path_c_seed_y, globalU, globalV);
            if (rc != 0) return rc;
        }

        {
            ScopedTimer pathA_timer(timings.pathA);
            run_path_a(ctx, guess, resultGrid, stats_pathA);
        }

        // Path A is done or abandoned; a cancelled field is incomplete, so stop
        // before spending anything on strain and packing.
        if (cancel_requested()) return kCancelled;

        {
            ScopedTimer pathB_timer(timings.pathB);
            run_path_b(ctx, anchor_ref_pts, anchor_def_pts, globalU, globalV,
                       path_c_seed_x, path_c_seed_y, resultGrid, stats_pathB);
        }

        // Path B's workers have been joined by their guard. A cancelled field is
        // partial, so there is nothing worth deriving strain from.
        if (cancel_requested()) return kCancelled;

        auto packed = pack_full_field_output(
                resultGrid, gridW, gridH, params.step, params.strain_window,
                output_ptr, output_capacity);
        StrainField& strainField = packed.strain;
        int valid_count = packed.valid_count;
        timings.strain = packed.time_strain_ms;

        // 🚀 DIAGNOSTIC: Print exactly how many points the filter caught
        LOGD("DIAGNOSTIC POST-FILTER: Dropped %d noisy points. Final Valid Output: %d", packed.dropped_by_post_filter, valid_count);
        if (packed.output_truncated) {
            LOGE("Output buffer full at %d points (capacity %d floats); remaining points dropped.", valid_count, output_capacity);
        }

        if (on_progress) {
            try { on_progress(100); } catch (...) {
                LOGE("on_progress(100) callback threw; ignoring");
            }
        }

        // ==========================================
        // ⏱️ AGGREGATE PROFILING METRICS
        // ==========================================
        SolveSummary summary = aggregate_thread_stats(stats_pathA, stats_pathB, safe_cores);
        timings.total = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
        log_profiling_summary(timings, summary, valid_count);

        // =========================================================
        // 🐛 EXPORT FULL DEBUG SUITE (extracted — behaviour unchanged)
        // =========================================================
        if (!local_debug_dir.empty()) {
            export_full_field_debug_suite(
                    local_debug_dir, gridW, gridH, params, roiMask,
                    cache.width, cache.height, cache.ref_img,
                    resultGrid, strainField, affTriangles);
        }

        // ==========================================
        // 🚀 NEW: FULL ENGINE TELEMETRY EXPORT TO KOTLIN
        // ==========================================
        fill_engine_metrics(metrics, metrics_len, timings, summary,
                            total_valid_points, valid_count, mesh_quality);

        defMat.release(); roiMask.release();
        return valid_count;
}


} // namespace pipeline
} // namespace Semper
