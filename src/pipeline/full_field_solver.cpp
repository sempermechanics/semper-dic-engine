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
#include <memory>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {

// Cooperative cancel. The token is passed down to every phase (SolveContext)
// and polled in the hot point loops: the cost is one relaxed atomic load per
// point against a full ICGN solve, and a late poll only means one more point,
// never a wrong result.

using namespace Semper::pipeline::internal;

// Re-entrant overload: the solve polls the caller-owned token and nothing else.
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
    return run_full_field_impl(cache, def_gray, roi_mask, params, output_ptr,
                               output_capacity, metrics, metrics_len,
                               std::move(on_progress), nullptr, cancel);
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
    return run_full_field_impl(cache, def_gray, roi_mask, params, output_ptr,
                               output_capacity, metrics, metrics_len,
                               std::move(on_progress), nullptr,
                               detail::global_cancel_token());
}

namespace internal {

namespace {

// Clips [lo, lo + len) to [0, extent) on one axis, moving `lo` forward by
// whole steps so the lattice lo + i * step is kept. 64-bit throughout: a
// caller's lo + len can overflow int. False if nothing is left.
bool clip_axis(int &lo, int &len, int step, int extent) {
    int64_t a = lo;
    const int64_t b = std::min<int64_t>((int64_t)lo + (int64_t)len, extent);
    if (a < 0) a += ((-a + step - 1) / step) * (int64_t)step;
    if (b <= a) return false;
    lo = (int)a;
    len = (int)(b - a);
    return true;
}

} // namespace

int validate_params(FullFieldParams &p, int width, int height) {
    if (width <= 0 || height <= 0) return -3;
    // Every pixel index in the engine is an int.
    if ((int64_t)width * (int64_t)height > (int64_t)std::numeric_limits<int>::max()) return -3;
    // step <= 0 would divide by zero forming the grid; it has always been
    // reported as an invalid ROI.
    if (p.step <= 0) return -2;
    if (p.subset_size < 3 || p.subset_size % 2 == 0) return -3;
    if (p.rect_w <= 0 || p.rect_h <= 0) return -2;
    if (!clip_axis(p.rect_x, p.rect_w, p.step, width)) return -2;
    if (!clip_axis(p.rect_y, p.rect_h, p.step, height)) return -2;
    if (p.rect_w / p.step <= 0 || p.rect_h / p.step <= 0) return -2;
    if ((int64_t)p.strain_window > max_strain_window(p)) return -3;
    return 0;
}

int64_t max_strain_window(const FullFieldParams &p) {
    if (p.step <= 0) return 0;
    const int64_t grid_min = std::min(p.rect_w / p.step, p.rect_h / p.step);
    return 2 * (grid_min + 1) * (int64_t)p.step;
}

NodeSolve solve_node(OptimizationEngine &engine, SubsetData &subset, const Image &ref_img,
                     const Image &def_img, int realX, int realY, int subset_size,
                     const CachedHessianData &cached, const float guess[6],
                     InitializationMode mode) {
    NodeSolve out;
    const auto t0 = std::chrono::high_resolution_clock::now();
    SubsetPrecomputer::precompute_subset_fast(subset, ref_img, realX, realY, subset_size, cached);
    out.hessian_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    out.initialized = subset.is_initialized;
    if (!out.initialized) return out;

    const int simplex_before = engine.count_simplex;
    out.res = engine.calculate_deformation(subset, def_img, guess[0], guess[1], guess[2],
                                           guess[3], guess[4], guess[5], mode);
    // VSG protector: a subset that reaches into the Ghost Wall must not pass,
    // least of all on Path B, where an accepted cell seeds its neighbours.
    reject_if_ghosted(out.res, subset_size);
    out.needed_rescue = engine.count_simplex > simplex_before;
    return out;
}

int run_full_field_impl(
    ReferenceCache& cache,
    const cv::Mat& def_gray,
    const cv::Mat& roi_mask,
    const FullFieldParams& params_in,
    float* output_ptr,
    int output_capacity,
    float* metrics,
    int metrics_len,
    ProgressCallback on_progress,
    FieldCapture* capture,
    CancelToken& cancel) {

        std::string local_debug_dir = cache.debug_dir;
        cache.debug_dir.clear();

        auto t_total_start = std::chrono::high_resolution_clock::now();
        PhaseTimings timings;

        // 🚀 PRIORITY 3: Return -3 for memory/init errors
        if (output_ptr == nullptr || cache.ref_img == nullptr || def_gray.empty()) return -3;
        // The deformed frame is correlated pixel for pixel against the cached
        // reference, as 8-bit gray: another size used to skip phase
        // correlation and then solve anyway, another type was read as bytes.
        if (def_gray.type() != CV_8UC1 || def_gray.cols != cache.width ||
            def_gray.rows != cache.height) {
            return -3;
        }
        // The mask is read as bytes too (it may be any size; it is resized).
        if (!roi_mask.empty() && roi_mask.type() != CV_8UC1) return -3;
        FullFieldParams params = params_in;
        const int param_rc = validate_params(params, cache.width, cache.height);
        if (param_rc != 0) return param_rc;
        SEMPER_ASSERT(output_capacity >= 0);
        // Clear any leftover cancel from a previous solve so this run starts fresh.
        // Without this, a caller that cancelled and then started a new solve
        // without resetting would get an instant -99.
        cancel.clear();

        LOGD("=== computeFullFieldDirect (HYBRID CORE) START ===");

        auto t_prep_start = std::chrono::high_resolution_clock::now();
        // Image reads rows * cols contiguous bytes; a view into a larger Mat
        // is not.
        cv::Mat defMat = def_gray.isContinuous() ? def_gray : def_gray.clone();
        cv::Mat roiMask;
        if (!roi_mask.empty()) {
            roiMask = roi_mask;
            if (roiMask.cols != cache.width || roiMask.rows != cache.height) {
                cv::resize(roiMask, roiMask, cv::Size(cache.width, cache.height), 0, 0, cv::INTER_NEAREST);
            }
        }

        // Heap-allocated so incremental DIC can take it over as the next
        // reference (FieldCapture::keep_deformed) instead of preparing it again.
        auto defImgOwner = std::make_unique<Image>(defMat.cols, defMat.rows, defMat.data);
        Image& defImg = *defImgOwner;
        defImg.prepare_data(false);
        timings.img_prep = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_prep_start).count();

        int safe_cores = std::max(1, (int)std::thread::hardware_concurrency());

        // validate_params clipped the rect to the image, so the grid holds at
        // most one node per pixel and every count below fits an int.
        const int gridW = params.rect_w / params.step;
        const int gridH = params.rect_h / params.step;

        // The VSG strain fit needs 3 grid nodes inside the strain window to
        // solve for a plane. That count is fixed by (step, strain_window)
        // alone, so when it is short every point is rejected by the strain
        // post-filter no matter how well the correlation went, and the caller
        // gets an empty field. Until 0.3.0 that came back as a *success* with
        // zero points, which is indistinguishable from a genuinely blank ROI
        // and left the caller nothing to debug from. Check it before solving.
        const int vsg_nodes = StrainCalculator::vsg_window_node_count(
                params.step, params.strain_window);
        if (vsg_nodes < 3) {
            // Derive the remedy from the function the gate itself calls, so the
            // advice cannot drift from the rule. strain_window == 2 * step always
            // clears it (radius == step puts 5 nodes in the window), which bounds
            // the search; this runs only on the error path.
            int min_window = 1;
            while (StrainCalculator::vsg_window_node_count(params.step, min_window) < 3)
                ++min_window;
            LOGE("strain_window=%d spans only %d grid node(s) at step=%d; the VSG "
                 "plane fit needs 3, so every point would be discarded. Raise "
                 "strain_window to at least %d, or lower step.",
                 params.strain_window, vsg_nodes, params.step, min_window);
            return kBadStrainWindow;
        }

        // Now that gridW and gridH exist, we can set their default fallbacks
        int path_c_seed_x = gridW / 2;
        int path_c_seed_y = gridH / 2;

        ResultGrid resultGrid((size_t)gridH, std::vector<GridPoint>((size_t)gridW));
        int total_valid_points = 0;

        // The scan below and the Hessian pre-pass depend only on the reference,
        // the grid and the mask, so a batch against one reference computes them
        // once (RefPrecompute). A hit replays the stored per-node verdicts into
        // the same initializer the scan uses.
        if (!cache.precompute) cache.precompute = std::make_unique<RefPrecompute>();
        RefPrecompute& pre = *cache.precompute;
        const bool grid_cached = pre.grid_matches(params, roiMask) &&
                                 pre.gridW == gridW && pre.gridH == gridH;
        if (grid_cached) {
            for (int y = 0; y < gridH; ++y) {
                for (int x = 0; x < gridW; ++x) {
                    const bool shouldSkip = pre.skip[(size_t)y * gridW + x] != 0;
                    int realX = params.rect_x + x * params.step;
                    int realY = params.rect_y + y * params.step;
                    resultGrid[y][x] = {(float)realX, (float)realY, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, CORR_INVALID, shouldSkip, -1, -1, 0, false, 0};
                }
            }
            total_valid_points = pre.total_valid_points;
        } else {
        pre.grid_valid = false;

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
            void stop() {
                stop_flag.store(true, std::memory_order_release);
                {
                    std::unique_lock<std::mutex> lk(cv_mutex);
                    cv.wait_for(lk, std::chrono::seconds(2), [&]{ return detached_flag; });
                }
                if (t.joinable()) t.join();
            }
            ~ThreadJoinGuard() { stop(); }
        };
        ThreadJoinGuard progressGuard{progress_thread, progress_thread_should_stop, progress_cv, progress_cv_mutex, progress_thread_detached};

        // =========================================================
        // 🚀 GLOBAL HESSIAN PRE-PASS & CONTRAST THRESHOLDING
        // =========================================================
        // Runs before seeding: the anchor lattice sits on grid nodes and reuses
        // this pool, so its solves cost little more than the field costs anyway.
        auto t_prepass_start = std::chrono::high_resolution_clock::now();
        if (!grid_cached) {
        pre.hessian_pool.assign((size_t)gridW * gridH, CachedHessianData());
        HessianPool& hessian_pool = pre.hessian_pool;
        RegionErrors prepass_errors;

    #pragma omp parallel for schedule(static) num_threads(safe_cores)
        for (int pool_idx = 0; pool_idx < gridW * gridH; ++pool_idx) {
            // OpenMP forbids breaking out of a parallel for, so a cancel (or
            // an exception, see RegionErrors) skips the remaining iterations.
            if (cancel.requested() || prepass_errors.failed()) continue;
            try {
                int gx = pool_idx % gridW, gy = pool_idx / gridW;
                if (!resultGrid[gy][gx].solved) {
                    int realX = params.rect_x + gx * params.step, realY = params.rect_y + gy * params.step;
                    hessian_pool[pool_idx] = SubsetPrecomputer::compute_hessian_only(*cache.ref_img, realX, realY, params.subset_size);
                }
            } catch (...) {
                prepass_errors.capture();
            }
        }
        // The pool is left marked stale (grid_valid is false until below).
        prepass_errors.rethrow_if_any();

        // The cancel flag only ever goes up during a solve, so if it is still
        // clear here every iteration above ran: the pool is complete and safe
        // to reuse. Otherwise it stays marked stale for the next solve.
        if (!cancel.requested()) {
            pre.rect_x = params.rect_x; pre.rect_y = params.rect_y;
            pre.rect_w = params.rect_w; pre.rect_h = params.rect_h;
            pre.step = params.step; pre.subset_size = params.subset_size;
            pre.roi_mask = roiMask.empty() ? cv::Mat() : roiMask.clone();
            pre.gridW = gridW; pre.gridH = gridH;
            pre.skip.assign((size_t)gridW * gridH, 0);
            for (int y = 0; y < gridH; ++y)
                for (int x = 0; x < gridW; ++x)
                    pre.skip[(size_t)y * gridW + x] = resultGrid[y][x].solved ? 1 : 0;
            pre.total_valid_points = total_valid_points;
            pre.grid_valid = true;
        }
        }
        const HessianPool& hessian_pool = pre.hessian_pool;

        timings.prepass = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_prepass_start).count();

        // Anchor-lattice seeding: an AKAZE homography for the global motion,
        // then IC-GN on a lattice of grid nodes. It records but publishes
        // nothing; see AnchorResult for why that has to be a separate step.
        std::vector<ThreadStats> stats_anchors(safe_cores);
        MeshSeedResult seeds = solve_anchor_seeds(
                cache, defMat, roiMask, defImg, params, gridW, gridH, safe_cores,
                hessian_pool, resultGrid, stats_anchors, timings, cancel);

        // Now apply them. Nodes that clear the result gate *and* survive the
        // median test are marked solved, so Path A skips them and Path B takes
        // them as boundary seeds.
        const int anchor_published = publish_anchor_results(
                seeds, global_points_solved, compute_order_counter,
                resultGrid, stats_anchors);
        LOGD("Anchor lattice published %d of %d lattice nodes (%d converged)",
             anchor_published, (int)seeds.anchors.size(), seeds.attempted);

        const std::vector<cv::Point2f>& anchor_ref_pts = seeds.ref_pts;
        const std::vector<cv::Point2f>& anchor_def_pts = seeds.def_pts;
        MeshQuality mesh_quality = seeds.quality;
        float globalU = seeds.globalU, globalV = seeds.globalV;

        // Path C is the "we have no seed at all" fallback, and its failure aborts
        // the solve. That was right when AKAZE fed the mesh directly and either
        // produced a mesh or produced nothing. The anchor lattice also yields a
        // global model and often some solved nodes, so a mesh too sparse to
        // guide Path A is not the same as having no seed: Path B can
        // flood-fill from what the lattice already found.
        //
        // A model on its own is not a seed, though. AKAZE can lock on the
        // global motion while every anchor fails the result gate:
        // then solved == 0, accepted == 0, and Path B has nothing to flood
        // fill from but the blind grid-centre node. Require an actual point,
        // and let the lock inform Path C's search rather than replace it.
        const bool have_seed = seeds.solved > 0 || seeds.accepted >= 3;
        bool execute_path_c = (mesh_quality == MeshQuality::NONE) && !have_seed;

        if (cancel.requested()) return kCancelled;

        std::vector<AffineTriangle> affTriangles;
        MeshGuessField guess = build_mesh_guess_field(
                cache, params, anchor_ref_pts, anchor_def_pts, mesh_quality,
                globalU, globalV, gridW, gridH, local_debug_dir,
                resultGrid, affTriangles, timings, seeds.homography);

        const SolveContext ctx{cache, params, defImg, gridW, gridH, safe_cores,
                               hessian_pool, global_points_solved, compute_order_counter,
                               cancel};

        if (execute_path_c) {
            // Path C runs in cases a bare model lock does not suppress (see
            // have_seed above), so its failure can no longer abort the solve:
            // that would turn fields Path B can deliver into an error return.
            // When AKAZE locked, fall through instead -- Path B still has the
            // global shift and the grid-centre seed. Without a lock there is
            // nothing left to try, so the abort stands.
            int rc = run_path_c(ctx, anchor_ref_pts, resultGrid, path_c_seed_x, path_c_seed_y, globalU, globalV);
            if (rc != 0) {
                if (!seeds.locked) return rc;
                LOGE("Path C failed (%d); AKAZE locked, continuing to Path B with the global shift", rc);
            }
        }

        {
            ScopedTimer pathA_timer(timings.pathA);
            run_path_a(ctx, guess, resultGrid, stats_pathA);
        }

        // Path A is done or abandoned; a cancelled field is incomplete, so stop
        // before spending anything on strain and packing.
        if (cancel.requested()) return kCancelled;

        {
            ScopedTimer pathB_timer(timings.pathB);
            run_path_b(ctx, anchor_ref_pts, anchor_def_pts, globalU, globalV,
                       path_c_seed_x, path_c_seed_y, resultGrid, stats_pathB);
        }

        // Path B's workers have been joined by their guard. A cancelled field is
        // partial, so there is nothing worth deriving strain from.
        if (cancel.requested()) return kCancelled;

        if (capture) {
            capture->gridW = gridW;
            capture->gridH = gridH;
            capture->rect_x = params.rect_x;
            capture->rect_y = params.rect_y;
            capture->step = params.step;
            capture->attempted = total_valid_points;
            capture->grid = resultGrid;
        }

        auto packed = pack_full_field_output(
                resultGrid, gridW, gridH, params.step, params.strain_window,
                output_ptr, output_capacity);
        StrainField& strainField = packed.strain;
        int valid_count = packed.valid_count;
        timings.strain = packed.time_strain_ms;

        // 🚀 DIAGNOSTIC: Print exactly how many points the filter caught
        LOGD("DIAGNOSTIC POST-FILTER: Dropped %d displacement outliers and %d noisy points. Final Valid Output: %d",
             packed.dropped_as_outlier, packed.dropped_by_post_filter, valid_count);
        if (packed.output_truncated) {
            LOGE("Output buffer full at %d points (capacity %d floats); remaining points dropped.", valid_count, output_capacity);
        }

        // The ticker is joined first, so the callback is never entered from
        // two threads at once (the final 100 used to race the last tick).
        progressGuard.stop();
        if (on_progress) {
            try { on_progress(100); } catch (...) {
                LOGE("on_progress(100) callback threw; ignoring");
            }
        }

        // ==========================================
        // ⏱️ AGGREGATE PROFILING METRICS
        // ==========================================
        SolveSummary summary = aggregate_thread_stats(stats_pathA, stats_pathB, stats_anchors, safe_cores);
        timings.total = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
        log_profiling_summary(timings, summary, seeds, valid_count);

        // =========================================================
        // 🐛 EXPORT FULL DEBUG SUITE (extracted — behaviour unchanged)
        // =========================================================
        if (!local_debug_dir.empty()) {
            export_full_field_debug_suite(
                    local_debug_dir, gridW, gridH, params, roiMask,
                    cache.width, cache.height, cache.ref_img.get(),
                    resultGrid, strainField, affTriangles);
        }

        // ==========================================
        // 🚀 NEW: FULL ENGINE TELEMETRY EXPORT TO KOTLIN
        // ==========================================
        fill_engine_metrics(metrics, metrics_len, timings, summary,
                            total_valid_points, valid_count,
                            packed.dropped_by_post_filter, packed.dropped_as_outlier,
                            seeds, mesh_quality);

        if (capture && capture->keep_deformed) {
            capture->deformed = std::move(defImgOwner);
            capture->deformed_gray = def_gray.clone();
            capture->akaze_deformed = std::move(seeds.def_features);
        }

        defMat.release(); roiMask.release();
        return valid_count;
}

} // namespace internal


} // namespace pipeline
} // namespace Semper
