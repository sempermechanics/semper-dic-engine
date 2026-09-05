#ifndef SEMPER_PIPELINE_FULL_FIELD_INTERNAL_HPP
#define SEMPER_PIPELINE_FULL_FIELD_INTERNAL_HPP

// Internal contract between the full-field solver translation units.
//
// run_full_field used to be one ~1500-line function; the phases below were
// lifted out of it verbatim so each lives in its own .cpp. Nothing here is part
// of the public API (that stays in <semper/pipeline.hpp>) and nothing here
// changes the algorithm, the OpenMP scheduling, or the arithmetic — the types
// are the former function-local structs, promoted to namespace scope so more
// than one TU can name them.

#include <semper/pipeline.hpp>
#include <semper/solver.hpp>
#include <semper/strain.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include <semper/types.hpp>

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace Semper {
namespace pipeline {
namespace internal {

struct ThreadStats {
    double icgn_time_ms = 0.0;
    int icgn_iters = 0;
    double simplex_time_ms = 0.0;
    int simplex_iters = 0;
    double hessian_time_ms = 0.0;
    double queue_wait_time_ms = 0.0;
    int points_solved = 0;

    // Detailed Simplex Telemetry
    int simplex_calls = 0;
    int simplex_saved = 0;
    int simplex_dead = 0;

    // Track WHY it went to Simplex
    int simplex_from_crash = 0;   // Failed in < 20 iters (Bad guess / out of bounds)
    int simplex_from_timeout = 0; // Failed at exactly 20 iters (Non-convergence)
};

struct EngineStatFlusher {
    OptimizationEngine &engine;
    ThreadStats &bucket;
    int &local_points;
    double &local_hessian;
    double &local_wait;

    EngineStatFlusher(OptimizationEngine &e, ThreadStats &b, int &lp, double &lh, double &lw)
            : engine(e), bucket(b), local_points(lp), local_hessian(lh), local_wait(lw) {}

    ~EngineStatFlusher() {
        bucket.icgn_time_ms += engine.time_icgn_ms;
        //bucket.icgn_iters += engine.count_icgn;
        bucket.simplex_time_ms += engine.time_simplex_ms;
        bucket.simplex_iters += engine.count_simplex;
        bucket.points_solved += local_points;
        bucket.hessian_time_ms += local_hessian;
        bucket.queue_wait_time_ms += local_wait;
    }
};

struct ScopedTimer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    double &out_ms;
    explicit ScopedTimer(double &out) : out_ms(out) {
        start = std::chrono::high_resolution_clock::now();
    }
    ~ScopedTimer() {
        out_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
    }
};

// Sentinel for "no valid correlation here" (skipped or failed point).
// ZNSSD is >= 0 for every real solve, so a negative value can never be
// confused with a genuinely perfect match (ZNSSD == 0.0), which the old
// corr==0 convention silently discarded.
inline constexpr float CORR_INVALID = tuning::kCorrInvalid;

// Toggle Simplex ON or OFF. constexpr so the dead branches fold away exactly
// as they did when this was a function-local `bool ... = true`.
inline constexpr bool ALLOW_SIMPLEX_RESCUE = true;

// Default-initialized so a new field can never leave a call site
// reading indeterminate memory. The aggregate-init sites below fill the
// first 15 members positionally, in this exact order:
//   0:x 1:y 2:u 3:v 4:ux 5:uy 6:vx 7:vy 8:corr 9:solved 10:thread_id
//   11:compute_order 12:mesh_assignment_type 13:used_simplex 14:icgn_iters
// (guess_* default to 0). Keep that order in sync with those sites.
// PointState (see point_state.hpp) documents skip/mesh/solved semantics;
// the hot GridPoint keeps a bool `solved` so layout/throughput stay stable.
struct GridPoint {
    float x = 0, y = 0, u = 0, v = 0, ux = 0, uy = 0, vx = 0, vy = 0, corr = 0;
    bool solved = false;
    int thread_id = 0;
    int compute_order = 0;
    int mesh_assignment_type = 0;
    bool used_simplex = false;
    int icgn_iters = 0;
    float guess_u = 0, guess_v = 0, guess_ux = 0, guess_uy = 0, guess_vx = 0, guess_vy = 0;
};

using ResultGrid = std::vector<std::vector<GridPoint>>;

struct AffineTriangle {
    cv::Point2f pts[3];
    double u, v, ux, uy, vx, vy;
    cv::Rect2f boundingBox;
};

// How much of the ROI the anchor mesh actually covers.
enum class MeshQuality { NONE, SPARSE, FULL };

using HessianPool = std::vector<CachedHessianData, Eigen::aligned_allocator<CachedHessianData>>;

// Everything the three solve paths read out of run_full_field's frame. Passed
// by const reference so the extracted phases keep the same view of the solve
// they had as inline blocks.
struct SolveContext {
    ReferenceCache &cache;
    const FullFieldParams &params;
    const Image &def_img;
    int gridW;
    int gridH;
    int safe_cores;
    const HessianPool &hessian_pool;
    std::atomic<int> &global_points_solved;
    std::atomic<int> &compute_order_counter;
};

// Per-phase wall clock, aggregated for the profiling log and the metrics array.
struct PhaseTimings {
    // `anchors` and `phase_corr` were `akaze` and `ransac`; they feed the same
    // Frozen metrics[10] slot, whose meaning is now "seeding time (ms)".
    double img_prep = 0, anchors = 0, phase_corr = 0, delaunay = 0, contour_assign = 0;
    double extrapolate = 0, smoothing = 0, prepass = 0, pathA = 0, pathB = 0;
    double strain = 0, total = 0;
};

// ---------------------------------------------------------------------------
// Shared per-point post-processing (identical at every calculate_deformation
// call site in Paths A and B — extracted, not rewritten).
// ---------------------------------------------------------------------------

// VSG PROTECTOR: reject if > 5% of the subset fell into the Ghost Wall.
inline void reject_if_ghosted(AnalysisResult &res, int subset_size) {
    float ghost_fraction = (float)res.invalid_ref_pixels / (float)(subset_size * subset_size);
    if (ghost_fraction > tuning::kGhostRejectFraction) {
        res.correlation_score = 2.0f;
        res.status = 1;
    }
}

// Classify a Simplex rescue on the REAL iteration count against kIcgnMaxIter,
// then record whether it saved the point or died.
inline void record_simplex_outcome(ThreadStats &bucket, const AnalysisResult &res) {
    bucket.simplex_calls++;

    if (res.iters >= tuning::kIcgnMaxIter) {
        bucket.simplex_from_timeout++;
    } else {
        bucket.simplex_from_crash++;
    }

    if (res.status == 0 && res.correlation_score <= tuning::kCorrAccept && ALLOW_SIMPLEX_RESCUE) {
        bucket.simplex_saved++;
    } else {
        bucket.simplex_dead++;
    }
}

// ---------------------------------------------------------------------------
// Anchor-lattice seeding, Delaunay mesh, mesh-guided solve
// (full_field_anchors.cpp, full_field_mesh.cpp, full_field_path_a.cpp)
// ---------------------------------------------------------------------------

// Outcome of the anchor lattice: the mesh vertices it produced, how well they
// cover the ROI, and the median rigid shift.
//
// The anchors are grid nodes, so they reuse the Hessian pool and their solved
// results are final for those nodes — Path A skips them and Path B picks them
// up as boundary seeds. Vertices are kept on a looser ZNSSD gate than results:
// a seed only has to be approximately right, an output point has to be right.
struct MeshSeedResult {
    std::vector<cv::Point2f> ref_pts;
    std::vector<cv::Point2f> def_pts;
    MeshQuality quality = MeshQuality::NONE;
    float globalU = 0.0f;
    float globalV = 0.0f;
    float coverage = 0.0f;
    int attempted = 0;
    int accepted = 0;      // kept as mesh vertices
    int solved = 0;        // also good enough to keep as output points
    bool phase_locked = false;
};

MeshSeedResult solve_anchor_seeds(
        ReferenceCache &cache,
        const cv::Mat &defMat,
        const Image &def_img,
        const FullFieldParams &params,
        int gridW,
        int gridH,
        int safe_cores,
        const HessianPool &hessian_pool,
        std::atomic<int> &global_points_solved,
        std::atomic<int> &compute_order_counter,
        ResultGrid &resultGrid,
        std::vector<ThreadStats> &stats,
        PhaseTimings &timings);

// Per-grid-point 6-DOF initial guess produced by the Delaunay mesh.
struct MeshGuessField {
    std::vector<float> u, v, ux, uy, vx, vy;
    std::vector<bool> in_mesh;
};

MeshGuessField build_mesh_guess_field(
        const ReferenceCache &cache,
        const FullFieldParams &params,
        const std::vector<cv::Point2f> &akaze_ref_pts,
        const std::vector<cv::Point2f> &akaze_def_pts,
        MeshQuality mesh_quality,
        float globalU,
        float globalV,
        int gridW,
        int gridH,
        const std::string &debug_dir,
        ResultGrid &resultGrid,
        std::vector<AffineTriangle> &affTriangles,
        PhaseTimings &timings);

void run_path_a(
        const SolveContext &ctx,
        const MeshGuessField &guess,
        ResultGrid &resultGrid,
        std::vector<ThreadStats> &stats_pathA);

// ---------------------------------------------------------------------------
// Path B — global priority queue flood fill (full_field_path_b.cpp)
// ---------------------------------------------------------------------------

void run_path_b(
        const SolveContext &ctx,
        const std::vector<cv::Point2f> &akaze_ref_pts,
        const std::vector<cv::Point2f> &akaze_def_pts,
        float globalU,
        float globalV,
        int path_c_seed_x,
        int path_c_seed_y,
        ResultGrid &resultGrid,
        std::vector<ThreadStats> &stats_pathB);

// ---------------------------------------------------------------------------
// Path C — central-tendency fallback seeding (full_field_path_c.cpp)
// ---------------------------------------------------------------------------

// Returns 0 on success (globalU/globalV and the seed indices are updated), or
// the negative error code run_full_field should return to its caller.
int run_path_c(
        const SolveContext &ctx,
        const std::vector<cv::Point2f> &akaze_ref_pts,
        ResultGrid &resultGrid,
        int &path_c_seed_x,
        int &path_c_seed_y,
        float &globalU,
        float &globalV);

// ---------------------------------------------------------------------------
// Output packing, telemetry aggregation and logging
// (full_field_solver_stats.cpp)
// ---------------------------------------------------------------------------

struct PackedFieldResult {
    StrainField strain;
    int valid_count = 0;
    int dropped_by_post_filter = 0;
    bool output_truncated = false;
    double time_strain_ms = 0.0;
};

PackedFieldResult pack_full_field_output(
        ResultGrid &resultGrid,
        int gridW,
        int gridH,
        int step,
        int strain_window,
        float *output_ptr,
        int output_capacity);

struct SolveSummary {
    double a_icgn = 0, a_simp = 0, b_icgn = 0, b_simp = 0, b_wait = 0;
    int total_icgn_iters = 0;

    int a_simp_calls = 0, a_simp_saved = 0, a_simp_dead = 0, a_simp_crash = 0, a_simp_timeout = 0;
    int b_simp_calls = 0, b_simp_saved = 0, b_simp_dead = 0, b_simp_crash = 0, b_simp_timeout = 0;

    double total_hessian = 0;
    int pathA_pts = 0, pathB_pts = 0;
};

SolveSummary aggregate_thread_stats(
        const std::vector<ThreadStats> &stats_pathA,
        const std::vector<ThreadStats> &stats_pathB,
        int safe_cores);

void log_profiling_summary(
        const PhaseTimings &timings,
        const SolveSummary &summary,
        int valid_count);

void fill_engine_metrics(
        float *metrics,
        int metrics_len,
        const PhaseTimings &timings,
        const SolveSummary &summary,
        int total_valid_points,
        int valid_count,
        MeshQuality mesh_quality);

// ---------------------------------------------------------------------------
// Debug-only export (maps/CSV). Production leaves debug_dir empty.
// (full_field_debug_export.cpp)
// ---------------------------------------------------------------------------

void export_full_field_debug_suite(
        const std::string &debug_dir,
        int gridW,
        int gridH,
        const FullFieldParams &params,
        const cv::Mat &roiMask,
        int cache_width,
        int cache_height,
        Image *ref_img,
        const ResultGrid &resultGrid,
        const StrainField &strainField,
        const std::vector<AffineTriangle> &affTriangles);

} // namespace internal
} // namespace pipeline
} // namespace Semper

#endif // SEMPER_PIPELINE_FULL_FIELD_INTERNAL_HPP
