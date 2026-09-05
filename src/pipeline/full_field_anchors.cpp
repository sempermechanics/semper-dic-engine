// Anchor-lattice seeding: the front-end that produces the Delaunay mesh vertices.
//
// Two layers, neither of which uses a feature detector or a descriptor:
//
//   L0  cv::phaseCorrelate over the ROI gives the global rigid shift from every
//       pixel, rather than from the median of sparse keypoint matches.
//   L1  a regular lattice of ROI grid nodes is solved with the engine's own
//       IC-GN, seeded from L0. The converged nodes become the mesh vertices.
//
// Anchors sit on grid nodes on purpose. They reuse the Hessian pool the solve
// already builds, and a node good enough to keep is a *final* result for that
// node — Path A skips it and Path B picks it up as a boundary seed — so the
// lattice adds almost no work beyond what the field costs anyway.
//
// Why this rather than descriptor matching: build_mesh_guess_field derives the
// guess gradients from cv::getAffineTransform over triangle vertices, so a
// vertex position error e over an edge of length L becomes a gradient error of
// order e/L. Keypoints give e ~ 1 px on short, clustered edges; IC-GN anchors
// give e ~ 0.01 px on edges fixed at stride * step. See docs/SEEDING_BENCHMARK.md.

#include "full_field_internal.hpp"

#include <semper/cancel.hpp>
#include <semper/seeding.hpp>
#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <omp.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

namespace {

float median_of(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + (long)mid, v.end());
    return v[mid];
}

// One lattice node's outcome, before the median test decides whether to keep it.
struct Anchor {
    float rx = 0, ry = 0, u = 0, v = 0;
    int gx = 0, gy = 0;
    bool vertex = false;   // passed the loose seed gate
    bool output = false;   // also passed the strict result gate
};

} // namespace

MeshSeedResult solve_anchor_seeds(
        ReferenceCache& cache,
        const cv::Mat& defMat,
        const Image& def_img,
        const FullFieldParams& params,
        int gridW,
        int gridH,
        int safe_cores,
        const HessianPool& hessian_pool,
        std::atomic<int>& global_points_solved,
        std::atomic<int>& compute_order_counter,
        ResultGrid& resultGrid,
        std::vector<ThreadStats>& stats,
        PhaseTimings& timings) {

    MeshSeedResult out;
    if (cache.ref_img == nullptr || gridW <= 0 || gridH <= 0) return out;

    // --- L0: global rigid shift ------------------------------------------
    double pu = 0.0, pv = 0.0, presp = 0.0;
    {
        ScopedTimer t(timings.phase_corr);
        if (!cache.gray.empty()) {
            out.phase_locked = seeding::phase_correlate_roi(
                    cache.gray, defMat,
                    cv::Rect(params.rect_x, params.rect_y, params.rect_w, params.rect_h),
                    pu, pv, presp);
        }
    }
    out.globalU = (float)pu;
    out.globalV = (float)pv;

    ScopedTimer anchor_timer(timings.anchors);

    // --- L1: lattice over grid nodes -------------------------------------
    // Aim for kAnchorTarget nodes, and always include the last row and column:
    // stopping at stride multiples leaves the convex hull short of the ROI
    // edge, which costs coverage exactly where the mesh has no triangle.
    int stride = (int)std::lround(std::sqrt((double)(gridW * gridH) /
                                            (double)tuning::kAnchorTarget));
    stride = std::max(tuning::kAnchorStrideMin, stride);
    stride = std::min(stride, std::max(1, std::min(gridW, gridH) / 3));

    std::vector<int> lat_x, lat_y;
    for (int gx = 0; gx < gridW; gx += stride) lat_x.push_back(gx);
    if (lat_x.back() != gridW - 1) lat_x.push_back(gridW - 1);
    for (int gy = 0; gy < gridH; gy += stride) lat_y.push_back(gy);
    if (lat_y.back() != gridH - 1) lat_y.push_back(gridH - 1);

    const int LW = (int)lat_x.size(), LH = (int)lat_y.size();
    if (LW < 2 || LH < 2) return out;

    std::vector<Anchor> anchors((size_t)LW * LH);
    const int n_anchor = LW * LH;

    // With a phase-correlation lock the guess is already sub-pixel, so pure
    // ICGN converges and the 6-DOF Simplex rescue is pure cost. Without a lock
    // each anchor has to find its own offset, so the coarse search stays.
    const auto init_mode = out.phase_locked ? INIT_NO_SIMPLEX : INIT_AUTO_SEARCH;

    std::atomic<int> attempted(0);

#pragma omp parallel num_threads(safe_cores)
    {
        const int tid = omp_get_thread_num();
        OptimizationEngine engine;
        Semper::SubsetData subset;
        engine.lm_enabled = true;
        engine.lm_alpha = tuning::kLmAlpha;
        engine.use_6x6_interpolator = params.use_6x6_interpolator;
        double local_hessian = 0.0, local_wait = 0.0;
        int local_pts = 0;
        EngineStatFlusher flusher(engine, stats[tid], local_pts, local_hessian, local_wait);

#pragma omp for schedule(dynamic, 4)
        for (int idx = 0; idx < n_anchor; ++idx) {
            if (cancel_requested()) continue;

            const int gx = lat_x[(size_t)(idx % LW)];
            const int gy = lat_y[(size_t)(idx / LW)];
            // No ROI mask check here: run_full_field's strict per-pixel scan
            // already pre-set `solved` on every node the mask or the boundary
            // reserve rejects, and the Hessian pool is only built for the rest.
            if (resultGrid[gy][gx].solved) continue;

            const int pool_idx = gy * gridW + gx;
            const int realX = params.rect_x + gx * params.step;
            const int realY = params.rect_y + gy * params.step;
            attempted.fetch_add(1, std::memory_order_relaxed);

            auto th1 = std::chrono::high_resolution_clock::now();
            SubsetPrecomputer::precompute_subset_fast(subset, *cache.ref_img, realX, realY,
                                                      params.subset_size,
                                                      hessian_pool[(size_t)pool_idx]);
            local_hessian += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - th1).count();
            if (!subset.is_initialized) continue;

            const int simplex_before = engine.count_simplex;
            AnalysisResult res = engine.calculate_deformation(
                    subset, def_img, out.globalU, out.globalV,
                    0.0f, 0.0f, 0.0f, 0.0f, init_mode);
            reject_if_ghosted(res, params.subset_size);
            stats[tid].icgn_iters += res.iters;
            const bool needed_rescue = (engine.count_simplex > simplex_before);
            if (needed_rescue) record_simplex_outcome(stats[tid], res);

            if (res.status != 0) continue;

            Anchor& a = anchors[(size_t)idx];
            a.rx = (float)realX; a.ry = (float)realY;
            a.u = res.u; a.v = res.v;
            a.gx = gx; a.gy = gy;

            // Two gates. A mesh vertex only has to be approximately right, and
            // the median test below rejects blunders; an output point has to
            // clear the same bar Path A applies.
            a.vertex = res.correlation_score <= tuning::kAnchorAcceptScore;
            a.output = res.correlation_score <= tuning::kCorrAccept;

            if (a.output) {
                const int order = compute_order_counter.fetch_add(1, std::memory_order_relaxed);
                resultGrid[gy][gx] = {(float)realX, (float)realY, res.u, res.v,
                                      res.ux, res.uy, res.vx, res.vy,
                                      res.correlation_score, true, tid, order,
                                      resultGrid[gy][gx].mesh_assignment_type,
                                      needed_rescue, res.iters};
                global_points_solved.fetch_add(1, std::memory_order_relaxed);
                local_pts++;
            }
        }
    }

    out.attempted = attempted.load(std::memory_order_relaxed);

    // Universal median test (Westerweel & Scarano) over the lattice. A global
    // affine or homography fit would be the wrong model: under a real strain
    // field the displacements are not affine, so a global fit rejects signal.
    // The 8-neighbourhood median assumes no field shape, and the lattice makes
    // the lookup O(1).
    std::vector<float> us, vs;
    for (int lj = 0; lj < LH; ++lj) {
        for (int li = 0; li < LW; ++li) {
            const size_t idx = (size_t)lj * LW + li;
            const Anchor& a = anchors[idx];
            if (!a.vertex) continue;

            std::vector<float> nu, nv;
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0) continue;
                    const int nj = lj + dj, ni = li + di;
                    if (ni < 0 || ni >= LW || nj < 0 || nj >= LH) continue;
                    const Anchor& nb = anchors[(size_t)nj * LW + ni];
                    if (!nb.vertex) continue;
                    nu.push_back(nb.u);
                    nv.push_back(nb.v);
                }
            }
            if (nu.size() >= 3) {
                const float mu = median_of(nu), mv = median_of(nv);
                if (std::abs(a.u - mu) > tuning::kAnchorMedianTol ||
                    std::abs(a.v - mv) > tuning::kAnchorMedianTol) {
                    continue;   // blunder
                }
            }
            out.ref_pts.emplace_back(a.rx, a.ry);
            out.def_pts.emplace_back(a.rx + a.u, a.ry + a.v);
            us.push_back(a.u);
            vs.push_back(a.v);
            if (a.output) out.solved++;
        }
    }
    out.accepted = (int)out.ref_pts.size();

    if (out.accepted >= 3) {
        out.globalU = median_of(us);
        out.globalV = median_of(vs);
        std::vector<cv::Point2f> hull;
        cv::convexHull(out.ref_pts, hull);
        const float roi_area = (float)(params.rect_w * params.rect_h);
        out.coverage = roi_area > 0.0f ? (float)cv::contourArea(hull) / roi_area : 0.0f;
    }

    const float accepted_fraction = out.attempted > 0
            ? (float)out.accepted / (float)out.attempted : 0.0f;
    // The old floor of 10 vertices came from AKAZE, where findHomography needed
    // that many correspondences. A Delaunay mesh needs three — one triangle —
    // so a sparse mesh is useful well below the old threshold, and on a small
    // ROI the lattice legitimately produces only a handful of nodes.
    if (out.accepted >= 10 && accepted_fraction >= tuning::kAnchorFullFraction &&
        out.coverage >= 0.30f) {
        out.quality = MeshQuality::FULL;
    } else if (out.accepted >= 3 && out.coverage >= tuning::kAnchorSparseCoverage) {
        out.quality = MeshQuality::SPARSE;
    } else {
        out.quality = MeshQuality::NONE;
    }

    LOGD("ROUTING: anchor lattice %d/%d kept (%d as output, cov %.2f, phase %s) -> quality %d",
         out.accepted, out.attempted, out.solved, (double)out.coverage,
         out.phase_locked ? "locked" : "none", (int)out.quality);
    return out;
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
