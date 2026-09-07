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
#include <array>
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

float median_of(float* v, int n) {
    if (n <= 0) return 0.0f;
    const int mid = n / 2;
    std::nth_element(v, v + mid, v + n);
    return v[mid];
}

float median_of(std::vector<float> v) {
    return median_of(v.data(), (int)v.size());
}

// How many lattice indices an axis of n grid nodes carries at this stride,
// including the last index, which is always appended. Mirrors the loop that
// fills lat_x / lat_y below; the two must agree.
int lattice_count(int n, int stride) {
    if (n <= 0 || stride <= 0) return 0;
    int c = (n + stride - 1) / stride;      // indices 0, s, 2s, ... < n
    if ((c - 1) * stride != n - 1) c++;     // ... plus the appended last one
    return c;
}

} // namespace

AnchorLattice plan_anchor_lattice(int gridW, int gridH) {
    AnchorLattice out;
    if (gridW <= 0 || gridH <= 0) return out;

    // Aim for kAnchorTarget nodes across the ROI.
    int stride = (int)std::lround(std::sqrt((double)(gridW * gridH) /
                                            (double)tuning::kAnchorTarget));
    stride = std::max(tuning::kAnchorStrideMin, stride);

    // The budget the isotropic stride implies. The per-axis clamps below can
    // only lower a stride, i.e. only add nodes, so this is what they are
    // allowed to spend.
    const int budget = lattice_count(gridW, stride) * lattice_count(gridH, stride);

    // A mesh needs 3 lattice indices on an axis to have anything to
    // triangulate there, so each axis is clamped on its own account. The old
    // rule clamped BOTH axes by the shorter one (min(gridW, gridH) / 3),
    // which shrank the long axis for a reason that only applied to the short
    // one: on a 1000 x 6 grid -- a beam, a weld seam -- the target stride of
    // 5 was forced to 2 everywhere and the lattice came out around 1500
    // anchors against a target of 256, roughly 6x the intended seeding cost.
    int sx = stride, sy = stride;
    while (sx > 1 && lattice_count(gridW, sx) < 3) --sx;
    while (sy > 1 && lattice_count(gridH, sy) < 3) --sy;

    // ... and the short axis does not get to charge its clamp to the long
    // one. Widen whichever axis still has room until the node count is back
    // inside the budget. When no clamp fired the count already equals the
    // budget, so this loop does nothing and square-ish ROIs keep the exact
    // lattice they had before.
    while (lattice_count(gridW, sx) * lattice_count(gridH, sy) > budget) {
        const bool grow_x = lattice_count(gridW, sx + 1) >= 3;
        const bool grow_y = lattice_count(gridH, sy + 1) >= 3;
        if (!grow_x && !grow_y) break;
        if (grow_x && (!grow_y ||
                       lattice_count(gridW, sx) >= lattice_count(gridH, sy))) {
            ++sx;
        } else {
            ++sy;
        }
    }

    out.stride_x = sx;
    out.stride_y = sy;
    out.nx = lattice_count(gridW, sx);
    out.ny = lattice_count(gridH, sy);
    return out;
}

MeshSeedResult solve_anchor_seeds(
        ReferenceCache& cache,
        const cv::Mat& defMat,
        const Image& def_img,
        const FullFieldParams& params,
        int gridW,
        int gridH,
        int safe_cores,
        const HessianPool& hessian_pool,
        const ResultGrid& resultGrid,
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
    // The last row and column are always included: stopping at stride
    // multiples leaves the convex hull short of the ROI edge, which costs
    // coverage exactly where the mesh then has no triangle.
    const AnchorLattice lat = plan_anchor_lattice(gridW, gridH);

    std::vector<int> lat_x, lat_y;
    lat_x.reserve((size_t)lat.nx);
    lat_y.reserve((size_t)lat.ny);
    for (int gx = 0; gx < gridW; gx += lat.stride_x) lat_x.push_back(gx);
    if (lat_x.back() != gridW - 1) lat_x.push_back(gridW - 1);
    for (int gy = 0; gy < gridH; gy += lat.stride_y) lat_y.push_back(gy);
    if (lat_y.back() != gridH - 1) lat_y.push_back(gridH - 1);

    const int LW = (int)lat_x.size(), LH = (int)lat_y.size();
    if (LW < 2 || LH < 2) return out;

    // The planner and the two loops above must agree, or the median test's
    // O(1) neighbour lookup indexes the wrong nodes.
    if (LW != lat.nx || LH != lat.ny) {
        LOGE("anchor lattice mismatch: planned %dx%d, built %dx%d",
             lat.nx, lat.ny, LW, LH);
    }

    std::vector<AnchorResult> anchors((size_t)LW * LH);
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
        // Stays 0: the anchor phase no longer publishes, so points_solved is
        // credited in publish_anchor_results once the median test has spoken.
        int local_pts = 0;
        EngineStatFlusher flusher(engine, stats[tid], local_pts, local_hessian, local_wait);

        // Chunk 4 measured against `static` and `dynamic, 16` on the DICe
        // pair at 1/4/8 cores: this is the best of the three on wall time
        // (`static` costs 14% more) and they tie on CPU. The residual
        // parallel overhead is fixed per-thread cost, not granularity -- it
        // does not move when the chunk does. docs/SEEDING_BENCHMARK.md 5.1.
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

            // Record, do not publish: the median test has not run yet, so it is
            // not yet known whether this node is a measurement or a blunder.
            AnchorResult& a = anchors[(size_t)idx];
            a.rx = (float)realX; a.ry = (float)realY;
            a.u = res.u; a.v = res.v;
            a.ux = res.ux; a.uy = res.uy; a.vx = res.vx; a.vy = res.vy;
            a.corr = res.correlation_score;
            a.gx = gx; a.gy = gy;
            a.thread_id = tid;
            a.icgn_iters = res.iters;
            a.used_simplex = needed_rescue;

            // Two gates. A mesh vertex only has to be approximately right, and
            // the median test below rejects blunders; an output point has to
            // clear the same bar Path A applies. kCorrAccept is the tighter of
            // the two, so output implies vertex.
            a.vertex = res.correlation_score <= tuning::kAnchorAcceptScore;
            a.output = res.correlation_score <= tuning::kCorrAccept;
        }
    }

    out.attempted = attempted.load(std::memory_order_relaxed);

    // Universal median test (Westerweel & Scarano) over the lattice. A global
    // affine or homography fit would be the wrong model: under a real strain
    // field the displacements are not affine, so a global fit rejects signal.
    // The 8-neighbourhood median assumes no field shape, and the lattice makes
    // the lookup O(1).
    std::vector<float> us, vs;
    us.reserve((size_t)n_anchor);
    vs.reserve((size_t)n_anchor);
    out.ref_pts.reserve((size_t)n_anchor);
    out.def_pts.reserve((size_t)n_anchor);
    for (int lj = 0; lj < LH; ++lj) {
        for (int li = 0; li < LW; ++li) {
            const size_t idx = (size_t)lj * LW + li;
            AnchorResult& a = anchors[idx];
            if (!a.vertex) continue;

            std::array<float, 8> nu{}, nv{};
            int nn = 0;
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0) continue;
                    const int nj = lj + dj, ni = li + di;
                    if (ni < 0 || ni >= LW || nj < 0 || nj >= LH) continue;
                    const AnchorResult& nb = anchors[(size_t)nj * LW + ni];
                    if (!nb.vertex) continue;
                    nu[nn] = nb.u;
                    nv[nn] = nb.v;
                    ++nn;
                }
            }
            if (nn >= 3) {
                const float mu = median_of(nu.data(), nn), mv = median_of(nv.data(), nn);
                if (std::abs(a.u - mu) > tuning::kAnchorMedianTol ||
                    std::abs(a.v - mv) > tuning::kAnchorMedianTol) {
                    // Withhold `kept` only. `vertex` stays set, so a rejected
                    // node still votes in its neighbours' medians -- dropping it
                    // from the sample would let one blunder cascade along a row.
                    continue;
                }
            }
            a.kept = true;
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
    if (out.accepted >= tuning::kAnchorFullMinVertices &&
        accepted_fraction >= tuning::kAnchorFullFraction &&
        out.coverage >= tuning::kAnchorFullCoverage) {
        out.quality = MeshQuality::FULL;
    } else if (out.accepted >= 3 && out.coverage >= tuning::kAnchorSparseCoverage) {
        out.quality = MeshQuality::SPARSE;
    } else {
        out.quality = MeshQuality::NONE;
    }

    LOGD("ROUTING: anchor lattice %d/%d kept (%d as output, cov %.2f, phase %s) -> quality %d",
         out.accepted, out.attempted, out.solved, (double)out.coverage,
         out.phase_locked ? "locked" : "none", (int)out.quality);
    out.anchors = std::move(anchors);
    return out;
}

int publish_anchor_results(
        const MeshSeedResult& seeds,
        std::atomic<int>& global_points_solved,
        std::atomic<int>& compute_order_counter,
        ResultGrid& resultGrid,
        std::vector<ThreadStats>& stats) {

    int published = 0;
    for (const AnchorResult& a : seeds.anchors) {
        // An anchor that clears the result gate but fails the median test is
        // the classic periodic-speckle blunder -- a confident match on the wrong
        // blob -- and publishing it would hand Path B a wrong boundary seed to
        // flood fill from. Both gates are required.
        if (!a.output || !a.kept) continue;
        if (a.gy < 0 || a.gy >= (int)resultGrid.size()) continue;
        if (a.gx < 0 || a.gx >= (int)resultGrid[a.gy].size()) continue;

        const int order = compute_order_counter.fetch_add(1, std::memory_order_relaxed);
        resultGrid[a.gy][a.gx] = {a.rx, a.ry, a.u, a.v,
                                  a.ux, a.uy, a.vx, a.vy,
                                  a.corr, true, a.thread_id, order,
                                  kMeshAnchor,
                                  a.used_simplex, a.icgn_iters};
        global_points_solved.fetch_add(1, std::memory_order_relaxed);
        // Credit the thread that actually did the ICGN, so the per-thread
        // balance in the profile still means something.
        if (a.thread_id >= 0 && a.thread_id < (int)stats.size())
            stats[(size_t)a.thread_id].points_solved++;
        published++;
    }
    return published;
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
