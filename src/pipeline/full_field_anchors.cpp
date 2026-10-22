// Anchor-lattice seeding: the front-end that produces the Delaunay mesh vertices.
//
// Two layers:
//
//   L0  AKAZE features are matched between the reference and the deformed
//       frame over the padded ROI, at the coarsest scale that yields enough
//       RANSAC inliers, and a homography is fitted to them
//       (akaze_features.cpp). That is the global model: translation, rotation
//       at any angle, scale, stretch and shear.
//   L1  a regular lattice of ROI grid nodes is solved with the engine's own
//       IC-GN, each node seeded with the displacement and gradients the model
//       puts there (only its translation when the model moves the ROI corners
//       by less than kSimilarityMinCornerPx). The converged nodes become the
//       mesh vertices, and the homography is refitted to them for the nodes
//       the mesh does not reach.
//
// Anchors sit on grid nodes on purpose. They reuse the Hessian pool the solve
// already builds, and a node good enough to keep is a *final* result for that
// node — Path A skips it and Path B picks it up as a boundary seed — so the
// lattice adds almost no work beyond what the field costs anyway.
//
// Why the keypoints only fit the model and are not mesh vertices themselves:
// build_mesh_guess_field derives the guess gradients from
// cv::getAffineTransform over triangle vertices, so a vertex position error e
// over an edge of length L becomes a gradient error of order e/L. Keypoints
// give e ~ 1 px (several at 0.25x) on short, clustered edges; IC-GN anchors give
// e ~ 0.01 px on edges fixed at stride * step. A homography fitted to all the
// inliers averages that keypoint error out instead. See docs/SEEDING_BENCHMARK.md.

#include "full_field_internal.hpp"

#include <semper/cancel.hpp>
#include <semper/seeding.hpp>
#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"
#include "util/median.hpp"
#include "util/omp.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

namespace {

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

void mark_kept_anchors(std::vector<AnchorResult>& anchors, int LW, int LH,
                       const std::vector<float>& res_u, const std::vector<float>& res_v) {
    for (int lj = 0; lj < LH; ++lj) {
        for (int li = 0; li < LW; ++li) {
            const size_t idx = (size_t)lj * LW + li;
            AnchorResult& a = anchors[idx];
            a.kept = false;
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
                    nu[nn] = res_u[(size_t)nj * LW + ni];
                    nv[nn] = res_v[(size_t)nj * LW + ni];
                    ++nn;
                }
            }
            // Fewer than three neighbours converged: there is no median to test
            // against, and nothing else vouches for the node. It used to be kept
            // on its ZNSSD alone, and ZNSSD does not tell a blunder apart where
            // the lattice is failing: under a 3-point bend's load nose the
            // translation lattice's few survivors matched speckle ~20 px off
            // at ZNSSD 0.17-0.30, and the triangles on them sent Path A to the
            // same wrong speckle (real_data_pmma_bending, frames 24-29); at the
            // stretched edge of SeedBench.StretchSeedsWithoutBlunders such
            // anchors are 25-50 px off at 0.095-0.36, inside the output gate as
            // well as outside it. Path A or B solves the node instead.
            if (nn < 3) continue;
            const float mu = util::median_inplace(nu.data(), (size_t)nn);
            const float mv = util::median_inplace(nv.data(), (size_t)nn);
            if (std::abs(res_u[idx] - mu) > tuning::kAnchorMedianTol ||
                std::abs(res_v[idx] - mv) > tuning::kAnchorMedianTol) {
                // Withhold `kept` only. `vertex` stays set, so a rejected node
                // still votes in its neighbours' medians -- dropping it from the
                // sample would let one blunder cascade along a row.
                continue;
            }
            a.kept = true;
        }
    }
}

namespace {

// The L0 global model: the AKAZE homography, its translation at the ROI
// centre, and whether its full form is worth seeding with.
struct GlobalModel {
    cv::Mat H;                    // empty: no model
    double cx = 0.0, cy = 0.0;    // ROI centre
    double t_u = 0.0, t_v = 0.0;  // H's displacement at the centre
    bool full = false;            // corners move > kSimilarityMinCornerPx beyond t
    int matches = 0;
    double scale = 0.0;
    double rotation_deg = 0.0, iso_scale = 1.0;
    seeding::AkazeRef def_side;   // the deformed frame's features at `scale`

    bool locked() const { return !H.empty(); }

    // Displacement and gradients H puts at (x, y); false if it maps to infinity.
    bool at(double x, double y, float g[6]) const {
        double X, Y, J[4];
        if (!seeding::homography_at(H, x, y, X, Y, J)) return false;
        g[0] = (float)(X - x); g[1] = (float)(Y - y);
        g[2] = (float)(J[0] - 1.0); g[3] = (float)J[1];
        g[4] = (float)J[2];         g[5] = (float)(J[3] - 1.0);
        return true;
    }
};

// AKAZE at the coarsest scale that gives kAkazeGoodInliers, else the scale
// with the most inliers if it has kAkazeMinInliers. The search starts at the
// scale an earlier solve settled on (the coarser ones failed there and the
// speckle has not changed) and tries the coarser ones only if nothing from
// there up fits. The reference keypoints of every scale tried are kept on the
// cache for the next solve.
GlobalModel match_global_model(ReferenceCache& cache, const cv::Mat& defMat,
                               const cv::Mat& roiMask, const FullFieldParams& params) {
    GlobalModel gm;
    gm.cx = params.rect_x + 0.5 * params.rect_w;
    gm.cy = params.rect_y + 0.5 * params.rect_h;
    if (cache.gray.empty() || defMat.cols != cache.gray.cols || defMat.rows != cache.gray.rows) {
        return gm;
    }

    const cv::Rect roi(params.rect_x, params.rect_y, params.rect_w, params.rect_h);
    if (!cache.precompute) cache.precompute = std::make_unique<RefPrecompute>();
    RefPrecompute& pre = *cache.precompute;
    const int levels = (int)std::size(tuning::kAkazeScales);
    if (pre.akaze_roi != roi || (int)pre.akaze_refs.size() != levels) {
        pre.akaze_refs.assign((size_t)levels, seeding::AkazeRef{});
        pre.akaze_roi = roi;
    }
    const int first = std::min(std::max(pre.akaze_first_level, 0), levels - 1);

    seeding::AkazeMatches best;
    int best_level = -1;
    auto try_level = [&](int lv) {
        seeding::AkazeRef& ref = pre.akaze_refs[(size_t)lv];
        if (ref.scale == 0.0) {
            const bool reused = pre.akaze_handover.scale == tuning::kAkazeScales[lv] &&
                    seeding::reuse_akaze_ref(pre.akaze_handover, roi, cache.gray.size(), ref);
            if (!reused) ref = seeding::prepare_akaze_ref(cache.gray, roi, tuning::kAkazeScales[lv]);
        }
        seeding::AkazeMatches m;
        const bool fit = seeding::match_akaze(ref, defMat, roiMask, m);
        LOGD("ROUTING: AKAZE at %.2fx: %d inliers%s", ref.scale, m.inliers(), fit ? "" : " (no model)");
        if (fit && m.inliers() > best.inliers()) {
            best = std::move(m);
            best_level = lv;
        }
        return best_level == lv && best.inliers() >= tuning::kAkazeGoodInliers;
    };
    bool good = false;
    for (int lv = first; lv < levels && !good; ++lv) good = try_level(lv);
    for (int lv = 0; lv < first && best_level < 0 && !good; ++lv) good = try_level(lv);
    if (best_level < 0) return gm;
    pre.akaze_first_level = best_level;

    gm.H = best.homography;
    gm.matches = best.inliers();
    gm.scale = best.scale;
    gm.def_side = std::move(best.def_side);
    float g[6];
    if (!gm.at(gm.cx, gm.cy, g)) {
        gm.H.release();
        return gm;
    }
    gm.t_u = g[0];
    gm.t_v = g[1];
    const double a = 1.0 + g[2], b = g[3], c = g[4], d = 1.0 + g[5];
    gm.rotation_deg = std::atan2(c - b, a + d) * 180.0 / CV_PI;
    gm.iso_scale = std::sqrt(std::abs(a * d - b * c));

    // The full model only where it changes the seed: at a corner it puts the
    // ROI more than kSimilarityMinCornerPx away from where its translation
    // does. Below that IC-GN pulls in from the translation alone.
    const double xs[2] = {(double)params.rect_x, (double)(params.rect_x + params.rect_w)};
    const double ys[2] = {(double)params.rect_y, (double)(params.rect_y + params.rect_h)};
    for (double x : xs) {
        for (double y : ys) {
            if (!gm.at(x, y, g) ||
                std::hypot(g[0] - gm.t_u, g[1] - gm.t_v) > tuning::kSimilarityMinCornerPx) {
                gm.full = true;
            }
        }
    }
    return gm;
}

} // namespace

MeshSeedResult solve_anchor_seeds(
        ReferenceCache& cache,
        const cv::Mat& defMat,
        const cv::Mat& roiMask,
        const Image& def_img,
        const FullFieldParams& params,
        int gridW,
        int gridH,
        int safe_cores,
        const HessianPool& hessian_pool,
        const ResultGrid& resultGrid,
        std::vector<ThreadStats>& stats,
        PhaseTimings& timings,
        const CancelToken& cancel) {

    MeshSeedResult out;
    if (cache.ref_img == nullptr || gridW <= 0 || gridH <= 0) return out;

    // --- L0: global motion ------------------------------------------------
    GlobalModel gm;
    {
        ScopedTimer t(timings.akaze);
        gm = match_global_model(cache, defMat, roiMask, params);
    }
    auto describe_model = [&](MeshSeedResult& r) {
        r.locked = gm.locked();
        r.matches = gm.matches;
        r.akaze_scale = (float)gm.scale;
        r.rotation_deg = (float)gm.rotation_deg;
        r.scale = (float)gm.iso_scale;
        r.homography = gm.H;
    };
    describe_model(out);

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
    if (LW < 2 || LH < 2) {
        out.model = gm.full;
        out.globalU = (float)gm.t_u;
        out.globalV = (float)gm.t_v;
        return out;
    }

    // The planner and the two loops above must agree, or the median test's
    // O(1) neighbour lookup indexes the wrong nodes.
    if (LW != lat.nx || LH != lat.ny) {
        LOGE("anchor lattice mismatch: planned %dx%d, built %dx%d",
             lat.nx, lat.ny, LW, LH);
    }

    const int n_anchor = LW * LH;

    // Solve the lattice from one global model and run the median test on it.
    // `sim` seeds every anchor with the displacement and gradients of the
    // homography at that anchor; otherwise every anchor gets its translation at
    // the ROI centre (zero without a model).
    auto solve_lattice = [&](bool sim, MeshSeedResult& r) {
        describe_model(r);
        r.model = sim;
        r.globalU = (float)gm.t_u;
        r.globalV = (float)gm.t_v;
        // The model's displacement at a reference point. Only evaluated under
        // `sim`; a point it maps to infinity gets the translation.
        auto model_at = [&](float x, float y, float g[6]) {
            if (!gm.at(x, y, g)) {
                g[0] = r.globalU; g[1] = r.globalV; g[2] = g[3] = g[4] = g[5] = 0.0f;
            }
        };

        std::vector<AnchorResult> anchors((size_t)n_anchor);

        // With a model the guess is within a pixel or so, so pure ICGN
        // converges and the 6-DOF Simplex rescue is pure cost. Without one each
        // anchor has to find its own offset, so the coarse search stays.
        const auto init_mode = r.locked ? INIT_NO_SIMPLEX : INIT_AUTO_SEARCH;

        std::atomic<int> attempted(0);
        RegionErrors errors;

#pragma omp parallel num_threads(safe_cores)
        {
            const int tid = util::omp_thread();
            OptimizationEngine engine = make_engine(params.use_6x6_interpolator);
            Semper::SubsetData subset;
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
                if (cancel.requested() || errors.failed()) continue;
                try {
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

                    // Under the model each anchor starts where the homography puts
                    // it, with its gradients there: under rotation the one global
                    // shift is off by 2 r sin(theta/2) at radius r.
                    float guess_sim[6];
                    if (sim) model_at((float)realX, (float)realY, guess_sim);
                    const float guess_shift[6] = {r.globalU, r.globalV, 0.0f, 0.0f, 0.0f, 0.0f};
                    const NodeSolve node = solve_node(engine, subset, *cache.ref_img, def_img,
                                                      realX, realY, params.subset_size,
                                                      hessian_pool[(size_t)pool_idx],
                                                      sim ? guess_sim : guess_shift, init_mode);
                    local_hessian += node.hessian_ms;
                    if (!node.initialized) continue;

                    const AnalysisResult& res = node.res;
                    const bool needed_rescue = node.needed_rescue;
                    stats[tid].icgn_iters += res.iters;
                    if (needed_rescue) record_simplex_outcome(stats[tid], res);

                    if (res.status != 0) continue;

                    // Record, do not publish: the median test has not run yet, so it
                    // is not yet known whether this node is a measurement or a blunder.
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
                } catch (...) {
                    errors.capture();
                }
            }
        }
        errors.rethrow_if_any();

        r.attempted = attempted.load(std::memory_order_relaxed);

        // The universal median test (mark_kept_anchors). Under the model it runs
        // on the residual from it. Rotation is a smooth field, but at 15 deg
        // lattice neighbours 20 px apart differ by 5 px, and on the lattice edge
        // -- where every neighbour sits on one side -- the raw median lands that
        // far off and each edge anchor fails a 1 px tolerance however good it
        // is. Subtracting the model removes exactly that. Translation-only, the
        // residual is the displacement itself and this is the original test.
        std::vector<float> res_u((size_t)n_anchor, 0.0f), res_v((size_t)n_anchor, 0.0f);
        for (size_t i = 0; i < anchors.size(); ++i) {
            const AnchorResult& a = anchors[i];
            if (!a.vertex) continue;
            float g[6] = {0, 0, 0, 0, 0, 0};
            if (sim) model_at(a.rx, a.ry, g);
            res_u[i] = a.u - g[0];
            res_v[i] = a.v - g[1];
        }
        mark_kept_anchors(anchors, LW, LH, res_u, res_v);

        std::vector<float> us, vs;
        us.reserve((size_t)n_anchor);
        vs.reserve((size_t)n_anchor);
        r.ref_pts.reserve((size_t)n_anchor);
        r.def_pts.reserve((size_t)n_anchor);
        for (const AnchorResult& a : anchors) {
            if (!a.kept) continue;
            r.ref_pts.emplace_back(a.rx, a.ry);
            r.def_pts.emplace_back(a.rx + a.u, a.ry + a.v);
            us.push_back(a.u);
            vs.push_back(a.v);
            if (a.output) r.solved++;
        }
        r.accepted = (int)r.ref_pts.size();

        if (r.accepted >= 3) {
            r.globalU = util::median_inplace(us.data(), us.size());
            r.globalV = util::median_inplace(vs.data(), vs.size());
            std::vector<cv::Point2f> hull;
            cv::convexHull(r.ref_pts, hull);
            const float roi_area = (float)(params.rect_w * params.rect_h);
            r.coverage = roi_area > 0.0f ? (float)cv::contourArea(hull) / roi_area : 0.0f;
        }

        const float accepted_fraction = r.attempted > 0
                ? (float)r.accepted / (float)r.attempted : 0.0f;
        // The old floor of 10 vertices came from AKAZE, where findHomography
        // needed that many correspondences. A Delaunay mesh needs three — one
        // triangle — so a sparse mesh is useful well below the old threshold,
        // and on a small ROI the lattice legitimately produces only a handful
        // of nodes.
        if (r.accepted >= tuning::kAnchorFullMinVertices &&
            accepted_fraction >= tuning::kAnchorFullFraction &&
            r.coverage >= tuning::kAnchorFullCoverage) {
            r.quality = MeshQuality::FULL;
        } else if (r.accepted >= 3 && r.coverage >= tuning::kAnchorSparseCoverage) {
            r.quality = MeshQuality::SPARSE;
        } else {
            r.quality = MeshQuality::NONE;
        }
        r.anchors = std::move(anchors);
        return accepted_fraction;
    };

    if (!gm.full) {
        solve_lattice(false, out);
    } else {
        const float kept = solve_lattice(true, out);
        // A model fitted to a few clustered matches can be wrong away from
        // them, and the lattice it seeds then comes out weak -- and not merely
        // sparse: anchors started from a wrong model can converge, as a
        // cluster, onto the wrong speckle. Such a cluster agrees with itself,
        // so neither the ZNSSD gate nor a median test of residuals from that
        // same wrong model catches it (seen with v0.5.0's log-polar similarity
        // under stretch, SeedBench.StretchSeedsWithoutBlunders: 25 px at p95).
        // Counting kept anchors cannot choose between the two lattices either,
        // because the blunders are kept. So the model is trusted only when its
        // lattice says it fits nearly everywhere; otherwise the
        // translation-only lattice is used. A model that fits keeps ~100% and
        // never pays for the second solve.
        if (out.quality != MeshQuality::FULL || kept < tuning::kSimilarityKeepFraction) {
            LOGD("ROUTING: homography lattice weak (%d/%d kept, quality %d); "
                 "re-solving from the translation alone",
                 out.accepted, out.attempted, (int)out.quality);
            out = MeshSeedResult{};
            solve_lattice(false, out);
            out.model_rejected = true;
        }
    }

    // Refit the global homography to the kept anchors: they are good to ~0.01 px
    // where the keypoints were good to the RANSAC tolerance at the detection
    // scale, and this is what the nodes outside the mesh start from. Without
    // enough anchors the AKAZE fit stands (or nothing, without a model) --
    // unless its own lattice just rejected it: the mesh would then seed every
    // node outside it from the model found wrong, which is how a wrong-speckle
    // cluster gets in, so those nodes are left to the flood fill instead.
    bool refitted = false;
    if (out.accepted >= tuning::kAnchorRefitMinPoints) {
        cv::Mat H = cv::findHomography(out.ref_pts, out.def_pts, cv::RANSAC,
                                       tuning::kAnchorRefitRansacPx);
        if (H.rows == 3 && H.cols == 3 && cv::checkRange(H)) {
            out.homography = H;
            refitted = true;
        }
    }
    if (out.model_rejected && !refitted) out.homography.release();
    out.def_features = std::move(gm.def_side);

    LOGD("ROUTING: anchor lattice %d/%d kept (%d as output, cov %.2f, AKAZE %s "
         "%d inliers at %.2fx, rot %.2f deg x%.4f %s) -> quality %d",
         out.accepted, out.attempted, out.solved, (double)out.coverage,
         out.locked ? "locked" : "none", out.matches, (double)out.akaze_scale,
         (double)out.rotation_deg, (double)out.scale,
         out.model ? "applied"
                   : out.model_rejected ? "rejected, translation lattice used"
                                        : "not applied",
         (int)out.quality);
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
