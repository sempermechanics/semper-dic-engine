// Anchor-lattice seeding — a seeder with no detector and no descriptor.
//
// Two layers:
//   L0  cv::phaseCorrelate over the ROI gives the global rigid shift from every
//       pixel, rather than from the median of sparse keypoint matches.
//   L1  a regular lattice of ROI grid nodes is solved with the engine's own
//       IC-GN, seeded from L0. The converged nodes become the Delaunay vertices.
//
// The point of the lattice is the guess *gradients*. build_mesh_guess_field
// derives (ux, uy, vx, vy) from cv::getAffineTransform over triangle vertices,
// so a vertex position error e over a triangle edge of length L produces a
// gradient error of order e/L. Keypoints give e ~ 1 px on short, clustered
// edges; IC-GN anchors give e ~ 0.01 px on edges fixed at stride * step.

#include <semper/seeding.hpp>
#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include <semper/types.hpp>

#include <opencv2/imgproc.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace Semper {
namespace seeding {

namespace {

// Median of a scratch vector. Takes by value so callers keep their ordering.
float median_of(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + (long)mid, v.end());
    return v[mid];
}

} // namespace

bool phase_correlate_roi(const cv::Mat &ref, const cv::Mat &def,
                         const cv::Rect &roi, double &out_u, double &out_v,
                         double &out_response) {
    cv::Rect r = roi & cv::Rect(0, 0, ref.cols, ref.rows);
    if (r.width < 16 || r.height < 16) return false;
    if (def.cols != ref.cols || def.rows != ref.rows) return false;

    cv::Mat ref32, def32;
    ref(r).convertTo(ref32, CV_32F);
    def(r).convertTo(def32, CV_32F);

    cv::Mat hann;
    cv::createHanningWindow(hann, ref32.size(), CV_32F);

    double response = 0.0;
    // Sign convention verified against the engine's warp on a synthetic shift
    // (SeedBench.PhaseCorrelateSignConvention): phaseCorrelate(ref, def)
    // returns (u, v) such that a point at reference p appears at p + (u, v).
    cv::Point2d shift = cv::phaseCorrelate(ref32, def32, hann, &response);

    out_response = response;
    if (!(response > (double)tuning::kPhaseCorrMinResponse)) return false;
    if (!(std::abs(shift.x) < 1e6 && std::abs(shift.y) < 1e6)) return false;

    out_u = shift.x;
    out_v = shift.y;
    return true;
}

AnchorSeedResult solve_anchor_lattice(
        const Image &ref_img, const Image &def_img, const cv::Mat &roi_mask,
        int rect_x, int rect_y, int rect_w, int rect_h, int step, int subset_size,
        bool use_6x6_interpolator, float guess_u, float guess_v,
        bool have_global_guess) {

    AnchorSeedResult out;
    out.globalU = guess_u;
    out.globalV = guess_v;
    out.phase_locked = have_global_guess;

    if (step <= 0 || subset_size <= 0) return out;

    const int gridW = rect_w / step;
    const int gridH = rect_h / step;
    if (gridW <= 0 || gridH <= 0) return out;

    // Lattice stride: aim for kAnchorTarget nodes, floored so the lattice is at
    // least 3x3 wherever the grid allows it.
    int stride = (int)std::lround(std::sqrt((double)(gridW * gridH) /
                                            (double)tuning::kAnchorTarget));
    stride = std::max(tuning::kAnchorStrideMin, stride);
    stride = std::min(stride, std::max(1, std::min(gridW, gridH) / 3));

    // The lattice must include the last row and column. Stopping at stride
    // multiples leaves the convex hull short of the ROI edge, which costs
    // coverage exactly where the mesh has no triangle to interpolate from.
    std::vector<int> lat_x, lat_y;
    for (int gx = 0; gx < gridW; gx += stride) lat_x.push_back(gx);
    if (lat_x.back() != gridW - 1) lat_x.push_back(gridW - 1);
    for (int gy = 0; gy < gridH; gy += stride) lat_y.push_back(gy);
    if (lat_y.back() != gridH - 1) lat_y.push_back(gridH - 1);
    const int LW = (int)lat_x.size(), LH = (int)lat_y.size();
    if (LW < 2 || LH < 2) return out;

    // The same boundary reserve run_full_field applies to grid points: half a
    // subset, DICe's 4-pixel interpolation buffer, and a 15 px deformation
    // allowance.
    const int half_subset = subset_size / 2;
    const int boundary = half_subset + 4 + 15;

    struct Anchor {
        float rx = 0, ry = 0, u = 0, v = 0;
        bool ok = false;
    };
    std::vector<Anchor> anchors((size_t)LW * LH);

    auto t_anchor_start = std::chrono::high_resolution_clock::now();

    const int n_anchor = LW * LH;
    // With a phase-correlation lock the guess is already sub-pixel, so pure
    // ICGN converges and the 6-DOF Simplex rescue is pure cost -- on noisy
    // images nearly every anchor triggers it and most still fail. Without a
    // lock each anchor has to find its own offset, so the coarse search stays.
    const auto init_mode = have_global_guess ? INIT_NO_SIMPLEX : INIT_AUTO_SEARCH;

    double precompute_ms = 0.0, icgn_ms = 0.0;

#pragma omp parallel reduction(+ : precompute_ms, icgn_ms)
    {
        OptimizationEngine engine;
        Semper::SubsetData subset;
        engine.lm_enabled = true;
        engine.lm_alpha = tuning::kLmAlpha;
        engine.use_6x6_interpolator = use_6x6_interpolator;

#pragma omp for schedule(dynamic, 4)
        for (int idx = 0; idx < n_anchor; ++idx) {
            const int li = idx % LW, lj = idx / LW;
            const int realX = rect_x + lat_x[(size_t)li] * step;
            const int realY = rect_y + lat_y[(size_t)lj] * step;

            if (realX - boundary < 0 || realX + boundary >= ref_img.width ||
                realY - boundary < 0 || realY + boundary >= ref_img.height) {
                continue;
            }
            if (!roi_mask.empty()) {
                if (realX < 0 || realX >= roi_mask.cols ||
                    realY < 0 || realY >= roi_mask.rows) continue;
                if (roi_mask.at<uchar>(realY, realX) < 128) continue;
            }

            // Full precompute: standalone, the lattice has no pooled Hessian to
            // reuse. On the shipping path anchors sit on grid nodes and would
            // use precompute_subset_fast against hessian_pool, so this is the
            // pessimistic reading -- hence the separate timer.
            auto t_pre = std::chrono::high_resolution_clock::now();
            SubsetPrecomputer::precompute_subset(subset, ref_img, realX, realY, subset_size);
            auto t_solve = std::chrono::high_resolution_clock::now();
            precompute_ms += std::chrono::duration<double, std::milli>(t_solve - t_pre).count();
            if (!subset.is_initialized) continue;

            AnalysisResult res = engine.calculate_deformation(
                    subset, def_img, guess_u, guess_v, 0.0f, 0.0f, 0.0f, 0.0f, init_mode);
            icgn_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_solve).count();

            const float ghost_fraction =
                    (float)res.invalid_ref_pixels / (float)(subset_size * subset_size);
            if (ghost_fraction > tuning::kGhostRejectFraction) continue;
            if (res.status != 0 || res.correlation_score > tuning::kAnchorAcceptScore) continue;

            Anchor &a = anchors[(size_t)idx];
            a.rx = (float)realX;
            a.ry = (float)realY;
            a.u = res.u;
            a.v = res.v;
            a.ok = true;
        }
    }

    out.attempted = n_anchor;
    for (const Anchor &a : anchors) if (a.ok) out.converged++;

    // Universal median test (Westerweel & Scarano) over the lattice. A global
    // affine or homography fit would be the wrong model here: under a real
    // strain field the displacements are not affine, so a global fit rejects
    // signal. The 8-neighbourhood median rejects blunders without assuming a
    // field shape, and the lattice makes the lookup O(1).
    std::vector<char> keep((size_t)LW * LH, 0);
    for (int lj = 0; lj < LH; ++lj) {
        for (int li = 0; li < LW; ++li) {
            const size_t idx = (size_t)lj * LW + li;
            if (!anchors[idx].ok) continue;

            std::vector<float> nu, nv;
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0) continue;
                    const int nj = lj + dj, ni = li + di;
                    if (ni < 0 || ni >= LW || nj < 0 || nj >= LH) continue;
                    const Anchor &nb = anchors[(size_t)nj * LW + ni];
                    if (!nb.ok) continue;
                    nu.push_back(nb.u);
                    nv.push_back(nb.v);
                }
            }
            if (nu.size() < 3) { keep[idx] = 1; continue; } // too isolated to judge

            const float mu = median_of(nu), mv = median_of(nv);
            if (std::abs(anchors[idx].u - mu) <= tuning::kAnchorMedianTol &&
                std::abs(anchors[idx].v - mv) <= tuning::kAnchorMedianTol) {
                keep[idx] = 1;
            }
        }
    }

    std::vector<float> us, vs;
    for (size_t idx = 0; idx < anchors.size(); ++idx) {
        if (!anchors[idx].ok || !keep[idx]) continue;
        const Anchor &a = anchors[idx];
        out.ref_pts.emplace_back(a.rx, a.ry);
        out.def_pts.emplace_back(a.rx + a.u, a.ry + a.v);
        us.push_back(a.u);
        vs.push_back(a.v);
    }
    out.accepted = (int)out.ref_pts.size();

    if (out.accepted >= 3) {
        out.globalU = median_of(us);
        out.globalV = median_of(vs);

        std::vector<cv::Point2f> hull;
        cv::convexHull(out.ref_pts, hull);
        const float hull_area = (float)cv::contourArea(hull);
        const float roi_area = (float)(rect_w * rect_h);
        out.coverage = roi_area > 0.0f ? hull_area / roi_area : 0.0f;
    }

    out.anchor_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_anchor_start).count();
    out.precompute_ms = precompute_ms;
    out.icgn_ms = icgn_ms;
    return out;
}

} // namespace seeding
} // namespace Semper
