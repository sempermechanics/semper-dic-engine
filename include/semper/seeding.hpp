#ifndef SEMPER_SEEDING_HPP
#define SEMPER_SEEDING_HPP

#include <semper/image.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

namespace Semper {
namespace seeding {

/**
 * Which seeding front-end produces the Delaunay mesh vertices.
 *
 * `AkazePyramid` is the shipping default and must stay bit-identical to the
 * pre-selector behaviour. The rest exist so `tests/integration/
 * test_seeding_bench.cpp` can measure candidates against it on identical
 * inputs; nothing in the shipping path selects them.
 *
 * `AnchorLattice` is structurally different from the descriptor candidates: it
 * has no detector and no descriptor. It takes the global rigid shift from
 * `cv::phaseCorrelate` and then solves every k-th grid node with the engine's
 * own IC-GN, using the converged nodes as mesh vertices.
 */
enum class SeedMethod {
    AkazePyramid = 0,  // AKAZE over the 0.25/0.5/1.0 scale ladder (shipping default)
    AkazeFull,         // AKAZE at scale 1.0 only — isolates the downscale penalty
    Sift,              // cv::SIFT, NORM_L2
    Orb,               // cv::ORB, NORM_HAMMING
    Brisk,             // cv::BRISK, NORM_HAMMING
    Kaze,              // cv::KAZE, NORM_L2 — AKAZE's non-accelerated parent
    AnchorLattice      // phase correlation + IC-GN anchor lattice
};

const char* seed_method_name(SeedMethod m);

/** True for the methods that run through extract_descriptor_features. */
bool is_descriptor_method(SeedMethod m);

/**
 * Detector/descriptor + Delaunay-vertex extraction.
 *
 * `method` selects the detector and the descriptor distance norm; every other
 * stage (ratio test, ROI gating, RANSAC filter, convex-hull coverage) is shared
 * so the candidates differ only in the thing under measurement.
 */
bool extract_descriptor_features(
    cv::Mat& ref, cv::Mat& def, cv::Mat& roi_mask, double scale,
    std::vector<cv::Point2f>& out_ref_pts, std::vector<cv::Point2f>& out_def_pts,
    float& out_bounding_box_area_ratio, double& out_detect_ms, double& out_ransac_ms,
    std::vector<cv::KeyPoint>& cached_kp, cv::Mat& cached_desc,
    int offset_x, int offset_y, SeedMethod method = SeedMethod::AkazePyramid,
    const std::string& debug_dir = "");

/** Pre-selector spelling. Retained so existing callers/tests keep compiling. */
inline bool extract_akaze_features(
    cv::Mat& ref, cv::Mat& def, cv::Mat& roi_mask, double scale,
    std::vector<cv::Point2f>& out_ref_pts, std::vector<cv::Point2f>& out_def_pts,
    float& out_bounding_box_area_ratio, double& out_akaze_ms, double& out_ransac_ms,
    std::vector<cv::KeyPoint>& cached_kp, cv::Mat& cached_desc,
    int offset_x, int offset_y, const std::string& debug_dir = "") {
    return extract_descriptor_features(ref, def, roi_mask, scale, out_ref_pts,
                                       out_def_pts, out_bounding_box_area_ratio,
                                       out_akaze_ms, out_ransac_ms, cached_kp,
                                       cached_desc, offset_x, offset_y,
                                       SeedMethod::AkazePyramid, debug_dir);
}

// -------------------------------------------------------------------------
// Anchor-lattice seeding (no detector, no descriptor)
// -------------------------------------------------------------------------

struct AnchorSeedResult {
    std::vector<cv::Point2f> ref_pts;
    std::vector<cv::Point2f> def_pts;
    float globalU = 0.0f;
    float globalV = 0.0f;
    float coverage = 0.0f;      // convex-hull area / ROI area
    int attempted = 0;
    int converged = 0;          // passed the ZNSSD gate
    int accepted = 0;           // also passed the median test
    bool phase_locked = false;
    double phase_ms = 0.0;
    double anchor_ms = 0.0;
};

/**
 * Global rigid shift of `roi` between the two images, from the normalized
 * cross-power spectrum. Returns false when the peak response is too weak to
 * trust, in which case out_u/out_v are left untouched.
 *
 * The returned (u, v) follows the engine's warp convention: a point at
 * reference position p appears at p + (u, v) in the deformed image.
 */
bool phase_correlate_roi(const cv::Mat& ref, const cv::Mat& def,
                         const cv::Rect& roi, double& out_u, double& out_v,
                         double& out_response);

/**
 * Solve a regular lattice of ROI grid nodes with IC-GN and keep the converged
 * ones as mesh vertices. `guess_u`/`guess_v` come from phase_correlate_roi when
 * `have_global_guess`; otherwise each anchor runs its own +/-15 px search.
 */
AnchorSeedResult solve_anchor_lattice(
    const Image& ref_img, const Image& def_img, const cv::Mat& roi_mask,
    int rect_x, int rect_y, int rect_w, int rect_h, int step, int subset_size,
    bool use_6x6_interpolator, float guess_u, float guess_v,
    bool have_global_guess);

void draw_outlined_text(cv::Mat& img, const std::string& text, cv::Point pt,
                        double scale = 0.5);

} // namespace seeding
} // namespace Semper

#endif
