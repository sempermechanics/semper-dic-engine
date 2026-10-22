#ifndef SEMPER_SEEDING_HPP
#define SEMPER_SEEDING_HPP

#include <opencv2/core.hpp>

#include <vector>

namespace Semper {
namespace seeding {

/**
 * The reference half of AKAZE matching at one detection scale: keypoints and
 * descriptors of `rect` (the ROI padded by tuning::kAkazePadPx / scale and
 * clipped to the image) resized by `scale`. Depends only on the reference, so
 * a batch against one reference detects it once.
 */
struct AkazeRef {
    cv::Rect rect;
    double scale = 0.0;
    std::vector<cv::KeyPoint> keypoints;   // in scaled rect coordinates
    cv::Mat descriptors;
    bool usable() const { return !keypoints.empty() && !descriptors.empty(); }
};

AkazeRef prepare_akaze_ref(const cv::Mat& ref_gray, const cv::Rect& roi, double scale);

/**
 * The reference half for `roi` at `from.scale`, taken from features already
 * detected on the same image over another padded rect (incremental DIC: the
 * deformed frame's features become the next reference's). Keeps the keypoints
 * that fall inside the new padded rect, re-expressed in its coordinates.
 * False, leaving `out` alone, when the old rect does not cover at least
 * tuning::kAkazeReuseCover of the new one -- detect afresh then.
 */
bool reuse_akaze_ref(const AkazeRef& from, const cv::Rect& roi, cv::Size image, AkazeRef& out);

/**
 * AKAZE matches between the reference and `def_gray` over `ref.rect`, filtered
 * by Lowe's ratio test and the ROI mask (on the reference keypoint), then by a
 * RANSAC homography. Points and the homography are in full-image coordinates,
 * reference -> deformed: a reference point p appears at H(p). Returns false
 * (and an empty homography) with fewer than tuning::kAkazeMinInliers inliers.
 */
struct AkazeMatches {
    std::vector<cv::Point2f> ref_pts, def_pts;   // RANSAC inliers
    cv::Mat homography;                          // 3x3 CV_64F; empty without a fit
    double scale = 0.0;
    double detect_ms = 0.0, ransac_ms = 0.0;
    // The deformed frame's own features over the same rect and scale, for
    // reuse_akaze_ref when that frame becomes a reference.
    AkazeRef def_side;
    int inliers() const { return (int)ref_pts.size(); }
};

bool match_akaze(const AkazeRef& ref, const cv::Mat& def_gray, const cv::Mat& roi_mask,
                 AkazeMatches& out);

/**
 * Image of (px, py) under the 3x3 CV_64F homography H, and H's Jacobian there
 * (dX/dx, dX/dy, dY/dx, dY/dy). False when the point maps to infinity.
 */
bool homography_at(const cv::Mat& H, double px, double py, double& X, double& Y, double J[4]);

/**
 * Global rigid shift of `roi` between the two images, from the normalized
 * cross-power spectrum. Returns false when the peak response is too weak to
 * trust, in which case out_u/out_v are left untouched.
 *
 * The returned (u, v) follows the engine's warp convention: a point at
 * reference position p appears at p + (u, v) in the deformed image. That sign
 * is pinned by SeedBench.PhaseCorrelateSignConvention, not inferred from the
 * OpenCV documentation.
 *
 * This is translation only. Under rotation the true displacement grows with
 * radius, so a single global shift is wrong for points far from the centre;
 * see docs/SEEDING_BENCHMARK.md section 4.5 for where that starts to bite.
 * phase_correlate_similarity below adds rotation and scale.
 */
bool phase_correlate_roi(const cv::Mat& ref, const cv::Mat& def,
                         const cv::Rect& roi, double& out_u, double& out_v,
                         double& out_response);

/**
 * The reference half of phase_correlate_roi, which depends only on the
 * reference and the ROI: the clipped rect, the CV_32F crop and the Hann window.
 * Preparing it once and calling the overload below gives the same result as
 * phase_correlate_roi, bit for bit, because cv::phaseCorrelate sees the same
 * inputs.
 */
struct PhaseCorrRef {
    cv::Rect rect;   // roi clipped to the image; empty if too small to correlate
    cv::Mat ref32;
    cv::Mat hann;
    bool usable() const { return rect.width >= 16 && rect.height >= 16; }

    // Reference half of the log-polar (Fourier-Mellin) rotation/scale
    // estimate; empty unless prepared with_log_polar and the ROI holds a
    // square of at least tuning::kLogPolarMinSide.
    cv::Rect lp_rect;      // centred square inside rect, even side
    cv::Mat lp_hann;       // window for the square crop
    cv::Mat lp_mapx, lp_mapy;  // log-polar sampling of the shifted spectrum
    cv::Mat lp_ref;        // log-polar log-magnitude of the reference
    cv::Mat lp_win;        // window along log-radius only (angle is periodic)
    double lp_dlogr = 0.0; // log-radius step per column
    bool lp_usable() const { return !lp_ref.empty(); }
};

PhaseCorrRef prepare_phase_correlate_ref(const cv::Mat& ref, const cv::Rect& roi,
                                         bool with_log_polar = true);

bool phase_correlate_roi(const PhaseCorrRef& ref, const cv::Mat& def,
                         double& out_u, double& out_v, double& out_response);

/**
 * Rotation and scale of the ROI, from phase correlation of the log-polar
 * resampled magnitude spectra (Fourier-Mellin). The magnitude spectrum does not
 * see translation, so this works before the translation is known. Returns
 * false when the peak is too weak to trust.
 *
 * Convention matches the engine's shape function: the deformed image holds a
 * reference point p at c + s R(theta) (p - c) + t, with R(theta) =
 * [cos -sin; sin cos] in image axes (y down). theta is in [-90, 90) deg: the
 * magnitude spectrum repeats every 180 deg, so a larger turn is reported
 * modulo 180, and phase_correlate_similarity decides between theta and
 * theta + 180. Pinned by SeedBench.LogPolarRotationScale and
 * SeedBench.AnchorSeedingHalfTurn.
 */
bool log_polar_rotation_scale(const PhaseCorrRef& ref, const cv::Mat& def,
                              double& out_theta_rad, double& out_scale,
                              double& out_response);

/**
 * The global motion the anchor lattice is seeded with: a translation, or a
 * similarity (rotation + scale + translation) about the ROI centre.
 *
 *   u(p) = u + ux (px - cx) + uy (py - cy)
 *   v(p) = v + vx (px - cx) + vy (py - cy)
 *
 * with [ux uy; vx vy] = s R - I. When `similarity` is false the gradients are
 * zero and (u, v, response, locked) are exactly what phase_correlate_roi
 * returns, bit for bit -- the rotation model is used only when it moves the
 * ROI corners by more than tuning::kSimilarityMinCornerPx and the
 * rotation-corrected translation correlates better than the uncorrected one.
 * The log-polar angle is ambiguous by 180 deg, so both theta and theta + 180
 * are corrected for and the stronger corrected peak is used.
 */
struct SimilarityShift {
    double u = 0.0, v = 0.0;          // displacement of (cx, cy)
    double cx = 0.0, cy = 0.0;
    double ux = 0.0, uy = 0.0, vx = 0.0, vy = 0.0;
    double response = 0.0;            // translation peak actually used
    bool locked = false;              // the translation is trusted
    bool similarity = false;          // the gradients above are in use
    // The uncorrected translation, kept even when the similarity replaces it,
    // so a caller can fall back to exactly the translation-only seeds.
    double t_u = 0.0, t_v = 0.0, t_response = 0.0;
    bool t_locked = false;
    // The log-polar estimate, reported whether or not it was used. theta_rad
    // is the turn the similarity uses when `similarity` is set, in
    // (-180, 180) deg; otherwise the raw estimate, in [-90, 90).
    double theta_rad = 0.0, scale = 1.0, lp_response = 0.0;
    bool lp_ok = false;
};

void phase_correlate_similarity(const PhaseCorrRef& ref, const cv::Mat& def,
                                SimilarityShift& out);

} // namespace seeding
} // namespace Semper

#endif
