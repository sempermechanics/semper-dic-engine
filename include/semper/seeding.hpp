#ifndef SEMPER_SEEDING_HPP
#define SEMPER_SEEDING_HPP

#include <opencv2/core.hpp>

namespace Semper {
namespace seeding {

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
    cv::Mat lp_ref_F;      // DFT of lp_ref * lp_win (CV_32FC2), reused per solve
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
 * false when the peak is too weak to trust. The peak is the point of largest
 * response (5x5 sum of the correlation surface), not the highest single
 * sample, which on a small noisy ROI is usually a noise spike
 * (docs/SEEDING_BENCHMARK.md 4.5.2).
 *
 * Convention matches the engine's shape function: the deformed image holds a
 * reference point p at c + s R(theta) (p - c) + t, with R(theta) =
 * [cos -sin; sin cos] in image axes (y down). |theta| < 90 deg: the magnitude
 * spectrum repeats every 180 deg, so a larger turn is reported modulo 180.
 * Pinned by SeedBench.LogPolarRotationScale.
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
    // The log-polar estimate, reported whether or not it was used. When the
    // similarity is used, theta_rad is the turn the image confirmed: within
    // 10 deg of the +-90 deg wrap that may be the reading +-180 deg, so it can
    // reach +-100 deg.
    double theta_rad = 0.0, scale = 1.0, lp_response = 0.0;
    bool lp_ok = false;
};

void phase_correlate_similarity(const PhaseCorrRef& ref, const cv::Mat& def,
                                SimilarityShift& out);

} // namespace seeding
} // namespace Semper

#endif
