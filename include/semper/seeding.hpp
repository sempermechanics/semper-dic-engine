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
};

PhaseCorrRef prepare_phase_correlate_ref(const cv::Mat& ref, const cv::Rect& roi);

bool phase_correlate_roi(const PhaseCorrRef& ref, const cv::Mat& def,
                         double& out_u, double& out_v, double& out_response);

} // namespace seeding
} // namespace Semper

#endif
