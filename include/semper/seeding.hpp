#ifndef SEMPER_SEEDING_HPP
#define SEMPER_SEEDING_HPP

#include <opencv2/core.hpp>
#include <string>

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

void draw_outlined_text(cv::Mat& img, const std::string& text, cv::Point pt,
                        double scale = 0.5);

} // namespace seeding
} // namespace Semper

#endif
