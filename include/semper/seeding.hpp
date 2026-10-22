#ifndef SEMPER_SEEDING_HPP
#define SEMPER_SEEDING_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

namespace Semper {
namespace seeding {

// out_homography (optional) receives the RANSAC homography fitted to the
// matches, in the scaled ROI coordinates the matching ran in; empty when the fit
// had too few matches.
bool extract_akaze_features(
    cv::Mat& ref, cv::Mat& def, cv::Mat& roi_mask, double scale,
    std::vector<cv::Point2f>& out_ref_pts, std::vector<cv::Point2f>& out_def_pts,
    float& out_bounding_box_area_ratio, double& out_akaze_ms, double& out_ransac_ms,
    std::vector<cv::KeyPoint>& cached_kp, cv::Mat& cached_desc,
    int offset_x, int offset_y, const std::string& debug_dir = "",
    cv::Mat* out_homography = nullptr);

void draw_outlined_text(cv::Mat& img, const std::string& text, cv::Point pt,
                        double scale = 0.5);

} // namespace seeding
} // namespace Semper

#endif
