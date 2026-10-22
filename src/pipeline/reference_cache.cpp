#include <semper/pipeline.hpp>
#include "full_field_internal.hpp"
#include "util/log.hpp"
#include <opencv2/imgproc.hpp>

namespace Semper {
namespace pipeline {

ReferenceCache::ReferenceCache() = default;
ReferenceCache::~ReferenceCache() { reset(); }

void ReferenceCache::reset() {
    precompute.reset();
    delete ref_img;
    ref_img = nullptr;
    width = 0;
    height = 0;
    gray.release();
    akaze_kp.clear();
    akaze_desc.release();
    akaze_scale = 0.25;
}

void ReferenceCache::set_from_gray(const cv::Mat& gray_in, const cv::Mat& roi_mask) {
    reset();
    if (gray_in.empty()) return;
    gray = gray_in.clone();
    width = gray.cols;
    height = gray.rows;
    ref_img = new Image(width, height, gray.data);
    ref_img->prepare_data(false);

    int sterilized_count = 0;
    if (!roi_mask.empty()) {
        cv::Mat mask = roi_mask;
        if (mask.cols != width || mask.rows != height) {
            cv::resize(mask, mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (mask.at<uchar>(y, x) < 128) {
                    ref_img->intensities[y * width + x] = -10.0f;
                    sterilized_count++;
                }
            }
        }
    }
    LOGD("Ghost Wall signature injected into %d pixels", sterilized_count);
    akaze_kp.clear();
    akaze_desc.release();
    akaze_scale = 0.25;
}

namespace internal {

bool RefPrecompute::grid_matches(const FullFieldParams& p, const cv::Mat& mask) const {
    if (!grid_valid) return false;
    if (rect_x != p.rect_x || rect_y != p.rect_y || rect_w != p.rect_w || rect_h != p.rect_h ||
        step != p.step || subset_size != p.subset_size) {
        return false;
    }
    if (roi_mask.empty() != mask.empty()) return false;
    if (mask.empty()) return true;
    if (roi_mask.size() != mask.size() || roi_mask.type() != mask.type()) return false;
    return cv::norm(roi_mask, mask, cv::NORM_INF) == 0.0;
}

} // namespace internal

} // namespace pipeline
} // namespace Semper
