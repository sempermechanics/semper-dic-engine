#include <semper/pipeline.hpp>
#include "util/log.hpp"
#include <opencv2/imgproc.hpp>

namespace Semper {
namespace pipeline {

void ReferenceCache::reset() {
    delete ref_img;
    ref_img = nullptr;
    width = 0;
    height = 0;
    gray.release();
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
}

} // namespace pipeline
} // namespace Semper
