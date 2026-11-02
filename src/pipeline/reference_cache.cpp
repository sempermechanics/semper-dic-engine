#include <semper/pipeline.hpp>
#include <semper/tuning.hpp>
#include "full_field_internal.hpp"
#include "util/log.hpp"
#include <opencv2/imgproc.hpp>
#include <cstdint>
#include <limits>

namespace Semper {
namespace pipeline {

namespace {

// Ghost Wall: masked reference pixels carry tuning::kGhostWallValue, which
// every subset routine skips (tuning::kGhostWallThreshold). Applied after
// prepare_data, so gradients next to the mask edge are still computed from
// the real intensities.
void stamp_ghost_wall(Image& img, const cv::Mat& roi_mask) {
    int sterilized_count = 0;
    if (!roi_mask.empty()) {
        cv::Mat mask = roi_mask;
        if (mask.cols != img.width || mask.rows != img.height) {
            cv::resize(mask, mask, cv::Size(img.width, img.height), 0, 0, cv::INTER_NEAREST);
        }
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                if (mask.at<uchar>(y, x) < 128) {
                    img.intensities[y * img.width + x] = tuning::kGhostWallValue;
                    sterilized_count++;
                }
            }
        }
    }
    LOGD("Ghost Wall signature injected into %d pixels", sterilized_count);
}

} // namespace

ReferenceCache::ReferenceCache() = default;
ReferenceCache::~ReferenceCache() { reset(); }

void ReferenceCache::reset() {
    ref_img.reset();
    precompute.reset();
    width = 0;
    height = 0;
    gray.release();
}

void ReferenceCache::set_from_gray(const cv::Mat& gray_in, const cv::Mat& roi_mask) {
    reset();
    // Image and the Ghost Wall read pixels as bytes; anything but 8-bit gray
    // (image or mask) leaves the cache empty, and the next solve returns -3.
    // io::to_gray8 converts.
    if (gray_in.empty() || gray_in.type() != CV_8UC1) return;
    if (!roi_mask.empty() && roi_mask.type() != CV_8UC1) return;
    // Every pixel index in the engine is an int (validate_params); refuse a
    // larger image before copying it rather than when the solve starts.
    if ((int64_t)gray_in.cols * (int64_t)gray_in.rows >
        (int64_t)std::numeric_limits<int>::max()) return;
    // Built aside and installed only once complete: the gradients and the
    // Ghost Wall can throw (bad_alloc, cv::Exception), and a throw used to
    // leave ref_img set over empty gradient planes, or unmasked, for the next
    // solve to read. Now the cache stays empty, as on any other failure.
    cv::Mat g = gray_in.clone();
    auto img = std::make_unique<Image>(g.cols, g.rows, g.data);
    img->prepare_data(false);
    stamp_ghost_wall(*img, roi_mask);
    gray = std::move(g);
    width = gray.cols;
    height = gray.rows;
    ref_img = std::move(img);
}

void ReferenceCache::adopt(std::unique_ptr<Image> prepared, cv::Mat gray_in, const cv::Mat& roi_mask) {
    internal::adopt_reference(*this, std::move(prepared), std::move(gray_in), roi_mask,
                              seeding::AkazeRef{});
}

namespace internal {

void adopt_reference(ReferenceCache& cache, std::unique_ptr<Image> prepared, cv::Mat gray_in,
                     const cv::Mat& roi_mask, seeding::AkazeRef handover) {
    if (!prepared || gray_in.empty()) {
        cache.reset();
        return;
    }
    // Everything that can throw runs before the cache changes: a throw leaves
    // the outgoing reference in place (IncrementalSession relies on it to
    // leave the session as it was).
    stamp_ghost_wall(*prepared, roi_mask);
    // The AKAZE detection scale the outgoing reference settled on depends on
    // the speckle, not on the frame: carry it over, so incremental DIC does not
    // climb the scale pyramid again at every reference update.
    const int akaze_level = cache.precompute ? cache.precompute->akaze_first_level : 0;
    std::unique_ptr<RefPrecompute> pre;
    if (akaze_level > 0 || handover.usable()) {
        pre = std::make_unique<RefPrecompute>();
        pre->akaze_first_level = akaze_level;
        pre->akaze_handover = std::move(handover);
    }
    cache.reset();
    cache.gray = std::move(gray_in);
    cache.width = cache.gray.cols;
    cache.height = cache.gray.rows;
    cache.ref_img = std::move(prepared);
    cache.precompute = std::move(pre);
}

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
