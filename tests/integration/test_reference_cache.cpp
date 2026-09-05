// Direct characterization of ReferenceCache — the ROI-mask "Ghost Wall"
// sterilization (masked pixels → -10.0f sentinel), mask resize, and the
// reset()/re-set lifecycle. These were only exercised incidentally before.
// Compiled only when DIC_HAVE_OPENCV is set (reference_cache.cpp is OpenCV-gated).
#include "framework/test_framework.h"

#include <semper/image.hpp>
#include <semper/pipeline.hpp>

#include <type_traits>

using Semper::pipeline::ReferenceCache;

// The struct's own contract: one process-wide instance owns a raw Image*, so it
// must never be copied or moved (double-free). Pin it at compile time.
static_assert(!std::is_copy_constructible<ReferenceCache>::value,
              "ReferenceCache must stay non-copyable (docs/CONTRACT.md A.2)");
static_assert(!std::is_move_constructible<ReferenceCache>::value,
              "ReferenceCache must stay non-movable (docs/CONTRACT.md A.2)");

#if defined(DIC_HAVE_OPENCV)

#include <opencv2/core.hpp>

namespace {

constexpr float kGhost = -10.0f;   // sterilization sentinel (reference_cache.cpp)

cv::Mat ramp_gray(int w, int h) {
    cv::Mat m(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            m.at<uchar>(y, x) = static_cast<uchar>((x + y) % 200 + 20);  // strictly > 0
    return m;
}

// Mask: left half black (< 128 → sterilized), right half white (kept).
cv::Mat left_half_black_mask(int w, int h) {
    cv::Mat m(h, w, CV_8UC1, cv::Scalar(255));
    m(cv::Rect(0, 0, w / 2, h)).setTo(0);
    return m;
}

} // namespace

TEST_CASE(ReferenceCache, RoiSterilization_MasksLeftHalf) {
    const int w = 40, h = 30;
    cv::Mat gray = ramp_gray(w, h);
    cv::Mat mask = left_half_black_mask(w, h);

    ReferenceCache cache;
    cache.set_from_gray(gray, mask);

    REQUIRE(cache.ref_img != nullptr);
    CHECK(cache.width == w);
    CHECK(cache.height == h);

    int masked_ok = 0, kept_ok = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = cache.ref_img->intensities[(size_t)y * w + x];
            if (x < w / 2) {
                if (v == kGhost) masked_ok++;          // sterilized
            } else {
                if (v >= 0.0f) kept_ok++;              // untouched, real pixel
            }
        }
    }
    CHECK(masked_ok == (w / 2) * h);
    CHECK(kept_ok == (w - w / 2) * h);
}

TEST_CASE(ReferenceCache, MaskResize_NearestKeepsHalfSplit) {
    // A mask at a different resolution must be INTER_NEAREST-resized to the
    // reference before sterilizing, so the split still lands at the midline.
    const int w = 40, h = 30;
    cv::Mat gray = ramp_gray(w, h);
    cv::Mat small_mask = left_half_black_mask(w / 4, h / 4);   // quarter size

    ReferenceCache cache;
    cache.set_from_gray(gray, small_mask);
    REQUIRE(cache.ref_img != nullptr);

    // Deep in each half, well clear of any 1-px nearest-resize boundary jitter.
    float left = cache.ref_img->intensities[(size_t)(h / 2) * w + (w / 4)];
    float right = cache.ref_img->intensities[(size_t)(h / 2) * w + (3 * w / 4)];
    CHECK(left == kGhost);
    CHECK(right >= 0.0f);
}

TEST_CASE(ReferenceCache, NoMask_LeavesAllPixelsReal) {
    const int w = 24, h = 16;
    cv::Mat gray = ramp_gray(w, h);

    ReferenceCache cache;
    cache.set_from_gray(gray, cv::Mat());   // empty mask → no sterilization
    REQUIRE(cache.ref_img != nullptr);

    int ghosts = 0;
    for (size_t i = 0; i < cache.ref_img->intensities.size(); ++i)
        if (cache.ref_img->intensities[i] == kGhost) ghosts++;
    CHECK(ghosts == 0);
}

TEST_CASE(ReferenceCache, Relifecycle_UpdatesDims) {
    ReferenceCache cache;
    cache.set_from_gray(ramp_gray(40, 30), cv::Mat());
    REQUIRE(cache.ref_img != nullptr);
    CHECK(cache.width == 40);
    CHECK(cache.height == 30);

    // Second call with a different size must rebuild cleanly, no stale state.
    cache.set_from_gray(ramp_gray(20, 50), cv::Mat());
    REQUIRE(cache.ref_img != nullptr);
    CHECK(cache.width == 20);
    CHECK(cache.height == 50);
    CHECK(cache.ref_img->width == 20);
    CHECK(cache.ref_img->height == 50);
}

TEST_CASE(ReferenceCache, EmptyInput_LeavesCacheCleared) {
    ReferenceCache cache;
    cache.set_from_gray(ramp_gray(24, 16), cv::Mat());
    REQUIRE(cache.ref_img != nullptr);

    // Empty gray returns silently (no error signal) but must clear, not retain
    // the previous reference — otherwise a failed re-init solves stale pixels.
    cache.set_from_gray(cv::Mat(), cv::Mat());
    CHECK(cache.ref_img == nullptr);
    CHECK(cache.width == 0);
    CHECK(cache.height == 0);
    CHECK(cache.gray.empty());
}

TEST_CASE(ReferenceCache, Reset_ClearsEverything) {
    ReferenceCache cache;
    cache.set_from_gray(ramp_gray(24, 16), cv::Mat());
    REQUIRE(cache.ref_img != nullptr);

    cache.reset();
    CHECK(cache.ref_img == nullptr);
    CHECK(cache.width == 0);
    CHECK(cache.height == 0);
    CHECK(cache.gray.empty());
}

#else

TEST_CASE(ReferenceCache, OpenCvRequired_SkippedWithoutOpenCV) {
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
