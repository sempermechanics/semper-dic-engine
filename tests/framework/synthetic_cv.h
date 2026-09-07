#ifndef DICTEST_SYNTHETIC_CV_H
#define DICTEST_SYNTHETIC_CV_H

// The OpenCV-dependent half of the synthetic-image helpers. Kept out of
// synthetic.h and image_io.h on purpose: those two expose no OpenCV types, so
// they stay usable by the tests that build without OpenCV.

#include <semper/image.hpp>

#include <opencv2/core.hpp>

#include <cstddef>
#include <random>

namespace dictest {

// Float intensities to CV_8UC1. The full-field golden fixture is generated
// through this, so the arithmetic is fixed: clamp in float, then truncate
// v + 0.5f. noise_sigma == 0 leaves the samples exactly as the field produced
// them and ignores noise_seed.
inline cv::Mat gray8(const Semper::Image &img, float noise_sigma = 0.0f,
                     unsigned noise_seed = 0) {
    cv::Mat m(img.height, img.width, CV_8UC1);
    std::mt19937 rng(noise_seed);
    std::normal_distribution<float> gauss(0.0f, noise_sigma);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float v = img.intensities[(std::size_t) y * img.width + x];
            if (noise_sigma > 0.0f) v += gauss(rng);
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    }
    return m;
}

} // namespace dictest

#endif // DICTEST_SYNTHETIC_CV_H
