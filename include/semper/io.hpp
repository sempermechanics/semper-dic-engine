#ifndef SEMPER_IO_HPP
#define SEMPER_IO_HPP

#include <opencv2/core.hpp>
#include <cstddef>
#include <cstdint>

namespace Semper {
namespace io {

/**
 * Decode grayscale image from encoded bytes or raw RGBA/ALPHA_8 buffer.
 * With expected_w/h > 0, a buffer of exactly w*h*4 or w*h bytes is raw unless
 * it starts with a known image signature (PNG, JPEG, TIFF, BMP, WebP, PNM)
 * and decodes as one. Encoded images go through to_gray8.
 */
cv::Mat decode_gray(const uint8_t* data, size_t len,
                    int expected_width = 0, int expected_height = 0);

/**
 * Any image to the engine's 8-bit single-channel input (added v0.5.0). Colour
 * collapses to gray; 8-bit passes through unchanged; any other depth (16-bit,
 * signed, float) is mapped linearly from its own min..max onto 0..255, which
 * keeps its contrast and which ZNSSD does not see (it is invariant to a*I + b
 * per image). A flat image maps to 128. Empty in, empty out.
 */
cv::Mat to_gray8(const cv::Mat& img);

/** Decode BGR (for preview thumbnails). Empty on failure. */
cv::Mat decode_bgr(const uint8_t* data, size_t len);

/** Image width/height without full decode when possible; zeros on failure. */
void image_dimensions(const uint8_t* data, size_t len, int& out_w, int& out_h);

} // namespace io
} // namespace Semper

#endif
