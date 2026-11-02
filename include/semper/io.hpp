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
 *
 * An image of more than 2^28 pixels is refused (empty Mat), before it is
 * decoded where the header gives its size (PNG, JPEG, BMP, TIFF): the engine
 * spends about 13 bytes a pixel on it (since v0.7.0).
 */
cv::Mat decode_gray(const uint8_t* data, size_t len,
                    int expected_width = 0, int expected_height = 0);

/**
 * Any image to the engine's 8-bit single-channel input (added v0.5.0). Colour
 * collapses to gray; 8-bit passes through unchanged; any other depth (16-bit,
 * signed, float) is mapped linearly onto 0..255, which keeps its contrast and
 * which ZNSSD does not see (it is invariant to a*I + b per image). The range
 * mapped is the image's 0.1th to 99.9th percentile (v0.7.0; min..max before),
 * so a hot pixel or a dead column does not set the scale; values outside it
 * saturate. On an image of up to 1000 pixels that is its min..max. A flat
 * image, or one with an infinite value, maps to 128. Empty in, empty out.
 */
cv::Mat to_gray8(const cv::Mat& img);

/**
 * Any ROI mask to the engine's 8-bit mask (added v0.7.0). The engine excludes
 * a pixel whose mask value is below 128. An 8-bit mask passes through
 * unchanged, so that rule applies to it as it always has (0 out, 255 in).
 * A mask of any other depth -- 16-bit, signed, float, a NumPy bool array --
 * is a boolean mask: nonzero is included (255), zero excluded (0). It used to
 * go through to_gray8, whose stretch turned an all-zero (or all-false) mask
 * into 128, i.e. everything included. Colour collapses to gray first. Empty
 * in, empty out.
 */
cv::Mat to_mask8(const cv::Mat& mask);

/**
 * decode_gray for a mask (added v0.7.0): the same raw/encoded rules, but a
 * decoded image of a depth other than 8-bit goes through to_mask8, not
 * to_gray8. The C ABI and JNI adapters decode masks with it.
 */
cv::Mat decode_mask(const uint8_t* data, size_t len,
                    int expected_width = 0, int expected_height = 0);

/** Decode BGR (for preview thumbnails). Empty on failure. */
cv::Mat decode_bgr(const uint8_t* data, size_t len);

/**
 * Image width/height; zeros on failure. Read from the header alone for PNG,
 * JPEG, BMP and TIFF (v0.7.0; it used to decode the whole image in colour);
 * other formats are decoded, in gray.
 */
void image_dimensions(const uint8_t* data, size_t len, int& out_w, int& out_h);

} // namespace io
} // namespace Semper

#endif
