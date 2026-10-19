// Boundary tests for Semper::io — the single point where every external image
// (C ABI, Python, JNI) enters the engine. image_codec.cpp was in NO test target
// before this suite; a pixel-corrupting bug here was invisible to the whole
// pipeline. Compiled only when DIC_HAVE_OPENCV is set (needs imgcodecs/imgproc),
// same gate as the DICe and full-field contract suites.
#include "framework/test_framework.h"

#include <semper/io.hpp>

#include <cstdint>
#include <vector>

#if defined(DIC_HAVE_OPENCV)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using Semper::io::decode_bgr;
using Semper::io::decode_gray;
using Semper::io::image_dimensions;

namespace {

// A small deterministic RGBA image with distinct per-channel gradients, so a
// channel swap or a wrong colour-conversion constant shows up on spot pixels.
cv::Mat make_rgba(int w, int h) {
    cv::Mat m(h, w, CV_8UC4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            auto& px = m.at<cv::Vec4b>(y, x);
            px[0] = static_cast<uchar>((x * 7) & 0xFF);   // R
            px[1] = static_cast<uchar>((y * 5) & 0xFF);   // G
            px[2] = static_cast<uchar>((x + y) & 0xFF);   // B
            px[3] = 255;                                  // A
        }
    }
    return m;
}

cv::Mat make_gray(int w, int h) {
    cv::Mat m(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            m.at<uchar>(y, x) = static_cast<uchar>((x * 3 + y * 11) & 0xFF);
    return m;
}

} // namespace

// --- Raw buffer paths ------------------------------------------------------

TEST_CASE(ImageCodec, RawRgba_MatchesOpenCvGray) {
    const int w = 8, h = 6;
    cv::Mat rgba = make_rgba(w, h);
    REQUIRE(rgba.isContinuous());

    cv::Mat got = decode_gray(rgba.data, (size_t)w * h * 4, w, h);
    REQUIRE(!got.empty());
    CHECK(got.cols == w);
    CHECK(got.rows == h);
    CHECK(got.type() == CV_8UC1);

    cv::Mat want;
    cv::cvtColor(rgba, want, cv::COLOR_RGBA2GRAY);
    // Spot-check corners and centre — exact, same conversion the codec uses.
    CHECK(got.at<uchar>(0, 0) == want.at<uchar>(0, 0));
    CHECK(got.at<uchar>(h - 1, w - 1) == want.at<uchar>(h - 1, w - 1));
    CHECK(got.at<uchar>(h / 2, w / 2) == want.at<uchar>(h / 2, w / 2));
}

TEST_CASE(ImageCodec, RawGray_IsExactClone) {
    const int w = 9, h = 7;
    cv::Mat gray = make_gray(w, h);
    REQUIRE(gray.isContinuous());

    cv::Mat got = decode_gray(gray.data, (size_t)w * h, w, h);
    REQUIRE(!got.empty());
    CHECK(got.cols == w);
    CHECK(got.rows == h);
    CHECK(got.type() == CV_8UC1);
    // Byte-for-byte identical to the input buffer.
    bool identical = (cv::countNonZero(got != gray) == 0);
    CHECK(identical);
    // ...and a genuine clone, not an alias of the caller's buffer.
    CHECK(got.data != gray.data);
}

// --- Encoded paths ---------------------------------------------------------

TEST_CASE(ImageCodec, EncodedPng_RoundTripsExactly) {
    const int w = 12, h = 10;
    cv::Mat gray = make_gray(w, h);
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", gray, png));   // PNG is lossless

    // expected_w/h left 0 → encoded path.
    cv::Mat got = decode_gray(png.data(), png.size());
    REQUIRE(!got.empty());
    CHECK(got.cols == w);
    CHECK(got.rows == h);
    CHECK(got.type() == CV_8UC1);
    CHECK(cv::countNonZero(got != gray) == 0);
}

TEST_CASE(ImageCodec, EncodedColor_ConvertsToGray) {
    const int w = 10, h = 8;
    cv::Mat bgr(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            bgr.at<cv::Vec3b>(y, x) =
                cv::Vec3b((uchar)(x * 9), (uchar)(y * 13), (uchar)((x + y) * 3));
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", bgr, png));

    cv::Mat got = decode_gray(png.data(), png.size());
    REQUIRE(!got.empty());
    CHECK(got.type() == CV_8UC1);   // colour collapsed to one channel
    CHECK(got.cols == w);
    CHECK(got.rows == h);
}

TEST_CASE(ImageCodec, EncodedNon8U_ConvertsTo8U) {
    const int w = 6, h = 5;
    cv::Mat u16(h, w, CV_16UC1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            u16.at<uint16_t>(y, x) = static_cast<uint16_t>((x + y) * 1000);
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", u16, png));   // 16-bit PNG

    cv::Mat got = decode_gray(png.data(), png.size());
    REQUIRE(!got.empty());
    CHECK(got.depth() == CV_8U);    // the non-8U → CV_8U branch
    CHECK(got.channels() == 1);
    CHECK(got.cols == w);
    CHECK(got.rows == h);
}

// A 16-bit image used to be convertTo(CV_8U) without a scale, which
// saturates: a 12-bit camera frame (0..4095) came out almost entirely 255.
// It is now mapped linearly from its own range onto 0..255, which ZNSSD does
// not see (it is invariant to a*I + b per image).
TEST_CASE(ImageCodec, Encoded16Bit_PreservesContrast) {
    const int w = 64, h = 8;
    cv::Mat u16(h, w, CV_16UC1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            u16.at<uint16_t>(y, x) = static_cast<uint16_t>(x * 64 + y);   // 0..4039, 12-bit
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", u16, png));

    cv::Mat got = decode_gray(png.data(), png.size());
    REQUIRE(!got.empty());
    REQUIRE(got.type() == CV_8UC1);
    double lo = 0, hi = 0;
    cv::minMaxLoc(got, &lo, &hi);
    CHECK(lo == 0.0);
    CHECK(hi == 255.0);
    // Monotone along the ramp, and every column distinct.
    for (int x = 1; x < w; ++x) CHECK(got.at<uchar>(0, x) > got.at<uchar>(0, x - 1));
}

TEST_CASE(ImageCodec, FloatAndSigned_ScaledToGray8) {
    cv::Mat f(4, 16, CV_32FC1);
    for (int x = 0; x < 16; ++x) f.col(x).setTo(x / 15.0f);   // a [0, 1] float image
    cv::Mat g = Semper::io::to_gray8(f);
    REQUIRE(g.type() == CV_8UC1);
    CHECK(g.at<uchar>(0, 0) == 0);
    CHECK(g.at<uchar>(0, 15) == 255);
    CHECK(g.at<uchar>(2, 5) == 85);
    // A flat image has no range to stretch: mid-grey, not a division by zero.
    cv::Mat flat(3, 3, CV_64FC1, cv::Scalar(0.25));
    CHECK(cv::countNonZero(Semper::io::to_gray8(flat) != 128) == 0);
    // 8-bit grey passes through untouched; colour collapses to one channel.
    cv::Mat g8 = make_gray(5, 4);
    CHECK(cv::countNonZero(Semper::io::to_gray8(g8) != g8) == 0);
    cv::Mat bgr(4, 5, CV_8UC3, cv::Scalar(10, 20, 30));
    CHECK(Semper::io::to_gray8(bgr).type() == CV_8UC1);
    CHECK(Semper::io::to_gray8(cv::Mat()).empty());
}

// Raw or encoded used to be decided by length alone. An encoded file whose
// size happens to equal w * h (or w * h * 4) for the expected dimensions was
// read as raw pixels -- the PNG's bytes, as an image.
TEST_CASE(ImageCodec, EncodedWithRawLength_StillDecoded) {
    const int w = 11, h = 9;
    cv::Mat gray = make_gray(w, h);
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", gray, png));
    const int n = (int)png.size();
    cv::Mat got = decode_gray(png.data(), png.size(), n, 1);   // n * 1 == len
    REQUIRE(!got.empty());
    CHECK(got.cols == w);
    CHECK(got.rows == h);
    CHECK(cv::countNonZero(got != gray) == 0);
    // ...and a raw buffer that merely starts like a JPEG is still raw.
    std::vector<uint8_t> raw((size_t)w * h, 7);
    raw[0] = 0xFF; raw[1] = 0xD8; raw[2] = 0xFF;
    cv::Mat r = decode_gray(raw.data(), raw.size(), w, h);
    REQUIRE(!r.empty());
    CHECK(r.cols == w);
    CHECK(r.at<uchar>(0, 0) == 0xFF);
    CHECK(r.at<uchar>(0, 3) == 7);
}

TEST_CASE(ImageCodec, MismatchedLength_FallsThroughToEncoded) {
    // expected_w/h are set, but len matches neither w*h nor w*h*4, so the raw
    // branches are skipped and the encoded decoder runs on the PNG bytes.
    const int w = 11, h = 9;
    cv::Mat gray = make_gray(w, h);
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", gray, png));
    REQUIRE(png.size() != (size_t)w * h);
    REQUIRE(png.size() != (size_t)w * h * 4);

    cv::Mat got = decode_gray(png.data(), png.size(), w, h);
    REQUIRE(!got.empty());
    CHECK(got.cols == w);
    CHECK(got.rows == h);
    CHECK(cv::countNonZero(got != gray) == 0);
}

// --- Guards ----------------------------------------------------------------

TEST_CASE(ImageCodec, NullAndEmpty_ReturnEmpty) {
    const uint8_t byte = 0;
    CHECK(decode_gray(nullptr, 16, 4, 4).empty());
    CHECK(decode_gray(&byte, 0, 4, 4).empty());
    CHECK(decode_gray(nullptr, 0).empty());
    CHECK(decode_bgr(nullptr, 16).empty());
    CHECK(decode_bgr(&byte, 0).empty());
}

TEST_CASE(ImageCodec, GarbageBytes_ReturnEmptyNoCrash) {
    // Not a valid image and not a raw w*h / w*h*4 buffer → empty, no crash.
    std::vector<uint8_t> junk(37);
    for (size_t i = 0; i < junk.size(); ++i) junk[i] = static_cast<uint8_t>(i * 7 + 1);
    CHECK(decode_gray(junk.data(), junk.size()).empty());
    CHECK(decode_gray(junk.data(), junk.size(), 100, 100).empty());
    CHECK(decode_bgr(junk.data(), junk.size()).empty());
}

TEST_CASE(ImageCodec, DecodeBgr_EncodedColorHasThreeChannels) {
    const int w = 7, h = 6;
    cv::Mat bgr(h, w, CV_8UC3, cv::Scalar(10, 20, 30));
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", bgr, png));

    cv::Mat got = decode_bgr(png.data(), png.size());
    REQUIRE(!got.empty());
    CHECK(got.channels() == 3);
    CHECK(got.cols == w);
    CHECK(got.rows == h);
}

// --- image_dimensions ------------------------------------------------------

TEST_CASE(ImageCodec, ImageDimensions_EncodedBuffer) {
    const int w = 13, h = 5;
    cv::Mat gray = make_gray(w, h);
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", gray, png));

    int ow = -1, oh = -1;
    image_dimensions(png.data(), png.size(), ow, oh);
    CHECK(ow == w);
    CHECK(oh == h);
}

TEST_CASE(ImageCodec, ImageDimensions_ZerosOnFailureAndNull) {
    std::vector<uint8_t> junk(29, 0xAB);
    int ow = 5, oh = 5;
    image_dimensions(junk.data(), junk.size(), ow, oh);
    CHECK(ow == 0);
    CHECK(oh == 0);

    ow = 5; oh = 5;
    image_dimensions(nullptr, 10, ow, oh);
    CHECK(ow == 0);
    CHECK(oh == 0);
}

TEST_CASE(ImageCodec, ImageDimensions_RawBufferYieldsZeros_KnownTrap) {
    // image_dimensions has no raw-buffer awareness: a raw RGBA/gray blob is not a
    // decodable container, so it reports 0x0. Callers with raw buffers already
    // know the dimensions; this pins the documented limitation rather than a fix.
    const int w = 8, h = 8;
    cv::Mat rgba = make_rgba(w, h);
    int ow = -1, oh = -1;
    image_dimensions(rgba.data, (size_t)w * h * 4, ow, oh);
    CHECK(ow == 0);
    CHECK(oh == 0);
}

#else

TEST_CASE(ImageCodec, OpenCvRequired_SkippedWithoutOpenCV) {
    // Suite still registers when OpenCV is absent so the filter surface is stable.
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
