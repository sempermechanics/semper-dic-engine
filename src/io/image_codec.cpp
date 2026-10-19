#include <semper/io.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstring>

namespace Semper {
namespace io {

namespace {

// Whether `data` starts like a container imdecode reads. Checked before the
// raw-length rule, which an encoded file can satisfy by coincidence.
bool has_image_signature(const uint8_t* data, size_t len) {
    auto starts = [&](const char* sig, size_t n) {
        return len >= n && std::memcmp(data, sig, n) == 0;
    };
    return starts("\x89PNG\r\n\x1a\n", 8) ||          // PNG
           starts("\xff\xd8\xff", 3) ||                 // JPEG
           starts("II*\0", 4) || starts("MM\0*", 4) ||   // TIFF
           starts("BM", 2) ||                            // BMP
           (starts("RIFF", 4) && len >= 12 && std::memcmp(data + 8, "WEBP", 4) == 0) ||
           (len >= 2 && data[0] == 'P' && data[1] >= '1' && data[1] <= '6');   // PNM
}

// The single encoded-image path.
cv::Mat decode_encoded(const uint8_t* data, size_t len) {
    if (len > (size_t)INT32_MAX) return cv::Mat();
    cv::Mat rawData(1, (int)len, CV_8UC1, const_cast<uint8_t*>(data));
    return to_gray8(cv::imdecode(rawData, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR));
}

} // namespace

cv::Mat to_gray8(const cv::Mat& img) {
    if (img.empty()) return cv::Mat();
    cv::Mat gray;
    if (img.channels() == 1) {
        gray = img;
    } else if (img.channels() == 3 || img.channels() == 4) {
        // cvtColor wants 8U, 16U or 32F; anything else is widened first.
        cv::Mat src = img;
        if (src.depth() != CV_8U && src.depth() != CV_16U && src.depth() != CV_32F)
            src.convertTo(src, CV_32F);
        cv::cvtColor(src, gray, img.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    } else {
        return cv::Mat();
    }
    if (gray.depth() == CV_8U) return gray.clone();

    // Not 8-bit: stretch the image's own range onto 0..255. convertTo without
    // a scale saturated instead -- a 12-bit camera frame came out nearly all
    // 255, and a [0, 1] float image all 0 or 1.
    double lo = 0.0, hi = 0.0;
    cv::minMaxLoc(gray, &lo, &hi);
    cv::Mat out;
    if (!(hi > lo)) {   // flat, or NaN in a float image
        out = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(128));
        return out;
    }
    const double scale = 255.0 / (hi - lo);
    gray.convertTo(out, CV_8U, scale, -lo * scale);
    return out;
}

cv::Mat decode_gray(const uint8_t* data, size_t len,
                    int expected_width, int expected_height) {
    if (data == nullptr || len == 0) return cv::Mat();
    if (expected_width > 0 && expected_height > 0) {
        const size_t px = (size_t)expected_width * (size_t)expected_height;
        const bool raw_rgba = len == px * 4, raw_gray = len == px;
        if (raw_rgba || raw_gray) {
            // A raw buffer can start like an image file; only a successful
            // decode makes it one.
            if (has_image_signature(data, len)) {
                cv::Mat img = decode_encoded(data, len);
                if (!img.empty()) return img;
            }
            cv::Mat img;
            if (raw_rgba) {
                cv::Mat rawData(expected_height, expected_width, CV_8UC4, const_cast<uint8_t*>(data));
                cv::cvtColor(rawData, img, cv::COLOR_RGBA2GRAY);
            } else {
                cv::Mat rawData(expected_height, expected_width, CV_8UC1, const_cast<uint8_t*>(data));
                img = rawData.clone();
            }
            return img;
        }
    }
    return decode_encoded(data, len);
}

cv::Mat decode_bgr(const uint8_t* data, size_t len) {
    if (data == nullptr || len == 0 || len > (size_t)INT32_MAX) return cv::Mat();
    cv::Mat rawData(1, (int)len, CV_8UC1, const_cast<uint8_t*>(data));
    return cv::imdecode(rawData, cv::IMREAD_COLOR);
}

void image_dimensions(const uint8_t* data, size_t len, int& out_w, int& out_h) {
    out_w = 0;
    out_h = 0;
    cv::Mat img = decode_bgr(data, len);
    if (img.empty()) {
        // try unchanged
        if (data == nullptr || len == 0 || len > (size_t)INT32_MAX) return;
        cv::Mat rawData(1, (int)len, CV_8UC1, const_cast<uint8_t*>(data));
        img = cv::imdecode(rawData, cv::IMREAD_UNCHANGED);
    }
    if (!img.empty()) {
        out_w = img.cols;
        out_h = img.rows;
    }
}

} // namespace io
} // namespace Semper
