#include <semper/io.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "util/finite.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace Semper {
namespace io {

namespace {

// The most pixels an image may have. The engine spends about 13 bytes on each
// (the 8-bit gray, then float intensities and two float gradient planes), so
// this keeps one image near 3.5 GB. OpenCV's own limit (2^30 pixels) let a
// file of a few kilobytes -- a highly compressible PNG or TIFF -- declare
// enough pixels to need 13 GB, and the process was killed before any error
// could be returned. Checked against the header before decoding where the
// format allows (PNG, JPEG, BMP, TIFF), and on the decoded size always.
constexpr uint64_t kMaxPixels = uint64_t(1) << 28;

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

uint32_t be16(const uint8_t* p) { return (uint32_t)p[0] << 8 | p[1]; }
uint32_t be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
uint32_t le16(const uint8_t* p) { return (uint32_t)p[1] << 8 | p[0]; }
uint32_t le32(const uint8_t* p) { return (uint32_t)p[3] << 24 | (uint32_t)p[2] << 16 | (uint32_t)p[1] << 8 | p[0]; }

// Width and height from the header of a PNG, JPEG, BMP or (classic) TIFF,
// without decoding anything. False for other formats or a short header.
bool header_dimensions(const uint8_t* d, size_t len, uint64_t& w, uint64_t& h) {
    if (d == nullptr) return false;
    // PNG: the IHDR chunk comes first, width and height big-endian.
    if (len >= 24 && std::memcmp(d, "\x89PNG\r\n\x1a\n", 8) == 0 && std::memcmp(d + 12, "IHDR", 4) == 0) {
        w = be32(d + 16);
        h = be32(d + 20);
        return true;
    }
    // BMP: BITMAPINFOHEADER width/height (a top-down BMP has a negative height).
    if (len >= 26 && d[0] == 'B' && d[1] == 'M') {
        const int32_t bw = (int32_t)le32(d + 18), bh = (int32_t)le32(d + 22);
        w = (uint64_t)std::llabs((long long)bw);
        h = (uint64_t)std::llabs((long long)bh);
        return true;
    }
    // JPEG: walk the marker segments to the first start-of-frame.
    if (len >= 4 && d[0] == 0xFF && d[1] == 0xD8) {
        size_t i = 2;
        while (i + 4 <= len) {
            if (d[i] != 0xFF) return false;
            const uint8_t m = d[i + 1];
            if (m == 0xFF) { ++i; continue; }                    // fill byte
            if (m == 0xD8 || m == 0x01 || (m >= 0xD0 && m <= 0xD7)) { i += 2; continue; }
            if (m == 0xD9 || m == 0xDA) return false;             // no frame before the scan
            const size_t seg = be16(d + i + 2);
            if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
                if (i + 9 > len) return false;
                h = be16(d + i + 5);
                w = be16(d + i + 7);
                return true;
            }
            if (seg < 2) return false;
            i += 2 + seg;
        }
        return false;
    }
    // TIFF: tags 256 (width) and 257 (height) of the first IFD.
    if (len >= 8 && (std::memcmp(d, "II*\0", 4) == 0 || std::memcmp(d, "MM\0*", 4) == 0)) {
        const bool le = d[0] == 'I';
        auto u16 = [&](size_t o) { return le ? le16(d + o) : be16(d + o); };
        auto u32 = [&](size_t o) { return le ? le32(d + o) : be32(d + o); };
        const size_t ifd = u32(4);
        if (ifd + 2 > len) return false;
        const size_t n = u16(ifd);
        bool got_w = false, got_h = false;
        for (size_t k = 0; k < n; ++k) {
            const size_t e = ifd + 2 + 12 * k;
            if (e + 12 > len) return false;
            const uint32_t tag = u16(e), type = u16(e + 2);
            const uint64_t v = type == 3 ? u16(e + 8) : type == 4 ? u32(e + 8) : 0;
            if (tag == 256) { w = v; got_w = type == 3 || type == 4; }
            if (tag == 257) { h = v; got_h = type == 3 || type == 4; }
        }
        return got_w && got_h;
    }
    return false;
}

// What the header promises is more than the engine will hold.
bool header_too_large(const uint8_t* data, size_t len) {
    uint64_t w = 0, h = 0;
    return header_dimensions(data, len, w, h) && (w == 0 || h == 0 || w > kMaxPixels || h > kMaxPixels ||
                                                  w * h > kMaxPixels);
}

bool too_large(const cv::Mat& m) {
    return (uint64_t)m.cols * (uint64_t)m.rows > kMaxPixels;
}

cv::Mat imdecode_bytes(const uint8_t* data, size_t len, int flags) {
    if (data == nullptr || len == 0 || len > (size_t)INT32_MAX) return cv::Mat();
    if (header_too_large(data, len)) return cv::Mat();
    cv::Mat rawData(1, (int)len, CV_8UC1, const_cast<uint8_t*>(data));
    cv::Mat img = cv::imdecode(rawData, flags);
    if (too_large(img)) return cv::Mat();
    return img;
}

// Encoded image, or a raw RGBA / ALPHA_8 buffer of expected_w x expected_h,
// to one 8-bit channel; `convert` takes a decoded image of any depth there.
template <class Convert>
cv::Mat decode_any(const uint8_t* data, size_t len, int expected_width, int expected_height,
                   Convert convert) {
    if (data == nullptr || len == 0) return cv::Mat();
    auto encoded = [&] {
        return convert(imdecode_bytes(data, len, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR));
    };
    if (expected_width > 0 && expected_height > 0) {
        // 64-bit: on a 32-bit size_t, w * h * 4 could wrap to a small number
        // and a short buffer would pass the length test.
        const uint64_t px = (uint64_t)expected_width * (uint64_t)expected_height;
        const bool fits = px <= kMaxPixels;
        const bool raw_rgba = fits && (uint64_t)len == px * 4, raw_gray = fits && (uint64_t)len == px;
        if (raw_rgba || raw_gray) {
            // A raw buffer can start like an image file; only a successful
            // decode makes it one.
            if (has_image_signature(data, len)) {
                cv::Mat img = encoded();
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
    return encoded();
}

// One channel, any depth; empty for a channel count that is not 1, 3 or 4.
cv::Mat to_one_channel(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    if (img.channels() != 3 && img.channels() != 4) return cv::Mat();
    // cvtColor wants 8U, 16U or 32F; anything else is widened first.
    cv::Mat src = img;
    if (src.depth() != CV_8U && src.depth() != CV_16U && src.depth() != CV_32F)
        src.convertTo(src, CV_32F);
    cv::Mat gray;
    cv::cvtColor(src, gray, img.channels() == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    return gray;
}

// The range a non-8-bit image is mapped onto 0..255 from: its 0.1th to 99.9th
// percentile, so a hot pixel or a dead column does not set the scale (one
// pixel at 65535 in a 16-bit frame whose speckle spans 1000..3000 left the
// speckle about 8 grey levels). On an image of up to 1000 pixels these are its
// minimum and maximum. Falls back to min..max when the percentiles meet.
void robust_range(const cv::Mat& gray, double min_v, double max_v, double& lo, double& hi) {
    const size_t n = gray.total();
    const size_t k_lo = (size_t)std::floor(0.001 * (double)(n - 1));
    const size_t k_hi = (size_t)std::ceil(0.999 * (double)(n - 1));
    lo = min_v;
    hi = max_v;
    if (gray.depth() == CV_16U) {
        // A histogram: two passes over the image, no copy of it.
        std::vector<size_t> hist(65536, 0);
        for (int y = 0; y < gray.rows; ++y) {
            const uint16_t* row = gray.ptr<uint16_t>(y);
            for (int x = 0; x < gray.cols; ++x) hist[row[x]]++;
        }
        size_t seen = 0;
        bool have_lo = false;
        for (size_t v = 0; v < hist.size(); ++v) {
            seen += hist[v];
            if (!have_lo && seen > k_lo) { lo = (double)v; have_lo = true; }
            if (seen > k_hi) { hi = (double)v; break; }
        }
    } else {
        cv::Mat d;
        gray.convertTo(d, CV_64F);
        std::vector<double> v;
        v.reserve(n);
        for (int y = 0; y < d.rows; ++y) {
            const double* row = d.ptr<double>(y);
            v.insert(v.end(), row, row + d.cols);
        }
        std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)k_lo, v.end());
        lo = v[k_lo];
        std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)k_hi, v.end());
        hi = v[k_hi];
    }
    if (!(hi > lo)) {
        lo = min_v;
        hi = max_v;
    }
}

} // namespace

cv::Mat to_gray8(const cv::Mat& img) {
    if (img.empty()) return cv::Mat();
    cv::Mat gray = to_one_channel(img);
    if (gray.empty()) return cv::Mat();
    if (gray.depth() == CV_8U) return gray.clone();

    // Not 8-bit: stretch the image's own range onto 0..255. convertTo without
    // a scale saturated instead -- a 12-bit camera frame came out nearly all
    // 255, and a [0, 1] float image all 0 or 1.
    double min_v = 0.0, max_v = 0.0;
    cv::minMaxLoc(gray, &min_v, &max_v);
    // Flat, or a non-finite extreme in a float image.
    if (!util::finite_bits(min_v) || !util::finite_bits(max_v) || !(max_v > min_v)) {
        return cv::Mat(gray.size(), CV_8UC1, cv::Scalar(128));
    }
    double lo = 0.0, hi = 0.0;
    robust_range(gray, min_v, max_v, lo, hi);
    const double scale = 255.0 / (hi - lo);
    cv::Mat out;
    gray.convertTo(out, CV_8U, scale, -lo * scale);   // saturates outside [lo, hi]
    return out;
}

cv::Mat to_mask8(const cv::Mat& mask) {
    if (mask.empty()) return cv::Mat();
    cv::Mat m = to_one_channel(mask);
    if (m.empty()) return cv::Mat();
    if (m.depth() == CV_8U) return m.clone();
    cv::Mat out;
    cv::compare(m, cv::Scalar::all(0), out, cv::CMP_NE);   // 255 where nonzero, else 0
    return out;
}

cv::Mat decode_gray(const uint8_t* data, size_t len,
                    int expected_width, int expected_height) {
    return decode_any(data, len, expected_width, expected_height,
                      [](const cv::Mat& m) { return to_gray8(m); });
}

cv::Mat decode_mask(const uint8_t* data, size_t len,
                    int expected_width, int expected_height) {
    return decode_any(data, len, expected_width, expected_height,
                      [](const cv::Mat& m) { return to_mask8(m); });
}

cv::Mat decode_bgr(const uint8_t* data, size_t len) {
    return imdecode_bytes(data, len, cv::IMREAD_COLOR);
}

void image_dimensions(const uint8_t* data, size_t len, int& out_w, int& out_h) {
    out_w = 0;
    out_h = 0;
    if (data == nullptr || len == 0) return;
    uint64_t w = 0, h = 0;
    if (header_dimensions(data, len, w, h)) {
        // From the header alone. A header that cannot describe a decodable
        // image (zero, or beyond int) reports zeros, as a failed decode does.
        if (w > 0 && h > 0 && w <= (uint64_t)INT32_MAX && h <= (uint64_t)INT32_MAX) {
            out_w = (int)w;
            out_h = (int)h;
        }
        return;
    }
    // Other formats (WebP, PNM): decode, at one byte a pixel rather than three.
    cv::Mat img = imdecode_bytes(data, len, cv::IMREAD_GRAYSCALE);
    if (!img.empty()) {
        out_w = img.cols;
        out_h = img.rows;
    }
}

} // namespace io
} // namespace Semper
