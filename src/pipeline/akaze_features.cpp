// AKAZE global-motion front end: descriptor matching between the reference and
// the deformed frame over a padded ROI, and a RANSAC homography over the
// matches. full_field_anchors.cpp seeds the anchor lattice from that
// homography; nothing here touches the field.
//
// A homography covers what the anchor lattice needs a global model for --
// translation, rotation at any angle, scale, anisotropic stretch, shear -- from
// the matches themselves, so unlike the log-polar similarity it has no 180 deg
// ambiguity and does not mistake stretch for rotation.

#include <semper/seeding.hpp>
#include <semper/tuning.hpp>
#include "util/finite.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>

namespace Semper {
namespace seeding {

namespace {

double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
}

cv::Rect padded_rect(const cv::Rect& roi, double scale, cv::Size image) {
    const int pad = (int)(tuning::kAkazePadPx / scale);
    cv::Rect r(roi.x - pad, roi.y - pad, roi.width + 2 * pad, roi.height + 2 * pad);
    return r & cv::Rect(0, 0, image.width, image.height);
}

// Runs f on the calling thread with OpenCV's thread pool left asleep: OpenCV
// runs a parallel_for_ that starts inside another one serially, and a
// one-stripe parallel_for_ runs its body inline. AKAZE on a few hundred
// pixels gains under 1 ms from the pool, but the pool's workers were still
// spinning when the engine's OpenMP region started right after, and on a host
// a small-ROI frame took up to 2-3x as long (wall time, same CPU work). The
// Android adapter pins OpenCV to one thread anyway.
template <class F>
void without_opencv_pool(F&& f) {
    cv::parallel_for_(cv::Range(0, 1), [&](const cv::Range&) { f(); }, 1.0);
}

void detect(const cv::Mat& gray, const cv::Rect& rect, double scale,
            std::vector<cv::KeyPoint>& kp, cv::Mat& desc) {
    without_opencv_pool([&] {
        cv::Mat small;
        cv::resize(gray(rect), small, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::AKAZE::create()->detectAndCompute(small, cv::noArray(), kp, desc);
    });
}

} // namespace

AkazeRef prepare_akaze_ref(const cv::Mat& ref_gray, const cv::Rect& roi, double scale) {
    AkazeRef out;
    out.scale = scale;
    out.rect = padded_rect(roi, scale, ref_gray.size());
    if (out.rect.width < 8 || out.rect.height < 8) return out;
    detect(ref_gray, out.rect, scale, out.keypoints, out.descriptors);
    return out;
}

bool reuse_akaze_ref(const AkazeRef& from, const cv::Rect& roi, cv::Size image, AkazeRef& out) {
    if (!from.usable() || from.scale <= 0.0) return false;
    const cv::Rect want = padded_rect(roi, from.scale, image);
    const double cover = (double)(want & from.rect).area();
    if (want.area() <= 0 || cover < tuning::kAkazeReuseCover * want.area()) return false;

    AkazeRef r;
    r.rect = want;
    r.scale = from.scale;
    // q' = s (P - o') = q + s (o - o'); kept if inside the new rect.
    const float dx = (float)(from.scale * (from.rect.x - want.x));
    const float dy = (float)(from.scale * (from.rect.y - want.y));
    const float w = (float)(from.scale * want.width), h = (float)(from.scale * want.height);
    std::vector<int> keep;
    for (size_t i = 0; i < from.keypoints.size(); ++i) {
        cv::KeyPoint k = from.keypoints[i];
        k.pt.x += dx;
        k.pt.y += dy;
        if (k.pt.x < 0 || k.pt.y < 0 || k.pt.x >= w || k.pt.y >= h) continue;
        r.keypoints.push_back(k);
        keep.push_back((int)i);
    }
    r.descriptors.create((int)keep.size(), from.descriptors.cols, from.descriptors.type());
    for (size_t j = 0; j < keep.size(); ++j) from.descriptors.row(keep[j]).copyTo(r.descriptors.row((int)j));
    if (!r.usable()) return false;
    out = std::move(r);
    return true;
}

bool match_akaze(const AkazeRef& ref, const cv::Mat& def_gray, const cv::Mat& roi_mask,
                 AkazeMatches& out) {
    out = AkazeMatches{};
    out.scale = ref.scale;
    if (!ref.usable() || def_gray.cols < ref.rect.br().x || def_gray.rows < ref.rect.br().y) {
        return false;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    detect(def_gray, ref.rect, ref.scale, kp, desc);
    if (kp.empty()) {
        out.detect_ms = ms_since(t0);
        return false;
    }

    std::vector<std::vector<cv::DMatch>> knn;
    without_opencv_pool([&] { cv::BFMatcher(cv::NORM_HAMMING).knnMatch(ref.descriptors, desc, knn, 2); });
    out.def_side.rect = ref.rect;
    out.def_side.scale = ref.scale;

    // Matching ran on q = s (P - o); in image coordinates P = o + q / s.
    const double s = ref.scale, ox = ref.rect.x, oy = ref.rect.y;
    std::vector<cv::Point2f> p1, p2;
    for (const auto& m : knn) {
        if (m.size() != 2 || !(m[0].distance < tuning::kAkazeRatio * m[1].distance)) continue;
        const cv::Point2f& q1 = ref.keypoints[(size_t)m[0].queryIdx].pt;
        if (!roi_mask.empty()) {
            const int mx = (int)(q1.x / s + ox), my = (int)(q1.y / s + oy);
            if (mx < 0 || my < 0 || mx >= roi_mask.cols || my >= roi_mask.rows ||
                roi_mask.at<uchar>(my, mx) < 128) {
                continue;
            }
        }
        p1.push_back(q1);
        p2.push_back(kp[(size_t)m[0].trainIdx].pt);
    }
    out.def_side.keypoints = std::move(kp);
    out.def_side.descriptors = desc;
    out.detect_ms = ms_since(t0);
    if ((int)p1.size() < tuning::kAkazeMinInliers) return false;

    const auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<uchar> inlier;
    cv::Mat H;
    without_opencv_pool([&] { H = cv::findHomography(p1, p2, cv::RANSAC, tuning::kAkazeRansacPx, inlier); });
    for (size_t i = 0; i < inlier.size(); ++i) {
        if (!inlier[i]) continue;
        out.ref_pts.emplace_back((float)(p1[i].x / s + ox), (float)(p1[i].y / s + oy));
        out.def_pts.emplace_back((float)(p2[i].x / s + ox), (float)(p2[i].y / s + oy));
    }
    if (H.rows == 3 && H.cols == 3 && cv::checkRange(H) && out.inliers() >= tuning::kAkazeMinInliers) {
        // P_def = o + H(s (P - o)) / s, i.e. T^-1 H T with T = s (P - o).
        const cv::Mat T(cv::Matx33d(s, 0, -s * ox, 0, s, -s * oy, 0, 0, 1));
        const cv::Mat Ti(cv::Matx33d(1.0 / s, 0, ox, 0, 1.0 / s, oy, 0, 0, 1));
        out.homography = Ti * H * T;
    }
    out.ransac_ms = ms_since(t1);
    return !out.homography.empty();
}

bool homography_at(const cv::Mat& H, double px, double py, double& X, double& Y, double J[4]) {
    if (H.rows != 3 || H.cols != 3 || H.type() != CV_64F) return false;
    const double* h = H.ptr<double>(0);
    const double w = h[6] * px + h[7] * py + h[8];
    // finite_bits, not std::isfinite: -ffast-math may fold the latter to true.
    if (!util::finite_bits(w) || std::abs(w) < 1e-12) return false;
    X = (h[0] * px + h[1] * py + h[2]) / w;
    Y = (h[3] * px + h[4] * py + h[5]) / w;
    J[0] = (h[0] - X * h[6]) / w;
    J[1] = (h[1] - X * h[7]) / w;
    J[2] = (h[3] - Y * h[6]) / w;
    J[3] = (h[4] - Y * h[7]) / w;
    return util::finite_bits(X) && util::finite_bits(Y) && util::finite_bits(J[0]) &&
           util::finite_bits(J[1]) && util::finite_bits(J[2]) && util::finite_bits(J[3]);
}

} // namespace seeding
} // namespace Semper
