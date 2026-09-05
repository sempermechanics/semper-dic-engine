#include <semper/seeding.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace Semper {
namespace seeding {

const char *seed_method_name(SeedMethod m) {
    switch (m) {
        case SeedMethod::AkazePyramid:  return "akaze_pyramid";
        case SeedMethod::AkazeFull:     return "akaze_full";
        case SeedMethod::Sift:          return "sift";
        case SeedMethod::Orb:           return "orb";
        case SeedMethod::Brisk:         return "brisk";
        case SeedMethod::Kaze:          return "kaze";
        case SeedMethod::AnchorLattice: return "anchor_lattice";
    }
    return "unknown";
}

bool is_descriptor_method(SeedMethod m) {
    return m != SeedMethod::AnchorLattice;
}

namespace {

// Detector + matching norm for one candidate. ORB is given a 5000-feature
// budget because every other detector here is uncapped; leaving it at OpenCV's
// 500 default would measure the cap rather than the detector.
cv::Ptr<cv::Feature2D> make_detector(SeedMethod method, int &out_norm) {
    switch (method) {
        case SeedMethod::Sift:
            out_norm = cv::NORM_L2;
            return cv::SIFT::create();
        case SeedMethod::Orb:
            out_norm = cv::NORM_HAMMING;
            return cv::ORB::create(5000);
        case SeedMethod::Brisk:
            out_norm = cv::NORM_HAMMING;
            return cv::BRISK::create();
        case SeedMethod::Kaze:
            out_norm = cv::NORM_L2;
            return cv::KAZE::create();
        case SeedMethod::AkazePyramid:
        case SeedMethod::AkazeFull:
        case SeedMethod::AnchorLattice:
        default:
            out_norm = cv::NORM_HAMMING;
            return cv::AKAZE::create();
    }
}

} // namespace

void draw_outlined_text(cv::Mat &img, const std::string &text, cv::Point pt, double scale) {
        cv::putText(img, text, pt, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(img, text, pt, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

bool extract_descriptor_features(cv::Mat &ref, cv::Mat &def, cv::Mat &roi_mask, double scale, std::vector<cv::Point2f> &out_ref_pts,
                              std::vector<cv::Point2f> &out_def_pts, float &out_bounding_box_area_ratio,
                              double &out_detect_ms, double &out_ransac_ms,
                              std::vector<cv::KeyPoint>& cached_kp, cv::Mat& cached_desc,
                              int offset_x, int offset_y, SeedMethod method,
                              const std::string &debug_dir) {
        (void)debug_dir;

        auto t_start_akaze = std::chrono::high_resolution_clock::now();
        cv::Mat smallRef, smallDef;
        cv::resize(ref, smallRef, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(def, smallDef, cv::Size(), scale, scale, cv::INTER_AREA);

        int norm_type = cv::NORM_HAMMING;
        auto detector = make_detector(method, norm_type);
        std::vector<cv::KeyPoint> kp2;
        cv::Mat desc2;

        if (cached_kp.empty() || cached_desc.empty()) {
            detector->detectAndCompute(smallRef, cv::noArray(), cached_kp, cached_desc);
        }
        detector->detectAndCompute(smallDef, cv::noArray(), kp2, desc2);

        if (cached_kp.empty() || kp2.empty()) {
            out_detect_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start_akaze).count();
            return false;
        }

        cv::BFMatcher matcher(norm_type);
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(cached_desc, desc2, matches, 2);

        std::vector<cv::Point2f> p1, p2;
        std::vector<cv::DMatch> good_matches;
        for (auto &m : matches) {
            if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
                float full_x = cached_kp[m[0].queryIdx].pt.x / scale;
                float full_y = cached_kp[m[0].queryIdx].pt.y / scale;

                bool is_valid = true;
                if (!roi_mask.empty()) {
                    // 🚀 FIX: Apply offsets to check the correct global mask location!
                    int mask_x = (int)(full_x + offset_x);
                    int mask_y = (int)(full_y + offset_y);
                    if (mask_x >= 0 && mask_x < roi_mask.cols && mask_y >= 0 && mask_y < roi_mask.rows) {
                        if (roi_mask.at<uchar>(mask_y, mask_x) < 128) {
                            is_valid = false;
                        }
                    } else {
                        is_valid = false;
                    }
                }

                if (is_valid) {
                    p1.push_back(cached_kp[m[0].queryIdx].pt);
                    p2.push_back(kp2[m[0].trainIdx].pt);
                    good_matches.push_back(m[0]);
                }
            }
        }

        auto t_end_akaze = std::chrono::high_resolution_clock::now();
        out_detect_ms = std::chrono::duration<double, std::milli>(t_end_akaze - t_start_akaze).count();

        if (p1.size() < 10) return false;

        auto t_start_ransac = std::chrono::high_resolution_clock::now();
        std::vector<uchar> inlier_mask;
        cv::findHomography(p1, p2, cv::RANSAC, 3.0, inlier_mask);

        out_ref_pts.clear(); out_def_pts.clear();
        for (size_t i = 0; i < inlier_mask.size(); ++i) {
            if (inlier_mask[i]) {
                out_ref_pts.push_back(cv::Point2f(p1[i].x / scale, p1[i].y / scale));
                out_def_pts.push_back(cv::Point2f(p2[i].x / scale, p2[i].y / scale));
            }
        }

        if (out_ref_pts.size() >= 3) {
            std::vector<cv::Point2f> hull;
            cv::convexHull(out_ref_pts, hull);
            float hull_area = (float)cv::contourArea(hull);
            float total_area = (float)(ref.cols * ref.rows);
            out_bounding_box_area_ratio = hull_area / total_area;
        } else {
            out_bounding_box_area_ratio = 0.0f;
        }

        out_ransac_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start_ransac).count();
        return out_ref_pts.size() >= 10;
    }

} // namespace seeding
} // namespace Semper
