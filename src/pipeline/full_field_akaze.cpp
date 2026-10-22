// Path A — AKAZE scale-pyramid routing / mesh seed detection.
// Extract-only split from full_field_path_a.cpp; algorithms unchanged.

#include "full_field_internal.hpp"

#include <semper/seeding.hpp>
#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

MeshSeedResult detect_mesh_seeds(
        ReferenceCache& cache,
        const cv::Mat& defMat,
        cv::Mat& roiMask,
        const FullFieldParams& params,
        const std::string& local_debug_dir) {

    MeshSeedResult out;
    float inlier_bb_area_ratio = 0.0f;
    bool has_good_akaze = false; // 🚀 RESTORED

    // 🚀 ADAPTIVE PADDING: Removed hardcoded padding from outside the loop

    if (params.rect_w > 32 && params.rect_h > 32) {
        try {
            // Prefer the cached gray reference from initializeReference; fall
            // back to decoding refBytes only if that cache is missing.
            cv::Mat refMat = cache.gray;
            if (!refMat.empty()) {

                // 🚀 PRIORITY 2: ADAPTIVE SCALE PYRAMID
                // Build the scale list based on the globally established baseline for this specimen
                std::vector<double> scales_to_try;
                if (cache.akaze_scale <= 0.25) scales_to_try = { 0.25,0.5,1.0};
                else if (cache.akaze_scale <= 0.5) scales_to_try = {0.5,1.0};
                else scales_to_try = {1.0};

                cv::Rect winning_padded_roi; // 🚀 Keep track of the offsets used for the winning scale

                for (double current_scale : scales_to_try) {

                    int adaptive_padding = (int)(40.0 / current_scale);
                    cv::Rect padded_roi(params.rect_x - adaptive_padding, params.rect_y - adaptive_padding, params.rect_w + 2 * adaptive_padding, params.rect_h + 2 * adaptive_padding);
                    padded_roi = padded_roi & cv::Rect(0, 0, cache.width, cache.height);

                    // 🚀 FIX: Must use padded_roi here, not winning_padded_roi!
                    cv::Mat refROI = refMat(padded_roi);
                    cv::Mat defROI = defMat(padded_roi);

                    if (current_scale != cache.akaze_scale) {
                        cache.akaze_kp.clear();
                        cache.akaze_desc.release();
                        cache.akaze_scale = current_scale;
                    }

                    double iter_akaze = 0, iter_ransac = 0;
                    cv::Mat H_scaled;
                    // 🚀 FIX: Pass padded_roi.x and padded_roi.y into the function!
                    bool success = seeding::extract_akaze_features(refROI, defROI, roiMask, current_scale, out.ref_pts, out.def_pts, inlier_bb_area_ratio, iter_akaze, iter_ransac, cache.akaze_kp, cache.akaze_desc, padded_roi.x, padded_roi.y, local_debug_dir, &H_scaled);
                    out.time_akaze_ms += iter_akaze;
                    out.time_ransac_ms += iter_ransac;

                    // 🚀 THE FIX: Separate Scale Escalation from Quality Routing
                    if (success && out.ref_pts.size() >= 25) {
                        // We found enough features! Zooming in further won't change the physical coverage area.
                        has_good_akaze = true;
                        winning_padded_roi = padded_roi; // 🚀 SAVE OFFSETS
                        // The fit ran on q = s (P - o); in image coordinates
                        // P_def = o + H(q) / s, i.e. T^-1 H T.
                        if (H_scaled.rows == 3 && H_scaled.cols == 3 && cv::checkRange(H_scaled)) {
                            const double sc = current_scale;
                            cv::Mat T = (cv::Mat_<double>(3, 3) << sc, 0, -sc * padded_roi.x, 0, sc, -sc * padded_roi.y, 0, 0, 1);
                            cv::Mat Ti = (cv::Mat_<double>(3, 3) << 1.0 / sc, 0, padded_roi.x, 0, 1.0 / sc, padded_roi.y, 0, 0, 1);
                            out.homography = Ti * H_scaled * T;
                        }

                        // Now, evaluate the structural integrity of the mesh
                        if (inlier_bb_area_ratio >= 0.30f) {
                            out.quality = MeshQuality::FULL;
                            LOGD("ROUTING: FULL Mesh at scale %.2fx (Pts: %d, Cov: %.2f)", current_scale, (int)out.ref_pts.size(), inlier_bb_area_ratio);
                        } else if (inlier_bb_area_ratio >= tuning::kAkazeSparseCoverage) {
                            out.quality = MeshQuality::SPARSE;
                            LOGD("ROUTING: SPARSE Mesh at scale %.2fx (Pts: %d, Cov: %.2f)", current_scale, (int)out.ref_pts.size(), inlier_bb_area_ratio);
                        } else {
                            out.quality = MeshQuality::NONE; // Too clustered, drop to Path C
                            LOGD("ROUTING: Features too clustered (Cov: %.2f). Forcing Path C.", inlier_bb_area_ratio);
                        }

                        break; // 🚀 CRITICAL: Stop escalating the scale! We have enough points.

                    } else {
                        LOGD("ROUTING: AKAZE Insufficient at scale %.2fx (Points: %d, Cov: %.2f). Escalating...", current_scale, (int)out.ref_pts.size(), inlier_bb_area_ratio);
                    }
                } // <--- END OF SCALE LOOP

                if (has_good_akaze) {
                    if (!local_debug_dir.empty()) {
                        cv::Mat akazeRefDraw, akazeDefDraw;
                        cv::Mat refROI = refMat(winning_padded_roi); // 🚀 Re-extract just for drawing
                        cv::Mat defROI = defMat(winning_padded_roi);
                        cv::cvtColor(refROI, akazeRefDraw, cv::COLOR_GRAY2BGR);
                        cv::cvtColor(defROI, akazeDefDraw, cv::COLOR_GRAY2BGR);

                        for(size_t i = 0; i < out.ref_pts.size(); ++i) {
                            cv::circle(akazeRefDraw, out.ref_pts[i], 3, cv::Scalar(0, 255, 0), -1);
                            cv::circle(akazeDefDraw, out.def_pts[i], 3, cv::Scalar(0, 255, 0), -1);
                        }

                        seeding::draw_outlined_text(akazeRefDraw, "AKAZE Reference Features: " + std::to_string(out.ref_pts.size()), cv::Point(10, 25), 0.6);
                        seeding::draw_outlined_text(akazeDefDraw, "AKAZE Deformed Features", cv::Point(10, 25), 0.6);

                        cv::imwrite(local_debug_dir + "/0_akaze_features_ref.jpg", akazeRefDraw);
                        cv::imwrite(local_debug_dir + "/0_akaze_features_def.jpg", akazeDefDraw);
                    }

                    for (size_t i = 0; i < out.ref_pts.size(); ++i) {
                        // 🚀 Apply the exact offset used during the successful extraction
                        out.ref_pts[i].x += winning_padded_roi.x;
                        out.ref_pts[i].y += winning_padded_roi.y;
                        out.def_pts[i].x += winning_padded_roi.x;
                        out.def_pts[i].y += winning_padded_roi.y;
                    }
                    std::vector<float> us, vs;
                    for (size_t i = 0; i < out.ref_pts.size(); i++) {
                        us.push_back(out.def_pts[i].x - out.ref_pts[i].x);
                        vs.push_back(out.def_pts[i].y - out.ref_pts[i].y);
                    }
                    std::sort(us.begin(), us.end()); std::sort(vs.begin(), vs.end());
                    out.globalU = us[us.size() / 2]; out.globalV = vs[vs.size() / 2];
                }
            }
        } catch (...) {
            // AKAZE/mesh seeding failed — fall through to Path C rather than abort.
            LOGE("AKAZE/scale pyramid threw; continuing with Path C fallback");
        }
    }

    return out;
}

// ==========================================
// 🚀 PATH A (DELAUNAY MESH SETUP)
// ==========================================
} // namespace internal
} // namespace pipeline
} // namespace Semper
