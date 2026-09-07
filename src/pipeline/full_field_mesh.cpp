// Path A — Delaunay 6-DOF guess field construction.
// Extract-only split from full_field_path_a.cpp; algorithms unchanged.

#include "full_field_internal.hpp"

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

MeshGuessField build_mesh_guess_field(
        const ReferenceCache& cache,
        const FullFieldParams& params,
        const std::vector<cv::Point2f>& seed_ref_pts,
        const std::vector<cv::Point2f>& seed_def_pts,
        MeshQuality mesh_quality,
        float globalU,
        float globalV,
        int gridW,
        int gridH,
        const std::string& local_debug_dir,
        ResultGrid& resultGrid,
        std::vector<AffineTriangle>& affTriangles,
        PhaseTimings& timings) {

    MeshGuessField guess;
    guess.u.assign(gridW * gridH, globalU);
    guess.v.assign(gridW * gridH, globalV);
    guess.ux.assign(gridW * gridH, 0.0f);
    guess.uy.assign(gridW * gridH, 0.0f);
    guess.vx.assign(gridW * gridH, 0.0f);
    guess.vy.assign(gridW * gridH, 0.0f);
    guess.in_mesh.assign(gridW * gridH, false);

    auto t_mesh_start = std::chrono::high_resolution_clock::now();

    cv::Subdiv2D subdiv(cv::Rect(0, 0, cache.width, cache.height));
    for (size_t i = 0; i < seed_ref_pts.size(); i++) {
        if (seed_ref_pts[i].x > 0 && seed_ref_pts[i].x < cache.width &&
            seed_ref_pts[i].y > 0 && seed_ref_pts[i].y < cache.height) {
            subdiv.insert(seed_ref_pts[i]);
        }
    }

    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    auto getDefPt = [&](cv::Point2f pt) -> cv::Point2f {
        float min_dist = 1e9; cv::Point2f best_pt = pt;
        for (size_t i = 0; i < seed_ref_pts.size(); i++) {
            float d = (seed_ref_pts[i].x - pt.x) * (seed_ref_pts[i].x - pt.x) + (seed_ref_pts[i].y - pt.y) * (seed_ref_pts[i].y - pt.y);
            if (d < min_dist) { min_dist = d; best_pt = seed_def_pts[i]; }
        }
        return best_pt;
    };

    try {
        for (size_t i = 0; i < triangleList.size(); i++) {
            cv::Vec6f t = triangleList[i]; cv::Point2f pt[3];
            pt[0] = cv::Point2f(t[0], t[1]); pt[1] = cv::Point2f(t[2], t[3]); pt[2] = cv::Point2f(t[4], t[5]);
            if (pt[0].x < 0 || pt[0].x >= cache.width || pt[1].x < 0 || pt[1].x >= cache.width || pt[2].x < 0 || pt[2].x >= cache.width) continue;
            cv::Point2f dst[3];
            dst[0] = getDefPt(pt[0]);
            dst[1] = getDefPt(pt[1]);
            dst[2] = getDefPt(pt[2]);

            // OpenCV returns: [ x_def ] = [ M00 M01 M02 ] * [ x_ref ]
            //                 [ y_def ]   [ M10 M11 M12 ]   [ y_ref ]
            //                                               [   1   ]
            cv::Mat warp_mat = cv::getAffineTransform(pt, dst);

            AffineTriangle at;
            at.pts[0] = pt[0]; at.pts[1] = pt[1]; at.pts[2] = pt[2];

            // ICGN Engine Expects: U(dx, dy) = U0 + Ux*dx + Uy*dy
            // Where dx, dy is relative to the Grid Point (0,0)

            double M00 = warp_mat.at<double>(0, 0);
            double M01 = warp_mat.at<double>(0, 1);
            double M02 = warp_mat.at<double>(0, 2);
            double M10 = warp_mat.at<double>(1, 0);
            double M11 = warp_mat.at<double>(1, 1);
            double M12 = warp_mat.at<double>(1, 2);

            // Strain/Shear derivatives (Ux = du/dx)
            at.ux = M00 - 1.0;
            at.uy = M01;
            at.vx = M10;
            at.vy = M11 - 1.0;

            // The Translation (U0, V0) evaluated at Coordinate (0,0) of the image!
            // Because OpenCV's matrix is global, the intercept M02 is the translation at absolute pixel (0,0).
            at.u = M02;
            at.v = M12;
            float minX = std::min({pt[0].x, pt[1].x, pt[2].x}); float maxX = std::max({pt[0].x, pt[1].x, pt[2].x});
            float minY = std::min({pt[0].y, pt[1].y, pt[2].y}); float maxY = std::max({pt[0].y, pt[1].y, pt[2].y});
            at.boundingBox = cv::Rect2f(minX - 15.0f, minY - 15.0f, (maxX - minX) + 30.0f, (maxY - minY) + 30.0f);
            affTriangles.push_back(at);
        }
    } catch (...) {
        LOGE("Delaunay triangle affine fit threw; mesh may be incomplete");
    }

    if (!local_debug_dir.empty()) {
        try {
            cv::Mat refFloat(cache.height, cache.width, CV_32FC1, (void *)cache.ref_img->intensities.data());
            cv::Mat ref8U, refColor;
            refFloat.convertTo(ref8U, CV_8UC1);
            cv::cvtColor(ref8U, refColor, cv::COLOR_GRAY2BGR);
            cv::Rect roiRect(params.rect_x, params.rect_y, params.rect_w, params.rect_h);
            roiRect = roiRect & cv::Rect(0, 0, cache.width, cache.height);
            cv::Mat meshDebug = refColor(roiRect) * 0.35; // Reduces opacity/brightness by 65%

            for (const auto &tri : affTriangles) {
                cv::Point pt1(tri.pts[0].x - params.rect_x, tri.pts[0].y - params.rect_y);
                cv::Point pt2(tri.pts[1].x - params.rect_x, tri.pts[1].y - params.rect_y);
                cv::Point pt3(tri.pts[2].x - params.rect_x, tri.pts[2].y - params.rect_y);
                cv::line(meshDebug, pt1, pt2, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
                cv::line(meshDebug, pt2, pt3, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
                cv::line(meshDebug, pt3, pt1, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
            }
            draw_outlined_text(meshDebug, "Delaunay 6-DOF Mesh", cv::Point(10, 25), 0.6);
            cv::imwrite(local_debug_dir + "/delaunay_mesh_debug.jpg", meshDebug);
        } catch (...) {
            LOGE("Delaunay mesh debug export threw");
        }
    }
    timings.delaunay = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_mesh_start).count();

    auto t_assign_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<cv::Point2f>> triContours;
    for (const auto &tri : affTriangles) triContours.push_back({tri.pts[0], tri.pts[1], tri.pts[2]});

    for (int y = 0; y < gridH; ++y) {
        for (int x = 0; x < gridW; ++x) {
            if (resultGrid[y][x].solved) continue;
            cv::Point2f gp(params.rect_x + x * params.step, params.rect_y + y * params.step);
            int idx = y * gridW + x;
            for (size_t ti = 0; ti < affTriangles.size(); ++ti) {
                const auto &tri = affTriangles[ti];
                if (gp.x < tri.boundingBox.x || gp.x > tri.boundingBox.x + tri.boundingBox.width || gp.y < tri.boundingBox.y || gp.y > tri.boundingBox.y + tri.boundingBox.height) continue;
                if (cv::pointPolygonTest(triContours[ti], gp, false) >= 0) {
                    guess.u[idx] = (float)(tri.ux * gp.x + tri.uy * gp.y + tri.u); guess.v[idx] = (float)(tri.vx * gp.x + tri.vy * gp.y + tri.v);
                    guess.ux[idx] = (float)tri.ux; guess.uy[idx] = (float)tri.uy; guess.vx[idx] = (float)tri.vx; guess.vy[idx] = (float)tri.vy;
                    guess.in_mesh[idx] = true; resultGrid[y][x].mesh_assignment_type = kMeshInTriangle; break;
                }
            }
        }
    }
    timings.contour_assign = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_assign_start).count();

    auto t_extrap_start = std::chrono::high_resolution_clock::now();
    // 🚀 IMPLEMENTATION: Priority 4 (Only extrapolate if the mesh is fully distributed)
    if (mesh_quality == MeshQuality::FULL) {
        float EXTRAP_LIMIT = -3.0f * params.step;
        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                int idx = y * gridW + x;
                if (guess.in_mesh[idx] || resultGrid[y][x].solved) continue;
                cv::Point2f gp(params.rect_x + x * params.step, params.rect_y + y * params.step);
                float best_dist = EXTRAP_LIMIT - 1.0f; int best_ti = -1;
                for (size_t ti = 0; ti < affTriangles.size(); ++ti) {
                    const auto &tri = affTriangles[ti];
                    if (gp.x < tri.boundingBox.x + EXTRAP_LIMIT || gp.x > tri.boundingBox.x + tri.boundingBox.width - EXTRAP_LIMIT || gp.y < tri.boundingBox.y + EXTRAP_LIMIT || gp.y > tri.boundingBox.y + tri.boundingBox.height - EXTRAP_LIMIT) continue;
                    double dist = cv::pointPolygonTest(triContours[ti], gp, true);
                    if (dist >= EXTRAP_LIMIT && (float)dist > best_dist) { best_dist = (float)dist; best_ti = (int)ti; }
                }
                if (best_ti >= 0) {
                    const auto &tri = affTriangles[best_ti];
                    guess.u[idx] = (float)(tri.ux * gp.x + tri.uy * gp.y + tri.u); guess.v[idx] = (float)(tri.vx * gp.x + tri.vy * gp.y + tri.v);
                    guess.ux[idx] = (float)tri.ux; guess.uy[idx] = (float)tri.uy; guess.vx[idx] = (float)tri.vx; guess.vy[idx] = (float)tri.vy;
                    guess.in_mesh[idx] = true; resultGrid[y][x].mesh_assignment_type = kMeshExtrapolated;
                }
            }
        }
    } else {
        LOGD("ROUTING: Mesh is SPARSE. Disabling extrapolation to prevent bad guesses.");
    }
    timings.extrapolate = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_extrap_start).count();

    auto t_smooth_start = std::chrono::high_resolution_clock::now();
    auto smoothGrid = [&](std::vector<float> &grid, int radius) {
        std::vector<float> temp = grid;
        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                if (!guess.in_mesh[y * gridW + x]) continue;
                float sum = 0.0f; int count = 0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int ny = y + dy, nx = x + dx;
                        if (nx >= 0 && nx < gridW && ny >= 0 && ny < gridH && guess.in_mesh[ny * gridW + nx]) { sum += temp[ny * gridW + nx]; count++; }
                    }
                }
                if (count > 0) grid[y * gridW + x] = sum / count;
            }
        }
    };
    smoothGrid(guess.u, 2); smoothGrid(guess.v, 2); smoothGrid(guess.ux, 2); smoothGrid(guess.uy, 2); smoothGrid(guess.vx, 2); smoothGrid(guess.vy, 2);
    timings.smoothing = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_smooth_start).count();

    return guess;
}

// ==========================================
// 🚀 PATH A (DELAUNAY MESH EXECUTION)
// ==========================================
} // namespace internal
} // namespace pipeline
} // namespace Semper
