// Debug-only export (maps/CSV) for the full-field solver. Production leaves
// debug_dir empty, so nothing in here runs on a normal solve. Lifted verbatim
// out of full_field_solver.cpp so the solve path stays readable.

#include "full_field_internal.hpp"

#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

void draw_outlined_text(cv::Mat &img, const std::string &text, cv::Point pt, double scale) {
    cv::putText(img, text, pt, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, text, pt, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

void export_full_field_debug_suite(
        const std::string& local_debug_dir,
        int gridW,
        int gridH,
        const FullFieldParams& params,
        const cv::Mat& roiMask,
        int cache_width,
        int cache_height,
        Image* ref_img,
        const ResultGrid& resultGrid,
        const StrainField& strainField,
        const std::vector<AffineTriangle>& affTriangles) {
if (!local_debug_dir.empty()) {
    try {
        int max_order = 1;
        float max_corr = 0.0001f, max_exx = -1e9f, min_exx = 1e9f;

        cv::Mat propMap(gridH, gridW, CV_8UC1, cv::Scalar(0));
        cv::Mat threadMap(gridH, gridW, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::Mat meshAssignMap(gridH, gridW, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat corrMap(gridH, gridW, CV_8UC1, cv::Scalar(0));
        cv::Mat strainMap(gridH, gridW, CV_8UC1, cv::Scalar(0));
        cv::Mat simplexMap(gridH, gridW, CV_8UC3, cv::Scalar(30, 30, 30));

        static const cv::Vec3b THREAD_COLORS[12] = {
                {60, 20, 220}, {20, 200, 20}, {200, 60, 20}, {200, 200, 20}, {20, 200, 200}, {200, 20, 200},
                {100, 180, 255}, {255, 140, 30}, {50, 255, 180}, {180, 50, 255}, {255, 50, 130}, {130, 255, 50}};

        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                const auto &gp = resultGrid[y][x];
                if (gp.solved && gp.corr > 0.0f) {
                    if (gp.compute_order > max_order) max_order = gp.compute_order;
                    if (gp.corr > max_corr) max_corr = gp.corr;
                    int idx = y * gridW + x;
                    if (strainField.exx[idx] > max_exx) max_exx = strainField.exx[idx];
                    if (strainField.exx[idx] < min_exx) min_exx = strainField.exx[idx];
                }
            }
        }

        if (max_exx == min_exx) { max_exx += 0.001f; min_exx -= 0.001f; }
        max_corr = std::min(max_corr, tuning::kCorrAccept);

        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                const auto &gp = resultGrid[y][x];

                bool is_masked = (!roiMask.empty() &&
                                  roiMask.at<uchar>(params.rect_y + y * params.step, params.rect_x + x * params.step) < 128);

                if (!is_masked) {
                    if (gp.solved && gp.corr > 0.0f) {
                        if (!gp.used_simplex) {
                            simplexMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                        } else if (gp.icgn_iters < tuning::kIcgnMaxIter) {
                            simplexMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 255);
                        } else {
                            simplexMap.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 0);
                        }
                    } else {
                        if (!gp.used_simplex) {
                            simplexMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                        } else if (gp.icgn_iters < tuning::kIcgnMaxIter) {
                            simplexMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 165, 255);
                        } else {
                            simplexMap.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 255);
                        }
                    }
                }

                if (!is_masked) {
                    if (gp.mesh_assignment_type == kMeshInTriangle) {
                        meshAssignMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
                    } else if (gp.mesh_assignment_type == kMeshExtrapolated) {
                        meshAssignMap.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 255);
                    } else if (gp.mesh_assignment_type == kMeshAnchor) {
                        meshAssignMap.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);
                    } else {
                        meshAssignMap.at<cv::Vec3b>(y, x) = cv::Vec3b(100, 100, 100);
                    }
                }

                if (!gp.solved || gp.compute_order < 0) continue;

                propMap.at<uchar>(y, x) = (uchar) ((float) gp.compute_order / max_order * 255.0f);
                threadMap.at<cv::Vec3b>(y, x) = THREAD_COLORS[std::max(0, std::min(gp.thread_id, 11))];
                corrMap.at<uchar>(y, x) = (uchar) ((gp.corr / max_corr) * 255.0f);

                int idx = y * gridW + x;
                strainMap.at<uchar>(y, x) = (uchar) (
                        ((strainField.exx[idx] - min_exx) / (max_exx - min_exx)) * 255.0f);
            }
        }

        cv::Mat propColor, corrColor, strainColor;
        cv::applyColorMap(propMap, propColor, cv::COLORMAP_JET);
        cv::applyColorMap(corrMap, corrColor, cv::COLORMAP_JET);
        cv::applyColorMap(strainMap, strainColor, cv::COLORMAP_JET);

        for (int y = 0; y < gridH; ++y) {
            for (int x = 0; x < gridW; ++x) {
                if (!resultGrid[y][x].solved || resultGrid[y][x].compute_order < 0) {
                    propColor.at<cv::Vec3b>(y, x) = {0, 0, 0};
                    threadMap.at<cv::Vec3b>(y, x) = {30, 30, 30};
                    corrColor.at<cv::Vec3b>(y, x) = {0, 0, 0};
                    strainColor.at<cv::Vec3b>(y, x) = {0, 0, 0};
                }
            }
        }

        cv::Mat outProp, outThread, outMesh, outCorr, outStrain, outSimplex;
        cv::Size sz(gridW * params.step, gridH * params.step);
        cv::resize(propColor, outProp, sz, 0, 0, cv::INTER_NEAREST);
        cv::resize(threadMap, outThread, sz, 0, 0, cv::INTER_NEAREST);
        cv::resize(meshAssignMap, outMesh, sz, 0, 0, cv::INTER_NEAREST);
        cv::resize(corrColor, outCorr, sz, 0, 0, cv::INTER_NEAREST);
        cv::resize(strainColor, outStrain, sz, 0, 0, cv::INTER_NEAREST);
        cv::resize(simplexMap, outSimplex, sz, 0, 0, cv::INTER_NEAREST);

        draw_outlined_text(outProp, "Propagation Debug", cv::Point(10, 25), 0.6);
        draw_outlined_text(outThread, "8-Core Thread Execution Map", cv::Point(10, 25), 0.6);
        draw_outlined_text(outMesh, "Mesh Assign (Grn=In, Yel=Ex, Blu=Anchor, Gry=PathB)", cv::Point(10, 25), 0.6);
        draw_outlined_text(outCorr, "ZNSSD Quality (Blue=Perfect, Red=Marginal)", cv::Point(10, 25), 0.6);
        draw_outlined_text(outStrain, "Exx Strain (Raw Plot)", cv::Point(10, 25), 0.6);
        draw_outlined_text(outSimplex, "Grn=Perfect | Yel=Save(Crash) | Cya=Save(Time) | Org=Dead(Crash) | Pur=Dead(Time) | Red=Insta-Dead", cv::Point(10, 25), 0.4);

        cv::imwrite(local_debug_dir + "/propagation_debug.png", outProp);
        cv::imwrite(local_debug_dir + "/thread_debug.png", outThread);
        cv::imwrite(local_debug_dir + "/mesh_assignment_map.png", outMesh);
        cv::imwrite(local_debug_dir + "/correlation_heatmap.png", outCorr);
        cv::imwrite(local_debug_dir + "/strain_exx_debug.png", outStrain);
        cv::imwrite(local_debug_dir + "/simplex_health_map.png", outSimplex);

        cv::Mat outSimplexOverlap = outSimplex.clone();

        for (const auto &tri : affTriangles) {
            cv::Point pt1(tri.pts[0].x - params.rect_x, tri.pts[0].y - params.rect_y);
            cv::Point pt2(tri.pts[1].x - params.rect_x, tri.pts[1].y - params.rect_y);
            cv::Point pt3(tri.pts[2].x - params.rect_x, tri.pts[2].y - params.rect_y);

            cv::line(outSimplexOverlap, pt1, pt2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            cv::line(outSimplexOverlap, pt2, pt3, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            cv::line(outSimplexOverlap, pt3, pt1, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        draw_outlined_text(outSimplexOverlap, "Simplex Health + Delaunay Overlap", cv::Point(10, 45), 0.4);
        cv::imwrite(local_debug_dir + "/simplex_mesh_overlap.png", outSimplexOverlap);

        std::string csvPath = local_debug_dir + "/debug_grid_data.csv";
        std::ofstream csvFile(csvPath);
        if (csvFile.is_open()) {
            csvFile << "# PIPELINE=2_HYBRID_CORE\n";
            csvFile << "RealX,RealY,GridX,GridY,ThreadID,ComputeOrder,U,V,Correlation,MeshType,UsedSimplex,ItersICGN,SolverState,GuessU,GuessV,GuessUx,GuessUy,GuessVx,GuessVy\n";
            for (int y = 0; y < gridH; ++y) {
                for (int x = 0; x < gridW; ++x) {
                    const auto &gp = resultGrid[y][x];

                    bool is_masked = (!roiMask.empty() && roiMask.at<uchar>(params.rect_y + y * params.step, params.rect_x + x * params.step) < 128);
                    if (is_masked) continue;

                    int solver_state = 0;
                    if (gp.solved && gp.corr > 0.0f) {
                        if (!gp.used_simplex) solver_state = 0;
                        else if (gp.icgn_iters < tuning::kIcgnMaxIter) solver_state = 1;
                        else solver_state = 2;
                    } else {
                        if (!gp.used_simplex) solver_state = 5;
                        else if (gp.icgn_iters < tuning::kIcgnMaxIter) solver_state = 3;
                        else solver_state = 4;
                    }

                    // 🚀 UPDATE THE EXPORT LINE
                    csvFile << gp.x << "," << gp.y << "," << x << "," << y << ","
                            << gp.thread_id << "," << gp.compute_order << "," << gp.u
                            << "," << gp.v << "," << gp.corr << ","
                            << gp.mesh_assignment_type << ","
                            << (gp.used_simplex ? 1 : 0) << ","
                            << gp.icgn_iters << ","
                            << solver_state << ","
                            << gp.guess_u << "," << gp.guess_v << ","
                            << gp.guess_ux << "," << gp.guess_uy << ","
                            << gp.guess_vx << "," << gp.guess_vy << "\n";
                }
            }
            csvFile.close();
        }

        if (!affTriangles.empty()) {
            std::string meshCsvPath = local_debug_dir + "/delaunay_mesh_data.csv";
            std::ofstream meshCsvFile(meshCsvPath);
            if (meshCsvFile.is_open()) {
                meshCsvFile << "TriangleID,Pt1_X,Pt1_Y,Pt2_X,Pt2_Y,Pt3_X,Pt3_Y\n";
                for (size_t i = 0; i < affTriangles.size(); ++i) {
                    const auto &tri = affTriangles[i];
                    meshCsvFile << i << ","
                                << tri.pts[0].x << "," << tri.pts[0].y << ","
                                << tri.pts[1].x << "," << tri.pts[1].y << ","
                                << tri.pts[2].x << "," << tri.pts[2].y << "\n";
                }
                meshCsvFile.close();
            }
        }

        try {
            cv::Mat refFloat(cache_height, cache_width, CV_32FC1, (void *)ref_img->intensities.data());
            cv::Mat ref8U, refColor;
            refFloat.convertTo(ref8U, CV_8UC1);
            cv::cvtColor(ref8U, refColor, cv::COLOR_GRAY2BGR);

            cv::Rect roiRect(params.rect_x, params.rect_y, params.rect_w, params.rect_h);
            roiRect = roiRect & cv::Rect(0, 0, cache_width, cache_height);

            cv::Mat imgAll = refColor(roiRect).clone();
            cv::Mat imgDead = imgAll.clone();

            int shiftX = roiRect.x;
            int shiftY = roiRect.y;

            cv::Scalar meshColor(214, 174, 107);
            for (const auto &tri : affTriangles) {
                cv::Point pt1(tri.pts[0].x - shiftX, tri.pts[0].y - shiftY);
                cv::Point pt2(tri.pts[1].x - shiftX, tri.pts[1].y - shiftY);
                cv::Point pt3(tri.pts[2].x - shiftX, tri.pts[2].y - shiftY);

                cv::line(imgAll, pt1, pt2, meshColor, 1, cv::LINE_AA);
                cv::line(imgAll, pt2, pt3, meshColor, 1, cv::LINE_AA);
                cv::line(imgAll, pt3, pt1, meshColor, 1, cv::LINE_AA);

                cv::line(imgDead, pt1, pt2, meshColor, 1, cv::LINE_AA);
                cv::line(imgDead, pt2, pt3, meshColor, 1, cv::LINE_AA);
                cv::line(imgDead, pt3, pt1, meshColor, 1, cv::LINE_AA);
            }

            for (int y = 0; y < gridH; ++y) {
                for (int x = 0; x < gridW; ++x) {
                    const auto &gp = resultGrid[y][x];
                    bool is_masked = (!roiMask.empty() && roiMask.at<uchar>(params.rect_y + y * params.step, params.rect_x + x * params.step) < 128);
                    if (is_masked) continue;

                    int solver_state = 0;
                    if (gp.solved && gp.corr > 0.0f) {
                        if (!gp.used_simplex) solver_state = 0;
                        else if (gp.icgn_iters < tuning::kIcgnMaxIter) solver_state = 1;
                        else solver_state = 2;
                    } else {
                        if (!gp.used_simplex) solver_state = 5;
                        else if (gp.icgn_iters < tuning::kIcgnMaxIter) solver_state = 3;
                        else solver_state = 4;
                    }

                    if (solver_state == 0) continue;

                    cv::Point pt(gp.x - shiftX, gp.y - shiftY);
                    int radius = 4;
                    int markerSize = 10;
                    int thickness = 2;

                    if (solver_state == 1) {
                        cv::circle(imgAll, pt, radius, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
                        cv::circle(imgAll, pt, radius, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
                    }
                    else if (solver_state == 2) {
                        cv::circle(imgAll, pt, radius, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
                        cv::circle(imgAll, pt, radius, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
                    }
                    else if (solver_state == 3) {
                        cv::Scalar col(0, 165, 255);
                        cv::drawMarker(imgAll, pt, col, cv::MARKER_TILTED_CROSS, markerSize, thickness, cv::LINE_AA);
                        cv::drawMarker(imgDead, pt, col, cv::MARKER_TILTED_CROSS, markerSize, thickness, cv::LINE_AA);
                    }
                    else if (solver_state == 4) {
                        cv::Scalar col(255, 0, 255);
                        cv::drawMarker(imgAll, pt, col, cv::MARKER_TILTED_CROSS, markerSize, thickness, cv::LINE_AA);
                        cv::drawMarker(imgDead, pt, col, cv::MARKER_TILTED_CROSS, markerSize, thickness, cv::LINE_AA);
                    }
                    else if (solver_state == 5) {
                        cv::Scalar col(0, 0, 255);
                        cv::drawMarker(imgAll, pt, col, cv::MARKER_TILTED_CROSS, markerSize, thickness, cv::LINE_AA);
                        cv::drawMarker(imgDead, pt, col, cv::MARKER_TILTED_CROSS, markerSize, thickness, cv::LINE_AA);
                    }
                }
            }

            draw_outlined_text(imgAll, "ALL Simplex Interventions vs. Mesh", cv::Point(10, 25), 0.6);
            draw_outlined_text(imgDead, "ONLY Dead Points vs. Mesh", cv::Point(10, 25), 0.6);

            cv::imwrite(local_debug_dir + "/mesh_overlap_ALL_simplex.jpg", imgAll);
            cv::imwrite(local_debug_dir + "/mesh_overlap_ONLY_dead.jpg", imgDead);

        } catch (...) {
            LOGE("Failed to generate C++ Mesh Overlap plots");
        }
    } catch (...) { LOGE("Unknown Exception during Debug Export"); }
}
} // export_full_field_debug_suite

} // namespace internal
} // namespace pipeline
} // namespace Semper
