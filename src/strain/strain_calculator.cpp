#include <semper/strain.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace Semper {

    StrainField StrainCalculator::compute_vsg_strain(const DisplacementField& disp, int window_pixels) {
        StrainField strain;
        int total_pts = disp.width * disp.height;

        // === 🚀 COMPILER-SAFE SENTINEL FIX ===
        // Android NDK fast-math strips out std::isnan checks.
        // Instead, we initialize with an impossible physical strain sentinel.
        // If the VSG window fails the 90% symmetry check, it leaves this sentinel.
        float sentinel = tuning::kStrainUninitSentinel;
        strain.exx.assign(total_pts, sentinel);
        strain.eyy.assign(total_pts, sentinel);
        strain.exy.assign(total_pts, sentinel);
        // ======================================

        float radius = window_pixels / 2.0f;
        float radius_sq = radius * radius;
        int grid_rad = std::ceil(radius / disp.step);

        // === 🚀 100% STRICT RULE: CALCULATE PERFECT CIRCLE ===
        // Before we process any pixels, calculate EXACTLY how many points
        // belong in a 100% mathematically full circular window.
        const double tiny = tuning::kVsgRadiusTiny;
        double d_radius_sq = static_cast<double>(radius_sq) + tiny;
        int expected_full_window_pts = 0;

        for (int dy = -grid_rad; dy <= grid_rad; ++dy) {
            for (int dx = -grid_rad; dx <= grid_rad; ++dx) {
                double d_phys_dx = static_cast<double>(dx * disp.step);
                double d_phys_dy = static_cast<double>(dy * disp.step);
                if ((d_phys_dx * d_phys_dx + d_phys_dy * d_phys_dy) <= d_radius_sq) {
                    expected_full_window_pts++;
                }
            }
        }
        // =====================================================

        for (int y = 0; y < disp.height; ++y) {
            for (int x = 0; x < disp.width; ++x) {
                int idx = y * disp.width + x;
                if (!disp.valid[idx]) continue;

                // 🚀 DICe PARITY: 64-bit precision for Least Squares Matrices
                Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
                Eigen::Vector3d AtU = Eigen::Vector3d::Zero();
                Eigen::Vector3d AtV = Eigen::Vector3d::Zero();
                int valid_pts = 0;

                // 🚀 DICe PARITY: Floating-point truncation buffer (tiny)
                const double tiny = tuning::kVsgRadiusTiny;
                double d_radius_sq = static_cast<double>(radius_sq) + tiny;

                for (int dy = -grid_rad; dy <= grid_rad; ++dy) {
                    for (int dx = -grid_rad; dx <= grid_rad; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx < 0 || nx >= disp.width || ny < 0 || ny >= disp.height) continue;
                        int nidx = ny * disp.width + nx;
                        if (!disp.valid[nidx]) continue;

                        double d_phys_dx = static_cast<double>(dx * disp.step);
                        double d_phys_dy = static_cast<double>(dy * disp.step);

                        if ((d_phys_dx * d_phys_dx + d_phys_dy * d_phys_dy) <= d_radius_sq) {
                            double d_u = static_cast<double>(disp.u[nidx]);
                            double d_v = static_cast<double>(disp.v[nidx]);

                            Eigen::Vector3d a(1.0, d_phys_dx, d_phys_dy);

                            AtA += a * a.transpose();
                            AtU += a * d_u;
                            AtV += a * d_v;
                            valid_pts++;
                        }
                    }
                }

                // === 🚀 90% STRUCTURAL SUPPORT RULE ===
                // DICe allows slightly truncated windows near complex boundaries,
                // relying on 'rcond' to catch instability. However, to prevent the
                // massive edge-explosions (which are exactly 50% half-circles),
                // we enforce a strict 90% structural fill ratio.
                // This guarantees the centroid is never severely shifted.
                double fill_ratio = static_cast<double>(valid_pts) / static_cast<double>(expected_full_window_pts);

                if (fill_ratio >= 0.90 && valid_pts >= 3) {

                    // 🚀 DICe PARITY: LAPACK GECON '1' (L1-Norm) Condition Number Estimation
                    double anorm = 0.0;
                    for (int col = 0; col < 3; ++col) {
                        double col_sum = std::abs(AtA(0, col)) + std::abs(AtA(1, col)) + std::abs(AtA(2, col));
                        if (col_sum > anorm) anorm = col_sum;
                    }

                    // Compute Inverse safely
                    Eigen::Matrix3d AtA_inv;
                    bool invertible;
                    double det;
                    AtA.computeInverseAndDetWithCheck(AtA_inv, det, invertible);

                    if (!invertible) continue;

                    // Calculate L1 Norm of the Inverse to find 'rcond' exactly like LAPACK
                    double inv_anorm = 0.0;
                    for (int col = 0; col < 3; ++col) {
                        double col_sum = std::abs(AtA_inv(0, col)) + std::abs(AtA_inv(1, col)) + std::abs(AtA_inv(2, col));
                        if (col_sum > inv_anorm) inv_anorm = col_sum;
                    }

                    double rcond = (anorm * inv_anorm > 0.0) ? (1.0 / (anorm * inv_anorm)) : 0.0;

                    // 🚀 DICe EXACT THRESHOLD: Abort if reciprocal condition number < 10^-12
                    if (rcond < 1e-12) continue;

                    // Solve for coefficients
                    Eigen::Vector3d Cu = AtA_inv * AtU;
                    Eigen::Vector3d Cv = AtA_inv * AtV;

                    double dudx = Cu(1);
                    double dudy = Cu(2);
                    double dvdx = Cv(1);
                    double dvdy = Cv(2);

                    // 🚀 DICe PARITY: Large-Deformation Green-Lagrange Strain Formula
                    strain.exx[idx] = static_cast<float>(0.5 * (2.0 * dudx + dudx * dudx + dvdx * dvdx));
                    strain.eyy[idx] = static_cast<float>(0.5 * (2.0 * dvdy + dudy * dudy + dvdy * dvdy));
                    strain.exy[idx] = static_cast<float>(0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy));
                }
            }
        }
        return strain;
    }

    namespace {

        // Median of vals (reordered). The mean of the middle two when the count
        // is even, so it matches numpy's median.
        float median_of(std::vector<float>& vals) {
            const size_t n = vals.size();
            const size_t mid = n / 2;
            std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
            float hi = vals[mid];
            if (n % 2 == 1) return hi;
            float lo = *std::max_element(vals.begin(), vals.begin() + mid);
            return 0.5f * (lo + hi);
        }

        // Normalized residual of centre against its neighbours: distance from
        // their median over the median of their own distances from it, plus eps.
        float normalized_residual(float centre, std::vector<float>& nb, std::vector<float>& scratch) {
            float med = median_of(nb);
            scratch.resize(nb.size());
            for (size_t i = 0; i < nb.size(); ++i) scratch[i] = std::abs(nb[i] - med);
            float spread = median_of(scratch);
            return std::abs(centre - med) / (spread + tuning::kOutlierEpsPx);
        }

    } // namespace

    int StrainCalculator::reject_displacement_outliers(DisplacementField& disp) {
        const int w = disp.width, h = disp.height;
        const int r = tuning::kOutlierRadius;
        // Flags from finished passes. A pass reads only these, so the order
        // points are visited in does not change the result.
        std::vector<bool> ok = disp.valid;
        std::vector<int> failed;
        std::vector<float> nu, nv, scratch;
        const size_t max_nb = static_cast<size_t>((2 * r + 1) * (2 * r + 1) - 1);
        nu.reserve(max_nb);
        nv.reserve(max_nb);
        scratch.reserve(max_nb);
        int rejected = 0;

        for (int pass = 0; pass < tuning::kOutlierMaxPasses; ++pass) {
            failed.clear();
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int idx = y * w + x;
                    if (!ok[idx]) continue;
                    nu.clear();
                    nv.clear();
                    for (int dy = -r; dy <= r; ++dy) {
                        const int ny = y + dy;
                        if (ny < 0 || ny >= h) continue;
                        for (int dx = -r; dx <= r; ++dx) {
                            const int nx = x + dx;
                            if (nx < 0 || nx >= w || (dx == 0 && dy == 0)) continue;
                            const int nidx = ny * w + nx;
                            if (!ok[nidx]) continue;
                            nu.push_back(disp.u[nidx]);
                            nv.push_back(disp.v[nidx]);
                        }
                    }
                    if (static_cast<int>(nu.size()) < tuning::kOutlierMinNeighbours) continue;
                    if (normalized_residual(disp.u[idx], nu, scratch) > tuning::kOutlierThreshold ||
                        normalized_residual(disp.v[idx], nv, scratch) > tuning::kOutlierThreshold) {
                        failed.push_back(idx);
                    }
                }
            }
            if (failed.empty()) break;
            for (int idx : failed) ok[idx] = false;
            rejected += static_cast<int>(failed.size());
        }

        for (size_t i = 0; i < ok.size(); ++i) {
            if (!ok[i]) disp.valid[i] = false;
        }
        return rejected;
    }
}
