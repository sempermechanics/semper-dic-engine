#include <semper/strain.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"
#include <semper/kernels/canonical_math.h>
#include <cmath>

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
                //
                // Row-major plain double, not Eigen. This normal-equation fit
                // is mirrored by src/gpu/kernels/strain_vsg.cl, and the two
                // must agree bit-for-bit; Eigen packs its fixed-size products
                // differently at different -march levels, which is one of the
                // three divergence causes docs/DETERMINISM.md records. Same
                // reason optimization_engine.cpp:100 dropped it. Layout is
                // AtA[r * 3 + c].
                double AtA[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                double AtU[3] = {0.0, 0.0, 0.0};
                double AtV[3] = {0.0, 0.0, 0.0};
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

                            const double a[3] = {1.0, d_phys_dx, d_phys_dy};

                            // AtA += a * aT, AtU += a * u, AtV += a * v.
                            // Every entry accumulates independently, so this
                            // is the same per-entry addition sequence the
                            // Eigen expression produced -- what is pinned is
                            // the dy-outer / dx-inner visit order above.
                            for (int r = 0; r < 3; ++r) {
                                for (int c = 0; c < 3; ++c) {
                                    AtA[r * 3 + c] += a[r] * a[c];
                                }
                                AtU[r] += a[r] * d_u;
                                AtV[r] += a[r] * d_v;
                            }
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
                    double anorm = semper_mat3_l1_norm(AtA);

                    // Compute Inverse safely. semper_inv3x3 rejects on
                    // fabs(det) <= 2.220446049250313e-16, which is the same
                    // test -- and the same threshold -- Eigen's
                    // computeInverseAndDetWithCheck applied by default.
                    double AtA_inv[9];
                    double det;
                    if (!semper_inv3x3(AtA, AtA_inv, &det)) continue;

                    // Calculate L1 Norm of the Inverse to find 'rcond' exactly like LAPACK
                    double inv_anorm = semper_mat3_l1_norm(AtA_inv);

                    double rcond = (anorm * inv_anorm > 0.0) ? (1.0 / (anorm * inv_anorm)) : 0.0;

                    // 🚀 DICe EXACT THRESHOLD: Abort if reciprocal condition number < 10^-12
                    if (rcond < 1e-12) continue;

                    // Solve for coefficients
                    double Cu[3];
                    double Cv[3];
                    semper_mat3_vec3(AtA_inv, AtU, Cu);
                    semper_mat3_vec3(AtA_inv, AtV, Cv);

                    double dudx = Cu[1];
                    double dudy = Cu[2];
                    double dvdx = Cv[1];
                    double dvdy = Cv[2];

                    // 🚀 DICe PARITY: Large-Deformation Green-Lagrange Strain Formula
                    strain.exx[idx] = static_cast<float>(0.5 * (2.0 * dudx + dudx * dudx + dvdx * dvdx));
                    strain.eyy[idx] = static_cast<float>(0.5 * (2.0 * dvdy + dudy * dudy + dvdy * dvdy));
                    strain.exy[idx] = static_cast<float>(0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy));
                }
            }
        }
        return strain;
    }
}