#include <semper/solver.hpp>
#include <semper/simd.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"
#include <algorithm>
#include <chrono>

namespace Semper {

    AnalysisResult OptimizationEngine::calculate_deformation(
            const SubsetData &subset, const Image &def_img, float guess_u,
            float guess_v, float guess_ux, float guess_uy, float guess_vx, float guess_vy, InitializationMode init_mode) {

        if (icgn_buffer.size() != subset.x_offsets.size()) {
            icgn_buffer.resize(subset.x_offsets.size());
        }
        if (simplex_buffer.size() != subset.x_offsets.size()) {
            simplex_buffer.resize(subset.x_offsets.size());
        }

        AnalysisResult res;

        auto run_icgn = [&](float start_u, float start_v, float start_ux, float start_uy, float start_vx, float start_vy) {
            auto t1 = std::chrono::high_resolution_clock::now();
            AnalysisResult r = solve_icgn(subset, def_img, start_u, start_v, start_ux, start_uy, start_vx, start_vy);
            auto t2 = std::chrono::high_resolution_clock::now();
            time_icgn_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            return r;
        };

        auto run_simplex = [&](AnalysisResult start_g, bool trans_only) {
            auto t1 = std::chrono::high_resolution_clock::now();
            AnalysisResult r = solve_simplex(subset, def_img, start_g, trans_only);
            auto t2 = std::chrono::high_resolution_clock::now();
            time_simplex_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            count_simplex++;
            return r;
        };

        if (init_mode == INIT_AUTO_SEARCH) {
            scalar_t est_u = guess_u, est_v = guess_v;
            if (est_u == 0.0f && est_v == 0.0f)
                estimate_initial_guess(subset, def_img, est_u, est_v);
            AnalysisResult start_guess = {(float)est_u, (float)est_v, 0.0f, 0.0f, 0.0f, 0.0f, 0, 1.0f};
            AnalysisResult coarse_res = run_simplex(start_guess, true);
            res = run_icgn(coarse_res.u, coarse_res.v, 0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            // 🚀 Pass the PERFECT Delaunay 6-DOF guess directly into ICGN!
            res = run_icgn(guess_u, guess_v, guess_ux, guess_uy, guess_vx, guess_vy);

            // 🚀 THE KILL SWITCH: Only run Simplex if the mode IS NOT INIT_NO_SIMPLEX!
            if ((res.status != 0 || res.correlation_score > tuning::kCorrSimplexTrigger) && init_mode != INIT_NO_SIMPLEX) {
                AnalysisResult start_guess = {guess_u, guess_v, guess_ux, guess_uy, guess_vx, guess_vy, 0, 1.0f};
                AnalysisResult rescue_res = run_simplex(start_guess, false);
                res = run_icgn(rescue_res.u, rescue_res.v, rescue_res.ux, rescue_res.uy, rescue_res.vx, rescue_res.vy);
            }
        }
        return res;
    }

    void OptimizationEngine::estimate_initial_guess(const SubsetData &subset,
                                                    const Image &def_img,
                                                    scalar_t &best_u,
                                                    scalar_t &best_v) {
        float min_ssd = 1e20f;
        int search_range = 15;

        for (int v = -search_range; v <= search_range; v += 2) {
            for (int u = -search_range; u <= search_range; u += 2) {
                float sum_sq_diff = 0.0f;
                for (size_t i = 0; i < subset.x_offsets.size(); i += 9) {
                    float x = subset.cx + subset.x_offsets[i];
                    float y = subset.cy + subset.y_offsets[i];
                    float def_val = use_6x6_interpolator ?
                                    def_img.interpolate_keys_fourth(x + (float)u, y + (float)v) :
                                    def_img.interpolate_bicubic(x + (float)u, y + (float)v);
                    float diff = subset.ref_intensities[i] - def_val;
                    sum_sq_diff += diff * diff;
                }
                if (sum_sq_diff < min_ssd) {
                    min_ssd = sum_sq_diff;
                    best_u = static_cast<scalar_t>(u);
                    best_v = static_cast<scalar_t>(v);
                }
            }
        }
    }

    // --- PURE ICGN SOLVER ---
    AnalysisResult OptimizationEngine::solve_icgn(const SubsetData &subset,
                                                  const Image &def_img,
                                                  float init_u, float init_v,
                                                  float init_ux, float init_uy,
                                                  float init_vx, float init_vy) {
        size_t n = subset.x_offsets.size();

        // 🚀 Form the initial shape matrix properly using the 6-DOF inputs!
        Eigen::Matrix3f W = Eigen::Matrix3f::Identity();
        W(0, 0) = 1.0f + init_ux;
        W(0, 1) = init_uy;
        W(0, 2) = init_u;
        W(1, 0) = init_vx;
        W(1, 1) = 1.0f + init_vy;
        W(1, 2) = init_v;
        // ── LM ADDITION ─────────────────────────────────────────────────────────
        // Precompute the matrix to use for the Newton step ONCE, before the loop.
        //
        // When LM is disabled (lm_enabled == false OR lm_alpha == 0):
        //   H_solve = subset.H_inv   ← identical to the original code path.
        //
        // When LM is enabled:
        //   H_solve = (H + diag([α, α, 0, 0, 0, 0]))⁻¹
        //   This exactly replicates DICe's computeUpdateFast:
        //     H(0,0) += alpha;   // u translation DOF
        //     H(1,1) += alpha;   // v translation DOF
        //   The four strain-gradient diagonal entries are NOT damped.
        //
        // Because α is a fixed scalar (not adaptive), this inversion is valid
        // for all iterations of this call. Cost: one 6×6 inversion per
        // calculate_deformation call, not per iteration.
        // ────────────────────────────────────────────────────────────────────────
        Eigen::Matrix<float, 6, 6> H_solve;

        if (lm_enabled && lm_alpha > 0.0f) {
            // Copy the raw (undamped) Hessian stored at precompute time.
            Eigen::Matrix<float, 6, 6> H_damped = subset.H;

            // Apply DICe-style selective damping: translation DOFs only.
            H_damped(0, 0) += lm_alpha;
            H_damped(1, 1) += lm_alpha;

            const float det = H_damped.determinant();
            if (std::abs(det) < 1e-6f) {
                // Damped matrix is still singular — fall back to the
                // pre-inverted H_inv so the point at least attempts a step.
                H_solve = subset.H_inv;
            } else {
                H_solve = H_damped.inverse();
            }
        } else {
            // LM disabled: identical to original behaviour.
            H_solve = subset.H_inv;
        }
        // ── END LM ADDITION ─────────────────────────────────────────────────────
        std::vector<float> &def_vals = this->icgn_buffer;
        float final_score = 1.0f;
        int max_iter = tuning::kIcgnMaxIter;

        // 🚀 Use the centralized builder
        std::vector<bool> ref_valid = build_ref_valid(subset);
        // invalid_ref_pixels is a pure function of ref_valid — the same count at
        // every return site below regardless of iteration/convergence outcome.
        // Computed once here instead of re-summed at each of the 5 returns.
        int invalid_ref_count = 0;
        for (size_t k = 0; k < n; ++k) { if (!ref_valid[k]) invalid_ref_count++; }
        for (int iter = 0; iter < max_iter; ++iter) {
            float def_sum = 0.0f;
            int valid_pixels = 0;

            // 1. Unrolled Affine Warp & Interpolation
            //
            // Batches interpolation 4 pixels at a time via interpolate_keys_fourth_x4
            // / interpolate_bicubic_x4 when 4 consecutive pixels are available and all
            // pass their own px/py guard — those _x4 functions do the identical
            // per-lane arithmetic in the identical order interpolate_keys_fourth /
            // interpolate_bicubic would (see image.hpp), so this changes no single
            // pixel's interpolated value, only how many are computed per call. Any
            // pixel that can't be batched (guard failure in the group, or fewer than
            // 4 pixels left) falls back to the original one-at-a-time scalar call —
            // same function, same result, just not batched. def_sum/valid_pixels
            // accumulate in the same i=0..n-1 order either way.
            size_t i = 0;
            while (i < n) {
                float x = subset.x_offsets_f[i];
                float y = subset.y_offsets_f[i];
                float final_x = subset.cx + W(0, 0) * x + W(0, 1) * y + W(0, 2);
                float final_y = subset.cy + W(1, 0) * x + W(1, 1) * y + W(1, 2);

                // 🚀 DICe PARITY: Explicit 4-Pixel Guard (WITH ROUNDING)
                int px = ((int)(final_x + 0.5f) == (int)(final_x)) ? (int)(final_x) : (int)(final_x) + 1;
                int py = ((int)(final_y + 0.5f) == (int)(final_y)) ? (int)(final_y) : (int)(final_y) + 1;
                if (px < 4 || px >= def_img.width - 4 ||
                    py < 4 || py >= def_img.height - 4) {
                    def_vals[i] = -1.0f;
                    ++i;
                    continue;
                }

                bool batched = false;
                if (i + 4 <= n) {
                    float fx[4] = {final_x, 0.0f, 0.0f, 0.0f};
                    float fy[4] = {final_y, 0.0f, 0.0f, 0.0f};
                    bool group_ok = true;
                    for (int k = 1; k < 4; ++k) {
                        size_t idx = i + static_cast<size_t>(k);
                        float xk = subset.x_offsets_f[idx];
                        float yk = subset.y_offsets_f[idx];
                        float fxk = subset.cx + W(0, 0) * xk + W(0, 1) * yk + W(0, 2);
                        float fyk = subset.cy + W(1, 0) * xk + W(1, 1) * yk + W(1, 2);
                        int pxk = ((int)(fxk + 0.5f) == (int)(fxk)) ? (int)(fxk) : (int)(fxk) + 1;
                        int pyk = ((int)(fyk + 0.5f) == (int)(fyk)) ? (int)(fyk) : (int)(fyk) + 1;
                        if (pxk < 4 || pxk >= def_img.width - 4 || pyk < 4 || pyk >= def_img.height - 4) {
                            group_ok = false;
                            break;
                        }
                        fx[k] = fxk;
                        fy[k] = fyk;
                    }

                    if (group_ok) {
                        float vals[4];
                        if (use_6x6_interpolator) {
                            def_img.interpolate_keys_fourth_x4(fx, fy, vals);
                        } else {
                            def_img.interpolate_bicubic_x4(fx, fy, vals);
                        }
                        for (int k = 0; k < 4; ++k) {
                            size_t idx = i + static_cast<size_t>(k);
                            float val = vals[k];
                            // DICe LAYER 2: The Mask-Aware Drop — only accept pixels
                            // valid in both the reference mask and the deformed image.
                            if (val > 0.0f && ref_valid[idx]) {
                                def_vals[idx] = val;
                                def_sum += val;
                                valid_pixels++;
                            } else {
                                def_vals[idx] = -1.0f;
                            }
                        }
                        i += 4;
                        batched = true;
                    }
                }

                if (!batched) {
                    float val = use_6x6_interpolator ?
                                def_img.interpolate_keys_fourth(final_x, final_y) :
                                def_img.interpolate_bicubic(final_x, final_y);

                    // DICe LAYER 2: The Mask-Aware Drop
                    // Only accept pixels that are physically valid in BOTH the reference mask and the deformed image
                    if (val > 0.0f && ref_valid[i]) {
                        def_vals[i] = val;
                        def_sum += val;
                        valid_pixels++;
                    } else {
                        def_vals[i] = -1.0f;
                    }
                    ++i;
                }
            }

            // === 🚀 PHASE 3 FIX: GUARD DIVISION BY ZERO ===
            // If the entire subset is dragged off the image (e.g., at high strain),
            // valid_pixels becomes 0. We must abort BEFORE the division to prevent
            // a NaN injection that destroys the Newton-Raphson matrix.
            // DICe PARITY: 90% survival threshold.
            if (valid_pixels < static_cast<int>(n * 0.90f)) {
                AnalysisResult res = {W(0, 2), W(1, 2), W(0, 0) - 1.0f, W(0, 1), W(1, 0), W(1, 1) - 1.0f, 1, 2.0f, iter};
                res.invalid_ref_pixels = invalid_ref_count;
                return res;
            }

            float def_mean = def_sum / static_cast<float>(valid_pixels);
            // ==============================================
            float def_sum_sq = 0.0f;
            Eigen::Matrix<float, 6, 1> dp_sum = Eigen::Matrix<float, 6, 1>::Zero();
            float error_sum_sq = 0.0f;

            // 🚀 FAST-PATH: All pixels are safely inside the image boundaries.
            // Portable SIMD kernels — one codepath compiled to NEON on ARM
            // and SSE on x86, replacing the old #if __aarch64__ fork.
            if (valid_pixels == static_cast<int>(n)) {
                def_sum_sq = simd::sum_sq_diff(def_vals.data(), n, def_mean);

                float def_std = std::sqrt(def_sum_sq / valid_pixels);
                if (def_std < 1e-5f)
                    def_std = 1.0f;
                float inv_std = 1.0f / def_std;

                // Fused error + 6-DOF gradient accumulation over the SoA
                // sdi_planes — fully vectorized, unlike the old per-pixel
                // Eigen axpy which even the NEON path ran scalar.
                float dp_out[6];
                error_sum_sq = simd::znssd_error_and_gradient(
                        def_vals.data(), subset.norm_ref_intensities.data(),
                        subset.sdi_planes.data(), n, def_mean, inv_std, dp_out);
                for (int k = 0; k < 6; ++k)
                    dp_sum(k) += dp_out[k];
            } else {
                Eigen::Matrix<float, 6, 6> H_dynamic = Eigen::Matrix<float, 6, 6>::Zero();
                for (size_t i = 0; i < n; ++i) {
                    if (def_vals[i] >= 0.0f) {
                        float diff = def_vals[i] - def_mean;
                        def_sum_sq += diff * diff;
                    }
                }
                float def_std = std::sqrt(def_sum_sq / valid_pixels);
                if (def_std < 1e-5f)
                    def_std = 1.0f;
                float inv_std = 1.0f / def_std;

                for (size_t i = 0; i < n; ++i) {
                    if (def_vals[i] >= 0.0f) {
                        float norm_def = (def_vals[i] - def_mean) * inv_std;
                        float diff = subset.norm_ref_intensities[i] - norm_def;
                        error_sum_sq += diff * diff;
                        dp_sum += subset.steepest_descent_images[i] * diff;
                        H_dynamic.noalias() += subset.steepest_descent_images[i] * subset.steepest_descent_images[i].transpose();
                    }
                }

                // DICe LAYER 3: 2x2 Sub-block Hessian Condition Number Guard
                float det_2x2 = H_dynamic(0,0)*H_dynamic(1,1) - H_dynamic(1,0)*H_dynamic(0,1);
                float norm_2x2 = H_dynamic(0,0)*H_dynamic(0,0) + H_dynamic(0,1)*H_dynamic(0,1) + H_dynamic(1,0)*H_dynamic(1,0) + H_dynamic(1,1)*H_dynamic(1,1);
                float cond_2x2 = (std::abs(det_2x2) > 1e-12f) ? (norm_2x2 / std::abs(det_2x2)) : 1.0e13f;

                if (cond_2x2 > 1.0e12f) {
                    return {W(0, 2), W(1, 2), W(0, 0) - 1.0f, W(0, 1), W(1, 0), W(1, 1) - 1.0f, 1, 2.0f, iter};
                }

                if (lm_enabled && lm_alpha > 0.0f) {
                    H_dynamic(0, 0) += lm_alpha;
                    H_dynamic(1, 1) += lm_alpha;
                }
                H_solve = H_dynamic.inverse();
            }

            final_score = error_sum_sq / valid_pixels;

            Eigen::Matrix<float, 6, 1> delta_p = -H_solve     * dp_sum;
            Eigen::Matrix3f dW = Eigen::Matrix3f::Identity();
            dW(0, 0) += delta_p(2);
            dW(0, 1) += delta_p(3);
            dW(0, 2) += delta_p(0);
            dW(1, 0) += delta_p(4);
            dW(1, 1) += delta_p(5);
            dW(1, 2) += delta_p(1);

            W = W * dW.inverse();

            if (delta_p.norm() < 0.001f) {
                // 🚀 DICe PARITY: Post-Match Sigma (Texture) Check
                // Compute sum of squared gradients over surviving active pixels
                float sum_gx_sq = 0.0f, sum_gy_sq = 0.0f;
                int valid_survivors = 0;

                for (size_t k = 0; k < n; ++k) {
                    // Skip pixels that hit the Ghost Wall (mask) OR the Out-of-Bounds guard
                    if (!ref_valid[k] || def_vals[k] < 0.0f) continue;

                    // Access the un-normalized gradients stored during pre-computation
                    sum_gx_sq += subset.gx_vec[k] * subset.gx_vec[k];
                    sum_gy_sq += subset.gy_vec[k] * subset.gy_vec[k];
                    valid_survivors++;
                }

                // 🚀 DICe 100% Deactivation Check
                if (valid_survivors == 0) {
                    AnalysisResult res = {W(0, 2), W(1, 2), W(0, 0) - 1.0f, W(0, 1), W(1, 0), W(1, 1) - 1.0f, 1, 2.0f, iter + 1};
                    res.invalid_ref_pixels = invalid_ref_count;
                    return res;
                }

                // 🚀 DICe Minimum Gradient Check (The Barcode Killer)
                float sum_grad = std::min(sum_gx_sq, sum_gy_sq);

                if (sum_grad <= 0.0f) {
                    // Reject: Surviving pixels lack 2D physical texture (sigma returns -1.0 in DICe)
                    AnalysisResult res = {W(0, 2), W(1, 2), W(0, 0) - 1.0f, W(0, 1), W(1, 0), W(1, 1) - 1.0f, 1, 2.0f, iter + 1};
                    res.invalid_ref_pixels = invalid_ref_count;
                    return res;
                }

                AnalysisResult res = {W(0, 2), W(1, 2), W(0, 0) - 1.0f, W(0, 1), W(1, 0), W(1, 1) - 1.0f, 0, final_score, iter + 1};
                res.invalid_ref_pixels = invalid_ref_count;
                return res;
            }
        }

        AnalysisResult res = {W(0, 2), W(1, 2), W(0, 0) - 1.0f, W(0, 1), W(1, 0), W(1, 1) - 1.0f, 1, final_score, max_iter};
        res.invalid_ref_pixels = invalid_ref_count;
        return res;
    }

    float OptimizationEngine::evaluate_znssd(const SubsetData &subset,
                                             const Image &def_img, float u, float v,
                                             float ux, float uy, float vx, float vy,
                                             std::vector<float> &buffer,
                                             const std::vector<bool> *ref_valid_mask) {
        size_t n = subset.x_offsets.size();
        float def_mean = 0.0f;
        int valid_pixels = 0;

        for (size_t i = 0; i < n; ++i) {
            // 🚀 Ghost Wall exclusion: skip pixels that are void in reference
            if (ref_valid_mask && !(*ref_valid_mask)[i]) {
                buffer[i] = -1.0f;
                continue;
            }

            float dx = subset.x_offsets_f[i];
            float dy = subset.y_offsets_f[i];

            float final_x = subset.cx + u + (1.0f + ux) * dx + uy * dy;
            float final_y = subset.cy + v + vx * dx + (1.0f + vy) * dy;

            // 🚀 DICe PARITY: Explicit 4-Pixel Guard (WITH ROUNDING)
            int px = ((int)(final_x + 0.5f) == (int)(final_x)) ? (int)(final_x) : (int)(final_x) + 1;
            int py = ((int)(final_y + 0.5f) == (int)(final_y)) ? (int)(final_y) : (int)(final_y) + 1;
            if (px < 4 || px >= def_img.width - 4 ||
                py < 4 || py >= def_img.height - 4) {
                buffer[i] = -1.0f;
                continue;
            }

            float val = use_6x6_interpolator ?
                        def_img.interpolate_keys_fourth(final_x, final_y) :
                        def_img.interpolate_bicubic(final_x, final_y);
            if (val >= 0.0f) {
                buffer[i] = val;
                def_mean += val;
                valid_pixels++;
            } else {
                buffer[i] = -1.0f;
            }
        }

        // === 🚀 PHASE 3 FIX: GUARD ZNSSD DIVISION BY ZERO ===
        if (valid_pixels == 0) {
            return 2.0f; // Return a terrible score so Simplex immediately rejects this guess
        }

        def_mean /= static_cast<float>(valid_pixels);
        float def_sum_sq = 0.0f;
        float znssd = 0.0f;

        if (valid_pixels == static_cast<int>(n)) {
            // 🚀 Portable SIMD kernels (NEON/SSE via one codepath) — see SimdKernels.h
            def_sum_sq = simd::sum_sq_diff(buffer.data(), n, def_mean);

            float def_std = std::sqrt(def_sum_sq / valid_pixels);
            if (def_std < 1e-5f)
                def_std = 1.0f;
            float inv_std = 1.0f / def_std;

            znssd = simd::znssd_sum(buffer.data(), subset.norm_ref_intensities.data(),
                                    n, def_mean, inv_std);
        } else {
            for (size_t i = 0; i < n; ++i) {
                if (buffer[i] >= 0.0f) {
                    float diff = buffer[i] - def_mean;
                    def_sum_sq += diff * diff;
                }
            }
            float def_std = std::sqrt(def_sum_sq / valid_pixels);
            if (def_std < 1e-5f)
                def_std = 1.0f;

            for (size_t i = 0; i < n; ++i) {
                if (buffer[i] >= 0.0f) {
                    float norm_def = (buffer[i] - def_mean) / def_std;
                    float diff = subset.norm_ref_intensities[i] - norm_def;
                    znssd += diff * diff;
                }
            }
        }

        return znssd / valid_pixels;
    }

// --- SIMPLEX RESCUE METHOD ---
    AnalysisResult OptimizationEngine::solve_simplex(const SubsetData &subset,
                                                     const Image &def_img,
                                                     AnalysisResult start,
                                                     bool translation_only) {

        const int DIM = translation_only ? 2 : 6;
        int n_pts = DIM + 1;

        float p[7][6] = {};
        float y[7] = {};

        float scale[] = {2.0f, 2.0f, 0.01f, 0.01f, 0.01f, 0.01f};
        std::vector<bool> ref_valid = build_ref_valid(subset);

        auto eval_pt = [&](const float *pt) {
            if (translation_only)
                return evaluate_znssd(subset, def_img, pt[0], pt[1], 0.0f, 0.0f, 0.0f,
                                      0.0f, this->simplex_buffer, &ref_valid);
            return evaluate_znssd(subset, def_img, pt[0], pt[1], pt[2], pt[3], pt[4],
                                  pt[5], this->simplex_buffer, &ref_valid);
        };

        p[0][0] = start.u;
        p[0][1] = start.v;
        if (!translation_only) {
            p[0][2] = start.ux;
            p[0][3] = start.uy;
            p[0][4] = start.vx;
            p[0][5] = start.vy;
        }
        y[0] = eval_pt(p[0]);

        for (int i = 1; i < n_pts; ++i) {
            for (int j = 0; j < DIM; ++j)
                p[i][j] = p[0][j];
            p[i][i - 1] += scale[i - 1];
            y[i] = eval_pt(p[i]);
        }

        const float alpha = 1.0f, gamma = 2.0f, rho = 0.5f, sigma = 0.5f;
        for (int iter = 0; iter < 100; ++iter) { // 🚀 Increased to 100
            int idx[7] = {0, 1, 2, 3, 4, 5, 6};
            std::sort(idx, idx + n_pts, [&](int a, int b) { return y[a] < y[b]; });

            if (std::abs(y[idx[0]] - y[idx[n_pts - 1]]) < 1e-5f)
                break;

            float p_bar[6] = {0.0f};
            for (int i = 0; i < DIM; ++i)
                for (int j = 0; j < DIM; ++j)
                    p_bar[j] += p[idx[i]][j];
            for (int j = 0; j < DIM; ++j)
                p_bar[j] /= DIM;

            float p_r[6] = {0.0f};
            for (int j = 0; j < DIM; ++j)
                p_r[j] = p_bar[j] + alpha * (p_bar[j] - p[idx[n_pts - 1]][j]);
            float y_r = eval_pt(p_r);

            if (y[idx[0]] <= y_r && y_r < y[idx[n_pts - 2]]) {
                for (int j = 0; j < DIM; ++j)
                    p[idx[n_pts - 1]][j] = p_r[j];
                y[idx[n_pts - 1]] = y_r;
            } else if (y_r < y[idx[0]]) {
                float p_e[6] = {0.0f};
                for (int j = 0; j < DIM; ++j)
                    p_e[j] = p_bar[j] + gamma * (p_r[j] - p_bar[j]);
                float y_e = eval_pt(p_e);
                if (y_e < y_r) {
                    for (int j = 0; j < DIM; ++j)
                        p[idx[n_pts - 1]][j] = p_e[j];
                    y[idx[n_pts - 1]] = y_e;
                } else {
                    for (int j = 0; j < DIM; ++j)
                        p[idx[n_pts - 1]][j] = p_r[j];
                    y[idx[n_pts - 1]] = y_r;
                }
            } else {
                float p_c[6] = {0.0f};
                bool outside = (y_r < y[idx[n_pts - 1]]);
                float *base_p = outside ? p_r : p[idx[n_pts - 1]];
                for (int j = 0; j < DIM; ++j)
                    p_c[j] = p_bar[j] + rho * (base_p[j] - p_bar[j]);
                float y_c = eval_pt(p_c);
                if (y_c < std::min(y_r, y[idx[n_pts - 1]])) {
                    for (int j = 0; j < DIM; ++j)
                        p[idx[n_pts - 1]][j] = p_c[j];
                    y[idx[n_pts - 1]] = y_c;
                } else {
                    for (int i = 1; i < n_pts; ++i) {
                        for (int j = 0; j < DIM; ++j)
                            p[idx[i]][j] = p[idx[0]][j] + sigma * (p[idx[i]][j] - p[idx[0]][j]);
                        y[idx[i]] = eval_pt(p[idx[i]]);
                    }
                }
            }
        }

        int best = 0;
        for (int k = 1; k < n_pts; ++k)
            if (y[k] < y[best])
                best = k;
        int final_status = (y[best] > tuning::kSimplexFailScore) ? -2 : 0;

        if (translation_only) {
            return {p[best][0], p[best][1], 0.0f, 0.0f, 0.0f, 0.0f, final_status, y[best]};
        }
        return {p[best][0], p[best][1], p[best][2], p[best][3], p[best][4], p[best][5], final_status, y[best]};
    }

} // namespace Semper