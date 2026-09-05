#include <semper/subset.hpp>
#include <semper/assert.hpp>
#include <semper/kernels/canonical_math.h>
#include "util/log.hpp"
#include <cmath>

namespace Semper {

    // =========================================================
    // EXISTING: Full Precompute (Fallback / Original logic)
    // =========================================================
    void SubsetPrecomputer::precompute_subset(SubsetData& data, const Image& ref_img, int_t cx, int_t cy, int_t dim) {
        SEMPER_ASSERT(dim > 0);
        int n = dim * dim;
        int half = dim / 2;

        if (data.dim != dim) {
            data.dim = dim;
            data.x_offsets.resize(n);
            data.y_offsets.resize(n);
            // 🚀 Resize the new float arrays
            data.x_offsets_f.resize(n);
            data.y_offsets_f.resize(n);
            data.ref_intensities.resize(n);
            data.norm_ref_intensities.resize(n);
            data.gx_vec.resize(n);
            data.gy_vec.resize(n);
            data.steepest_descent_images.resize(n);

            int idx = 0;
            for (int y = -half; y <= half; ++y) {
                for (int x = -half; x <= half; ++x) {
                    data.x_offsets[idx] = x;
                    data.y_offsets[idx] = y;
                    // 🚀 Cast and store the float versions once
                    data.x_offsets_f[idx] = static_cast<float>(x);
                    data.y_offsets_f[idx] = static_cast<float>(y);
                    idx++;
                }
            }
        }

        data.cx = cx;
        data.cy = cy;

        if (cx - half < 0 || cx + half >= ref_img.width ||
            cy - half < 0 || cy + half >= ref_img.height) {
            data.is_initialized = false;
            return;
        }

        // 🚀 Float conversion
        float sum = 0.0f;
        int valid_pixels = 0;
        int idx = 0;

        // ==========================================
        // 🚀 CONTIGUOUS MEMORY SCAN (Hardware Prefetching)
        // ==========================================
        int start_y = cy - half;
        int start_x = cx - half;

        for (int y = 0; y < dim; ++y) {
            const scalar_t* int_row = &ref_img.intensities[(start_y + y) * ref_img.width + start_x];
            const scalar_t* gx_row = &ref_img.grad_x[(start_y + y) * ref_img.width + start_x];
            const scalar_t* gy_row = &ref_img.grad_y[(start_y + y) * ref_img.width + start_x];

            for (int x = 0; x < dim; ++x) {
                data.ref_intensities[idx] = int_row[x];
                data.gx_vec[idx] = gx_row[x];
                data.gy_vec[idx] = gy_row[x];

                // 🚀 DICe PARITY: Intensity-only Ghost Wall detection
                if (int_row[x] >= -5.0f) {
                    sum += int_row[x];
                    valid_pixels++;
                }
                idx++;
            }
        }

        // Abort if subset is mostly hanging off into the void
        if (valid_pixels < n * 0.5f) {
            data.is_initialized = false;
            return;
        }

        data.mean_intensity = sum / static_cast<float>(valid_pixels);
        float sum_sq_diff = 0.0f;

        for (int i = 0; i < n; ++i) {
            // 🚀 DICe PARITY: Intensity-only check
            if (data.ref_intensities[i] < -5.0f) continue;
            float diff = data.ref_intensities[i] - data.mean_intensity;
            sum_sq_diff += diff * diff;
        }
        data.std_dev = std::sqrt(sum_sq_diff / static_cast<float>(valid_pixels));
        if (data.std_dev < 1e-5f) data.std_dev = 1.0f;

        for (int i = 0; i < n; ++i) {
            data.norm_ref_intensities[i] = (data.ref_intensities[i] - data.mean_intensity) / data.std_dev;
        }

        // Row-major plain float, not Eigen. This Hessian is what the GPU
        // pre-pass has to reproduce bit for bit, and Eigen packs a fixed-size
        // outer product differently at different -march levels -- the same
        // hazard optimization_engine.cpp:100 documents for the warp matrix.
        // semper_mat6_add_outer accumulates the upper triangle only and
        // mirrors afterwards, which is bit-identical to the full
        // sd*sd.transpose() it replaces: s[r]*s[c] and s[c]*s[r] are the same
        // product of the same two floats, added over the same scan.
        float H[36];
        for (int i = 0; i < 36; ++i) H[i] = 0.0f;
        idx = 0;

        // 🚀 SIMD SoA mirror must track n (see Types.h / SimdKernels.h)
        if (data.sdi_planes.size() != static_cast<size_t>(n) * 6)
            data.sdi_planes.resize(static_cast<size_t>(n) * 6);

        // Exact Hessian Computation using direct relative coordinates
        for (int y = -half; y <= half; ++y) {
            for (int x = -half; x <= half; ++x) {
                float gx = data.gx_vec[idx] / data.std_dev;
                float gy = data.gy_vec[idx] / data.std_dev;

                const float sd[6] = {gx, gy,
                                     gx * (float)x, gx * (float)y,
                                     gy * (float)x, gy * (float)y};

                // steepest_descent_images stays Eigen: it is 6x1, so storage
                // order is not a question, and it is part of the published
                // SubsetData layout.
                Eigen::Matrix<float, 6, 1> &sd_e = data.steepest_descent_images[idx];
                for (int k = 0; k < 6; ++k) sd_e(k) = sd[k];

                // 🚀 SoA mirror for the portable SIMD hot loop
                data.sdi_planes[idx]         = sd[0];
                data.sdi_planes[n + idx]     = sd[1];
                data.sdi_planes[2 * n + idx] = sd[2];
                data.sdi_planes[3 * n + idx] = sd[3];
                data.sdi_planes[4 * n + idx] = sd[4];
                data.sdi_planes[5 * n + idx] = sd[5];

                // 🚀 DICe PARITY: Intensity-only check
                if (data.ref_intensities[idx] >= -5.0f) {
                    semper_mat6_add_outer(H, sd);
                }
                idx++;
            }
        }

        semper_mat6_symmetrize(H);

        // Gauss-Jordan with partial pivoting, not Eigen's PartialPivLU. It
        // returns 0 only on an exactly-zero pivot, where det is 0 and the
        // threshold below would reject the point anyway -- but it is checked
        // FIRST, because H_inv_rm is left untouched on failure.
        float H_inv_rm[36], det = 0.0f;
        const int invertible = semper_inv6x6(H, H_inv_rm, &det);

        float det_2x2 = H[0*6+0]*H[1*6+1] - H[1*6+0]*H[0*6+1];
        float norm_2x2 = H[0*6+0]*H[0*6+0] + H[0*6+1]*H[0*6+1]
                       + H[1*6+0]*H[1*6+0] + H[1*6+1]*H[1*6+1];
        float cond_2x2 = (std::abs(det_2x2) > 1e-12f) ? (norm_2x2 / std::abs(det_2x2)) : 1.0e13f;

        if (!invertible || std::abs(det) < 1e-6f || cond_2x2 > 1.0e12f) {
            data.H_inv = Eigen::Matrix<float, 6, 6>::Zero();
            data.is_initialized = false;
            return;
        } else {
            // Element-wise, NOT a memcpy: SubsetData::H_inv is Eigen and so
            // column-major, which optimization_engine.cpp:126 transposes
            // straight back on read. The inverse of a symmetric matrix is
            // symmetric in exact arithmetic but NOT bit for bit after
            // Gauss-Jordan, so getting this transpose wrong would be silent.
            for (int r = 0; r < 6; ++r)
                for (int c = 0; c < 6; ++c) data.H_inv(r, c) = H_inv_rm[r * 6 + c];
        }
        // ── LM ADDITION ─────────────────────────────────────────────────────────
        // Same row-major -> column-major transpose as H_inv above.
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c) data.H(r, c) = H[r * 6 + c];
        // ────────────────────────────────────────────────────────────────────────
        data.is_initialized = true;
    }


    // =========================================================
    // 🚀 NEW: compute_hessian_only
    //    Builds H_inv + stats WITHOUT constructing a SubsetData.
    //    Designed to be called in bulk from an OMP pre-pass.
    //    Uses only contiguous pointer arithmetic — no heap allocation.
    // =========================================================
    CachedHessianData SubsetPrecomputer::compute_hessian_only(
            const Image& ref_img, int_t cx, int_t cy, int_t dim) {

        CachedHessianData result; // valid = false by default
        int n    = dim * dim;
        int half = dim / 2;

        if (cx - half < 0 || cx + half >= ref_img.width  ||
            cy - half < 0 || cy + half >= ref_img.height) {
            return result;
        }

        // ── Pass 1: Mean intensity ───────────────────────────────
        float sum = 0.0f;
        int valid_pixels = 0;
        for (int oy = -half; oy <= half; ++oy) {
            const float* int_row = &ref_img.intensities[(cy + oy) * ref_img.width + cx - half];
            // gx_row and gy_row unused in Pass 1, removed for clarity/speed
            for (int ox = 0; ox < dim; ++ox) {
                // === 🚀 PHASE 2 FIX: UNIFIED GHOST WALL THRESHOLD ===
                // Previously < 1e-6f, which killed natural black speckles.
                // The Ghost Wall uses -10.0f. The safe threshold is -5.0f.
                if (int_row[ox] < -5.0f) continue;
                // ====================================================
                sum += int_row[ox];
                valid_pixels++;
            }
        }

        // Abort if subset is mostly hanging off into the void
        if (valid_pixels < n * 0.5f) {
            return result; // Invalid
        }

        const float mean = sum / static_cast<float>(valid_pixels);

        // ── Pass 2: Std dev ──────────────────────────────────────
        float sum_sq = 0.0f;
        for (int oy = -half; oy <= half; ++oy) {
            const float* int_row = &ref_img.intensities[(cy + oy) * ref_img.width + cx - half];
            for (int ox = 0; ox < dim; ++ox) {
                // 🚀 DICe PARITY: Negative Signature check for the Mask
                if (int_row[ox] < -5.0f) continue;
                float d = int_row[ox] - mean;
                sum_sq += d * d;
            }
        }
        float std_dev = std::sqrt(sum_sq / static_cast<float>(valid_pixels));
        if (std_dev < 1e-5f) std_dev = 1.0f;
        const float inv_std = 1.0f / std_dev;

        result.mean_intensity = mean;
        result.std_dev        = std_dev;
        // 🚀 REMOVED: result.sssig

        // ── Pass 3: Hessian accumulation (upper triangle only → 21 muls instead of 36) ──
        // Row-major plain float for the same reason as precompute_subset.
        float H[36];
        for (int i = 0; i < 36; ++i) H[i] = 0.0f;

        for (int oy = -half; oy <= half; ++oy) {
            const float* int_row = &ref_img.intensities[(cy + oy) * ref_img.width + cx - half];
            const float* gx_row = &ref_img.grad_x[(cy + oy) * ref_img.width + cx - half];
            const float* gy_row = &ref_img.grad_y[(cy + oy) * ref_img.width + cx - half];
            const float fy = static_cast<float>(oy);

            for (int ox = -half; ox <= half; ++ox) {
                // === 🚀 PHASE 2 FIX: UNIFIED GHOST WALL THRESHOLD ===
                // Allow valid black speckles (intensity 0.0) into the Hessian gradients.
                if (int_row[ox + half] < -5.0f) continue;
                // ====================================================

                const float gx = gx_row[ox + half] * inv_std;
                const float gy = gy_row[ox + half] * inv_std;
                const float fx = static_cast<float>(ox);

                const float sd[6] = {gx, gy, gx * fx, gx * fy, gy * fx, gy * fy};

                // 21-op upper-triangle accumulation, in the identical order
                // the six explicit rows here used to be written out in.
                semper_mat6_add_outer(H, sd);
            }
        }

        // Symmetrize lower triangle
        semper_mat6_symmetrize(H);

        float H_inv_rm[36], det = 0.0f;
        const int invertible = semper_inv6x6(H, H_inv_rm, &det);

        float det_2x2 = H[0*6+0]*H[1*6+1] - H[1*6+0]*H[0*6+1];
        float norm_2x2 = H[0*6+0]*H[0*6+0] + H[0*6+1]*H[0*6+1]
                       + H[1*6+0]*H[1*6+0] + H[1*6+1]*H[1*6+1];
        float cond_2x2 = (std::abs(det_2x2) > 1e-12f) ? (norm_2x2 / std::abs(det_2x2)) : 1.0e13f;

        // ── LM ADDITION ─────────────────────────────────────────────────────────
        // preserve raw Hessian before it is inverted; element-wise because
        // CachedHessianData::H is Eigen and therefore column-major
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c) result.H(r, c) = H[r * 6 + c];
        // ────────────────────────────────────────────────────────────────────────

        if (!invertible || std::abs(det) < 1e-6f || cond_2x2 > 1.0e12f) {
            result.H_inv = Eigen::Matrix<float, 6, 6>::Zero();
            result.valid = false;
            return result;
        } else {
            for (int r = 0; r < 6; ++r)
                for (int c = 0; c < 6; ++c) result.H_inv(r, c) = H_inv_rm[r * 6 + c];
        }

        result.valid = true;
        return result;
    }


    // =========================================================
    // 🚀 NEW: precompute_subset_fast
    //    Drops the H accumulation + inversion entirely by reading
    //    from the pre-built pool. Still rebuilds steepest_descent_images
    //    (required by solve_icgn) but skips mean/std recomputation.
    //    Net savings: ~65% of original precompute FP work per call.
    // =========================================================
    void SubsetPrecomputer::precompute_subset_fast(
            SubsetData& data, const Image& ref_img,
            int_t cx, int_t cy, int_t dim,
            const CachedHessianData& cached) {

        // Safety: degrade gracefully if the pool entry was invalid (boundary overshoot, etc.)
        if (!cached.valid) {
            precompute_subset(data, ref_img, cx, cy, dim);
            return;
        }

        int n    = dim * dim;
        int half = dim / 2;

        // ── Resize buffers once when dim changes (amortised O(1)) ───
        if (data.dim != dim) {
            data.dim = dim;
            data.x_offsets.resize(n);
            data.y_offsets.resize(n);
            // 🚀 Resize the new float arrays
            data.x_offsets_f.resize(n);
            data.y_offsets_f.resize(n);
            data.ref_intensities.resize(n);
            data.norm_ref_intensities.resize(n);
            data.gx_vec.resize(n);
            data.gy_vec.resize(n);
            data.steepest_descent_images.resize(n);

            int oi = 0;
            for (int y = -half; y <= half; ++y)
                for (int x = -half; x <= half; ++x) {
                    data.x_offsets[oi] = x;
                    data.y_offsets[oi] = y;
                    // 🚀 Cast and store the float versions once
                    data.x_offsets_f[oi] = static_cast<float>(x);
                    data.y_offsets_f[oi] = static_cast<float>(y);
                    oi++;
                }
        }

        data.cx = cx;
        data.cy = cy;

        if (cx - half < 0 || cx + half >= ref_img.width  ||
            cy - half < 0 || cy + half >= ref_img.height) {
            data.is_initialized = false;
            return;
        }

        // ── Fast contiguous load via memcpy (hardware prefetch-friendly) ──
        int idx = 0;
        for (int oy = 0; oy < dim; ++oy) {
            const int row_base = (cy - half + oy) * ref_img.width + (cx - half);
            memcpy(&data.ref_intensities[idx], &ref_img.intensities[row_base], dim * sizeof(float));
            memcpy(&data.gx_vec[idx],          &ref_img.grad_x    [row_base], dim * sizeof(float));
            memcpy(&data.gy_vec[idx],          &ref_img.grad_y    [row_base], dim * sizeof(float));
            idx += dim;
        }

        // ── Use cached stats — zero recomputation ───────────────────
        data.mean_intensity = cached.mean_intensity;
        data.std_dev        = cached.std_dev;
        const float mean    = cached.mean_intensity;
        const float inv_std = 1.0f / cached.std_dev;

        // ── Normalize + build steepest_descent_images in one fused loop ──
        //    (SD images are needed for the ICGN Newton step — can't skip these)

        // 🚀 SIMD SoA mirror must track n (see Types.h / SimdKernels.h)
        if (data.sdi_planes.size() != static_cast<size_t>(n) * 6)
            data.sdi_planes.resize(static_cast<size_t>(n) * 6);

        idx = 0;
        for (int oy = -half; oy <= half; ++oy) {
            const float fy = static_cast<float>(oy);
            for (int ox = -half; ox <= half; ++ox) {
                data.norm_ref_intensities[idx] = (data.ref_intensities[idx] - mean) * inv_std;

                const float gx = data.gx_vec[idx] * inv_std;
                const float gy = data.gy_vec[idx] * inv_std;
                const float fx = static_cast<float>(ox);

                // Direct coefficient assignment avoids the << operator overhead
                auto& sd = data.steepest_descent_images[idx];
                sd(0) = gx;      sd(1) = gy;
                sd(2) = gx * fx; sd(3) = gx * fy;
                sd(4) = gy * fx; sd(5) = gy * fy;

                // 🚀 SoA mirror for the portable SIMD hot loop
                data.sdi_planes[idx]         = gx;
                data.sdi_planes[n + idx]     = gy;
                data.sdi_planes[2 * n + idx] = gx * fx;
                data.sdi_planes[3 * n + idx] = gx * fy;
                data.sdi_planes[4 * n + idx] = gy * fx;
                data.sdi_planes[5 * n + idx] = gy * fy;
                idx++;
            }
        }

        // 🚀 THE KEY SKIP: use pre-built inverse Hessian
        data.H_inv = cached.H_inv;
        // ── LM ADDITION ─────────────────────────────────────────────────────────
        data.H = cached.H;   // propagate raw Hessian for LM damping
        // ────────────────────────────────────────────────────────────────────────
        data.is_initialized = true;
    }

} // namespace Semper