#ifndef SEMPER_SIMD_HPP
#define SEMPER_SIMD_HPP

// =====================================================================
// 🚀 PORTABLE SIMD KERNELS (OpenCV Universal Intrinsics)
//
// One codepath for every ABI. OpenCV's HAL maps these intrinsics to:
//   • NEON  on arm64-v8a / armeabi-v7a
//   • SSE   on x86 / x86_64 (emulators, ChromeOS, etc.)
// and falls back to plain scalar loops if CV_SIMD is unavailable.
//
// This replaces the previous hand-written #if defined(__aarch64__)
// arm_neon.h blocks in OptimizationEngine.cpp, which left every
// non-ARM64 device on unvectorized scalar code.
// =====================================================================

#include <cstddef>
#include <opencv2/core/hal/intrin.hpp>

// Multiply, then add -- never cv::v_fma. NEON fuses v_fma into one rounding,
// SSE2 (the x86-64 baseline) cannot, so with it the same input gave different
// sums on arm64 and x86-64. Two roundings everywhere keep the targets in step;
// the build disables compiler contraction for the scalar code likewise
// (-ffp-contract=off).

namespace Semper {
    namespace simd {

// Σ (vals[i] - mean)²  over the full range [0, n)
        inline float sum_sq_diff(const float *vals, size_t n, float mean) {
            float sum = 0.0f;
            size_t i = 0;
#if CV_SIMD
            const size_t step = (size_t) cv::VTraits<cv::v_float32>::vlanes();
            cv::v_float32 mean_v = cv::vx_setall_f32(mean);
            cv::v_float32 acc = cv::vx_setzero_f32();
            for (; i + step <= n; i += step) {
                cv::v_float32 d = cv::v_sub(cv::vx_load(vals + i), mean_v);
                acc = cv::v_add(cv::v_mul(d, d), acc);
            }
            sum = cv::v_reduce_sum(acc);
#endif
            // Fresh index for the scalar tail: GCC's -Waggressive-loop-optimizations
            // falsely claims UB when the same `i` continues after `i + step <= n`.
            for (size_t j = i; j < n; ++j) {
                float d = vals[j] - mean;
                sum += d * d;
            }
            return sum;
        }

// Σ (ref[i] - (vals[i] - mean) * inv_std)²  — the ZNSSD error reduction
        inline float znssd_sum(const float *vals, const float *ref, size_t n,
                               float mean, float inv_std) {
            float sum = 0.0f;
            size_t i = 0;
#if CV_SIMD
            const size_t step = (size_t) cv::VTraits<cv::v_float32>::vlanes();
            cv::v_float32 mean_v = cv::vx_setall_f32(mean);
            cv::v_float32 inv_v = cv::vx_setall_f32(inv_std);
            cv::v_float32 acc = cv::vx_setzero_f32();
            for (; i + step <= n; i += step) {
                cv::v_float32 norm_def =
                        cv::v_mul(cv::v_sub(cv::vx_load(vals + i), mean_v), inv_v);
                cv::v_float32 diff = cv::v_sub(cv::vx_load(ref + i), norm_def);
                acc = cv::v_add(cv::v_mul(diff, diff), acc);
            }
            sum = cv::v_reduce_sum(acc);
#endif
            for (size_t j = i; j < n; ++j) {
                float norm_def = (vals[j] - mean) * inv_std;
                float diff = ref[j] - norm_def;
                sum += diff * diff;
            }
            return sum;
        }

// Fused ZNSSD error + steepest-descent gradient accumulation.
//
//   error   = Σ diff²                    where diff = ref[i] - (vals[i]-mean)*inv_std
//   dp[k]  += Σ sdi_plane_k[i] * diff    for k = 0..5
//
// sdi_planes is the Structure-of-Arrays mirror of steepest_descent_images:
// plane k occupies [k*n, (k+1)*n). SoA is what makes the six gradient
// accumulations vectorizable — the old AoS layout (stride-6 Eigen vectors)
// defeated both hand-written NEON and the auto-vectorizer.
        inline float znssd_error_and_gradient(const float *vals, const float *ref,
                                              const float *sdi_planes, size_t n,
                                              float mean, float inv_std,
                                              float dp_out[6]) {
            float err = 0.0f;
            float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f, dp3 = 0.0f, dp4 = 0.0f, dp5 = 0.0f;
            size_t i = 0;
            // Plane base pointers shared by the SIMD body and the scalar
            // tail. Indexing the tail as p_k[i] (instead of
            // sdi_planes[k*n + i]) also keeps GCC's aggressive loop
            // optimizer from flagging the recomputed k*n+i offsets.
            const float *p0 = sdi_planes;
            const float *p1 = sdi_planes + n;
            const float *p2 = sdi_planes + 2 * n;
            const float *p3 = sdi_planes + 3 * n;
            const float *p4 = sdi_planes + 4 * n;
            const float *p5 = sdi_planes + 5 * n;
#if CV_SIMD
            const size_t step = (size_t) cv::VTraits<cv::v_float32>::vlanes();
            cv::v_float32 mean_v = cv::vx_setall_f32(mean);
            cv::v_float32 inv_v = cv::vx_setall_f32(inv_std);
            cv::v_float32 err_acc = cv::vx_setzero_f32();
            cv::v_float32 a0 = cv::vx_setzero_f32(), a1 = cv::vx_setzero_f32(),
                    a2 = cv::vx_setzero_f32(), a3 = cv::vx_setzero_f32(),
                    a4 = cv::vx_setzero_f32(), a5 = cv::vx_setzero_f32();
            for (; i + step <= n; i += step) {
                cv::v_float32 norm_def =
                        cv::v_mul(cv::v_sub(cv::vx_load(vals + i), mean_v), inv_v);
                cv::v_float32 diff = cv::v_sub(cv::vx_load(ref + i), norm_def);
                err_acc = cv::v_add(cv::v_mul(diff, diff), err_acc);
                a0 = cv::v_add(cv::v_mul(cv::vx_load(p0 + i), diff), a0);
                a1 = cv::v_add(cv::v_mul(cv::vx_load(p1 + i), diff), a1);
                a2 = cv::v_add(cv::v_mul(cv::vx_load(p2 + i), diff), a2);
                a3 = cv::v_add(cv::v_mul(cv::vx_load(p3 + i), diff), a3);
                a4 = cv::v_add(cv::v_mul(cv::vx_load(p4 + i), diff), a4);
                a5 = cv::v_add(cv::v_mul(cv::vx_load(p5 + i), diff), a5);
            }
            err = cv::v_reduce_sum(err_acc);
            dp0 = cv::v_reduce_sum(a0);
            dp1 = cv::v_reduce_sum(a1);
            dp2 = cv::v_reduce_sum(a2);
            dp3 = cv::v_reduce_sum(a3);
            dp4 = cv::v_reduce_sum(a4);
            dp5 = cv::v_reduce_sum(a5);
#endif
            for (size_t j = i; j < n; ++j) {
                float norm_def = (vals[j] - mean) * inv_std;
                float diff = ref[j] - norm_def;
                err += diff * diff;
                dp0 += p0[j] * diff;
                dp1 += p1[j] * diff;
                dp2 += p2[j] * diff;
                dp3 += p3[j] * diff;
                dp4 += p4[j] * diff;
                dp5 += p5[j] * diff;
            }
            dp_out[0] = dp0;
            dp_out[1] = dp1;
            dp_out[2] = dp2;
            dp_out[3] = dp3;
            dp_out[4] = dp4;
            dp_out[5] = dp5;
            return err;
        }

    } // namespace simd
} // namespace Semper

#endif // SEMPER_SIMD_HPP
