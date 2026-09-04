#ifndef SEMPER_SIMD_HPP
#define SEMPER_SIMD_HPP

// =====================================================================
// CANONICAL SIMD KERNELS
//
// These are the innermost hot loops of the ICGN solver. They are the
// vectorized implementation of the reductions defined canonically in
// include/semper/kernels/canonical_math.h, and they must produce BIT-IDENTICAL
// results to that reference — tests/unit/test_simd_kernels.cpp asserts
// exact equality, not a tolerance.
//
// Two properties make that possible, and both are load-bearing:
//
//  1. THE VECTOR IS PINNED TO 128 BITS (4 float lanes).
//     CV__SIMD_FORCE_WIDTH below stops OpenCV selecting a 256-bit
//     (AVX2) or 512-bit register. That matters because lane k of the
//     accumulator holds the partial sum of elements k, k+W, k+2W, ...
//     — so the vector width IS the summation order. With the width
//     left to the hardware, this engine produced different bits on
//     AVX2 (8 lanes) than on SSE/NEON (4 lanes), which is precisely
//     why the golden-corpus tolerances had been widened for "runner
//     FP noise". Four lanes is the canonical order on every backend,
//     including inside an OpenCL work-item.
//
//  2. NO FMA. The kernels below use v_mul + v_add, never v_fma.
//     A fused multiply-add rounds once; a separate multiply and add
//     round twice, and the two give different results. The canonical
//     reference is written as `acc += d * d` under -ffp-contract=off,
//     i.e. two roundings, so the vector path must not fuse either.
//     This also keeps the kernels correct on baseline x86-64, where
//     hardware FMA is not guaranteed.
//
// Reductions finish with an explicit ((a0+a1)+(a2+a3)) lane combine
// rather than v_reduce_sum, whose internal association is unspecified.
//
// The host build compiles these under -fno-fast-math -ffp-contract=off
// (see CMakeLists.txt). Explicit intrinsics are used rather than a
// plain loop precisely because strict FP forbids the compiler from
// auto-vectorizing a float reduction — hand-written intrinsics are how
// this stays both deterministic and fast.
//
// See docs/DETERMINISM.md.
// =====================================================================

// Must precede the intrin.hpp include: it selects which register width
// namespace OpenCV exposes as cv::v_float32.
#ifndef CV__SIMD_FORCE_WIDTH
#define CV__SIMD_FORCE_WIDTH 128
#endif

#include <cstddef>
#include <opencv2/core/hal/intrin.hpp>

#include <semper/kernels/canonical_math.h>

#if CV_SIMD
// If some other header pulled in intrin.hpp before this one, the force
// width above was ignored and the lane count — and therefore the
// summation order — is whatever the hardware offered. Fail loudly.
static_assert(cv::VTraits<cv::v_float32>::max_nlanes == 4,
              "SIMD width is not pinned to 4 lanes: include semper/simd.hpp "
              "before any other OpenCV intrin.hpp user, or the reduction "
              "order stops matching canonical_math.h.");
#endif

namespace Semper {
    namespace simd {

// Combine four lane accumulators in the canonical pairwise order.
#if CV_SIMD
        inline float combine4(const cv::v_float32 &acc) {
            float a[4];
            cv::v_store(a, acc);
            return SEMPER_COMBINE4(a[0], a[1], a[2], a[3]);
        }
#endif

// Sigma (vals[i] - mean)^2 over [0, n)
        inline float sum_sq_diff(const float *vals, size_t n, float mean) {
            const size_t nb = n & ~(size_t) 3;
            float sum;
            size_t i = 0;
#if CV_SIMD
            cv::v_float32 mean_v = cv::vx_setall_f32(mean);
            cv::v_float32 acc = cv::vx_setzero_f32();
            for (; i < nb; i += 4) {
                cv::v_float32 d = cv::v_sub(cv::vx_load(vals + i), mean_v);
                acc = cv::v_add(acc, cv::v_mul(d, d));
            }
            sum = combine4(acc);
#else
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (; i < nb; i += 4) {
                const float d0 = vals[i] - mean, d1 = vals[i + 1] - mean;
                const float d2 = vals[i + 2] - mean, d3 = vals[i + 3] - mean;
                a0 += d0 * d0; a1 += d1 * d1; a2 += d2 * d2; a3 += d3 * d3;
            }
            sum = SEMPER_COMBINE4(a0, a1, a2, a3);
#endif
            // Fresh index for the scalar tail: GCC's
            // -Waggressive-loop-optimizations falsely claims UB when the
            // same `i` continues after the vector bound.
            for (size_t j = nb; j < n; ++j) {
                const float d = vals[j] - mean;
                sum += d * d;
            }
            return sum;
        }

// Sigma (ref[i] - (vals[i] - mean) * inv_std)^2 — the ZNSSD residual
        inline float znssd_sum(const float *vals, const float *ref, size_t n,
                               float mean, float inv_std) {
            const size_t nb = n & ~(size_t) 3;
            float sum;
            size_t i = 0;
#if CV_SIMD
            cv::v_float32 mean_v = cv::vx_setall_f32(mean);
            cv::v_float32 inv_v = cv::vx_setall_f32(inv_std);
            cv::v_float32 acc = cv::vx_setzero_f32();
            for (; i < nb; i += 4) {
                cv::v_float32 norm_def =
                        cv::v_mul(cv::v_sub(cv::vx_load(vals + i), mean_v), inv_v);
                cv::v_float32 diff = cv::v_sub(cv::vx_load(ref + i), norm_def);
                acc = cv::v_add(acc, cv::v_mul(diff, diff));
            }
            sum = combine4(acc);
#else
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (; i < nb; i += 4) {
                const float e0 = ref[i] - (vals[i] - mean) * inv_std;
                const float e1 = ref[i + 1] - (vals[i + 1] - mean) * inv_std;
                const float e2 = ref[i + 2] - (vals[i + 2] - mean) * inv_std;
                const float e3 = ref[i + 3] - (vals[i + 3] - mean) * inv_std;
                a0 += e0 * e0; a1 += e1 * e1; a2 += e2 * e2; a3 += e3 * e3;
            }
            sum = SEMPER_COMBINE4(a0, a1, a2, a3);
#endif
            for (size_t j = nb; j < n; ++j) {
                const float e = ref[j] - (vals[j] - mean) * inv_std;
                sum += e * e;
            }
            return sum;
        }

// Fused ZNSSD residual + steepest-descent gradient accumulation.
//
//   error   = Sigma diff^2         diff = ref[i] - (vals[i]-mean)*inv_std
//   dp[k]  += Sigma sdi_plane_k[i] * diff    for k = 0..5
//
// sdi_planes is the Structure-of-Arrays mirror of
// steepest_descent_images: plane k occupies [k*n, (k+1)*n). SoA is what
// makes the six gradient accumulations vectorizable — the old AoS
// layout (stride-6 Eigen vectors) defeated both hand-written NEON and
// the auto-vectorizer.
        inline float znssd_error_and_gradient(const float *vals, const float *ref,
                                              const float *sdi_planes, size_t n,
                                              float mean, float inv_std,
                                              float dp_out[6]) {
            const size_t nb = n & ~(size_t) 3;
            float err;
            size_t i = 0;

            const float *p0 = sdi_planes;
            const float *p1 = sdi_planes + n;
            const float *p2 = sdi_planes + 2 * n;
            const float *p3 = sdi_planes + 3 * n;
            const float *p4 = sdi_planes + 4 * n;
            const float *p5 = sdi_planes + 5 * n;
#if CV_SIMD
            cv::v_float32 mean_v = cv::vx_setall_f32(mean);
            cv::v_float32 inv_v = cv::vx_setall_f32(inv_std);
            cv::v_float32 err_acc = cv::vx_setzero_f32();
            cv::v_float32 g0 = cv::vx_setzero_f32(), g1 = cv::vx_setzero_f32(),
                    g2 = cv::vx_setzero_f32(), g3 = cv::vx_setzero_f32(),
                    g4 = cv::vx_setzero_f32(), g5 = cv::vx_setzero_f32();
            for (; i < nb; i += 4) {
                cv::v_float32 norm_def =
                        cv::v_mul(cv::v_sub(cv::vx_load(vals + i), mean_v), inv_v);
                cv::v_float32 d = cv::v_sub(cv::vx_load(ref + i), norm_def);
                err_acc = cv::v_add(err_acc, cv::v_mul(d, d));
                g0 = cv::v_add(g0, cv::v_mul(cv::vx_load(p0 + i), d));
                g1 = cv::v_add(g1, cv::v_mul(cv::vx_load(p1 + i), d));
                g2 = cv::v_add(g2, cv::v_mul(cv::vx_load(p2 + i), d));
                g3 = cv::v_add(g3, cv::v_mul(cv::vx_load(p3 + i), d));
                g4 = cv::v_add(g4, cv::v_mul(cv::vx_load(p4 + i), d));
                g5 = cv::v_add(g5, cv::v_mul(cv::vx_load(p5 + i), d));
            }
            err = combine4(err_acc);
            dp_out[0] = combine4(g0);
            dp_out[1] = combine4(g1);
            dp_out[2] = combine4(g2);
            dp_out[3] = combine4(g3);
            dp_out[4] = combine4(g4);
            dp_out[5] = combine4(g5);
#else
            float e[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float g[6][4] = {};
            const float *pp[6] = {p0, p1, p2, p3, p4, p5};
            for (; i < nb; i += 4) {
                float d[4];
                for (int L = 0; L < 4; ++L) {
                    d[L] = ref[i + L] - (vals[i + L] - mean) * inv_std;
                    e[L] += d[L] * d[L];
                }
                for (int k = 0; k < 6; ++k)
                    for (int L = 0; L < 4; ++L)
                        g[k][L] += pp[k][i + L] * d[L];
            }
            err = SEMPER_COMBINE4(e[0], e[1], e[2], e[3]);
            for (int k = 0; k < 6; ++k)
                dp_out[k] = SEMPER_COMBINE4(g[k][0], g[k][1], g[k][2], g[k][3]);
#endif
            for (size_t j = nb; j < n; ++j) {
                const float d = ref[j] - (vals[j] - mean) * inv_std;
                err += d * d;
                dp_out[0] += p0[j] * d;
                dp_out[1] += p1[j] * d;
                dp_out[2] += p2[j] * d;
                dp_out[3] += p3[j] * d;
                dp_out[4] += p4[j] * d;
                dp_out[5] += p5[j] * d;
            }
            return err;
        }

    } // namespace simd
} // namespace Semper

#endif // SEMPER_SIMD_HPP
