// =====================================================================
// SUITE: SimdKernels — native/core/SimdKernels.h
//
// The portable SIMD kernels are the innermost hot loops of the DIC
// engine. Every test validates the vectorized path against a plain
// double-precision scalar reference implementation, across sizes that
// exercise full vector lanes, partial tails, and degenerate inputs.
//
// If CV_SIMD is active (NEON/SSE), these tests prove the SIMD and
// scalar results agree; on a compiler without SIMD they still validate
// the scalar fallback.
// =====================================================================
#include "framework/test_framework.h"
#include <semper/simd.hpp>

#include <random>
#include <vector>

namespace {

    std::vector<float> random_vec(size_t n, unsigned seed, float lo = 0.0f,
                                  float hi = 255.0f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(lo, hi);
        std::vector<float> v(n);
        for (auto &x : v) x = d(rng);
        return v;
    }

    // Double-precision scalar references (the "ground truth" oracles)
    double ref_sum_sq_diff(const std::vector<float> &vals, float mean) {
        double s = 0.0;
        for (float v : vals) {
            double d = (double) v - (double) mean;
            s += d * d;
        }
        return s;
    }

    double ref_znssd(const std::vector<float> &vals,
                     const std::vector<float> &ref, float mean, float inv_std) {
        double s = 0.0;
        for (size_t i = 0; i < vals.size(); ++i) {
            double nd = ((double) vals[i] - mean) * inv_std;
            double diff = (double) ref[i] - nd;
            s += diff * diff;
        }
        return s;
    }

} // namespace

// The kernel must be exact for every n: full SIMD blocks (n % lanes == 0),
// ragged tails (n % lanes != 0), and sub-vector sizes (n < lanes).
TEST_CASE(SimdKernels, SumSqDiff_AllSizes) {
    for (size_t n : {(size_t) 1, (size_t) 3, (size_t) 4, (size_t) 5,
                     (size_t) 7, (size_t) 8, (size_t) 16, (size_t) 31,
                     (size_t) 961, (size_t) 1000}) {
        auto vals = random_vec(n, (unsigned) (100 + n));
        float mean = 127.3f;
        double expect = ref_sum_sq_diff(vals, mean);
        float got = Semper::simd::sum_sq_diff(vals.data(), n, mean);
        CHECK_REL(got, expect, 1e-4);
    }
}

TEST_CASE(SimdKernels, SumSqDiff_ZeroWhenAllEqualMean) {
    std::vector<float> vals(64, 42.5f);
    float got = Semper::simd::sum_sq_diff(vals.data(), vals.size(), 42.5f);
    CHECK_NEAR(got, 0.0f, 1e-3);
}

TEST_CASE(SimdKernels, Znssd_AllSizes) {
    for (size_t n : {(size_t) 1, (size_t) 4, (size_t) 5, (size_t) 31,
                     (size_t) 961, (size_t) 1023}) {
        auto vals = random_vec(n, (unsigned) (200 + n));
        auto ref = random_vec(n, (unsigned) (300 + n), -2.0f, 2.0f);
        float mean = 128.0f, inv_std = 1.0f / 40.0f;
        double expect = ref_znssd(vals, ref, mean, inv_std);
        float got = Semper::simd::znssd_sum(vals.data(), ref.data(), n,
                                                 mean, inv_std);
        CHECK_REL(got, expect, 1e-4);
    }
}

// Perfect correlation: ref IS the normalized version of vals → ZNSSD = 0.
TEST_CASE(SimdKernels, Znssd_PerfectMatchIsZero) {
    size_t n = 961;
    auto vals = random_vec(n, 7);
    double mean = 0.0;
    for (float v : vals) mean += v;
    mean /= (double) n;
    double var = 0.0;
    for (float v : vals) var += (v - mean) * (v - mean);
    float std_dev = (float) std::sqrt(var / n);
    float inv_std = 1.0f / std_dev;

    std::vector<float> ref(n);
    for (size_t i = 0; i < n; ++i)
        ref[i] = (vals[i] - (float) mean) * inv_std;

    float got = Semper::simd::znssd_sum(vals.data(), ref.data(), n,
                                             (float) mean, inv_std);
    CHECK_NEAR(got, 0.0f, 1e-2); // n=961 accumulated float roundoff
}

// The fused kernel must return BOTH the same error as znssd_sum AND
// gradient projections matching a scalar double-precision oracle.
TEST_CASE(SimdKernels, ErrorAndGradient_MatchesScalarOracle) {
    for (size_t n : {(size_t) 5, (size_t) 31, (size_t) 961}) {
        auto vals = random_vec(n, (unsigned) (400 + n));
        auto ref = random_vec(n, (unsigned) (500 + n), -2.0f, 2.0f);
        auto planes = random_vec(6 * n, (unsigned) (600 + n), -5.0f, 5.0f);
        float mean = 126.0f, inv_std = 1.0f / 35.0f;

        // Oracle
        double e_expect = 0.0;
        double dp_expect[6] = {0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < n; ++i) {
            double nd = ((double) vals[i] - mean) * inv_std;
            double diff = (double) ref[i] - nd;
            e_expect += diff * diff;
            for (int k = 0; k < 6; ++k)
                dp_expect[k] += (double) planes[k * n + i] * diff;
        }

        float dp_out[6];
        float e_got = Semper::simd::znssd_error_and_gradient(
                vals.data(), ref.data(), planes.data(), n, mean, inv_std,
                dp_out);

        CHECK_REL(e_got, e_expect, 1e-4);
        for (int k = 0; k < 6; ++k)
            CHECK_NEAR(dp_out[k], dp_expect[k],
                       std::max(1e-3, std::fabs(dp_expect[k]) * 1e-4));
    }
}

// Consistency contract: the fused kernel's error term must equal the
// standalone znssd_sum for identical inputs (they share callers that
// assume this).
TEST_CASE(SimdKernels, FusedErrorEqualsStandaloneZnssd) {
    size_t n = 500;
    auto vals = random_vec(n, 42);
    auto ref = random_vec(n, 43, -2.0f, 2.0f);
    auto planes = random_vec(6 * n, 44, -1.0f, 1.0f);
    float mean = 130.0f, inv_std = 0.02f;

    float dp[6];
    float fused = Semper::simd::znssd_error_and_gradient(
            vals.data(), ref.data(), planes.data(), n, mean, inv_std, dp);
    float standalone =
            Semper::simd::znssd_sum(vals.data(), ref.data(), n, mean, inv_std);
    CHECK_REL(fused, standalone, 1e-5);
}

// =====================================================================
// CANONICAL EXACTNESS
//
// The tests above prove the SIMD kernels are *accurate* against a
// double-precision oracle. These prove something stricter and, for the
// GPU work, more important: that they are BIT-IDENTICAL to the canonical
// reference in <semper/kernels/canonical_math.h>.
//
// That reference is the definition the OpenCL kernels will reproduce, so
// if the CPU vector path drifts from it by even one ulp there is no
// single answer for the GPU to match. Hence exact ==, never CHECK_NEAR.
//
// These are what catch the two ways the contract silently breaks:
//   - a vector width other than 4 lanes (changes the summation order), and
//   - reintroducing v_fma (one rounding instead of two).
// =====================================================================

#include <semper/kernels/canonical_math.h>

namespace {
    // Deliberately includes 729 (a 27 px subset, tail of exactly 1) and
    // sizes covering every tail residue 0..3.
    const int kCanonSizes[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 121, 625, 729, 961};
} // namespace

TEST_CASE(SimdKernels, SumSqDiff_BitIdenticalToCanonical) {
    for (int n : kCanonSizes) {
        auto vals = random_vec((size_t) n, (unsigned) (21000 + n));
        const float mean = 127.3f;
        CHECK(Semper::simd::sum_sq_diff(vals.data(), (size_t) n, mean)
              == semper_canon_sum_sq_diff(vals.data(), n, mean));
    }
}

TEST_CASE(SimdKernels, Znssd_BitIdenticalToCanonical) {
    for (int n : kCanonSizes) {
        auto vals = random_vec((size_t) n, (unsigned) (22000 + n));
        auto ref = random_vec((size_t) n, (unsigned) (23000 + n), -3.0f, 3.0f);
        const float mean = 118.7f, inv_std = 0.031f;
        CHECK(Semper::simd::znssd_sum(vals.data(), ref.data(), (size_t) n, mean, inv_std)
              == semper_canon_znssd_sum(vals.data(), ref.data(), n, mean, inv_std));
    }
}

TEST_CASE(SimdKernels, ErrorAndGradient_BitIdenticalToCanonical) {
    for (int n : kCanonSizes) {
        auto vals = random_vec((size_t) n, (unsigned) (24000 + n));
        auto ref = random_vec((size_t) n, (unsigned) (25000 + n), -3.0f, 3.0f);
        auto sdi = random_vec((size_t) n * 6, (unsigned) (26000 + n), -1.0f, 1.0f);
        const float mean = 118.7f, inv_std = 0.031f;

        float dp_simd[6], dp_canon[6];
        const float e_simd = Semper::simd::znssd_error_and_gradient(
                vals.data(), ref.data(), sdi.data(), (size_t) n, mean, inv_std, dp_simd);
        const float e_canon = semper_canon_znssd_error_and_gradient(
                vals.data(), ref.data(), sdi.data(), n, mean, inv_std, dp_canon);

        CHECK(e_simd == e_canon);
        for (int k = 0; k < 6; ++k) CHECK(dp_simd[k] == dp_canon[k]);
    }
}

// The lane count is the summation order. If a future build lets OpenCV
// pick a 256-bit register, every reduction above silently changes
// association and the GPU parity tests start failing for reasons that
// look like kernel bugs. Assert the pin directly so the failure names
// its actual cause.
TEST_CASE(SimdKernels, VectorWidthIsPinnedTo4Lanes) {
#if CV_SIMD
    CHECK(cv::VTraits<cv::v_float32>::vlanes() == 4);
#else
    // Scalar fallback still honours the 4-accumulator form; nothing to pin.
    CHECK(true);
#endif
}
