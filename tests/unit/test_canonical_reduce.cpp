// =====================================================================
// SUITE: CanonicalReduce — include/semper/kernels/canonical_math.h
//
// The canonical reductions define the bit-exactness contract between the
// CPU and the OpenCL backend: every long summation accumulates into four
// stride-4 accumulators combined as ((a0+a1)+(a2+a3)), with the tail
// added to the combined result in index order.
//
// These tests pin the ORDER, not merely the value. A reduction that
// produces the mathematically-correct sum by a different association is
// a failure here, because the GPU would then have no fixed target to
// reproduce. That is why the assertions below are exact float equality
// against an independently written accumulator, never CHECK_NEAR.
// =====================================================================
#include "framework/test_framework.h"

// Included directly, not wrapped in extern "C": the header is written in
// the C99/OpenCL common subset but is valid C++ as-is, and the functions
// are inline, so linkage never enters into it.
#include <semper/kernels/canonical_math.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

    std::vector<float> random_vec(int n, unsigned seed, float lo = 0.0f,
                                  float hi = 255.0f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(lo, hi);
        std::vector<float> v((size_t) n);
        for (auto &x : v) x = d(rng);
        return v;
    }

    // Independent restatement of the documented contract. Deliberately
    // written from the specification rather than copied from the header,
    // so a change to the header's association is actually caught.
    float spec_sum_sq_diff(const std::vector<float> &v, float mean) {
        const int n = (int) v.size();
        const int nb = n & ~3;
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < nb; i += 4)
            for (int k = 0; k < 4; ++k) {
                const float d = v[(size_t) (i + k)] - mean;
                a[k] += d * d;
            }
        float s = (a[0] + a[1]) + (a[2] + a[3]);
        for (int i = nb; i < n; ++i) {
            const float d = v[(size_t) i] - mean;
            s += d * d;
        }
        return s;
    }

    float spec_znssd(const std::vector<float> &v, const std::vector<float> &r,
                     float mean, float inv_std) {
        const int n = (int) v.size();
        const int nb = n & ~3;
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < nb; i += 4)
            for (int k = 0; k < 4; ++k) {
                const float e = r[(size_t) (i + k)]
                                - (v[(size_t) (i + k)] - mean) * inv_std;
                a[k] += e * e;
            }
        float s = (a[0] + a[1]) + (a[2] + a[3]);
        for (int i = nb; i < n; ++i) {
            const float e = r[(size_t) i] - (v[(size_t) i] - mean) * inv_std;
            s += e * e;
        }
        return s;
    }

    // Sizes chosen to exercise every tail residue 0..3, sub-block inputs,
    // and n = 729 — the real subset size for a 27 px subset, which has a
    // tail of exactly 1 (729 = 4*182 + 1).
    const int kSizes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 121, 625, 729, 961};

} // namespace

TEST_CASE(CanonicalReduce, SumSqDiff_MatchesSpecExactly) {
    for (int n : kSizes) {
        auto v = random_vec(n, (unsigned) (7000 + n));
        const float mean = 127.3f;
        const float got = semper_canon_sum_sq_diff(v.data(), n, mean);
        const float want = spec_sum_sq_diff(v, mean);
        CHECK(got == want);
    }
}

TEST_CASE(CanonicalReduce, ZnssdSum_MatchesSpecExactly) {
    for (int n : kSizes) {
        auto v = random_vec(n, (unsigned) (8000 + n));
        auto r = random_vec(n, (unsigned) (9000 + n), -3.0f, 3.0f);
        const float mean = 118.7f, inv_std = 0.031f;
        const float got = semper_canon_znssd_sum(v.data(), r.data(), n, mean, inv_std);
        const float want = spec_znssd(v, r, mean, inv_std);
        CHECK(got == want);
    }
}

// The fused kernel must return exactly what the two separate reductions
// would: same residual, and each of the six gradient planes independently
// following the same 4-accumulator association.
TEST_CASE(CanonicalReduce, FusedGradient_ResidualMatchesZnssdSum) {
    for (int n : kSizes) {
        auto v = random_vec(n, (unsigned) (11000 + n));
        auto r = random_vec(n, (unsigned) (12000 + n), -3.0f, 3.0f);
        auto sdi = random_vec(n * 6, (unsigned) (13000 + n), -1.0f, 1.0f);
        const float mean = 118.7f, inv_std = 0.031f;

        float dp[6];
        const float err = semper_canon_znssd_error_and_gradient(
                v.data(), r.data(), sdi.data(), n, mean, inv_std, dp);

        CHECK(err == semper_canon_znssd_sum(v.data(), r.data(), n, mean, inv_std));

        for (int k = 0; k < 6; ++k) {
            const int nb = n & ~3;
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < nb; i += 4)
                for (int L = 0; L < 4; ++L) {
                    const float d = r[(size_t) (i + L)]
                                    - (v[(size_t) (i + L)] - mean) * inv_std;
                    a[L] += sdi[(size_t) (k * n + i + L)] * d;
                }
            float want = (a[0] + a[1]) + (a[2] + a[3]);
            for (int i = nb; i < n; ++i) {
                const float d = r[(size_t) i] - (v[(size_t) i] - mean) * inv_std;
                want += sdi[(size_t) (k * n + i)] * d;
            }
            CHECK(dp[k] == want);
        }
    }
}

// Pinning the order costs nothing in accuracy. Blocked (pairwise)
// summation has better expected and worst-case error growth than a single
// running accumulator -- O(sqrt(log n)) against O(n) -- but that is a
// property of the DISTRIBUTION, not of any one input: on an individual
// sample naive summation can land closer to the oracle by luck of
// rounding. Asserting it per-sample would be a flaky test asserting the
// wrong thing, so the claim is made across many trials, where it is the
// error growth that actually shows up.
TEST_CASE(CanonicalReduce, BlockedBeatsNaiveOnAverageAgainstDoubleOracle) {
    const int n = 729; // 27 px subset
    const int trials = 300;
    double sum_err_canon = 0.0, sum_err_naive = 0.0;
    int canon_wins = 0;

    for (int t = 0; t < trials; ++t) {
        auto v = random_vec(n, (unsigned) (40000 + t));
        const float mean = 127.3f;

        double oracle = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = (double) v[(size_t) i] - (double) mean;
            oracle += d * d;
        }

        float naive = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float d = v[(size_t) i] - mean;
            naive += d * d;
        }

        const float canon = semper_canon_sum_sq_diff(v.data(), n, mean);
        const double ec = std::fabs((double) canon - oracle);
        const double en = std::fabs((double) naive - oracle);
        sum_err_canon += ec;
        sum_err_naive += en;
        if (ec <= en) ++canon_wins;
    }

    // Mean absolute error must be no worse than the naive accumulator.
    CHECK(sum_err_canon <= sum_err_naive);
    // And it should win outright on a clear majority of samples, not
    // merely tie on aggregate.
    CHECK(canon_wins * 2 > trials);
}

// Degenerate inputs must not produce NaN or trap: n == 0 is a legitimate
// call (a fully masked subset) and must return exactly zero.
TEST_CASE(CanonicalReduce, EmptyInputReturnsZero) {
    float dummy = 0.0f;
    CHECK(semper_canon_sum_sq_diff(&dummy, 0, 5.0f) == 0.0f);
    CHECK(semper_canon_znssd_sum(&dummy, &dummy, 0, 5.0f, 1.0f) == 0.0f);
    float dp[6];
    CHECK(semper_canon_znssd_error_and_gradient(&dummy, &dummy, &dummy, 0,
                                                5.0f, 1.0f, dp) == 0.0f);
    for (int k = 0; k < 6; ++k) CHECK(dp[k] == 0.0f);
}
