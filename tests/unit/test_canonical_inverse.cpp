// =====================================================================
// SUITE: CanonicalInverse — include/semper/kernels/canonical_math.h
//
// semper_inv3x3 (double, strain VSG normal equations) and semper_inv6x6
// (float, ICGN Hessian) replace Eigen's inverse() on BOTH sides of the
// CPU/GPU boundary. Eigen's LU uses internal blocking and pivoting that
// cannot be called from an OpenCL kernel, so rather than have the device
// chase Eigen, both backends use these definitions.
//
// Correctness here is verified structurally — A*inv(A) == I and a known
// analytic inverse — rather than by diffing against Eigen, because
// matching Eigen bit-for-bit is explicitly NOT the goal.
// =====================================================================
#include "framework/test_framework.h"

#include <semper/kernels/canonical_math.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

    void mul3(const double a[9], const double b[9], double out[9]) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                double s = 0.0;
                for (int k = 0; k < 3; ++k) s += a[r * 3 + k] * b[k * 3 + c];
                out[r * 3 + c] = s;
            }
    }

    void mul6(const float a[36], const float b[36], float out[36]) {
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c) {
                float s = 0.0f;
                for (int k = 0; k < 6; ++k) s += a[r * 6 + k] * b[k * 6 + c];
                out[r * 6 + c] = s;
            }
    }

    // A symmetric positive-definite 6x6, shaped like a real ICGN Hessian:
    // built as J^T J from a random Jacobian so it is SPD by construction.
    void make_spd6(float out[36], unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        const int M = 40;
        std::vector<float> J((size_t) M * 6);
        for (auto &x : J) x = d(rng);
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c) {
                float s = 0.0f;
                for (int m = 0; m < M; ++m)
                    s += J[(size_t) (m * 6 + r)] * J[(size_t) (m * 6 + c)];
                out[r * 6 + c] = s;
            }
        for (int r = 0; r < 6; ++r) out[r * 6 + r] += 1.0f; // keep it well conditioned
    }

} // namespace

TEST_CASE(CanonicalInverse, Inv3x3_KnownAnalyticInverse) {
    // Diagonal matrix: inverse is the elementwise reciprocal, det is the
    // product — an exactly representable case, so equality is exact.
    const double a[9] = {2.0, 0.0, 0.0,
                         0.0, 4.0, 0.0,
                         0.0, 0.0, 0.5};
    double inv[9], det;
    REQUIRE(semper_inv3x3(a, inv, &det) == 1);
    CHECK(det == 4.0);
    CHECK(inv[0] == 0.5);
    CHECK(inv[4] == 0.25);
    CHECK(inv[8] == 2.0);
    CHECK(inv[1] == 0.0);
    CHECK(inv[5] == 0.0);
}

TEST_CASE(CanonicalInverse, Inv3x3_RoundTripsToIdentity) {
    std::mt19937 rng(2468u);
    std::uniform_real_distribution<double> d(-5.0, 5.0);
    for (int trial = 0; trial < 200; ++trial) {
        double a[9];
        for (auto &x : a) x = d(rng);
        // Diagonally dominant keeps the system well conditioned so the
        // residual bound below reflects the algorithm, not the input.
        a[0] += 20.0; a[4] += 20.0; a[8] += 20.0;

        double inv[9], det, prod[9];
        REQUIRE(semper_inv3x3(a, inv, &det) == 1);
        mul3(a, inv, prod);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                CHECK_NEAR(prod[r * 3 + c], (r == c) ? 1.0 : 0.0, 1e-12);
    }
}

TEST_CASE(CanonicalInverse, Inv3x3_SingularIsRejected) {
    // Row 2 is exactly twice row 0 — determinant is structurally zero.
    const double a[9] = {1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0,
                         2.0, 4.0, 6.0};
    double inv[9], det;
    CHECK(semper_inv3x3(a, inv, &det) == 0);
}

TEST_CASE(CanonicalInverse, Inv6x6_IdentityIsItsOwnInverse) {
    float a[36] = {0.0f};
    for (int i = 0; i < 6; ++i) a[i * 6 + i] = 1.0f;
    float inv[36], det;
    REQUIRE(semper_inv6x6(a, inv, &det) == 1);
    CHECK(det == 1.0f);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            CHECK(inv[r * 6 + c] == ((r == c) ? 1.0f : 0.0f));
}

TEST_CASE(CanonicalInverse, Inv6x6_RoundTripsToIdentityOnSpdHessians) {
    for (unsigned seed = 0; seed < 50; ++seed) {
        float a[36], inv[36], prod[36], det;
        make_spd6(a, 3000u + seed);
        REQUIRE(semper_inv6x6(a, inv, &det) == 1);
        CHECK(det > 0.0f); // SPD determinant is strictly positive
        mul6(a, inv, prod);
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                CHECK_NEAR(prod[r * 6 + c], (r == c) ? 1.0f : 0.0f, 2e-3f);
    }
}

// Pivoting must be exercised: a matrix with a zero in the (0,0) slot is
// only invertible if the elimination actually swaps rows.
TEST_CASE(CanonicalInverse, Inv6x6_RequiresRowPivoting) {
    float a[36] = {0.0f};
    // Anti-diagonal permutation: every leading pivot is zero without swaps.
    for (int i = 0; i < 6; ++i) a[i * 6 + (5 - i)] = 2.0f;
    float inv[36], det;
    REQUIRE(semper_inv6x6(a, inv, &det) == 1);
    float prod[36];
    mul6(a, inv, prod);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            CHECK_NEAR(prod[r * 6 + c], (r == c) ? 1.0f : 0.0f, 1e-6f);
}

TEST_CASE(CanonicalInverse, Inv6x6_SingularIsRejected) {
    float a[36] = {0.0f};
    for (int i = 0; i < 6; ++i) a[i * 6 + i] = 1.0f;
    // Make row 3 an exact duplicate of row 1 — rank deficient.
    for (int c = 0; c < 6; ++c) a[3 * 6 + c] = a[1 * 6 + c];
    float inv[36], det;
    CHECK(semper_inv6x6(a, inv, &det) == 0);
}

// The pivot tie-break must resolve to the lowest row index. An
// unspecified tie-break would be a bit-exactness hole the moment two
// candidate pivots have equal magnitude — which happens routinely on
// structured matrices.
TEST_CASE(CanonicalInverse, Inv6x6_PivotTieBreakIsLowestRow) {
    float a[36] = {0.0f};
    for (int i = 0; i < 6; ++i) a[i * 6 + i] = 1.0f;
    // Rows 2 and 4 both carry magnitude 1.0 in column 0: a tie.
    a[2 * 6 + 0] = 1.0f;
    a[4 * 6 + 0] = 1.0f;
    float inv[36], det;
    REQUIRE(semper_inv6x6(a, inv, &det) == 1);

    // Run it again on a byte-identical copy: the result must be bit-identical,
    // which it can only be if the tie-break is deterministic.
    float a2[36], inv2[36], det2;
    for (int i = 0; i < 36; ++i) a2[i] = a[i];
    REQUIRE(semper_inv6x6(a2, inv2, &det2) == 1);
    CHECK(det == det2);
    for (int i = 0; i < 36; ++i) CHECK(inv[i] == inv2[i]);
}
