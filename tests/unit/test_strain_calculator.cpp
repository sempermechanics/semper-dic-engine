// =====================================================================
// SUITE: Strain — native/postprocessing/StrainCalculator.cpp
//
// The VSG plane-fit strain algorithm is fed analytically exact
// displacement fields, so the Green-Lagrange strain it should output is
// known in closed form:
//
//   u = a·X + b·Y,  v = c·X + d·Y   (X, Y in physical pixels)
//   exx = a + ½(a² + c²)
//   eyy = d + ½(b² + d²)
//   exy = ½(b + c + a·b + c·d)
//
// A linear field is fit EXACTLY by VSG's linear least squares, so
// interior points must match to float precision, independent of window
// size. The suite also guards the two safety mechanisms: the -1000
// sentinel on unsupported windows and the 90% structural fill rule.
// =====================================================================
#include "framework/test_framework.h"
#include <semper/strain.hpp>

#include <cstdint>

using Semper::DisplacementField;
using Semper::StrainCalculator;
using Semper::StrainField;

namespace {

    // Build a grid with the linear displacement field u=aX+bY, v=cX+dY.
    DisplacementField linear_field(int gw, int gh, int step, float a, float b,
                                   float c, float d) {
        DisplacementField f;
        f.width = gw;
        f.height = gh;
        f.step = step;
        f.u.resize((size_t) gw * gh);
        f.v.resize((size_t) gw * gh);
        f.valid.assign((size_t) gw * gh, true);
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                float X = (float) (x * step), Y = (float) (y * step);
                f.u[(size_t) y * gw + x] = a * X + b * Y;
                f.v[(size_t) y * gw + x] = c * X + d * Y;
            }
        }
        return f;
    }

    struct GL { float exx, eyy, exy; };

    GL green_lagrange(float a, float b, float c, float d) {
        return {a + 0.5f * (a * a + c * c),
                d + 0.5f * (b * b + d * d),
                0.5f * (b + c + a * b + c * d)};
    }

} // namespace

TEST_CASE(Strain, VsgRecoversUniaxialStrainExactly) {
    const float a = 0.01f; // 1% exx
    auto f = linear_field(21, 21, 5, a, 0, 0, 0);
    auto s = StrainCalculator::compute_vsg_strain(f, 25);
    auto gt = green_lagrange(a, 0, 0, 0);

    int idx = 10 * 21 + 10; // grid center — full circular window support
    CHECK_NEAR(s.exx[idx], gt.exx, 1e-5);
    CHECK_NEAR(s.eyy[idx], gt.eyy, 1e-5);
    CHECK_NEAR(s.exy[idx], gt.exy, 1e-5);
}

TEST_CASE(Strain, VsgRecoversGeneralLinearField) {
    const float a = 0.012f, b = 0.004f, c = -0.006f, d = 0.009f;
    auto f = linear_field(25, 25, 4, a, b, c, d);
    auto s = StrainCalculator::compute_vsg_strain(f, 20);
    auto gt = green_lagrange(a, b, c, d);

    // Every interior point of a linear field must fit exactly
    for (int y = 8; y <= 16; y += 4) {
        for (int x = 8; x <= 16; x += 4) {
            int idx = y * 25 + x;
            CHECK_NEAR(s.exx[idx], gt.exx, 1e-5);
            CHECK_NEAR(s.eyy[idx], gt.eyy, 1e-5);
            CHECK_NEAR(s.exy[idx], gt.exy, 1e-5);
        }
    }
}

TEST_CASE(Strain, VsgRigidBodyTranslationGivesZeroStrain) {
    auto f = linear_field(15, 15, 5, 0, 0, 0, 0);
    for (auto &u : f.u) u += 4.2f;  // constant displacement everywhere
    for (auto &v : f.v) v -= 1.7f;
    auto s = StrainCalculator::compute_vsg_strain(f, 25);

    int idx = 7 * 15 + 7;
    CHECK_NEAR(s.exx[idx], 0.0f, 1e-6);
    CHECK_NEAR(s.eyy[idx], 0.0f, 1e-6);
    CHECK_NEAR(s.exy[idx], 0.0f, 1e-6);
}

TEST_CASE(Strain, VsgLeavesSentinelWhereWindowUnsupported) {
    // Grid corner: the circular window is only ~25% filled — the 90%
    // structural rule must refuse to output strain (sentinel remains).
    auto f = linear_field(21, 21, 5, 0.01f, 0, 0, 0);
    auto s = StrainCalculator::compute_vsg_strain(f, 25);
    CHECK_NEAR(s.exx[0], -1000.0f, 1e-3);

    // Invalid points must also keep the sentinel
    auto f2 = linear_field(21, 21, 5, 0.01f, 0, 0, 0);
    int center = 10 * 21 + 10;
    f2.valid[center] = false;
    auto s2 = StrainCalculator::compute_vsg_strain(f2, 25);
    CHECK_NEAR(s2.exx[center], -1000.0f, 1e-3);
}

// ---------------------------------------------------------------------
// Displacement median test (reject_displacement_outliers). Wrong matches
// can pass the correlation check in clusters, all off by about the same
// amount, and tilt every strain window they fall in. The test must drop
// those and leave smooth fields, noise and straight jumps alone.
// ---------------------------------------------------------------------

namespace {

    // Deterministic noise in [-amp, amp].
    void add_noise(DisplacementField &f, float amp) {
        uint32_t s = 12345u;
        auto next = [&s, amp]() {
            s = s * 1664525u + 1013904223u;
            return amp * (2.0f * (float) (s >> 8) / 16777216.0f - 1.0f);
        };
        for (auto &u : f.u) u += next();
        for (auto &v : f.v) v += next();
    }

    int count_invalid(const DisplacementField &f) {
        int n = 0;
        for (bool ok : f.valid) n += ok ? 0 : 1;
        return n;
    }

} // namespace

TEST_CASE(Strain, OutlierTestLeavesSmoothNoisyFieldAlone) {
    auto f = linear_field(40, 40, 5, 0.01f, 0.003f, -0.004f, 0.008f);
    add_noise(f, 0.1f);
    CHECK(StrainCalculator::reject_displacement_outliers(f) == 0);
    CHECK(count_invalid(f) == 0);
}

TEST_CASE(Strain, OutlierTestRejectsASingleSpike) {
    auto f = linear_field(30, 30, 5, 0.01f, 0, 0, 0.005f);
    add_noise(f, 0.02f);
    const int spike = 12 * 30 + 17;
    f.u[spike] += 3.0f;
    CHECK(StrainCalculator::reject_displacement_outliers(f) == 1);
    CHECK(!f.valid[spike]);
    CHECK(count_invalid(f) == 1);
}

TEST_CASE(Strain, OutlierTestRejectsAWholeCluster) {
    // A 4x4 block of matches 20 px off, as seen on real bending images. Its
    // inner points outnumber good ones in their own 5x5, so only repeated
    // passes reach them.
    auto f = linear_field(30, 30, 5, 0.01f, 0, 0, 0.005f);
    add_noise(f, 0.02f);
    for (int y = 10; y < 14; ++y)
        for (int x = 15; x < 19; ++x) f.v[y * 30 + x] -= 20.0f;
    CHECK(StrainCalculator::reject_displacement_outliers(f) == 16);
    for (int y = 10; y < 14; ++y)
        for (int x = 15; x < 19; ++x) CHECK(!f.valid[y * 30 + x]);
    CHECK(count_invalid(f) == 16);

    // And the strain beside it is the field's again.
    auto s = StrainCalculator::compute_vsg_strain(f, 21);
    auto gt = green_lagrange(0.01f, 0, 0, 0.005f);
    CHECK_NEAR(s.eyy[12 * 30 + 21], gt.eyy, 2e-3);
}

TEST_CASE(Strain, OutlierTestKeepsAStraightJump) {
    // A crack opening 5 px along a column: both sides are consistent within
    // their own majority, so nothing is an outlier.
    auto f = linear_field(30, 30, 5, 0, 0, 0, 0);
    add_noise(f, 0.02f);
    for (int y = 0; y < 30; ++y)
        for (int x = 15; x < 30; ++x) f.u[y * 30 + x] += 5.0f;
    CHECK(StrainCalculator::reject_displacement_outliers(f) == 0);
}

TEST_CASE(Strain, OutlierTestSkipsSparseAndInvalidPoints) {
    // Two neighbours are too few to judge; an already invalid point is not
    // counted again and never judges its neighbours.
    auto f = linear_field(3, 1, 5, 0, 0, 0, 0);
    f.u[2] = 50.0f;
    CHECK(StrainCalculator::reject_displacement_outliers(f) == 0);

    auto g = linear_field(20, 20, 5, 0, 0, 0, 0);
    const int bad = 10 * 20 + 10;
    g.valid[bad] = false;
    g.u[bad] = 99.0f;
    CHECK(StrainCalculator::reject_displacement_outliers(g) == 0);
    CHECK(count_invalid(g) == 1);
}
