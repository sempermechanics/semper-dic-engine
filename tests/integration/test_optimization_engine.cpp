// =====================================================================
// SUITE: Engine — native/core/OptimizationEngine.cpp
//
// THE SYNTHETIC DEFORMATION REGRESSION SUITE.
//
// Method: a continuous analytic speckle field g(x,y) is sampled to
// create the reference image; the deformed image is created by exact
// analytic inverse warping (see framework/synthetic.h). Ground truth is
// therefore known to machine precision — the recovered 6-DOF solution
// (u, v, ux, uy, vx, vy) is compared directly against it.
//
// Accuracy expectations (literature-standard for ICGN + bicubic on
// smooth synthetic speckle):
//   translation  |Δu|, |Δv|   ≤ 0.02 px
//   gradients    |Δux| … |Δvy| ≤ 2e-3
//
// These tolerances catch: interpolation regressions, SIMD kernel bugs,
// Hessian/steepest-descent sign errors, convergence-criteria breakage,
// and cross-ABI numerical drift (run the suite per-ABI to compare).
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"
#include <semper/solver.hpp>
#include <semper/subset.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Semper::Image;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;
using Semper::OptimizationEngine;
using Semper::AnalysisResult;
using Semper::INIT_NO_SEARCH;
using Semper::INIT_NO_SIMPLEX;
using Semper::INIT_AUTO_SEARCH;

namespace {

    constexpr int W = 160, H = 160;
    constexpr int CX = 80, CY = 80;
    constexpr int DIM = 31;

    constexpr float TOL_TRANS = 0.02f;  // px
    constexpr float TOL_GRAD = 2e-3f;

    // Run one full synthetic experiment and return the engine's answer.
    AnalysisResult run_engine(const dictest::AffineDeformation &truth,
                              float guess_u, float guess_v,
                              Semper::InitializationMode mode,
                              unsigned seed = 1234) {
        dictest::SpeckleField field(seed, W, H);
        Image ref = dictest::make_reference_image(field, W, H);
        Image def = dictest::make_deformed_image(field, W, H, truth);

        SubsetData subset;
        SubsetPrecomputer::precompute_subset(subset, ref, CX, CY, DIM);
        REQUIRE(subset.is_initialized);

        OptimizationEngine engine;
        return engine.calculate_deformation(subset, def, guess_u, guess_v,
                                            0.0f, 0.0f, 0.0f, 0.0f, mode);
    }

    dictest::AffineDeformation centered(float u, float v, float ux = 0.0f,
                                        float uy = 0.0f, float vx = 0.0f,
                                        float vy = 0.0f) {
        dictest::AffineDeformation d;
        d.u = u; d.v = v;
        d.ux = ux; d.uy = uy; d.vx = vx; d.vy = vy;
        d.cx = (float) CX; d.cy = (float) CY;
        return d;
    }

} // namespace

// --- Sanity anchor: zero deformation must be recovered as zero -------
TEST_CASE(Engine, ZeroDeformation_RecoversZero) {
    auto res = run_engine(centered(0.0f, 0.0f), 0.0f, 0.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.u, 0.0f, TOL_TRANS);
    CHECK_NEAR(res.v, 0.0f, TOL_TRANS);
    CHECK_NEAR(res.correlation_score, 0.0f, 1e-3);
}

// --- Integer-pixel translation ---------------------------------------
TEST_CASE(Engine, PureTranslation_IntegerPixel) {
    auto res = run_engine(centered(3.0f, -2.0f), 3.0f, -2.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.u, 3.0f, TOL_TRANS);
    CHECK_NEAR(res.v, -2.0f, TOL_TRANS);
}

// --- THE canonical DIC benchmark: sub-pixel translation --------------
TEST_CASE(Engine, PureTranslation_Subpixel) {
    const float u_true = 2.25f, v_true = -1.50f;
    // Start from the nearest integer guess: ICGN must pull in the
    // fractional remainder purely from image gradients.
    auto res = run_engine(centered(u_true, v_true), 2.0f, -2.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.u, u_true, TOL_TRANS);
    CHECK_NEAR(res.v, v_true, TOL_TRANS);
    CHECK_NEAR(res.ux, 0.0f, TOL_GRAD);
    CHECK_NEAR(res.uy, 0.0f, TOL_GRAD);
    CHECK_NEAR(res.vx, 0.0f, TOL_GRAD);
    CHECK_NEAR(res.vy, 0.0f, TOL_GRAD);
}

// Repeat across several sub-pixel phases — interpolation bias is
// phase-dependent, so a single phase can hide systematic error.
TEST_CASE(Engine, PureTranslation_SubpixelPhaseSweep) {
    const float phases[] = {0.1f, 0.25f, 0.4f, 0.6f, 0.75f, 0.9f};
    for (float p : phases) {
        auto res = run_engine(centered(1.0f + p, 0.0f), 1.0f, 0.0f,
                              INIT_NO_SIMPLEX, /*seed=*/500 + (unsigned) (p * 100));
        REQUIRE(res.status == 0);
        CHECK_NEAR(res.u, 1.0f + p, TOL_TRANS);
        CHECK_NEAR(res.v, 0.0f, TOL_TRANS);
    }
}

// --- Uniaxial strain ---------------------------------------------------
TEST_CASE(Engine, UniaxialStrain_Recovered) {
    const float exx = 0.01f; // 1% strain — typical DIC test magnitude
    auto res = run_engine(centered(0.0f, 0.0f, exx, 0.0f, 0.0f, 0.0f),
                          0.0f, 0.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.ux, exx, TOL_GRAD);
    CHECK_NEAR(res.uy, 0.0f, TOL_GRAD);
    CHECK_NEAR(res.vx, 0.0f, TOL_GRAD);
    CHECK_NEAR(res.vy, 0.0f, TOL_GRAD);
    CHECK_NEAR(res.u, 0.0f, TOL_TRANS);
    CHECK_NEAR(res.v, 0.0f, TOL_TRANS);
}

// --- Simple shear ------------------------------------------------------
TEST_CASE(Engine, SimpleShear_Recovered) {
    const float gamma = 0.008f;
    auto res = run_engine(centered(0.0f, 0.0f, 0.0f, gamma, 0.0f, 0.0f),
                          0.0f, 0.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.uy, gamma, TOL_GRAD);
    CHECK_NEAR(res.ux, 0.0f, TOL_GRAD);
}

// --- Fully general 6-DOF deformation ----------------------------------
TEST_CASE(Engine, General6DOF_Recovered) {
    auto truth = centered(1.3f, -0.7f, 0.006f, -0.004f, 0.003f, 0.008f);
    auto res = run_engine(truth, 1.0f, -1.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.u, truth.u, TOL_TRANS);
    CHECK_NEAR(res.v, truth.v, TOL_TRANS);
    CHECK_NEAR(res.ux, truth.ux, TOL_GRAD);
    CHECK_NEAR(res.uy, truth.uy, TOL_GRAD);
    CHECK_NEAR(res.vx, truth.vx, TOL_GRAD);
    CHECK_NEAR(res.vy, truth.vy, TOL_GRAD);
}

// --- DICe-style field consistency: many subsets, one rigid motion -----
// Pattern from DICe's custom_app regression test (dicengine/dice): solve a
// GRID of subsets under a single rigid translation and require every one to
// recover the same answer. The rest of this suite solves only the center
// subset, which cannot see position-dependent defects — coordinate-origin
// mistakes, asymmetric boundary handling, or row-stride bugs that grow with
// distance from the center.
TEST_CASE(Engine, MultiSubsetGrid_ConsistentRigidTranslation) {
    const float u_true = 0.40f, v_true = -0.25f; // DICe's canonical ~0.4 px

    // TOL_TRANS (0.02 px) is calibrated at the CENTER subset. Off-center
    // subsets see a different local speckle realization, and measured error
    // varies to ~0.024 px at individual POIs (seed-dependent, direction-free —
    // sampling variance, not a positional defect). 0.03 px still fails hard on
    // anything structural: a stride/origin bug shows up as whole pixels.
    const float TOL_GRID = 0.03f;

    dictest::SpeckleField field(1234, W, H);
    Image ref = dictest::make_reference_image(field, W, H);
    Image def = dictest::make_deformed_image(field, W, H,
                                             centered(u_true, v_true));

    const int poi[] = {40, 80, 120}; // 3x3 grid spanning the field
    for (int py : poi) {
        for (int px : poi) {
            SubsetData subset;
            SubsetPrecomputer::precompute_subset(subset, ref, px, py, DIM);
            REQUIRE(subset.is_initialized);

            OptimizationEngine engine;
            auto res = engine.calculate_deformation(subset, def, 0.0f, 0.0f,
                                                    0.0f, 0.0f, 0.0f, 0.0f,
                                                    INIT_NO_SIMPLEX);
            REQUIRE(res.status == 0);
            CHECK_NEAR(res.u, u_true, TOL_GRID);
            CHECK_NEAR(res.v, v_true, TOL_GRID);
        }
    }
}

// --- Auto-search path: engine must find a LARGE unknown translation ---
TEST_CASE(Engine, AutoSearch_FindsLargeTranslation) {
    // No guess given (0,0): the coarse SSD search (±15 px) + simplex +
    // ICGN pipeline must find it alone.
    auto res = run_engine(centered(6.4f, -8.3f), 0.0f, 0.0f, INIT_AUTO_SEARCH);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.u, 6.4f, 0.05f);  // full pipeline: slightly looser
    CHECK_NEAR(res.v, -8.3f, 0.05f);
}

// --- Photometric invariance: ZNSSD must ignore brightness/contrast ----
TEST_CASE(Engine, ZnssdInvariantToBrightnessAndContrast) {
    const float u_true = 1.75f;
    dictest::SpeckleField field(999, W, H);
    Image ref = dictest::make_reference_image(field, W, H);

    auto truth = centered(u_true, 0.0f);
    Image def = dictest::make_deformed_image(field, W, H, truth);

    // Apply gain + offset to the deformed image: I' = 0.7·I + 30
    for (auto &v : def.intensities) v = 0.7f * v + 30.0f;
    def.prepare_data(false);

    SubsetData subset;
    SubsetPrecomputer::precompute_subset(subset, ref, CX, CY, DIM);
    REQUIRE(subset.is_initialized);

    OptimizationEngine engine;
    auto res = engine.calculate_deformation(subset, def, 2.0f, 0.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.u, u_true, TOL_TRANS);
    CHECK_NEAR(res.v, 0.0f, TOL_TRANS);
}

// --- Failure semantics: subset dragged off the image must NOT "converge"
TEST_CASE(Engine, SubsetOffImage_ReportsFailureStatus) {
    // Deformed content is a 200 px translation — far outside the image,
    // so >10% of warped pixels land outside and the engine must abort
    // with status != 0 rather than fabricate a solution.
    auto res = run_engine(centered(0.0f, 0.0f), 200.0f, 0.0f, INIT_NO_SIMPLEX);
    CHECK(res.status != 0);
}

// --- Interpolator selector: both kernels must solve the same problem --
TEST_CASE(Engine, BothInterpolatorsConverge) {
    const float u_true = 0.35f;
    dictest::SpeckleField field(4321, W, H);
    Image ref = dictest::make_reference_image(field, W, H);
    Image def = dictest::make_deformed_image(field, W, H, centered(u_true, 0.0f));

    SubsetData subset;
    SubsetPrecomputer::precompute_subset(subset, ref, CX, CY, DIM);
    REQUIRE(subset.is_initialized);

    for (bool use6x6 : {false, true}) {
        OptimizationEngine engine;
        engine.use_6x6_interpolator = use6x6;
        auto res = engine.calculate_deformation(subset, def, 0.0f, 0.0f, 0.0f,
                                                0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);
        REQUIRE(res.status == 0);
        CHECK_NEAR(res.u, u_true, TOL_TRANS);
    }
}

// --- Determinism: the same solve twice must be bit-identical ----------
// Guards against threading races, uninitialized buffers, and any future
// change that makes results vary run-to-run on the same device.
TEST_CASE(Engine, RepeatSolve_BitIdentical) {
    auto truth = centered(1.3f, -0.7f, 0.006f, -0.004f, 0.003f, 0.008f);
    auto r1 = run_engine(truth, 1.0f, -1.0f, INIT_NO_SIMPLEX);
    auto r2 = run_engine(truth, 1.0f, -1.0f, INIT_NO_SIMPLEX);
    REQUIRE(r1.status == 0);
    REQUIRE(r2.status == 0);
    CHECK(r1.u == r2.u);
    CHECK(r1.v == r2.v);
    CHECK(r1.ux == r2.ux);
    CHECK(r1.uy == r2.uy);
    CHECK(r1.vx == r2.vx);
    CHECK(r1.vy == r2.vy);
    CHECK(r1.correlation_score == r2.correlation_score);
}

// --- Contract: a successful solve always reports ZNSSD >= 0 -----------
// The JNI layer marks failed/skipped points with a negative sentinel
// (CORR_INVALID = -1), so a real solve must never produce a negative
// score — including the perfect-match case (ZNSSD == 0.0 exactly).
TEST_CASE(Engine, SuccessfulSolve_CorrelationNonNegative) {
    auto res = run_engine(centered(0.0f, 0.0f), 0.0f, 0.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK(res.correlation_score >= 0.0f);
}

// --- Levenberg-Marquardt damping must not change a well-posed answer --
TEST_CASE(Engine, LmDampingPreservesWellPosedSolution) {
    const float u_true = 1.6f, v_true = 0.8f;
    dictest::SpeckleField field(2468, W, H);
    Image ref = dictest::make_reference_image(field, W, H);
    Image def = dictest::make_deformed_image(field, W, H, centered(u_true, v_true));

    SubsetData subset;
    SubsetPrecomputer::precompute_subset(subset, ref, CX, CY, DIM);
    REQUIRE(subset.is_initialized);

    OptimizationEngine engine;
    engine.lm_enabled = true;
    engine.lm_alpha = 1e-3f;
    auto res = engine.calculate_deformation(subset, def, 2.0f, 1.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);
    REQUIRE(res.status == 0);
    CHECK_NEAR(res.u, u_true, TOL_TRANS);
    CHECK_NEAR(res.v, v_true, TOL_TRANS);
}

// ---------------------------------------------------------------------
// IC-GN validity, inverses and degeneracy (v0.5.0)
// ---------------------------------------------------------------------
namespace {

    // std::isfinite folds to true under -ffast-math, which this suite is
    // built with; look at the exponent bits instead.
    bool finite_bits(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof u);
        return (u & 0x7f800000u) != 0x7f800000u;
    }

    bool all_finite(const AnalysisResult &r) {
        return finite_bits(r.u) && finite_bits(r.v) && finite_bits(r.ux) && finite_bits(r.uy) &&
               finite_bits(r.vx) && finite_bits(r.vy) && finite_bits(r.correlation_score);
    }

    // A pointwise intensity map commutes with the warp, so applying it to
    // both images keeps the ground truth exact.
    void saturate_dark(Image &img, float floor) {
        for (float &v : img.intensities) v = std::max(0.0f, v - floor);
        img.prepare_data(false);
    }

} // namespace

// Black is a valid intensity. IC-GN used to drop every deformed sample that
// was not > 0 -- a speckle saturated to 0, or a cubic undershoot next to one
// -- as if it had left the image, and failed the 90% valid-pixel guard once a
// tenth of the subset was black. Validity is now where the sample was taken,
// not what it read.
TEST_CASE(Engine, SaturatedBlackSpeckle_Converges) {
    const auto truth = centered(2.25f, -1.5f);
    dictest::SpeckleField field(1234, W, H);
    Image ref = dictest::make_reference_image(field, W, H);
    Image def = dictest::make_deformed_image(field, W, H, truth);
    saturate_dark(ref, 100.0f);
    saturate_dark(def, 100.0f);

    SubsetData subset;
    SubsetPrecomputer::precompute_subset(subset, ref, CX, CY, DIM);
    REQUIRE(subset.is_initialized);
    int black = 0;
    for (float v : subset.ref_intensities) black += (v == 0.0f) ? 1 : 0;
    std::printf("    black pixels in subset: %d of %d\n", black, DIM * DIM);
    REQUIRE(black > DIM * DIM / 10);

    for (bool use6x6 : {false, true}) {
        OptimizationEngine engine;
        engine.use_6x6_interpolator = use6x6;
        auto res = engine.calculate_deformation(subset, def, 2.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                                                0.0f, INIT_NO_SIMPLEX);
        // The clip at 0 is a kink no cubic reproduces, so the bias near it
        // is larger than on smooth speckle (measured 0.026 px in v).
        CHECK(res.status == 0);
        CHECK_NEAR(res.u, 2.25f, 0.05f);
        CHECK_NEAR(res.v, -1.5f, 0.05f);
    }
}

// Once an iteration has every pixel back on the image, IC-GN must step with
// the reference Hessian again. It kept the inverse of the last masked
// Hessian instead. To make that visible, the AoS steepest-descent images
// (which only the masked branch reads) are scaled by 10, so a masked
// Hessian is 100x the real one: a full iteration stepping with its inverse
// moves 1/100 of the way and runs out of iterations.
TEST_CASE(Engine, MaskedThenFullIteration_MatchesFullOnlyPath) {
    const auto truth = centered(-60.5f, 0.0f);
    dictest::SpeckleField field(97, W, H);
    Image ref = dictest::make_reference_image(field, W, H);
    Image def = dictest::make_deformed_image(field, W, H, truth);

    SubsetData subset;
    SubsetPrecomputer::precompute_subset(subset, ref, CX, CY, DIM);
    REQUIRE(subset.is_initialized);

    OptimizationEngine engine;
    // Reference: started with every pixel on the image, the full path only.
    auto full_only = engine.calculate_deformation(subset, def, -60.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                                  0.0f, INIT_NO_SIMPLEX);
    REQUIRE(full_only.status == 0);
    // Half-pixel phase 4.5 px from the image edge: the usual 0.02 is too tight.
    CHECK_NEAR(full_only.u, -60.5f, 0.05f);

    for (auto &sd : subset.steepest_descent_images) sd *= 10.0f;
    // Started with the left column past the 4 px guard: the first
    // iterations are masked, the rest are not.
    auto res = engine.calculate_deformation(subset, def, -62.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, INIT_NO_SIMPLEX);
    CHECK(res.status == 0);
    CHECK_NEAR(res.u, full_only.u, 1e-3);
    CHECK_NEAR(res.v, full_only.v, 1e-3);
}

// A masked Hessian that is singular in its strain block passes the 2x2
// translation check. Its inverse was taken anyway, and the step came out
// huge or NaN -- (int)NaN in the pixel guard is undefined behaviour, and
// the result carried the garbage out. It must fail as a failure: status 1,
// finite values, not far from where it started.
TEST_CASE(Engine, DegenerateMaskedHessian_FailsCleanly) {
    dictest::SpeckleField field(55, W, H);
    Image ref = dictest::make_reference_image(field, W, H);
    Image def = dictest::make_deformed_image(field, W, H, centered(-60.0f, 0.0f));

    SubsetData subset;
    SubsetPrecomputer::precompute_subset(subset, ref, CX, CY, DIM);
    REQUIRE(subset.is_initialized);
    // Column 3 (d/d uy) a copy of column 0 times 2: exactly rank-deficient,
    // while the translation block is untouched.
    for (auto &sd : subset.steepest_descent_images) sd(3) = 2.0f * sd(0);

    OptimizationEngine engine;
    auto res = engine.calculate_deformation(subset, def, -62.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, INIT_NO_SIMPLEX);
    CHECK(res.status != 0);
    CHECK(all_finite(res));
    CHECK(std::fabs(res.u + 62.0f) < 5.0f);
    CHECK(std::fabs(res.uy) < 0.5f);
}
