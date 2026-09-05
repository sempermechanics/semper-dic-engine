// The VSG strain pipeline on real speckle, against a prescribed strain.
//
// Where test_strain_gradients.cpp reads raw per-subset ICGN gradients, this
// feeds the displacement field through StrainCalculator's virtual strain
// gauge — the post-processor a real analysis actually uses — and checks the
// smoothed exx against the known applied 1% strain (def_exx.tif, truth
// exx = 0.01).
//
// The Strain unit suite proves VSG is exact on synthetic linear fields; this
// proves it holds up with real correlation noise feeding it. Smoothing over a
// window should be markedly tighter than the raw per-subset scatter.
#include "framework/test_framework.h"
#include "framework/image_io.h"
#include <semper/solver.hpp>
#include <semper/strain.hpp>
#include <semper/image.hpp>
#include <semper/subset.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(DIC_HAVE_OPENCV)

using Semper::Image;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;
using Semper::OptimizationEngine;
using Semper::AnalysisResult;
using Semper::INIT_NO_SIMPLEX;
using Semper::DisplacementField;
using Semper::StrainField;
using Semper::StrainCalculator;
using dictest::GrayImage;
using dictest::load_gray;

namespace {
    constexpr int SUBSET_SIZE = 27;
    constexpr float EXX_TRUE = 0.01f;
    constexpr float CX = 256.0f;

    // Regular grid the DisplacementField requires (spacing == step).
    constexpr int GRID_ORIGIN = 60, GRID_STEP = 20, GRID_N = 20; // 60..440

    // VSG window in pixels. Must span several grid points for the plane fit to
    // average noise; 120 px covers ~6 steps.
    constexpr int VSG_WINDOW_PX = 120;

    // VSG smooths, so we hold it to a much tighter bound than the raw
    // per-subset gradient scatter.
    constexpr float VSG_MEAN_TOL = 1.0e-3f;
    constexpr double VSG_RMS_MAX = 2.0e-3;
} // namespace

TEST_CASE(DiceStrainVsg, UniaxialStrain_ThroughStrainCalculator) {
    GrayImage r, d;
    REQUIRE(load_gray(std::string(DICE_FIXTURES_DIR) + "/ref.tif", r));
    REQUIRE(load_gray(std::string(DICE_FIXTURES_DIR) + "/def_exx.tif", d));
    Image ref(r.w, r.h, r.px.data());
    ref.prepare_data(false);
    Image def(d.w, d.h, d.px.data());
    def.prepare_data(false);

    // 1. Solve the displacement field on a regular grid.
    DisplacementField disp;
    disp.width = GRID_N;
    disp.height = GRID_N;
    disp.step = GRID_STEP;
    disp.u.assign(GRID_N * GRID_N, 0.0f);
    disp.v.assign(GRID_N * GRID_N, 0.0f);
    disp.valid.assign(GRID_N * GRID_N, false);

    int solved = 0;
    for (int gy = 0; gy < GRID_N; ++gy) {
        for (int gx = 0; gx < GRID_N; ++gx) {
            const int x = GRID_ORIGIN + gx * GRID_STEP;
            const int y = GRID_ORIGIN + gy * GRID_STEP;
            SubsetData subset;
            SubsetPrecomputer::precompute_subset(subset, ref, x, y, SUBSET_SIZE);
            if (!subset.is_initialized) continue;

            OptimizationEngine engine;
            // Seed translation as RGDIC propagation would; strain is not seeded.
            const float guess_u = EXX_TRUE * (static_cast<float>(x) - CX);
            AnalysisResult res = engine.calculate_deformation(
                subset, def, guess_u, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);
            if (res.status != 0) continue;

            const int idx = gy * GRID_N + gx;
            disp.u[idx] = res.u;
            disp.v[idx] = res.v;
            disp.valid[idx] = true;
            ++solved;
        }
    }
    REQUIRE(solved > GRID_N * GRID_N / 2);

    // 2. Run the real post-processor.
    const StrainField strain = StrainCalculator::compute_vsg_strain(disp, VSG_WINDOW_PX);
    REQUIRE(strain.exx.size() == disp.u.size());

    // 3. Compare against the applied strain, over points VSG could actually
    //    support (it leaves a sentinel where the window is under-filled).
    int n = 0;
    double sum = 0, sq = 0, worst = 0;
    for (size_t i = 0; i < strain.exx.size(); ++i) {
        if (!disp.valid[i]) continue;
        const float e = strain.exx[i];
        if (e < -100.0f) continue; // VSG sentinel for unsupported windows
        const double dev = static_cast<double>(e) - EXX_TRUE;
        sum += e;
        sq += dev * dev;
        worst = std::max(worst, std::fabs(dev));
        ++n;
    }
    REQUIRE(n > 0);

    const double mean = sum / n, rms = std::sqrt(sq / n);
    std::printf("  DiceStrainVsg: %d/%d subsets solved, %d VSG points\n"
                "     exx mean=%.5f (truth %.5f)  rms dev=%.5f  worst=%.5f\n",
                solved, GRID_N * GRID_N, n, mean, (double) EXX_TRUE, rms, worst);

    CHECK_NEAR(static_cast<float>(mean), EXX_TRUE, VSG_MEAN_TOL);
    CHECK(rms < VSG_RMS_MAX);
}

#endif // DIC_HAVE_OPENCV
