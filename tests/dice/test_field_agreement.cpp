// Our displacement field vs DICe's own solved field.
//
// DICe ships solved fields for its dic_challenge_12 regression case (open-hole
// tension on CFRP) as DICe_solution_NN.txt. We run our engine at DICe's exact
// subset coordinates and diff. Because DICe's gold is fixed and external, this
// both anchors us to the reference implementation and detects drift in ours.
//
// The metric is AGREEMENT, not correctness: oht_cfrp is a real experiment with
// no analytic truth, so DICe's field is a reference result. Independent DIC
// codes legitimately differ, so the bounds below are inter-code agreement
// bounds taken from the measured spread.
//
// Matched to DICe where possible (subset 27, Keys-fourth, its coordinates).
// It used translation + normal-strain shape functions and neighbour seeding;
// we solve full 6-DOF affine.
//
// From the gold header: subset 27 | step 35 | ZNSSD | KEYS_FOURTH |
// (0,0) upper-left, x right, y down.
#include "framework/test_framework.h"
#include "framework/image_io.h"
#include <semper/solver.hpp>
#include <semper/image.hpp>
#include <semper/subset.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(DIC_HAVE_OPENCV)

using Semper::Image;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;
using Semper::OptimizationEngine;
using Semper::AnalysisResult;
using Semper::INIT_NO_SIMPLEX;
using Semper::INIT_AUTO_SEARCH;
using dictest::GrayImage;
using dictest::load_gray;

namespace {

    constexpr int SUBSET_SIZE = 27;      // DICe gold header: "Subset size: 27"

    // Inter-code agreement bounds (px), set from the MEASURED spread rather than
    // guessed. Our engine reproduces DICe's field on this pair to
    // rms 0.0006 px / max 0.0033 px, converging at 230/230 of its points — so
    // these bounds keep ~8x headroom while still being tight enough to catch a
    // regression. Cross-build drift no longer eats into that headroom at all:
    // the engine is bit-identical across -march levels (docs/DETERMINISM.md),
    // so the whole margin absorbs genuine inter-code disagreement.
    constexpr double RMS_TOL = 0.005;
    constexpr double MAX_TOL = 0.02;
    constexpr double MIN_COMPARED_FRACTION = 0.95;

    struct GoldPt {
        int x, y;
        double u, v;
    };

    // Parse DICe's solution file: '***' header block, then a column-name line,
    // then CSV rows of COORDINATE_X,COORDINATE_Y,DISPLACEMENT_X,DISPLACEMENT_Y,...
    bool load_dice_gold(const std::string &path, std::vector<GoldPt> &out) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line.rfind("***", 0) == 0) continue;
            if (line.rfind("COORDINATE_X", 0) == 0) continue; // column names
            std::stringstream ss(line);
            std::string cell;
            double col[4];
            int n = 0;
            while (n < 4 && std::getline(ss, cell, ',')) {
                col[n++] = std::atof(cell.c_str());
            }
            if (n < 4) continue;
            out.push_back({(int) std::lround(col[0]), (int) std::lround(col[1]), col[2], col[3]});
        }
        return !out.empty();
    }

} // namespace

TEST_CASE(DiceFieldAgreement, OhtCfrp_AgreesWithDiceSolution) {
    GrayImage r, d;
    REQUIRE(load_gray(std::string(DICE_FIXTURES_DIR) + "/oht_cfrp_00.tiff", r));
    REQUIRE(load_gray(std::string(DICE_FIXTURES_DIR) + "/oht_cfrp_01.tiff", d));
    Image ref(r.w, r.h, r.px.data());
    ref.prepare_data(false);
    Image def(d.w, d.h, d.px.data());
    def.prepare_data(false);

    std::vector<GoldPt> gold;
    REQUIRE(load_dice_gold(std::string(DICE_FIXTURES_DIR) + "/DICe_solution_01.txt", gold));
    std::printf("  DiceFieldAgreement: %zu DICe reference points\n", gold.size());

    int compared = 0;
    double su = 0, sv = 0, su2 = 0, sv2 = 0, maxdu = 0, maxdv = 0;
    for (const auto &g : gold) {
        SubsetData subset;
        SubsetPrecomputer::precompute_subset(subset, ref, g.x, g.y, SUBSET_SIZE);
        if (!subset.is_initialized) continue;

        OptimizationEngine engine;
        engine.use_6x6_interpolator = true; // DICe used KEYS_FOURTH
        AnalysisResult res = engine.calculate_deformation(
            subset, def, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);
        if (res.status != 0) continue;

        const double du = static_cast<double>(res.u) - g.u;
        const double dv = static_cast<double>(res.v) - g.v;
        su += du; sv += dv;
        su2 += du * du; sv2 += dv * dv;
        if (std::fabs(du) > std::fabs(maxdu)) maxdu = du;
        if (std::fabs(dv) > std::fabs(maxdv)) maxdv = dv;
        ++compared;
    }

    REQUIRE(compared > 0);
    const double frac = static_cast<double>(compared) / static_cast<double>(gold.size());
    const double rms_u = std::sqrt(su2 / compared), rms_v = std::sqrt(sv2 / compared);
    std::printf("  DiceFieldAgreement: compared %d/%zu (%.0f%%)\n"
                "     du: mean=%+.4f rms=%.4f max=%+.4f\n"
                "     dv: mean=%+.4f rms=%.4f max=%+.4f\n",
                compared, gold.size(), frac * 100.0,
                su / compared, rms_u, maxdu,
                sv / compared, rms_v, maxdv);

    CHECK(frac >= MIN_COMPARED_FRACTION);
    CHECK(rms_u < RMS_TOL);
    CHECK(rms_v < RMS_TOL);
    CHECK(std::fabs(maxdu) < MAX_TOL);
    CHECK(std::fabs(maxdv) < MAX_TOL);
}

// ---------------------------------------------------------------------
// Load-step ladder. DICe ships a solved field for every step of the same
// experiment, so the deformation grows while the reference stays fixed:
//
//   step 01 -> max |d| ~  0.9 px      step 06 -> max |d| ~  6.6 px
//   step 03 -> max |d| ~  3.2 px      step 11 -> max |d| ~ 12.0 px
//
// Only step 01 is within a zero-guess ICGN's ~1 px pull-in range, so the rest
// exercise the COARSE SEARCH path (INIT_AUTO_SEARCH) — the seeding behaviour
// the rest of the host suite never touches. Each rung is measured against
// DICe's own field, and per-rung agreement is reported so the bounds stay
// evidence-based.
// ---------------------------------------------------------------------
namespace {
    struct Rung {
        const char *image;
        const char *gold;
        double max_disp_px; // from DICe's own field, for context in the log
    };

    // Agreement bounds for searched (multi-pixel) rungs. Looser than step 01:
    // the coarse search lands on a slightly different basin than DICe's
    // neighbour-seeded guess, and these are real experimental images.
    constexpr double LADDER_RMS_TOL = 0.05;
    constexpr double LADDER_MIN_FRACTION = 0.90;
} // namespace

TEST_CASE(DiceFieldAgreement, OhtCfrp_LoadStepLadder) {
    GrayImage r;
    REQUIRE(load_gray(std::string(DICE_FIXTURES_DIR) + "/oht_cfrp_00.tiff", r));
    Image ref(r.w, r.h, r.px.data());
    ref.prepare_data(false);

    const Rung rungs[] = {
        {"/oht_cfrp_03.tiff", "/DICe_solution_03.txt", 3.16},
        {"/oht_cfrp_06.tiff", "/DICe_solution_06.txt", 6.61},
        {"/oht_cfrp_11.tiff", "/DICe_solution_11.txt", 12.03},
    };

    for (const auto &rung : rungs) {
        GrayImage d;
        REQUIRE(load_gray(std::string(DICE_FIXTURES_DIR) + rung.image, d));
        Image def(d.w, d.h, d.px.data());
        def.prepare_data(false);

        std::vector<GoldPt> gold;
        REQUIRE(load_dice_gold(std::string(DICE_FIXTURES_DIR) + rung.gold, gold));

        int compared = 0;
        double su2 = 0, sv2 = 0, maxd = 0;
        for (const auto &g : gold) {
            SubsetData subset;
            SubsetPrecomputer::precompute_subset(subset, ref, g.x, g.y, SUBSET_SIZE);
            if (!subset.is_initialized) continue;

            OptimizationEngine engine;
            engine.use_6x6_interpolator = true;
            // No prior knowledge: the engine must FIND a multi-pixel shift.
            AnalysisResult res = engine.calculate_deformation(
                subset, def, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, INIT_AUTO_SEARCH);
            if (res.status != 0) continue;

            const double du = static_cast<double>(res.u) - g.u;
            const double dv = static_cast<double>(res.v) - g.v;
            su2 += du * du;
            sv2 += dv * dv;
            maxd = std::max(maxd, std::max(std::fabs(du), std::fabs(dv)));
            ++compared;
        }

        REQUIRE(compared > 0);
        const double frac = static_cast<double>(compared) / static_cast<double>(gold.size());
        const double rms_u = std::sqrt(su2 / compared), rms_v = std::sqrt(sv2 / compared);
        std::printf("  DiceFieldAgreement ladder %s (DICe max|d|=%.2f px):"
                    " converged %d/%zu (%.0f%%)  rms du=%.4f dv=%.4f  max|d|=%.4f\n",
                    rung.gold, rung.max_disp_px, compared, gold.size(),
                    frac * 100.0, rms_u, rms_v, maxd);

        CHECK(frac >= LADDER_MIN_FRACTION);
        CHECK(rms_u < LADDER_RMS_TOL);
        CHECK(rms_v < LADDER_RMS_TOL);
    }
}

#endif // DIC_HAVE_OPENCV
