// =====================================================================
// SEEDING BENCHMARK — measure before replacing
//
// The engine's displacements come from IC-GN; the seeding front-end only
// supplies the initial guess. So "which feature matcher is best" is a question
// about seed accuracy, ROI coverage and cost, not about sub-pixel precision.
// This suite measures the candidates on identical inputs and prints the table;
// it asserts nothing about which one wins.
//
// Two quantities matter, and they are reported separately:
//
//   vertex error   |u - u_true| at the mesh vertices the front-end produced.
//   guess-gradient error
//                  |ux - ux_true| over the grid, taken from the Delaunay mesh
//                  the vertices feed. build_mesh_guess_field derives the
//                  gradients with cv::getAffineTransform over triangle
//                  vertices, so a vertex error e over an edge of length L
//                  becomes a gradient error of order e/L. This is the quantity
//                  the guess actually hands to ICGN.
//
// Ground truth is exact: reference and deformed images are both sampled from
// the same continuous analytic speckle function (framework/synthetic.h), so
// no resampling error enters the comparison.
//
// The always-on contracts that pin the seeding front-end live in
// test_seeding_contracts.cpp; the harness both use is framework/seed_bench.h.
//
// Run:  SEMPER_RUN_SEEDBENCH=1 ./dic_tests SeedBench
//       SEMPER_SEEDBENCH_CSV=/tmp/seed.csv  (optional machine-readable dump)
// =====================================================================

#include "framework/test_framework.h"

#if DIC_HAVE_OPENCV

#include "framework/seed_bench.h"

using namespace dictest::seedbench;


// ---------------------------------------------------------------------------
// The sweep. Measurement only — no assertions on the winner.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, CandidateSweep) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1 to run the sweep)\n");
        return;
    }

    std::vector<Row> rows;
    for (const Scenario &sc : scenarios()) {
        dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
        const Image ref = dictest::make_reference_image(field, W, H);
        const Image def = dictest::make_deformed_image(field, W, H, sc.def);
        const cv::Mat ref_gray = dictest::gray8(ref, sc.noise_sigma, 101);
        const cv::Mat def_gray = dictest::gray8(def, sc.noise_sigma, 202);

        run_scenario(sc.name, affine_truth(sc.def), ref_gray, def_gray,
                       bench_params(), rows);
    }

    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// The same sweep on DICe's real speckle (ref.tif / def.tif, 512x512).
//
// The pair is a rigid translation, so the true displacement-gradient tensor is
// exactly zero regardless of the translation magnitude — mesh_med_dP is
// rigorous ground truth here. The displacement columns use DICe's published
// U = 0.4 px and V = 0.4 px; DICe publishes no V reference, so read the
// displacement columns as indicative and the gradient columns as exact.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, RealSpeckleSweep) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1 to run the sweep)\n");
        return;
    }

    const std::string dir = DICE_FIXTURES_DIR;
    dictest::GrayImage r, d;
    REQUIRE(dictest::load_gray(dir + "/ref.tif", r));
    REQUIRE(dictest::load_gray(dir + "/def.tif", d));
    REQUIRE(r.w == 512 && r.h == 512);
    REQUIRE(d.w == r.w && d.h == r.h);

    cv::Mat ref_gray(r.h, r.w, CV_8UC1, r.px.data());
    cv::Mat def_gray(d.h, d.w, CV_8UC1, d.px.data());

    Scenario sc{};
    sc.name = "REAL_dice_0.4px";
    sc.def.u = 0.4f;
    sc.def.v = 0.4f;
    sc.def.cx = 256.0f;
    sc.def.cy = 256.0f;
    sc.noise_sigma = 0.0f;

    FullFieldParams params;
    params.rect_x = 56;
    params.rect_y = 56;
    params.rect_w = 400;
    params.rect_h = 400;
    params.step = STEP;
    params.subset_size = SUBSET;
    params.strain_window = STRAIN_WIN;
    params.use_6x6_interpolator = false;

    std::vector<Row> rows;
    run_scenario(sc.name, affine_truth(sc.def), ref_gray, def_gray, params, rows);

    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_REAL")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// DIC Challenge Sample 5 — real speckle, exact rigid translation (0.10 px).
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, Challenge5) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1)\n");
        return;
    }
    const char *repo = dice_repo();
    if (repo == nullptr) {
        std::printf("    skipped (set SEMPER_DICE_REPO to a dicengine/dice checkout)\n");
        return;
    }
    const std::string dir = std::string(repo) + "/tests/regression/dic_challenge_5_znssd/images/";

    cv::Mat ref_gray, def_gray;
    REQUIRE(load_gray_mat(dir + "Sample5 Reference.tif", ref_gray));
    REQUIRE(load_gray_mat(dir + "Sample5-001 X0.10 Y0.10 N2 C100 R20.tif", def_gray));
    REQUIRE(ref_gray.cols == 512 && ref_gray.rows == 512);

    dictest::AffineDeformation d;
    d.u = 0.10f; d.v = 0.10f;
    d.cx = 256.0f; d.cy = 256.0f;

    std::vector<Row> rows;
    run_scenario("C5_real_0.10px", affine_truth(d), ref_gray, def_gray,
                 params_for(512, 512, SUBSET, STEP), rows);
    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_C5")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// DIC Challenge Sample 14 — real speckle, u(x) = 0.1 sin(2 pi x / L).
//
// This is the decisive set for the guess gradients: the field is non-affine,
// the amplitude is 0.100 px, and the shortest period (L5, ~205 px) puts a
// continuously varying strain of ~3e-3 across the ROI. A mesh built on
// clustered keypoints has no way to represent that; a uniform lattice does.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, Challenge14Sinusoid) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1)\n");
        return;
    }
    const char *repo = dice_repo();
    if (repo == nullptr) {
        std::printf("    skipped (set SEMPER_DICE_REPO to a dicengine/dice checkout)\n");
        return;
    }
    const std::string base = std::string(repo) + "/tests/regression/dic_challenge_14_vsg/";

    cv::Mat ref_gray;
    REQUIRE(load_gray_mat(base + "images/Sample14 Reference.tif", ref_gray));
    REQUIRE(ref_gray.cols == 2048 && ref_gray.rows == 589);

    // L1 is the longest period and L5 the shortest; series index selects the
    // column pair in command.csv.
    const struct { const char *tag; const char *file; int series; } sets[] = {
            {"C14_L3_sin356px", "images/Sample14 L3 Amp0.1b.tif", 1},
            {"C14_L5_sin205px", "images/Sample14 L5 Amp0.1b.tif", 2},
    };

    std::vector<Row> rows;
    for (const auto &set : sets) {
        cv::Mat def_gray;
        REQUIRE(load_gray_mat(base + set.file, def_gray));

        CommandCurve curve;
        REQUIRE(load_command_csv(base + "command.csv", set.series, curve));

        // 589 rows is short, so the ROI is height-limited; keep the full width
        // where the sinusoid lives.
        run_scenario(set.tag, sinusoid_truth(curve), ref_gray, def_gray,
                       params_for(2048, 589, SUBSET, 10), rows);
    }
    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_C14")) write_csv(rows, csv);
}

// ---------------------------------------------------------------------------
// Rotation and large-motion sweep.
//
// This is the regime the descriptor front-end was actually built for: DICe uses
// its AKAZE Feature_Matching_Initializer for stereo and for re-acquiring lost
// subsets, not for small-displacement 2D DIC. AKAZE is scale- and
// rotation-invariant; phase correlation is not, and the anchor lattice's
// fallback when the correlation peak is untrustworthy is only a +/-15 px
// per-anchor search. So this sweep is expected to be where the lattice loses,
// and it exists to find out where the crossover actually is.
//
// The `lock` column is the diagnostic: `NO` means phaseCorrelate's peak was
// below kPhaseCorrMinResponse and every anchor fell back to its own coarse
// search.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, LargeMotionSweep) {
    if (std::getenv("SEMPER_RUN_SEEDBENCH") == nullptr) {
        std::printf("    skipped (set SEMPER_RUN_SEEDBENCH=1)\n");
        return;
    }

    std::vector<Scenario> sweep;
    sweep.push_back({"R_rot_2deg",    rotation(2.0),  0.0f});
    sweep.push_back({"R_rot_5deg",    rotation(5.0),  0.0f});
    sweep.push_back({"R_rot_10deg",   rotation(10.0), 0.0f});
    sweep.push_back({"R_rot_15deg",   rotation(15.0), 0.0f});
    sweep.push_back({"R_rot_30deg",   rotation(30.0), 0.0f});
    // Handheld capture: the phone turns, steps closer and shifts at once.
    sweep.push_back({"S_scale_1.05",  similarity(0.0, 1.05), 0.0f});
    sweep.push_back({"RST_10deg_1.05_20px", similarity(10.0, 1.05, 20.0f, -12.0f), 0.0f});
    sweep.push_back({"T_trans_25px",  affine(25.0f, -15.0f),  0.0f});
    sweep.push_back({"T_trans_50px",  affine(50.0f, -30.0f),  0.0f});
    sweep.push_back({"T_trans_100px", affine(100.0f, -60.0f), 0.0f});

    const FullFieldParams params = big_motion_params();
    std::vector<Row> rows;
    for (const Scenario &sc : sweep) {
        dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
        const Image ref = dictest::make_reference_image(field, W, H);
        const Image def = dictest::make_deformed_image(field, W, H, sc.def);
        const cv::Mat ref_gray = dictest::gray8(ref, sc.noise_sigma, 101);
        const cv::Mat def_gray = dictest::gray8(def, sc.noise_sigma, 202);
        run_scenario(sc.name, affine_truth(sc.def), ref_gray, def_gray, params, rows);
    }

    print_table(rows);
    if (const char *csv = std::getenv("SEMPER_SEEDBENCH_CSV_BIG")) write_csv(rows, csv);
}

#endif // DIC_HAVE_OPENCV
