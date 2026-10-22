// =====================================================================
// SEEDING CONTRACTS — the always-on half of the seeding suite
//
// The sweeps in test_seeding_bench.cpp measure the anchor-lattice front-end
// and assert nothing; they are gated behind SEMPER_RUN_SEEDBENCH, and three
// of them behind an external DICe checkout. What CI defends on every run is
// here: the phase-correlation sign convention, the log-polar rotation/scale
// estimate, the rotation and quality gates, the similarity fallback, the
// median test and its publication rule, and the lattice sizing. Harness:
// framework/seed_bench.h.
// =====================================================================

#include "framework/test_framework.h"

#if DIC_HAVE_OPENCV

#include "framework/seed_bench.h"

using namespace dictest::seedbench;

// ---------------------------------------------------------------------------
// Sign convention of cv::phaseCorrelate against the engine's warp.
//
// The engine's shape function puts a reference point p at p + (u, v) in the
// deformed image. This pins which way round phaseCorrelate reports that, so
// the anchor lattice cannot silently seed with an inverted global guess.
// Cheap and deterministic, so it runs in the default suite.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, PhaseCorrelateSignConvention) {
    const float U = 5.0f, V = -3.0f;
    dictest::SpeckleField field(/*seed=*/7, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const Image def = dictest::make_deformed_image(field, W, H, affine(U, V));

    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 0);
    const cv::Mat def_gray = dictest::gray8(def, 0.0f, 0);

    double u = 0.0, v = 0.0, resp = 0.0;
    const bool ok = Semper::seeding::phase_correlate_roi(
            ref_gray, def_gray, cv::Rect(RECT_ORIGIN, RECT_ORIGIN, ROI_SIZE, ROI_SIZE),
            u, v, resp);

    REQUIRE(ok);
    CHECK(resp > 0.5);
    CHECK_NEAR(u, U, 0.05);
    CHECK_NEAR(v, V, 0.05);
    std::printf("    phaseCorrelate -> (%.4f, %.4f), truth (%.1f, %.1f), response %.4f\n",
                u, v, (double)U, (double)V, resp);
}

// ---------------------------------------------------------------------------
// The log-polar (Fourier-Mellin) half of L0: sign and magnitude of the
// rotation and scale it reports, the translation re-measured once they are
// undone, and the switch that keeps small motion on the translation-only path.
// Always on: one DFT pair per case.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, LogPolarRotationScale) {
    using Semper::seeding::SimilarityShift;
    dictest::SpeckleField field(/*seed=*/7, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 0);
    const cv::Rect roi(BIG_ORIGIN, BIG_ORIGIN, BIG_RECT, BIG_RECT);
    const auto pre = Semper::seeding::prepare_phase_correlate_ref(ref_gray, roi);
    REQUIRE(pre.lp_usable());

    struct Case { double deg, s; float u, v; };
    const Case turned[] = {
        {10.0, 1.00, 0.0f, 0.0f},
        {-7.0, 1.04, 6.0f, -4.0f},   // the other way, zoomed in, and shifted
        {3.0, 0.97, -2.5f, 1.5f},
        {25.0, 1.00, 12.0f, 8.0f},
    };
    for (const Case &c : turned) {
        const dictest::AffineDeformation d = similarity(c.deg, c.s, c.u, c.v);
        const cv::Mat def_gray = dictest::gray8(
                dictest::make_deformed_image(field, W, H, d), 0.0f, 0);
        SimilarityShift gm;
        Semper::seeding::phase_correlate_similarity(pre, def_gray, gm);
        const Truth t = affine_truth(d)((float)gm.cx, (float)gm.cy);
        std::printf("    %+6.1f deg x%.3f -> %+8.4f deg x%.5f (lp resp %.3f), "
                    "centre (%.3f, %.3f) truth (%.3f, %.3f), sim=%d\n",
                    c.deg, c.s, gm.theta_rad * 180.0 / CV_PI, gm.scale, gm.lp_response,
                    gm.u, gm.v, (double)t.u, (double)t.v, gm.similarity ? 1 : 0);
        REQUIRE(gm.lp_ok);
        CHECK_NEAR(gm.theta_rad * 180.0 / CV_PI, c.deg, 0.15);
        CHECK_NEAR(gm.scale, c.s, 0.005);
        REQUIRE(gm.similarity);
        CHECK(gm.locked);
        CHECK_NEAR(gm.u, t.u, 0.25);
        CHECK_NEAR(gm.v, t.v, 0.25);
        CHECK_NEAR(gm.ux, d.ux, 3e-3);
        CHECK_NEAR(gm.uy, d.uy, 3e-3);
        CHECK_NEAR(gm.vx, d.vx, 3e-3);
        CHECK_NEAR(gm.vy, d.vy, 3e-3);
    }

    // Below kSimilarityMinCornerPx the answer must be the translation-only
    // one exactly -- this is what keeps small-motion fields bit-identical. The
    // second case is FullFieldGolden's affine scenario, whose rotation and
    // dilation move this ROI's corners by ~2.8 px.
    const dictest::AffineDeformation small[] = {
        affine(5.0f, -3.0f),
        affine(2.0f, -1.5f, 0.012f, -0.006f, 0.005f, 0.010f),
    };
    for (const dictest::AffineDeformation &d : small) {
        const cv::Mat def_gray = dictest::gray8(
                dictest::make_deformed_image(field, W, H, d), 0.0f, 0);
        SimilarityShift gm;
        Semper::seeding::phase_correlate_similarity(pre, def_gray, gm);
        double u = 0.0, v = 0.0, resp = 0.0;
        const bool ok = Semper::seeding::phase_correlate_roi(pre, def_gray, u, v, resp);
        std::printf("    small: lp %+.4f deg x%.5f -> sim=%d\n",
                    gm.theta_rad * 180.0 / CV_PI, gm.scale, gm.similarity ? 1 : 0);
        CHECK(!gm.similarity);
        CHECK(gm.locked == ok);
        CHECK(gm.u == u);
        CHECK(gm.v == v);
        CHECK(gm.response == resp);
        CHECK(gm.ux == 0.0 && gm.uy == 0.0 && gm.vx == 0.0 && gm.vy == 0.0);
    }
}

// ---------------------------------------------------------------------------
// The rotation half of the quality gate: LargeMotionSweep's rows that the
// translation-only seeder lost (docs/SEEDING_BENCHMARK.md 4.5 -- 0.450 mesh
// coverage at 5 deg, 0.152 and no lock at 15 deg), as always-on assertions.
// Same bounds as AnchorSeedingQualityGate for coverage; vertex error is
// bounded at 0.05 px against a measured ~0.009.
//
// The 120 and 180 deg rows are v0.6.0's: the log-polar similarity before it
// could not tell theta from theta + 180 deg, and the AKAZE homography can.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AnchorSeedingRotationGate) {
    const struct { const char *name; dictest::AffineDeformation def; } gates[] = {
        {"R_rot_5deg", rotation(5.0)},
        {"R_rot_15deg", rotation(15.0)},
        {"RST_10deg_1.05_20px", similarity(10.0, 1.05, 20.0f, -12.0f)},
        {"R_rot_120deg", rotation(120.0)},
        {"R_rot_180deg", rotation(180.0)},
    };

    dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 101);

    for (const auto &g : gates) {
        const Image def = dictest::make_deformed_image(field, W, H, g.def);
        const cv::Mat def_gray = dictest::gray8(def, 0.0f, 202);

        Row row;
        row.scenario = g.name;
        measure_seed(row, affine_truth(g.def), ref_gray, def_gray, big_motion_params());
        std::printf("    %-20s lock=%s model=%d rot=%+.3f x%.4f q=%d cov=%.3f "
                    "vtx=%.4f px kept %d/%d meshfr=%.3f\n",
                    g.name, row.locked ? "yes" : "NO", row.model ? 1 : 0,
                    row.model_rot_deg, row.model_scale, row.quality, (double)row.coverage,
                    row.vtx_med_du, row.anchors_accepted, row.anchors_attempted,
                    row.mesh_frac);

        CHECK(row.locked);
        CHECK(row.model);
        CHECK(row.quality == 2);
        CHECK(row.coverage >= 0.90f);
        CHECK(row.mesh_frac >= 0.95);
        REQUIRE(row.vtx_med_du >= 0.0);
        CHECK(row.vtx_med_du <= 0.05);
    }
}

// ---------------------------------------------------------------------------
// Strong anisotropic stretch: IncBench S5 ("stretch to 60%"), frame 150, a
// 45% stretch with Poisson 0.3 on a 110 px ROI, noise-free.
//
// Until v0.6.0 this was the log-polar similarity's failure mode: it read the
// stretch as about -5.8 deg x 0.918, the lattice seeded from that kept 123 of
// 484 anchors with blunders no gate caught (0.215 px field rms), and the
// translation-only fallback kept 86 with a p95 vertex error of ~25.7 px.
//
// AKAZE descriptors do not survive a 45% stretch either -- no model here --
// so every anchor runs its own offset search, the path v0.5.0 also took
// without a phase lock. On this frame that search is what does the work, and
// it keeps over 300 anchors with no blunders: the p95 vertex error is the
// bound that matters.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, StretchSeedsWithoutBlunders) {
    const int w = 384, h = 256;
    const dictest::SmoothSpeckleField field(101, w, h);
    const cv::Mat ref = dictest::render_frame(field, w, h, dictest::identity_motion(), 0.0f, 7919u);
    const dictest::Motion m =
            dictest::stretch_rotate(0.45, 0.3, 0.0, dictest::Vec2{}, dictest::Vec2{192, 128});
    const cv::Mat def = dictest::render_frame(field, w, h, m, 0.0f, 7919u + 150u);

    FullFieldParams p;   // IncBench's grid: the app defaults, 110 px ROI centred
    p.rect_x = (w - 110) / 2;
    p.rect_y = (h - 110) / 2;
    p.rect_w = 110;
    p.rect_h = 110;
    p.step = 5;
    p.subset_size = 21;
    p.strain_window = 15;
    p.use_6x6_interpolator = false;

    const TruthFn truth = [&m](float x, float y) {
        const dictest::Vec2 d = m.displacement(x, y);
        const dictest::Mat2 F = m.grad(x, y);
        Truth t;
        t.u = (float)d.x; t.v = (float)d.y;
        t.ux = (float)(F.a - 1.0); t.uy = (float)F.b;
        t.vx = (float)F.c; t.vy = (float)(F.d - 1.0);
        return t;
    };

    Row row;
    row.scenario = "stretch_45pct";
    measure_seed(row, truth, ref, def, p);
    std::printf("    lock=%s model %+.3f deg x%.4f applied=%d rejected=%d kept %d/%d q=%d "
                "cov=%.3f vtx med %.4f p95 %.4f px\n",
                row.locked ? "yes" : "NO", row.model_rot_deg, row.model_scale,
                row.model ? 1 : 0, row.model_rejected ? 1 : 0, row.anchors_accepted,
                row.anchors_attempted, row.quality, (double)row.coverage, row.vtx_med_du,
                row.vtx_p95_du);

    // Measured: FULL, 337/484 kept, cov 0.834, p95 0.026 px.
    CHECK(row.quality == 2);
    CHECK(row.anchors_accepted >= 200);
    REQUIRE(row.vtx_p95_du >= 0.0);
    CHECK(row.vtx_p95_du <= 0.25);

    // And the field: 0.215 px from the misfired similarity, 0.014 px from
    // v0.5.0's translation-only fallback.
    measure_full_field(row, truth, ref, def, p);
    std::printf("    field: %d points, rms %.4f px\n", row.ff_rc, row.ff_rms_u);
    REQUIRE(row.ff_rc > 0);
    CHECK(row.ff_rms_u <= 0.05);
}

// ---------------------------------------------------------------------------
// The guess field at the grid edge. build_mesh_guess_field box-averages the
// mesh guess over 5x5 in-mesh neighbours; on the outer two rings the box is
// one-sided, and a plain average moved u/v by gradient x offset (about 5 px
// at a quarter turn), outside IC-GN's reach. Every such node went to a
// Simplex rescue: 70 of them at 85 deg (83 at 90), at 2.7x the field CPU.
// Carrying the average back to the node along the smoothed gradient keeps an
// affine guess exact, and the count at 0.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, GuessSmoothingKeepsEdgesInReach) {
    dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 101);
    const dictest::AffineDeformation d = rotation(85.0);
    const Image def = dictest::make_deformed_image(field, W, H, d);
    const cv::Mat def_gray = dictest::gray8(def, 0.0f, 202);

    Row row;
    row.scenario = "R_rot_85deg";
    measure_full_field(row, affine_truth(d), ref_gray, def_gray, big_motion_params());
    std::printf("    field: %d points, %d Simplex rescues, %.2f mean iters, rms %.4f px\n",
                row.ff_rc, row.ff_simplex_calls, row.ff_mean_iters, row.ff_rms_u);
    REQUIRE(row.ff_rc > 0);
    CHECK(row.ff_simplex_calls <= 5);
    CHECK(row.ff_rms_u <= 0.05);
}

// ---------------------------------------------------------------------------
// A half turn, end to end. v0.5.0 read 180 deg as 0 deg (the log-polar
// spectrum is the same for both), seeded every anchor from the wrong side of
// the ROI and returned no usable field. The homography carries the whole
// field, mesh and the nodes outside it alike.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, HalfTurnSolvesTheField) {
    dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 101);
    const dictest::AffineDeformation d = rotation(180.0);
    const Image def = dictest::make_deformed_image(field, W, H, d);
    const cv::Mat def_gray = dictest::gray8(def, 0.0f, 202);

    Row row;
    row.scenario = "R_rot_180deg";
    measure_full_field(row, affine_truth(d), ref_gray, def_gray, big_motion_params());
    std::printf("    field: %d points, %d Simplex rescues, %.2f mean iters, rms %.4f px\n",
                row.ff_rc, row.ff_simplex_calls, row.ff_mean_iters, row.ff_rms_u);
    REQUIRE(row.ff_rc > 0);
    CHECK(row.ff_simplex_calls <= 5);
    CHECK(row.ff_rms_u <= 0.05);
}

// ---------------------------------------------------------------------------
// The incremental handover: keypoints detected on a deformed frame become the
// next reference's instead of being detected again. reuse_akaze_ref must give
// the same keypoints and descriptors, in the new rect's own coordinates, as a
// fresh detection would for every keypoint far enough inside both rects that
// the crop did not change it -- and refuse a rect it covers too little of.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AkazeHandoverMatchesFreshDetection) {
    using namespace Semper::seeding;
    dictest::SpeckleField field(/*seed=*/7, W, H, BLOBS);
    const cv::Mat gray = dictest::gray8(dictest::make_reference_image(field, W, H), 0.0f, 0);
    const cv::Rect roi(BIG_ORIGIN, BIG_ORIGIN, BIG_RECT, BIG_RECT);
    const cv::Rect moved = roi + cv::Point(6, -4);
    const double s = 0.5;

    const AkazeRef from = prepare_akaze_ref(gray, roi, s);
    REQUIRE(from.usable());
    AkazeRef reused;
    REQUIRE(reuse_akaze_ref(from, moved, gray.size(), reused));
    const AkazeRef fresh = prepare_akaze_ref(gray, moved, s);
    CHECK(reused.rect == fresh.rect);
    CHECK(reused.scale == fresh.scale);
    CHECK(reused.descriptors.rows == (int)reused.keypoints.size());

    // A keypoint of the reused set, mapped back to image coordinates, must
    // also be in the fresh set at the same place with the same descriptor.
    const float margin = 20.0f;   // AKAZE's support at this scale, in scaled px
    const float w = (float)(s * reused.rect.width), h = (float)(s * reused.rect.height);
    int inside = 0, found = 0;
    for (size_t i = 0; i < reused.keypoints.size(); ++i) {
        const cv::Point2f q = reused.keypoints[i].pt;
        const cv::Point2f img(q.x / (float)s + reused.rect.x, q.y / (float)s + reused.rect.y);
        const cv::Point2f in_from((img.x - from.rect.x) * (float)s, (img.y - from.rect.y) * (float)s);
        if (q.x < margin || q.y < margin || q.x > w - margin || q.y > h - margin ||
            in_from.x < margin || in_from.y < margin ||
            in_from.x > s * from.rect.width - margin || in_from.y > s * from.rect.height - margin) {
            continue;
        }
        ++inside;
        for (size_t j = 0; j < fresh.keypoints.size(); ++j) {
            if (cv::norm(fresh.keypoints[j].pt - q) < 0.01 &&
                cv::norm(fresh.descriptors.row((int)j), reused.descriptors.row((int)i),
                         cv::NORM_HAMMING) == 0) {
                ++found;
                break;
            }
        }
    }
    std::printf("    reused %zu, fresh %zu keypoints; %d well inside, %d identical\n",
                reused.keypoints.size(), fresh.keypoints.size(), inside, found);
    // Measured 113 of 117: AKAZE sets its contrast factor from the whole crop,
    // so a few keypoints near threshold come and go with the crop.
    REQUIRE(inside >= 20);
    CHECK(found >= (int)(0.90 * inside));

    // Too little overlap: a rect shifted by most of its width is refused.
    AkazeRef far;
    CHECK(!reuse_akaze_ref(from, roi + cv::Point(BIG_RECT / 2, 0), gray.size(), far));
}

// ---------------------------------------------------------------------------
// Quality gate.
//
// Every sweep in test_seeding_bench.cpp is gated behind SEMPER_RUN_SEEDBENCH,
// and three of them additionally behind an external DICe checkout, which meant
// that before this case the largest behavioural change in the repo's history
// was defended in CI by exactly one assertion -- the sign convention above.
//
// This case promotes three of the seven CandidateSweep scenarios to always-on
// assertions. They are chosen to cover the three regimes the anchor lattice
// is claimed to handle, not to re-measure the sweep:
//
//   S1b_trans_0.4px  the sub-pixel regime the engine actually ships for
//   S4_rot_0.5deg    small rotation -- docs/SEEDING_BENCHMARK.md 4.5 records
//                    that the lattice degrades above ~5 deg, so the lower end
//                    of that range is the boundary worth pinning
//   S5_trans_12px    motion far outside IC-GN's own capture range, which only
//                    succeeds if the L0 global-motion lock is real
//
// The bounds are deliberately loose -- 3x to 5x the measured value, listed
// per row below. This is a regression gate, not a benchmark: it should fire
// when seeding stops working, and stay silent when it merely gets 20% worse
// on one host. The sweep remains the place to read exact numbers.
//
// Reference values are the medians reported by CandidateSweep and the
// criteria in docs/SEEDING_BENCHMARK.md section 9. For scale, the AKAZE
// front-end this replaced measured 0.157 px vertex error and 9.39e-3 gradient
// error at 0.446 mesh coverage on the DICe pair (section 4.1), so every bound
// here is still far tighter than the behaviour that shipped before it.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AnchorSeedingQualityGate) {
    struct Gate {
        const char *name;
        dictest::AffineDeformation def;
        double max_vtx_med_du;   // px, median vertex displacement error
        double max_mesh_med_dux; // guess displacement-gradient error
        double max_mesh_med_du;  // px, median guess displacement error
    };

    const Gate gates[] = {
        // measured: 0.0103 px / 5.52e-4 / 0.0083 px
        {"S1b_trans_0.4px", affine(0.4f, 0.0f), 0.05, 2.0e-3, 0.04},
        // measured: 0.0106 px / 5.41e-4 / 0.0083 px
        {"S4_rot_0.5deg", rotation(0.5), 0.05, 2.0e-3, 0.04},
        // measured: 0.0000 px / 1.31e-6 / 0.0000 px. Bounded well above the
        // measurement because what is being asserted is that the global lock
        // carried a 12 px motion at all, not the last digit of an exact hit.
        {"S5_trans_12px", affine(12.0f, -7.0f), 0.02, 5.0e-5, 0.02},
    };

    dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 101);

    for (const Gate &g : gates) {
        const Image def = dictest::make_deformed_image(field, W, H, g.def);
        const cv::Mat def_gray = dictest::gray8(def, 0.0f, 202);

        Row row;
        row.scenario = g.name;
        measure_seed(row, affine_truth(g.def), ref_gray, def_gray,
                     bench_params());

        std::printf("    %-16s lock=%s q=%d cov=%.3f vtx=%.4f px "
                    "dP=%.3e du=%.4f px meshfr=%.3f\n",
                    g.name, row.locked ? "yes" : "NO", row.quality,
                    (double) row.coverage, row.vtx_med_du, row.mesh_med_dux,
                    row.mesh_med_du, row.mesh_frac);

        // A lost global lock is the single failure that explains most of the
        // others, so check it first and by itself.
        CHECK(row.locked);
        // MeshQuality::FULL. SPARSE means the hull did not cover the ROI and
        // Path B is doing work the mesh was supposed to do.
        CHECK(row.quality == 2);
        CHECK(row.coverage >= 0.90f);
        CHECK(row.mesh_frac >= 0.95);
        // percentile() returns -1 for an empty sample; the bounds below would
        // then pass vacuously.
        REQUIRE(row.vtx_med_du >= 0.0);
        REQUIRE(row.mesh_med_dux >= 0.0);
        CHECK(row.vtx_med_du <= g.max_vtx_med_du);
        CHECK(row.mesh_med_dux <= g.max_mesh_med_dux);
        CHECK(row.mesh_med_du <= g.max_mesh_med_du);
    }
}

// ---------------------------------------------------------------------------
// A model its own lattice rejects must not seed the nodes outside the mesh.
// AnchorSeedingRotationGate's 15 deg turn, with the centre of the ROI blank in
// both frames: AKAZE locks on the speckle, the full model is tried, and its
// lattice keeps the textured three quarters -- under kSimilarityKeepFraction,
// so the model is rejected. Started from the translation alone, the anchors
// left on texture sit 20 px and more from where they start and none is kept.
// With no anchors there is nothing to refit, and the AKAZE homography the
// lattice had just rejected used to go on to build_mesh_guess_field, which
// seeds every node outside the mesh from it.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, RejectedModelDoesNotSeedOutsideMesh) {
    dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
    cv::Mat ref_gray = dictest::gray8(dictest::make_reference_image(field, W, H), 0.0f, 101);
    cv::Mat def_gray = dictest::gray8(dictest::make_deformed_image(field, W, H, rotation(15.0)),
                                      0.0f, 202);
    // A 160 px blank square at the centre; in the deformed frame it also
    // covers where the turn takes the square's corners.
    ref_gray(cv::Rect(240, 240, 160, 160)).setTo(128);
    def_gray(cv::Rect(222, 222, 196, 196)).setTo(128);

    Row row;
    row.scenario = "blank_centre_rot_15deg";
    measure_seed(row, affine_truth(rotation(15.0)), ref_gray, def_gray, big_motion_params());
    std::printf("    lock=%s model=%d rejected=%d kept %d/%d q=%d homography=%d\n",
                row.locked ? "yes" : "NO", row.model ? 1 : 0, row.model_rejected ? 1 : 0,
                row.anchors_accepted, row.anchors_attempted, row.quality,
                row.homography ? 1 : 0);

    // The scenario, so the check below cannot pass vacuously.
    REQUIRE(row.locked);
    REQUIRE(row.model_rejected);
    REQUIRE(row.anchors_accepted < Semper::tuning::kAnchorRefitMinPoints);
    CHECK(!row.homography);
}

// ---------------------------------------------------------------------------
// The metrics identity (CONTRACT.md, slots 19, 20, 23) on the Path C route.
// StretchSeedsWithoutBlunders' 45% stretch leaves AKAZE no model, and a
// one-row grid has no anchor lattice, so nothing is published, the mesh is
// NONE, Path C finds the seed and Path B solves the row from it. Path B
// solved its seeds without counting them in any slot, so slot 1 came out
// ahead of 3 + 4 + 19 by the number of seeds.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, MetricsIdentityHoldsOnPathCRoute) {
    const int w = 384, h = 256;
    const dictest::SmoothSpeckleField field(101, w, h);
    const cv::Mat ref = dictest::render_frame(field, w, h, dictest::identity_motion(), 0.0f, 7919u);
    const dictest::Motion m =
            dictest::stretch_rotate(0.45, 0.3, 0.0, dictest::Vec2{}, dictest::Vec2{192, 128});
    const cv::Mat def = dictest::render_frame(field, w, h, m, 0.0f, 7919u + 150u);

    FullFieldParams p;
    p.rect_x = 172;
    p.rect_y = 128;
    p.rect_w = 8 * 5;
    p.rect_h = 5;
    p.step = 5;
    p.subset_size = 21;
    p.strain_window = 11;
    p.use_6x6_interpolator = false;

    ReferenceCache cache;
    cache.set_from_gray(ref, cv::Mat());
    std::vector<float> out(8 * 8, 0.0f);
    float metrics[Semper::pipeline::kMetricsLen];
    for (float &v : metrics) v = -1.0f;
    const int n = Semper::pipeline::run_full_field(cache, def, cv::Mat(), p, out.data(),
                                                   (int)out.size(), metrics,
                                                   Semper::pipeline::kMetricsLen, nullptr);
    std::printf("    rc %d  seeding %g lock %g  3:%g + 4:%g + 19:%g  vs  1:%g + 20:%g + 23:%g\n",
                n, metrics[16], metrics[21], metrics[3], metrics[4], metrics[19], metrics[1],
                metrics[20], metrics[23]);
    REQUIRE(n >= 0);
    // The route under test: no lock, no anchors, Path C seeded Path B.
    REQUIRE(metrics[21] == 0.0f);
    REQUIRE(metrics[16] == 0.0f);
    CHECK(metrics[19] == 0.0f);
    CHECK(metrics[4] > 0.0f);
    CHECK(metrics[3] + metrics[4] + metrics[19] == metrics[1] + metrics[20] + metrics[23]);
}

// ---------------------------------------------------------------------------
// Path C used to stake the whole solve on one node. The scene above, at a
// 15 px step, with the deformed frame blank where the centre node lands (the
// stretch is centred there, so it does not move): that node's search has
// nothing to match, and with no AKAZE lock the solve returned kSeedFailed
// though every other node of the row is fine. The next nearest are tried now.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, PathCTriesNextNodeWhenCentreFails) {
    const int w = 384, h = 256;
    const dictest::SmoothSpeckleField field(101, w, h);
    const cv::Mat ref = dictest::render_frame(field, w, h, dictest::identity_motion(), 0.0f, 7919u);
    const dictest::Motion m =
            dictest::stretch_rotate(0.45, 0.3, 0.0, dictest::Vec2{}, dictest::Vec2{192, 128});
    cv::Mat def = dictest::render_frame(field, w, h, m, 0.0f, 7919u + 150u);

    FullFieldParams p;
    p.rect_x = 132;
    p.rect_y = 128;
    p.rect_w = 8 * 15;
    p.rect_h = 15;
    p.step = 15;
    p.subset_size = 21;
    p.strain_window = 31;
    p.use_6x6_interpolator = false;
    // Path C's first candidate is grid node (4, 0): (192, 128).
    def(cv::Rect(182, 118, 21, 21)).setTo(128);

    ReferenceCache cache;
    cache.set_from_gray(ref, cv::Mat());
    std::vector<float> out(8 * 8, 0.0f);
    float metrics[Semper::pipeline::kMetricsLen];
    for (float &v : metrics) v = -1.0f;
    const int n = Semper::pipeline::run_full_field(cache, def, cv::Mat(), p, out.data(),
                                                   (int)out.size(), metrics,
                                                   Semper::pipeline::kMetricsLen, nullptr);
    std::printf("    rc %d  seeding %g lock %g  path B %g\n", n, metrics[16], metrics[21],
                metrics[4]);
    // The route under test: no lock, no anchors, so a Path C failure is final.
    REQUIRE(metrics[21] == 0.0f);
    CHECK(n != Semper::pipeline::kSeedFailed);
    CHECK(n >= 0);
    CHECK(metrics[16] == 0.0f);
    CHECK(metrics[4] > 0.0f);
}

// ---------------------------------------------------------------------------
// The publish gate. solve_anchor_seeds records outcomes and publishes nothing;
// publish_anchor_results applies them once the universal median test has run.
// Nothing in the fixtures below produces a blunder that clears kCorrAccept and
// then fails the median test -- that is a periodic-speckle failure and it is
// hard to stage deterministically -- so the gate itself is checked directly.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, PublishAnchorResultsRespectsMedianTest) {
    using Semper::pipeline::internal::AnchorResult;
    using Semper::pipeline::internal::GridPoint;
    using Semper::pipeline::internal::MeshSeedResult;
    using Semper::pipeline::internal::ResultGrid;
    using Semper::pipeline::internal::ThreadStats;
    using Semper::pipeline::internal::publish_anchor_results;

    auto make = [](int gx, int gy, bool output, bool kept, int tid) {
        AnchorResult a;
        a.gx = gx; a.gy = gy;
        a.rx = (float)gx; a.ry = (float)gy;
        a.u = 1.0f + (float)gx; a.v = 2.0f;
        a.corr = 0.05f;
        a.thread_id = tid;
        a.output = output;
        a.kept = kept;
        return a;
    };

    MeshSeedResult seeds;
    seeds.anchors = {
            make(1, 1, true,  true,  0),   // published
            make(2, 1, true,  false, 1),   // the blunder: cleared kCorrAccept,
                                           // rejected by the median test
            make(3, 1, false, true,  1),   // a vertex, but too loose to publish
            make(0, 0, false, false, 0),   // never converged: default slot
            make(9, 9, true,  true,  0),   // outside the grid entirely
    };

    ResultGrid grid(4, std::vector<GridPoint>(4));
    std::atomic<int> solved(0), order(0);
    std::vector<ThreadStats> stats(2);

    const int published = publish_anchor_results(seeds, solved, order, grid, stats);

    CHECK(published == 1);
    CHECK(solved.load() == 1);
    CHECK(order.load() == 1);   // one order number consumed, not five

    CHECK(grid[1][1].solved);
    CHECK_NEAR(grid[1][1].u, 2.0f, 1e-6f);
    CHECK(grid[1][1].mesh_assignment_type == Semper::pipeline::internal::kMeshAnchor);

    // The blunder must not reach the field: Path B would take it as a boundary
    // seed and flood fill from a wrong displacement.
    CHECK(!grid[1][2].solved);
    CHECK(!grid[1][3].solved);
    CHECK(!grid[0][0].solved);

    // Credit lands on the thread that ran the ICGN, not on whoever published.
    CHECK(stats[0].points_solved == 1);
    CHECK(stats[1].points_solved == 0);
}

// ---------------------------------------------------------------------------
// The median test itself, on a hand-made 4 x 3 lattice. Its blind spot was a
// vertex with fewer than three converged neighbours: nothing to take a median
// of, so it was kept on its ZNSSD alone. On the real PMMA 3-point bend
// (material_testing real_data_pmma_bending, frames 24-29) the translation
// lattice under the load nose kept exactly such anchors, ~20 px off at ZNSSD
// 0.17-0.30, and the mesh on them sent Path A to the wrong speckle: up to
// 3,861 of the 9,431 deflection-probe points were lost. Requiring the output
// gate instead is not enough: StretchSeedsWithoutBlunders has untested
// blunders at 0.095 and 0.150. An untested node is now never a vertex.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, MedianTestKeepsNoUntestedAnchor) {
    using Semper::pipeline::internal::AnchorResult;
    using Semper::pipeline::internal::mark_kept_anchors;

    const int LW = 4, LH = 3;
    std::vector<AnchorResult> anchors((size_t)LW * LH);   // default: never converged
    std::vector<float> res_u((size_t)LW * LH, 0.0f), res_v((size_t)LW * LH, 0.0f);
    auto set = [&](int li, int lj, bool output, float u) {
        AnchorResult &a = anchors[(size_t)lj * LW + li];
        a.vertex = true;
        a.output = output;
        a.kept = true;   // stale: the test must decide afresh
        res_u[(size_t)lj * LW + li] = u;
        res_v[(size_t)lj * LW + li] = 2.0f;
    };
    // Column 2 never converged except its bottom node.
    set(0, 0, true, 1.0f);  set(1, 0, true, 1.0f);                         set(3, 0, true, 1.0f);
    set(0, 1, true, 1.0f);  set(1, 1, false, 1.2f);                        set(3, 1, true, 1.0f);
    set(0, 2, true, 6.0f);  set(1, 2, true, 1.0f);  set(2, 2, true, 1.0f); set(3, 2, false, -18.0f);

    mark_kept_anchors(anchors, LW, LH, res_u, res_v);
    auto kept = [&](int li, int lj) { return anchors[(size_t)lj * LW + li].kept; };

    // Tested and consistent, including the loose vertex (1, 1): where the
    // median test can run, the looser vertex gate is still enough.
    CHECK(kept(0, 0));
    CHECK(kept(1, 0));
    CHECK(kept(0, 1));
    CHECK(kept(1, 1));
    CHECK(kept(1, 2));
    CHECK(kept(2, 2));
    // A blunder with three neighbours is rejected.
    CHECK(!kept(0, 2));
    // (3, 1) has the far-off (3, 2) among its three neighbours and is kept on
    // the other two.
    CHECK(kept(3, 1));
    // Untested (one neighbour), however good its ZNSSD: not kept.
    CHECK(!kept(3, 0));
    // Untested (two neighbours) and only a loose vertex: the PMMA case.
    CHECK(!kept(3, 2));
    // Never converged.
    CHECK(!kept(2, 0));
    CHECK(!kept(2, 1));
}

// ---------------------------------------------------------------------------
// Lattice sizing. No image, no solve -- plan_anchor_lattice is exposed so the
// stride arithmetic can be checked directly, because the failure it guards
// against is a cost blow-up on a geometry no fixture in this repo has.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AnchorLatticeSizing) {
    using Semper::pipeline::internal::AnchorLattice;
    using Semper::pipeline::internal::plan_anchor_lattice;
    namespace tuning = Semper::tuning;

    // Square ROI: the stride is the isotropic sqrt(nodes / target) and both
    // axes keep it. 64 x 64 = 4096 nodes, 4096 / 256 = 16, sqrt = 4. Each axis
    // carries indices 0, 4, ... 60 plus the appended last index 63 = 17, so
    // 289 anchors -- which is exactly the vertex count CandidateSweep reports.
    // This case pins the per-axis clamp as a no-op on square-ish ROIs.
    {
        const AnchorLattice lat = plan_anchor_lattice(64, 64);
        CHECK(lat.stride_x == 4);
        CHECK(lat.stride_y == 4);
        CHECK(lat.nx == 17);
        CHECK(lat.ny == 17);
        CHECK(lat.count() == 289);
    }

    // The regression this test exists for. A long thin ROI -- a beam, a weld
    // seam -- used to have the stride clamped from the *shorter* axis: 6 / 3 = 2
    // forced on both, giving a lattice of ~2000 nodes against a target of 256.
    // Clamping per axis keeps the short axis at its 3-node floor and pays for
    // it by coarsening the long one.
    {
        const AnchorLattice lat = plan_anchor_lattice(1000, 6);
        CHECK(lat.ny >= 3);   // the floor the old clamp was protecting
        CHECK(lat.nx >= 3);
        // Not 256 exactly: the short axis cannot go below 3, so the product
        // overshoots. It must stay the same order as the target, not 8x it.
        CHECK(lat.count() <= 2 * tuning::kAnchorTarget);
        CHECK(lat.stride_x > lat.stride_y);
    }

    // Same ROI on its side. Nothing in the planner should prefer an axis.
    {
        const AnchorLattice a = plan_anchor_lattice(1000, 6);
        const AnchorLattice b = plan_anchor_lattice(6, 1000);
        CHECK(b.stride_x == a.stride_y);
        CHECK(b.stride_y == a.stride_x);
        CHECK(b.count() == a.count());
    }

    // A grid too small to hold 3 lattice nodes per axis: the floor cannot be
    // met, and asking for it must not loop or return an empty lattice.
    {
        const AnchorLattice lat = plan_anchor_lattice(2, 2);
        CHECK(lat.stride_x == 1);
        CHECK(lat.stride_y == 1);
        CHECK(lat.count() == 4);
    }

    // Degenerate input: no grid, no lattice, no arithmetic on it.
    {
        CHECK(plan_anchor_lattice(0, 10).count() == 0);
        CHECK(plan_anchor_lattice(10, 0).count() == 0);
        CHECK(plan_anchor_lattice(-1, -1).count() == 0);
    }
}

#endif // DIC_HAVE_OPENCV
