// =====================================================================
// SEEDING CONTRACTS — the always-on half of the seeding suite
//
// The sweeps in test_seeding_bench.cpp measure the anchor-lattice front-end
// and assert nothing; they are gated behind SEMPER_RUN_SEEDBENCH, and three
// of them behind an external DICe checkout. What CI defends on every run is
// here: the phase-correlation sign convention, the log-polar rotation/scale
// estimate, the rotation and quality gates, the similarity fallback, the
// median-test publication rule, and the lattice sizing. Harness:
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
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AnchorSeedingRotationGate) {
    const struct { const char *name; dictest::AffineDeformation def; } gates[] = {
        {"R_rot_5deg", rotation(5.0)},
        {"R_rot_15deg", rotation(15.0)},
        {"RST_10deg_1.05_20px", similarity(10.0, 1.05, 20.0f, -12.0f)},
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
        std::printf("    %-20s lock=%s sim=%d rot=%+.3f x%.4f q=%d cov=%.3f "
                    "vtx=%.4f px kept %d/%d meshfr=%.3f\n",
                    g.name, row.phase_locked ? "yes" : "NO", row.similarity ? 1 : 0,
                    row.lp_rot_deg, row.lp_scale, row.quality, (double)row.coverage,
                    row.vtx_med_du, row.anchors_accepted, row.anchors_attempted,
                    row.mesh_frac);

        CHECK(row.phase_locked);
        CHECK(row.similarity);
        CHECK(row.quality == 2);
        CHECK(row.coverage >= 0.90f);
        CHECK(row.mesh_frac >= 0.95);
        REQUIRE(row.vtx_med_du >= 0.0);
        CHECK(row.vtx_med_du <= 0.05);
    }
}

// ---------------------------------------------------------------------------
// Turns of 90 deg and beyond. The magnitude spectrum is point-symmetric, so the
// log-polar peak gives theta only modulo 180 deg: 100 deg reads as -80, 135 as
// -45, 179 as -1, and -90 as +90. Before L0 tried theta + 180 as well, each of
// these seeded from the wrong half-turn and the solve returned kSeedFailed
// (-5). The pixel-exact quarter turn (cv::rotate) puts the log-polar peak
// exactly on the wrap of the angle axis, which a range check excluding +-90
// threw away: the solve fell back to translation-only and returned no points.
//
// A node counts only if the solve returns it within 0.5 px of the truth, out
// of the nodes the same solve returns at zero motion (the strain window drops
// the outer ring), so a dropped node counts against the fraction.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, AnchorSeedingHalfTurn) {
    struct Turn { double deg; bool exact; };
    const Turn turns[] = {
        {90.0, true}, {90.0, false}, {-90.0, false}, {100.0, false},
        {135.0, false}, {179.0, false}, {-120.0, false},
    };

    dictest::SpeckleField field(/*seed=*/11, W, H, BLOBS);
    const Image ref = dictest::make_reference_image(field, W, H);
    const cv::Mat ref_gray = dictest::gray8(ref, 0.0f, 101);
    const FullFieldParams p = big_motion_params();
    const cv::Rect roi(p.rect_x, p.rect_y, p.rect_w, p.rect_h);
    const auto pre = Semper::seeding::prepare_phase_correlate_ref(ref_gray, roi);
    REQUIRE(pre.lp_usable());

    const int gridW = p.rect_w / p.step, gridH = p.rect_h / p.step;
    std::vector<float> out((size_t)gridW * gridH * 8, 0.0f);
    auto solve = [&](const cv::Mat &def_gray) {
        ReferenceCache cache;
        cv::Mat no_mask;
        cache.set_from_gray(ref_gray, no_mask);
        float metrics[19] = {};
        return Semper::pipeline::run_full_field(cache, def_gray, no_mask, p, out.data(),
                                                (int)out.size(), metrics, 19, nullptr);
    };
    const int nodes = solve(ref_gray);
    REQUIRE(nodes > 0);

    for (const Turn &c : turns) {
        dictest::AffineDeformation d = rotation(c.deg);
        cv::Mat def_gray;
        if (c.exact) {
            // cv::rotate turns about the centre of the pixel grid.
            d.cx = d.cy = 0.5f * (float)(W - 1);
            cv::rotate(ref_gray, def_gray, cv::ROTATE_90_CLOCKWISE);
        } else {
            def_gray = dictest::gray8(dictest::make_deformed_image(field, W, H, d), 0.0f, 202);
        }
        const TruthFn truth = affine_truth(d);

        // L0: the half-turn is resolved, and the centre lands where it should.
        Semper::seeding::SimilarityShift gm;
        Semper::seeding::phase_correlate_similarity(pre, def_gray, gm);
        const double got = gm.theta_rad * 180.0 / CV_PI;
        const Truth tc = truth((float)gm.cx, (float)gm.cy);

        const int rc = solve(def_gray);
        int good = 0;
        for (int i = 0; i < rc; ++i) {
            const float *o = &out[(size_t)i * 8];
            const Truth t = truth(o[0], o[1]);
            if (std::hypot(o[2] - t.u, o[3] - t.v) <= 0.5f) good++;
        }
        const double frac = (double)good / (double)nodes;
        std::printf("    %+7.1f deg%s -> L0 %+9.4f deg (lp resp %.3f, sim=%d), centre "
                    "(%.3f, %.3f) truth (%.3f, %.3f); rc=%d, %d/%d within 0.5 px\n",
                    c.deg, c.exact ? " exact" : "      ", got, gm.lp_response,
                    gm.similarity ? 1 : 0, gm.u, gm.v, (double)tc.u, (double)tc.v, rc, good,
                    nodes);

        REQUIRE(gm.lp_ok);
        CHECK(gm.similarity);
        CHECK(gm.locked);
        CHECK(std::abs(std::remainder(got - c.deg, 360.0)) <= 0.15);
        CHECK_NEAR(gm.u, tc.u, 0.25);
        CHECK_NEAR(gm.v, tc.v, 0.25);
        REQUIRE(rc > 0);
        CHECK(frac >= 0.95);
    }
}

// ---------------------------------------------------------------------------
// The similarity's failure mode, and its fallback. The log-polar estimate is
// a rotation plus an isotropic scale; a strong uniaxial stretch is neither,
// and IncBench found frames where it reported one anyway and the corrected
// translation peak still beat a translation peak the stretch had already
// weakened. This is the noise-free one: IncBench S5 ("stretch to 60%"),
// frame 150, a 45% stretch with Poisson 0.3 on a 110 px ROI, which the
// estimate reads as about -5.8 deg x 0.918. The lattice that seeds keeps
// 123 of 484 anchors and publishes 114 of them; the field built on that
// measured 0.215 px rms in IncBench against 0.014 px translation-only. The
// fallback must hand this frame back to the translation-only lattice, which
// keeps fewer anchors (86) -- so a rule that kept the larger lattice would
// pick the wrong one.
//
// What the case does not fix: at 45% stretch the edge anchors' true
// displacement is ~25 px, and from a zero-gradient seed some stick near zero
// and still clear the loose vertex gate, in either lattice (p95 vertex error
// ~25.7 px both ways). That is the translation-only seeder's behaviour too,
// so it is reported here and not asserted.
// ---------------------------------------------------------------------------
TEST_CASE(SeedBench, SimilarityFallsBackUnderStretch) {
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
    std::printf("    lp %+.3f deg x%.4f: sim=%d rejected=%d kept %d/%d q=%d cov=%.3f "
                "vtx med %.4f p95 %.4f px\n",
                row.lp_rot_deg, row.lp_scale, row.similarity ? 1 : 0,
                row.similarity_rejected ? 1 : 0, row.anchors_accepted,
                row.anchors_attempted, row.quality, (double)row.coverage, row.vtx_med_du,
                row.vtx_p95_du);

    // The estimate must still be the misfire this case was chosen for, or the
    // case no longer exercises the fallback.
    REQUIRE(std::abs(row.lp_rot_deg) > 2.0);
    CHECK(row.similarity_rejected);
    CHECK(!row.similarity);
    CHECK(row.phase_locked);

    // The outcome that matters is the field: 0.215 px with the similarity
    // lattice, 0.014 px translation-only. Bounded between the two.
    measure_full_field(row, truth, ref, def, p);
    std::printf("    field: %d points, rms %.4f px\n", row.ff_rc, row.ff_rms_u);
    REQUIRE(row.ff_rc > 0);
    CHECK(row.ff_rms_u <= 0.05);
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
//                    succeeds if the L0 phase-correlation lock is real
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
        // measurement because what is being asserted is that the phase lock
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
                    g.name, row.phase_locked ? "yes" : "NO", row.quality,
                    (double) row.coverage, row.vtx_med_du, row.mesh_med_dux,
                    row.mesh_med_du, row.mesh_frac);

        // A lost phase lock is the single failure that explains most of the
        // others, so check it first and by itself.
        CHECK(row.phase_locked);
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
