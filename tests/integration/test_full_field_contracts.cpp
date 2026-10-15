// Host characterization for run_full_field — mirrors EnginePipelineSmokeTest
// contract block (degenerate ROI → -2, empty def → -3, stale cancel, undersized
// buffer). Compiled only when DIC_HAVE_OPENCV is set (same gate as DICe tests).
#include "framework/test_framework.h"

#include <cstdio>
#include <cstring>
#include "framework/synthetic.h"
#include "framework/synthetic_cv.h"

#include <semper/cancel.hpp>
#include <semper/strain.hpp>
#include <semper/pipeline.hpp>

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::clear_cancel;
using Semper::pipeline::kCancelled;
using Semper::pipeline::request_cancel;
using Semper::pipeline::run_full_field;

#if defined(DIC_HAVE_OPENCV)

namespace {

constexpr int W = 256;
constexpr int H = 256;
constexpr int STEP = 15;
constexpr int SUBSET = 21;
// 2 * STEP + 1 is the smallest window that puts more than one grid node inside
// the VSG circle. At the old value of 15 the window spanned exactly one node,
// the strain fit was rank-deficient everywhere, and every test below solved an
// empty field without noticing. run_full_field now rejects that pair outright
// (kBadStrainWindow), which is what turned the silent version into a failure.
constexpr int STRAIN_WIN = 2 * STEP + 1;

void make_pair(cv::Mat &ref_gray, cv::Mat &def_gray) {
    dictest::SpeckleField field(/*seed=*/11, W, H, /*blob_count=*/600);
    dictest::AffineDeformation def;
    def.u = 3.0f;
    def.v = 2.0f;
    def.cx = W / 2.0f;
    def.cy = H / 2.0f;
    const Semper::Image ref = dictest::make_reference_image(field, W, H);
    const Semper::Image deformed = dictest::make_deformed_image(field, W, H, def);
    ref_gray = dictest::gray8(ref);
    def_gray = dictest::gray8(deformed);
}

FullFieldParams params_for(int rect_w, int rect_h) {
    FullFieldParams p;
    p.rect_x = 0;
    p.rect_y = 0;
    p.rect_w = rect_w;
    p.rect_h = rect_h;
    p.step = STEP;
    p.subset_size = SUBSET;
    p.strain_window = STRAIN_WIN;
    p.use_6x6_interpolator = false;
    return p;
}

int call_solver(ReferenceCache &cache, const cv::Mat &def_gray,
                const FullFieldParams &params, int buffer_floats) {
    std::vector<float> out(std::max(1, buffer_floats), 0.f);
    float metrics[17] = {};
    metrics[16] = -1.f;
    return run_full_field(cache, def_gray, cv::Mat(), params,
                          out.data(), buffer_floats, metrics, 17, nullptr);
}

} // namespace

TEST_CASE(FullField, DegenerateRoi_ReturnsRoiError) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    // rectW < step ⇒ gridW == 0 ⇒ documented ROI error (-2).
    const int code = call_solver(cache, def_gray, params_for(STEP - 1, STEP - 1),
                                 8 * 4);
    CHECK(code == -2);
}

TEST_CASE(FullField, StrainWindowTooSmallForStep_IsRejected) {
    // The VSG plane fit needs 3 grid nodes inside the strain window. When the
    // window spans fewer, no point can ever survive the strain post-filter --
    // and until 0.3.0 that came back as 0 points with a *success* code, which a
    // caller cannot tell apart from a genuinely blank ROI. It is now -4.
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());

    FullFieldParams params = params_for(W, H);
    params.strain_window = STEP;   // one node: the centre and nothing else
    REQUIRE(Semper::StrainCalculator::vsg_window_node_count(params.step,
                                                            params.strain_window) < 3);
    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    std::vector<float> out((size_t)cap, 0.0f);
    float metrics[23] = {};
    CHECK(run_full_field(cache, def_gray, cv::Mat(), params, out.data(), cap,
                         metrics, 23, nullptr) == Semper::pipeline::kBadStrainWindow);

    // The guard must not be so eager that it rejects a working pair: the
    // smallest window the error message tells the caller to use has to pass.
    params.strain_window = 2 * params.step + 1;
    REQUIRE(Semper::StrainCalculator::vsg_window_node_count(params.step,
                                                            params.strain_window) >= 3);
    CHECK(run_full_field(cache, def_gray, cv::Mat(), params, out.data(), cap,
                         metrics, 23, nullptr) > 0);
}

TEST_CASE(FullField, DegenerateStep_ReturnsRoiError) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    // step == 0 would divide-by-zero at `rect_w / step`. In a Release/NDEBUG
    // build the SEMPER_ASSERT is compiled out, so without the guard this SIGFPEs;
    // the guard returns the -2 ROI code instead.
    FullFieldParams p = params_for(W, H);
    p.step = 0;
    const int code = call_solver(cache, def_gray, p, 8 * 4);
    CHECK(code == -2);
}

TEST_CASE(FullField, EmptyDeformed_ReturnsInitError) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    const int code = call_solver(cache, cv::Mat(), params_for(W, H), 8 * 4);
    CHECK(code == -3);
}

TEST_CASE(FullField, StaleCancelDoesNotAbortNextSolve) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    request_cancel(); // stale — run_full_field must clear on entry
    const int n = call_solver(cache, def_gray, params_for(W, H),
                              8 * ((W / STEP) * (H / STEP) + 8));
    clear_cancel();
    // Contract: must not return the cancel sentinel. Packed-point count may be
    // zero on this analytic field after the strain post-filter; field quality
    // is covered by Engine.* and emulator smoke, not this cancel pin.
    CHECK(n != kCancelled);
    CHECK(n >= 0);
}

TEST_CASE(FullField, UndersizedBuffer_TruncatesWithoutOverflow) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    // Room for only 4 packed points (8 floats each).
    const int n = call_solver(cache, def_gray, params_for(W, H), 8 * 4);
    CHECK(n >= 0);
    CHECK(n <= 4);
}

TEST_CASE(FullField, MetricsLayout_Contract) {
    // The metrics array's slot layout is Frozen (docs/CONTRACT.md §A.4) but only
    // metrics[0] was ever checked. Pin the slot invariants a downstream telemetry
    // reader relies on. Runs a real solve on the 256x256 speckle pair.
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());

    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    std::vector<float> out(cap, 0.f);
    float metrics[23];
    for (int i = 0; i < 23; ++i) metrics[i] = -12345.f;

    const int n = run_full_field(cache, def_gray, cv::Mat(), params_for(W, H),
                                 out.data(), cap, metrics, 23, nullptr);
    REQUIRE(n >= 0);

    // Counts: attempted >= solved >= 0, rejected is the exact complement.
    CHECK(metrics[0] >= metrics[1]);            // attempted >= solved
    CHECK(metrics[1] >= 0.f);                   // solved >= 0
    CHECK(metrics[2] == metrics[0] - metrics[1]); // rejected identity
    // Convergence % is a bounded ratio.
    CHECK(metrics[15] >= 0.f);
    CHECK(metrics[15] <= 100.f);
    // Seed-quality flag is one of the three documented values.
    CHECK((metrics[16] == 0.f || metrics[16] == 1.f || metrics[16] == 2.f));
    // Seeding health (slots 21/22): the phase lock is a flag, coverage is a
    // fraction of the ROI. On this synthetic pure-translation pair both should
    // be healthy -- if the lock ever fails here, the seeder broke.
    CHECK((metrics[21] == 0.f || metrics[21] == 1.f));
    CHECK(metrics[22] >= 0.f);
    CHECK(metrics[22] <= 1.f);
    CHECK(metrics[21] == 1.f);
    CHECK(metrics[22] > 0.25f);
}

TEST_CASE(FullField, MetricsLen16_LeavesSlot16Untouched) {
    // metrics_len == 16 is the documented minimum; the writer must fill 0..15 and
    // never touch slot 16 (that would be a write past a 16-float caller buffer).
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());

    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    std::vector<float> out(cap, 0.f);
    const float S = -98765.f;   // "not written" sentinel
    float metrics[17];
    for (int i = 0; i < 17; ++i) metrics[i] = S;

    const int n = run_full_field(cache, def_gray, cv::Mat(), params_for(W, H),
                                 out.data(), cap, metrics, 16, nullptr);
    REQUIRE(n >= 0);
    CHECK(metrics[0] != S);     // slot 0 written
    CHECK(metrics[15] != S);    // slot 15 written
    CHECK(metrics[16] == S);    // slot 16 must stay the sentinel
}

TEST_CASE(FullField, NullMetrics_DoesNotCrash) {
    // metrics == nullptr is legal (the writer guards on it); the solve must still
    // run and return a valid point count.
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());

    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    std::vector<float> out(cap, 0.f);
    const int n = run_full_field(cache, def_gray, cv::Mat(), params_for(W, H),
                                 out.data(), cap, nullptr, 17, nullptr);
    CHECK(n >= 0);
}

// ---------------------------------------------------------------------------
// Full-field repeat determinism.
//
// The positive guarantee behind the sanitizer work: docs/TESTING.md promises
// that the same binary on the same device with the same input is bit-identical.
// A data race between the parallel workers would break exactly that, so this is
// the property that actually guards against one, and unlike a sanitizer run it
// needs no special toolchain.
//
// It covers all four concurrent regions in one solve — the Hessian pre-pass,
// the anchor lattice and Path A (OpenMP), plus Path B's std::thread workers.
//
// Thread count is deliberately not varied: run_full_field pins every OpenMP
// region with `num_threads(safe_cores)`, which overrides omp_set_num_threads
// and OMP_NUM_THREADS, so a test cannot change it without a production hook.
// Cross-thread-count invariance is therefore not a contract this engine makes;
// Path B's reliability-guided propagation is order-dependent by construction.
// ---------------------------------------------------------------------------
TEST_CASE(FullField, RepeatSolve_BitIdenticalField) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);

    // A finer step than the shared default, to get a denser field under test.
    // strain_window must stay >= 2 * step or run_full_field rejects the pair.
    FullFieldParams params = params_for(W, H);
    params.step = 5;
    params.strain_window = 31;

    const int gridW = params.rect_w / params.step;
    const int gridH = params.rect_h / params.step;
    const int cap = gridW * gridH * 8;

    auto solve = [&](std::vector<float> &out) {
        ReferenceCache cache;
        cache.set_from_gray(ref_gray, cv::Mat());
        out.assign((size_t)cap, 0.0f);
        float metrics[17] = {};
        return run_full_field(cache, def_gray, cv::Mat(), params,
                              out.data(), cap, metrics, 17, nullptr);
    };

    std::vector<float> reference;
    const int n_ref = solve(reference);
    REQUIRE(n_ref > 0);

    for (int rep = 1; rep <= 4; ++rep) {
        std::vector<float> field;
        const int n = solve(field);
        CHECK(n == n_ref);
        if (n != n_ref) continue;

        // Bit-identical, not near-equal: a race would perturb some point, and an
        // epsilon comparison would let a small perturbation through.
        size_t first_diff = (size_t)-1;
        for (size_t i = 0; i < (size_t)n * 8; ++i) {
            if (std::memcmp(&reference[i], &field[i], sizeof(float)) != 0) {
                first_diff = i;
                break;
            }
        }
        if (first_diff != (size_t)-1) {
            std::printf("    repeat %d: first difference at float %zu "
                        "(point %zu, field %zu): %.9g vs %.9g\n",
                        rep, first_diff, first_diff / 8, first_diff % 8,
                        (double)reference[first_diff], (double)field[first_diff]);
        }
        CHECK(first_diff == (size_t)-1);
    }
}

// ---------------------------------------------------------------------------
// Point-accounting contract for the metrics array.
//
// docs/CONTRACT.md freezes slot 3 as "solved via mesh" (Path A) and slot 4 as
// "solved via flood fill" (Path B). The anchor lattice added a third producer
// of solved points, so those two no longer account for slot 1 on their own and
// slot 19 carries the remainder. The failure this guards against is not a crash
// but a quiet one: the anchor phase filled a ThreadStats vector that was never
// passed to aggregate_thread_stats, so its points, iterations and rescue counts
// were dropped and every "total" in the report was short by a whole IC-GN phase.
// ---------------------------------------------------------------------------
TEST_CASE(FullField, MetricsPointCountsAccountForEverySolvedPoint) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);

    FullFieldParams params = params_for(W, H);
    params.step = 5;
    params.strain_window = 31;

    const int gridW = params.rect_w / params.step;
    const int gridH = params.rect_h / params.step;
    const int cap = gridW * gridH * 8;

    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    std::vector<float> out((size_t)cap, 0.0f);
    float metrics[23] = {};
    const int n = run_full_field(cache, def_gray, cv::Mat(), params,
                                 out.data(), cap, metrics, 23, nullptr);
    REQUIRE(n > 0);

    const int total_solved = (int)metrics[1];
    const int via_mesh     = (int)metrics[3];
    const int via_flood    = (int)metrics[4];
    const int via_anchors  = (int)metrics[19];
    const int post_dropped = (int)metrics[20];

    std::printf("    anchors=%d + pathA=%d + pathB=%d = %d  vs  solved=%d + dropped=%d = %d\n",
                via_anchors, via_mesh, via_flood,
                via_anchors + via_mesh + via_flood,
                total_solved, post_dropped, total_solved + post_dropped);

    // The anchor lattice solves grid points outright, so this fails if its
    // bucket is dropped again — which is exactly how the defect presented.
    CHECK(via_anchors > 0);

    // Every point some path solved is either in the output or was discarded by
    // the strain post-filter. Nothing may fall out of the accounting in between.
    CHECK(via_anchors + via_mesh + via_flood == total_solved + post_dropped);

    // Mean ICGN iterations divides total iterations by valid_count. Dropping a
    // phase's iterations while keeping its points understates it; a solve that
    // converges at all cannot average below one iteration per point.
    CHECK(metrics[8] >= 1.0f);

    // Slot 18 is labelled total ICGN time and slot 17 total simplex time; both
    // must cover the anchor phase, which runs before either path.
    CHECK(metrics[18] > 0.0f);
}

// ---------------------------------------------------------------------------
// Reference precompute cache. A batch against one reference reuses the grid
// scan, the Hessian pool and the reference side of the phase correlation.
// Reuse must never change a result: every solve below is compared bit for bit
// against a solve on a fresh cache, which recomputes everything.
// ---------------------------------------------------------------------------
namespace {

struct SolveOut {
    int rc = 0;
    std::vector<float> out;
};

SolveOut solve_on(ReferenceCache &cache, const cv::Mat &def_gray, const cv::Mat &mask,
                  const FullFieldParams &p) {
    SolveOut s;
    const int cap = (p.rect_w / p.step) * (p.rect_h / p.step) * 8;
    s.out.assign((size_t)std::max(cap, 1), 0.0f);
    float metrics[23] = {};
    s.rc = run_full_field(cache, def_gray, mask, p, s.out.data(), cap, metrics, 23, nullptr);
    return s;
}

SolveOut solve_fresh(const cv::Mat &ref_gray, const cv::Mat &def_gray, const cv::Mat &mask,
                     const FullFieldParams &p) {
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, mask);
    return solve_on(cache, def_gray, mask, p);
}

bool bit_equal(const SolveOut &a, const SolveOut &b) {
    if (a.rc != b.rc) return false;
    if (a.rc <= 0) return true;
    return std::memcmp(a.out.data(), b.out.data(), (size_t)a.rc * 8 * sizeof(float)) == 0;
}

// Frames of a slowly growing affine deformation of one speckle field.
std::vector<cv::Mat> affine_frames(cv::Mat &ref_gray, int frames) {
    dictest::SpeckleField field(/*seed=*/23, W, H, /*blob_count=*/600);
    ref_gray = dictest::gray8(dictest::make_reference_image(field, W, H));
    std::vector<cv::Mat> out;
    for (int f = 1; f <= frames; ++f) {
        dictest::AffineDeformation d;
        d.u = 1.3f * f;
        d.v = -0.7f * f;
        d.ux = 0.006f * f;
        d.vy = -0.002f * f;
        d.uy = 0.003f * f;
        d.vx = -0.003f * f;
        d.cx = W / 2.0f;
        d.cy = H / 2.0f;
        out.push_back(dictest::gray8(dictest::make_deformed_image(field, W, H, d)));
    }
    return out;
}

cv::Mat ring_mask() {
    cv::Mat m(H, W, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float r2 = (x - W / 2.0f) * (x - W / 2.0f) + (y - H / 2.0f) * (y - H / 2.0f);
            if (r2 < 110.0f * 110.0f && r2 > 20.0f * 20.0f) m.at<uchar>(y, x) = 255;
        }
    return m;
}

} // namespace

TEST_CASE(FullField, RefCache_SequenceOnOneCache_BitIdenticalToFresh) {
    cv::Mat ref_gray;
    const std::vector<cv::Mat> frames = affine_frames(ref_gray, 4);

    // 240 is a DFT-optimal size: cv::phaseCorrelate windows such inputs in
    // place, which is exactly what a cached reference crop must survive.
    FullFieldParams p = params_for(W, H);
    p.rect_x = 8; p.rect_y = 8; p.rect_w = 240; p.rect_h = 240;
    p.step = 8; p.strain_window = 2 * 8 + 1;
    p.use_6x6_interpolator = true;

    for (int with_mask = 0; with_mask <= 1; ++with_mask) {
        const cv::Mat mask = with_mask ? ring_mask() : cv::Mat();
        ReferenceCache shared;
        shared.set_from_gray(ref_gray, mask);
        for (size_t f = 0; f < frames.size(); ++f) {
            const SolveOut fresh = solve_fresh(ref_gray, frames[f], mask, p);
            REQUIRE(fresh.rc > 0);
            // Twice on the shared cache: a hit after a previous frame, then a
            // hit on the same frame.
            for (int rep = 0; rep < 2; ++rep) {
                const SolveOut hit = solve_on(shared, frames[f], mask, p);
                CHECK(bit_equal(hit, fresh));
            }
        }
    }
}

TEST_CASE(FullField, RefCache_InvalidatesOnParamsOrMask) {
    cv::Mat ref_gray;
    const std::vector<cv::Mat> frames = affine_frames(ref_gray, 1);
    const cv::Mat &def = frames[0];

    FullFieldParams a = params_for(W, H);
    a.step = 8; a.strain_window = 17;
    FullFieldParams b = a;
    b.rect_x = 16; b.rect_w = W - 32;       // different grid origin and size
    FullFieldParams c = a;
    c.subset_size = 25;                      // different Hessian per node
    const cv::Mat ring = ring_mask();

    ReferenceCache shared;
    shared.set_from_gray(ref_gray, cv::Mat());
    CHECK(bit_equal(solve_on(shared, def, cv::Mat(), a), solve_fresh(ref_gray, def, cv::Mat(), a)));
    CHECK(bit_equal(solve_on(shared, def, cv::Mat(), b), solve_fresh(ref_gray, def, cv::Mat(), b)));
    CHECK(bit_equal(solve_on(shared, def, cv::Mat(), c), solve_fresh(ref_gray, def, cv::Mat(), c)));

    // A mask passed to the solve (not the reference) changes the grid scan.
    ReferenceCache plain;
    plain.set_from_gray(ref_gray, cv::Mat());
    const SolveOut no_mask = solve_on(plain, def, cv::Mat(), a);
    const SolveOut masked = solve_on(plain, def, ring, a);
    ReferenceCache plain_fresh;
    plain_fresh.set_from_gray(ref_gray, cv::Mat());
    CHECK(bit_equal(masked, solve_on(plain_fresh, def, ring, a)));
    CHECK(masked.rc < no_mask.rc);
    // And back again.
    CHECK(bit_equal(solve_on(plain, def, cv::Mat(), a), no_mask));

    // A new reference drops everything.
    shared.set_from_gray(frames[0], cv::Mat());
    CHECK(bit_equal(solve_on(shared, def, cv::Mat(), a), solve_fresh(frames[0], def, cv::Mat(), a)));
}

TEST_CASE(FullField, RefCache_CancelledSolveDoesNotPoisonCache) {
    cv::Mat ref_gray;
    const std::vector<cv::Mat> frames = affine_frames(ref_gray, 1);
    FullFieldParams p = params_for(W, H);
    p.step = 8; p.strain_window = 17;

    ReferenceCache shared;
    shared.set_from_gray(ref_gray, cv::Mat());
    Semper::pipeline::CancelToken token;
    std::vector<float> out((size_t)(W / 8) * (H / 8) * 8, 0.0f);
    float metrics[23] = {};
    // Cancel from the first progress tick: the pre-pass may be cut short.
    const int rc = run_full_field(shared, frames[0], cv::Mat(), p, out.data(), (int)out.size(),
                                  metrics, 23, token, [&](int) { token.request(); });
    CHECK(rc == kCancelled || rc > 0);
    CHECK(bit_equal(solve_on(shared, frames[0], cv::Mat(), p),
                    solve_fresh(ref_gray, frames[0], cv::Mat(), p)));
}

#else

TEST_CASE(FullField, OpenCvRequired_SkippedWithoutOpenCV) {
    // Suite still registers when OpenCV is absent so the filter surface is stable.
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
