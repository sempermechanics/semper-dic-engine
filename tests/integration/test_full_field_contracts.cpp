// Host characterization for run_full_field — mirrors EnginePipelineSmokeTest
// contract block (degenerate ROI → -2, empty def → -3, stale cancel, undersized
// buffer). Compiled only when DIC_HAVE_OPENCV is set (same gate as DICe tests).
#include "framework/test_framework.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include "framework/synthetic.h"
#include "framework/synthetic_cv.h"

#include <semper/cancel.hpp>
#include <semper/strain.hpp>
#include <semper/pipeline.hpp>
#include <semper/solver.hpp>
#include <semper/subset.hpp>
#include <semper/tuning.hpp>
#include "pipeline/full_field_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
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

// A strain window far larger than the grid used to be accepted and sized
// the VSG loops: vsg_window_node_count walks (window / step)^2 offsets, so
// strain_window = INT_MAX spun for hours inside the -4 check, and a window
// merely much larger than the grid cost window^2 per node in the strain fit,
// only to reject every node (once the radius passes the grid's short side by
// a step, no node can have the 90% of its window on the grid the fit needs).
// Beyond 2 * (min(gridW, gridH) + 1) * step it is an argument error now.
TEST_CASE(FullField, HugeStrainWindow_ReturnsInitErrorPromptly) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    const FullFieldParams base = params_for(W, H);
    const int grid_min = std::min(base.rect_w / base.step, base.rect_h / base.step);
    const int max_window = 2 * (grid_min + 1) * base.step;

    // Run on a worker so a regression fails the test instead of hanging it.
    auto solve_within = [&](int strain_window, int seconds, int &rc) {
        auto done = std::make_shared<std::atomic<bool>>(false);
        auto result = std::make_shared<std::atomic<int>>(0);
        std::thread([=]() {
            ReferenceCache cache;
            cache.set_from_gray(ref_gray, cv::Mat());
            FullFieldParams p = base;
            p.strain_window = strain_window;
            const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
            std::vector<float> out((size_t)cap, 0.0f);
            result->store(run_full_field(cache, def_gray, cv::Mat(), p, out.data(), cap,
                                         nullptr, 0, nullptr));
            done->store(true);
        }).detach();
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (!done->load() && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        rc = result->load();
        return done->load();
    };

    int rc = 0;
    for (int window : {std::numeric_limits<int>::max(), 1 << 20, max_window + 1}) {
        const bool returned = solve_within(window, 20, rc);
        if (!returned) {
            std::printf("    strain_window=%d still running after 20 s\n", window);
            std::fflush(stdout);
            // The worker cannot be stopped or joined; end the run here.
            std::_Exit(1);
        }
        CHECK(rc == -3);
    }
    // The largest accepted window still solves (and, at that size, the strain
    // fit legitimately keeps few or no points).
    REQUIRE(solve_within(max_window, 60, rc));
    CHECK(rc >= 0);
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

TEST_CASE(FullField, EvenOrZeroSubsetSize_ReturnsInitError) {
    // The subset is built symmetrically around its centre pixel, so it holds
    // (2 * (dim / 2) + 1)^2 pixels. For an even dim that is (dim + 1)^2, and
    // the precompute wrote 441 values into the 400-float buffers of a subset
    // of 20 (a heap overflow under ASan, silent corruption without it); for 0
    // it wrote into empty vectors. Nothing below the C ABI checked it.
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    for (int subset : {20, 0, -21, 1, 2}) {
        FullFieldParams p = params_for(W, H);
        p.subset_size = subset;
        CHECK(call_solver(cache, def_gray, p, cap) == -3);
    }
    // The smallest subset that is accepted.
    FullFieldParams p = params_for(W, H);
    p.subset_size = 3;
    CHECK(call_solver(cache, def_gray, p, cap) >= 0);
}

TEST_CASE(FullField, HugeRect_ClippedToImage) {
    // The rect was never intersected with the image: rect_w = INT_MAX made a
    // grid of INT_MAX / step columns, and the ResultGrid allocation threw
    // std::bad_alloc straight through the C and JNI entry points. The rect is
    // now clipped to the image without moving the grid lattice, so any rect
    // that covers the image solves exactly the image-sized rect.
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    auto solve = [&](const FullFieldParams &p, std::vector<float> &out) {
        ReferenceCache cache;
        cache.set_from_gray(ref_gray, cv::Mat());
        out.assign((size_t)cap, 0.0f);
        float metrics[23] = {};
        return run_full_field(cache, def_gray, cv::Mat(), p, out.data(), cap,
                              metrics, 23, nullptr);
    };
    std::vector<float> want, got;
    const int n_want = solve(params_for(W, H), want);
    REQUIRE(n_want > 0);

    FullFieldParams huge = params_for(W, H);
    huge.rect_w = std::numeric_limits<int>::max();
    huge.rect_h = std::numeric_limits<int>::max();
    CHECK(solve(huge, got) == n_want);
    CHECK(std::memcmp(want.data(), got.data(), (size_t)n_want * 8 * sizeof(float)) == 0);

    // Hanging off the top-left by whole steps: same lattice, same field.
    FullFieldParams shifted = params_for(W + 3 * STEP, H + 2 * STEP);
    shifted.rect_x = -3 * STEP;
    shifted.rect_y = -2 * STEP;
    CHECK(solve(shifted, got) == n_want);
    CHECK(std::memcmp(want.data(), got.data(), (size_t)n_want * 8 * sizeof(float)) == 0);
}

TEST_CASE(FullField, RectOffImage_ReturnsRoiError) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    const int cap = 8 * 64;

    FullFieldParams right = params_for(W, H);
    right.rect_x = W + 10;
    CHECK(call_solver(cache, def_gray, right, cap) == -2);

    FullFieldParams above = params_for(W, H);
    above.rect_y = -2 * H;
    CHECK(call_solver(cache, def_gray, above, cap) == -2);

    // rect_x + rect_w overflows int.
    FullFieldParams overflow = params_for(std::numeric_limits<int>::max(), H);
    overflow.rect_x = std::numeric_limits<int>::max() - 5;
    CHECK(call_solver(cache, def_gray, overflow, cap) == -2);

    // On the image, but narrower than one step once clipped.
    FullFieldParams sliver = params_for(W, H);
    sliver.rect_x = W - STEP / 2;
    CHECK(call_solver(cache, def_gray, sliver, cap) == -2);
}

// The deformed image was never checked against the reference: a frame of
// another size skipped phase correlation (it checks) and was then correlated
// pixel by pixel anyway, returning a plausible field of nonsense; a frame of
// another type was read as bytes. Both are argument errors now.
TEST_CASE(FullField, DeformedSizeMismatch_ReturnsInitError) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    CHECK(call_solver(cache, def_gray(cv::Rect(0, 0, W - 16, H)).clone(), params_for(W, H), cap) == -3);
    cv::Mat bigger;
    cv::copyMakeBorder(def_gray, bigger, 0, 20, 0, 20, cv::BORDER_REFLECT);
    CHECK(call_solver(cache, bigger, params_for(W, H), cap) == -3);
    CHECK(call_solver(cache, def_gray, params_for(W, H), cap) > 0);
}

TEST_CASE(FullField, DeformedWrongType_ReturnsInitError) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    cv::Mat f32, bgr;
    def_gray.convertTo(f32, CV_32F);
    cv::cvtColor(def_gray, bgr, cv::COLOR_GRAY2BGR);
    CHECK(call_solver(cache, f32, params_for(W, H), cap) == -3);
    CHECK(call_solver(cache, bgr, params_for(W, H), cap) == -3);
    // The ROI mask is read as bytes too.
    std::vector<float> out((size_t)cap, 0.0f);
    cv::Mat mask_f(H, W, CV_32FC1, cv::Scalar(255.0f));
    CHECK(run_full_field(cache, def_gray, mask_f, params_for(W, H), out.data(), cap,
                         nullptr, 0, nullptr) == -3);
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
    // Seeding health (slots 21/22): the global lock is a flag, coverage is a
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

// The median test rejects converged-but-wrong points before packing. Those
// were counted by their solving path (slots 3/4/19) and then appeared in
// neither slot 1 nor slot 20, so the documented identity failed whenever it
// fired. Slot 23 (v0.5.0) counts them. Here a 44 px block of the deformed
// image moves 4 px further than the rest: its inner nodes converge cleanly to
// the block's motion and the median test peels them off as a cluster.
TEST_CASE(FullField, MetricsIdentityHoldsWithOutliers) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    cv::Mat block = def_gray(cv::Rect(100, 100, 44, 44)).clone();
    block.copyTo(def_gray(cv::Rect(104, 100, 44, 44)));

    FullFieldParams params = params_for(W, H);
    params.step = 5;
    params.strain_window = 31;
    const int cap = (W / 5) * (H / 5) * 8;
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    std::vector<float> out((size_t)cap, 0.0f);
    float metrics[Semper::pipeline::kMetricsLen];
    for (float &m : metrics) m = -1.0f;
    const int n = run_full_field(cache, def_gray, cv::Mat(), params, out.data(), cap,
                                 metrics, Semper::pipeline::kMetricsLen, nullptr);
    REQUIRE(n > 0);
    std::printf("    3:%g + 4:%g + 19:%g  vs  1:%g + 20:%g + 23:%g\n", metrics[3], metrics[4],
                metrics[19], metrics[1], metrics[20], metrics[23]);
    CHECK(metrics[23] > 0.0f);
    CHECK(metrics[3] + metrics[4] + metrics[19] == metrics[1] + metrics[20] + metrics[23]);
}

// Which thread the progress callback runs on (CONTRACT.md, ProgressCallback):
// ticks from the solve's own thread, the final 100 on the caller's, after the
// ticker has stopped. semper_c.h used to promise the caller's thread for all.
TEST_CASE(FullField, ProgressFinalTickOnCallingThread) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    FullFieldParams params = params_for(W, H);
    params.step = 5;
    params.strain_window = 31;
    const int cap = (W / 5) * (H / 5) * 8;
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    std::vector<float> out((size_t)cap, 0.0f);

    struct Call { int pct; std::thread::id tid; };
    std::vector<Call> calls;
    std::mutex calls_mutex;
    std::atomic<int> in_callback(0);
    bool overlapped = false;
    const int n = run_full_field(cache, def_gray, cv::Mat(), params, out.data(), cap,
                                 nullptr, 0, [&](int pct) {
        if (in_callback.fetch_add(1) != 0) overlapped = true;
        {
            std::lock_guard<std::mutex> lk(calls_mutex);
            calls.push_back({pct, std::this_thread::get_id()});
        }
        in_callback.fetch_sub(1);
    });
    REQUIRE(n > 0);
    REQUIRE(!calls.empty());
    CHECK(!overlapped);
    CHECK(calls.back().pct == 100);
    CHECK(calls.back().tid == std::this_thread::get_id());
    for (size_t i = 0; i + 1 < calls.size(); ++i) {
        CHECK(calls[i].tid != std::this_thread::get_id());
        CHECK(calls[i].pct <= calls[i + 1].pct);
    }
}

// Path C is the last resort when the anchor lattice found nothing and phase
// correlation did not lock. Its failure returned -1, a code no header,
// binding or document defined. It is kSeedFailed (-5) now.
TEST_CASE(FullField, UnseedableField_ReturnsDocumentedCode) {
    // A blank frame (lens cap, dropped frame): nothing to correlate with.
    // (An unrelated speckle frame is not enough -- the unlocked lattice's
    // own search finds a few spurious matches on it and seeds from those.)
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    const cv::Mat unrelated(H, W, CV_8UC1, cv::Scalar(128));
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    std::vector<float> out((size_t)cap, 0.0f);
    float metrics[Semper::pipeline::kMetricsLen] = {};
    const int rc = run_full_field(cache, unrelated, cv::Mat(), params_for(W, H), out.data(), cap,
                                  metrics, Semper::pipeline::kMetricsLen, nullptr);
    std::printf("    rc = %d\n", rc);
    CHECK(rc == Semper::pipeline::kSeedFailed);
    CHECK(Semper::pipeline::kSeedFailed == -5);
}

// The rescue telemetry (metrics 5-7 and the profile's crash/timeout split)
// classified a Simplex rescue by the iterations of the IC-GN run *after* the
// rescue, which usually converges quickly, so a rescued timeout was counted
// as a crash. It is classified by the run that triggered it now. Here the
// first run is made to time out by scaling the SoA steepest-descent planes
// (the full-subset gradient) to 1/20: every step is 1/20 of the way.
TEST_CASE(Telemetry, RescueClassifiedByTriggeringRun) {
    using namespace Semper;
    dictest::SpeckleField field(/*seed=*/31, 160, 160);
    dictest::AffineDeformation truth;
    truth.u = 1.0f; truth.v = -0.5f; truth.cx = 80.0f; truth.cy = 80.0f;
    const Image ref = dictest::make_reference_image(field, 160, 160);
    const Image def = dictest::make_deformed_image(field, 160, 160, truth);
    SubsetData subset;
    SubsetPrecomputer::precompute_subset(subset, ref, 80, 80, 31);
    REQUIRE(subset.is_initialized);
    for (float &v : subset.sdi_planes) v *= 0.05f;

    OptimizationEngine engine;
    const AnalysisResult res = engine.calculate_deformation(subset, def, 0.0f, 0.0f, 0.0f, 0.0f,
                                                           0.0f, 0.0f, INIT_NO_SEARCH);
    std::printf("    simplex runs %d, final iters %d, status %d\n", engine.count_simplex, res.iters,
                res.status);
    REQUIRE(engine.count_simplex == 1);
    REQUIRE(res.iters < tuning::kIcgnMaxIter);   // the run after the rescue was short

    Semper::pipeline::internal::ThreadStats bucket;
    Semper::pipeline::internal::record_simplex_outcome(bucket, res);
    CHECK(bucket.simplex_calls == 1);
    CHECK(bucket.simplex_from_timeout == 1);
    CHECK(bucket.simplex_from_crash == 0);
}

// Every solve in the pipeline -- anchors, Paths A, B and C, re-anchoring --
// builds its engine through make_engine. Path C used to set up its own and
// left LM damping off, so the one seed that has to work without any other
// help was solved by a differently configured engine.
TEST_CASE(PathC, UsesSameEngineConfig) {
    for (bool use6x6 : {false, true}) {
        const Semper::OptimizationEngine e = Semper::pipeline::internal::make_engine(use6x6);
        CHECK(e.lm_enabled);
        CHECK(e.lm_alpha == Semper::tuning::kLmAlpha);
        CHECK(e.use_6x6_interpolator == use6x6);
    }
}

// ---------------------------------------------------------------------------
// Reference precompute cache. A batch against one reference reuses the grid
// scan, the Hessian pool and the reference AKAZE keypoints.
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

// The debug export reads the ROI mask at every grid node. With a rect past
// the image those reads were out of bounds (roiMask.at<> is unchecked in
// Release). validate_params now clips the rect first, and the export's own
// lookup is bounds-checked too; under ASan this is the test for both.
TEST_CASE(DebugExport, RectBeyondImage_NoOOB) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("semper_debug_" + std::to_string((long long)std::chrono::steady_clock::now()
                                                                  .time_since_epoch().count()));
    REQUIRE(fs::create_directories(dir));

    cv::Mat mask(H, W, CV_8UC1, cv::Scalar(255));
    cv::circle(mask, cv::Point(W / 2, H / 2), 30, cv::Scalar(0), -1);
    FullFieldParams p = params_for(W + 4 * STEP, H + 4 * STEP);
    p.rect_x = -2 * STEP;
    p.rect_y = -2 * STEP;

    ReferenceCache cache;
    cache.set_from_gray(ref_gray, mask);
    cache.debug_dir = dir.string();
    const int cap = 8 * ((p.rect_w / STEP) * (p.rect_h / STEP) + 8);
    std::vector<float> out((size_t)cap, 0.0f);
    CHECK(run_full_field(cache, def_gray, mask, p, out.data(), cap, nullptr, 0, nullptr) > 0);
    // The export ran: its CSV is there.
    CHECK(fs::exists(dir / "debug_grid_data.csv"));
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// ---------------------------------------------------------------------------
// Two engines, two tokens, one process. Until 0.5.0 the token overload bound
// its token into one process-global "active" pointer for the duration of the
// solve, so with two solves in flight both polled whichever was bound last:
// cancelling one stopped the other, or nothing, and a solve that outlived its
// token left the global pointing at a dead object. Tokens now travel with the
// solve. Run under TSan this is also the test that sees two solves at once.
// ---------------------------------------------------------------------------
TEST_CASE(CancelToken, ConcurrentEnginesCancelIndependently) {
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    FullFieldParams p = params_for(W, H);
    p.step = 5;
    p.strain_window = 31;
    const int cap = (W / 5) * (H / 5) * 8;

    // The uncontested result B has to reproduce.
    std::vector<float> want((size_t)cap, 0.0f);
    int n_want = 0;
    {
        ReferenceCache cache;
        cache.set_from_gray(ref_gray, cv::Mat());
        float metrics[23] = {};
        n_want = run_full_field(cache, def_gray, cv::Mat(), p, want.data(), cap, metrics, 23, nullptr);
    }
    REQUIRE(n_want > 0);

    using Clock = std::chrono::steady_clock;
    auto wait_for = [](const std::atomic<bool> &flag) {
        const auto until = Clock::now() + std::chrono::seconds(20);
        while (!flag.load() && Clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };

    Semper::pipeline::CancelToken token_a, token_b;
    std::atomic<bool> b_running(false), a_done(false);
    int rc_a = 0, rc_b = 0;
    std::vector<float> out_b((size_t)cap, 0.0f);

    // B starts first and, from its progress thread, holds its solve open
    // (the solve joins that thread before returning) until A has finished,
    // so the two overlap for all of A.
    std::thread tb([&] {
        ReferenceCache cache;
        cache.set_from_gray(ref_gray, cv::Mat());
        float metrics[23] = {};
        bool first = true;
        rc_b = run_full_field(cache, def_gray, cv::Mat(), p, out_b.data(), cap, metrics, 23,
                              token_b, [&](int) {
                                  if (!first) return;
                                  first = false;
                                  b_running = true;
                                  wait_for(a_done);
                              });
    });
    wait_for(b_running);
    std::thread ta([&] {
        ReferenceCache cache;
        cache.set_from_gray(ref_gray, cv::Mat());
        std::vector<float> out((size_t)cap, 0.0f);
        float metrics[23] = {};
        rc_a = run_full_field(cache, def_gray, cv::Mat(), p, out.data(), cap, metrics, 23,
                              token_a, [&](int) { token_a.request(); });
        a_done = true;
    });
    ta.join();
    tb.join();

    CHECK(rc_a == kCancelled);
    CHECK(rc_b == n_want);
    CHECK(!token_b.requested());
    if (rc_b == n_want)
        CHECK(std::memcmp(want.data(), out_b.data(), (size_t)n_want * 8 * sizeof(float)) == 0);
}

#else

TEST_CASE(FullField, OpenCvRequired_SkippedWithoutOpenCV) {
    // Suite still registers when OpenCV is absent so the filter surface is stable.
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
