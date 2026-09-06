// Host characterization for run_full_field — mirrors EnginePipelineSmokeTest
// contract block (degenerate ROI → -2, empty def → -3, stale cancel, undersized
// buffer). Compiled only when DIC_HAVE_OPENCV is set (same gate as DICe tests).
#include "framework/test_framework.h"

#include <cstring>
#include "framework/synthetic.h"

#include <semper/cancel.hpp>
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
constexpr int STRAIN_WIN = 15;

cv::Mat gray8_from_image(const Semper::Image &img) {
    cv::Mat m(img.height, img.width, CV_8UC1);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float v = img.intensities[(size_t)y * img.width + x];
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    }
    return m;
}

void make_pair(cv::Mat &ref_gray, cv::Mat &def_gray) {
    dictest::SpeckleField field(/*seed=*/11, W, H, /*blob_count=*/600);
    dictest::AffineDeformation def;
    def.u = 3.0f;
    def.v = 2.0f;
    def.cx = W / 2.0f;
    def.cy = H / 2.0f;
    const Semper::Image ref = dictest::make_reference_image(field, W, H);
    const Semper::Image deformed = dictest::make_deformed_image(field, W, H, def);
    ref_gray = gray8_from_image(ref);
    def_gray = gray8_from_image(deformed);
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
    // The 17-float metrics array is Frozen (docs/CONTRACT.md §A.4) but only
    // metrics[0] was ever checked. Pin the slot invariants a downstream telemetry
    // reader relies on. Runs a real solve on the 256x256 speckle pair.
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());

    const int cap = 8 * ((W / STEP) * (H / STEP) + 8);
    std::vector<float> out(cap, 0.f);
    float metrics[17];
    for (int i = 0; i < 17; ++i) metrics[i] = -12345.f;

    const int n = run_full_field(cache, def_gray, cv::Mat(), params_for(W, H),
                                 out.data(), cap, metrics, 17, nullptr);
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

#else

TEST_CASE(FullField, OpenCvRequired_SkippedWithoutOpenCV) {
    // Suite still registers when OpenCV is absent so the filter surface is stable.
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV

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

    // The default STEP/STRAIN_WIN pair puts one grid node inside the VSG window,
    // so every point is rejected and the field is empty. Use a step and window
    // that actually produce points, or this proves nothing.
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
