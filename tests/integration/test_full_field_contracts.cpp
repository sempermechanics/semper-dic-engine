// Host characterization for run_full_field — mirrors EnginePipelineSmokeTest
// contract block (degenerate ROI → -2, empty def → -3, stale cancel, undersized
// buffer). Compiled only when DIC_HAVE_OPENCV is set (same gate as DICe tests).
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <semper/cancel.hpp>
#include <semper/pipeline.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

void set_env(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
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

TEST_CASE(FullField, PathBRoundTraceIsInert) {
    // SEMPER_PATHB_ROUNDS=1 turns on the Path B round-structure diagnostic
    // that GPU Phase 5 used to conclude Path B stays on the CPU (see the
    // header of src/pipeline/full_field_path_b.cpp). It reads per-round
    // counters that the solve is already maintaining, so it must not be able
    // to move a single output float -- a diagnostic that perturbs the thing
    // it measures is worse than no diagnostic.
    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);
    const auto p = params_for(W, H);
    const int grid_pts = (W / STEP) * (H / STEP);
    const int buf_floats = grid_pts * 8;

    std::vector<float> quiet(buf_floats, 0.f), traced(buf_floats, 0.f);
    float m_quiet[17] = {}, m_traced[17] = {};
    m_quiet[16] = -1.f;
    m_traced[16] = -1.f;

    set_env("SEMPER_PATHB_ROUNDS", nullptr);
    ReferenceCache cache_a;
    cache_a.set_from_gray(ref_gray, cv::Mat());
    const int rc_quiet = run_full_field(cache_a, def_gray, cv::Mat(), p,
                                        quiet.data(), buf_floats, m_quiet, 17, nullptr);

    set_env("SEMPER_PATHB_ROUNDS", "1");
    ReferenceCache cache_b;
    cache_b.set_from_gray(ref_gray, cv::Mat());
    const int rc_traced = run_full_field(cache_b, def_gray, cv::Mat(), p,
                                         traced.data(), buf_floats, m_traced, 17, nullptr);
    set_env("SEMPER_PATHB_ROUNDS", nullptr);

    REQUIRE(rc_quiet >= 0);
    CHECK(rc_quiet == rc_traced);

    int diffs = 0;
    for (int i = 0; i < buf_floats; ++i)
        if (quiet[(size_t) i] != traced[(size_t) i]) ++diffs;
    CHECK(diffs == 0);

    // Counts, the Path A/B split, the rescue tally and the mean iteration
    // count too, since those are exactly what the diagnostic reads. Slots
    // 9-14 are wall-clock timings and a throughput ratio derived from them
    // (full_field_solver_stats.cpp), so they are not reproducible between any
    // two solves and are excluded here rather than compared loosely.
    for (int i = 0; i <= 8; ++i) CHECK(m_quiet[i] == m_traced[i]);
    CHECK(m_quiet[15] == m_traced[15]);   // convergence %
    CHECK(m_quiet[16] == m_traced[16]);   // seeding route
}

#else

TEST_CASE(FullField, OpenCvRequired_SkippedWithoutOpenCV) {
    // Suite still registers when OpenCV is absent so the filter surface is stable.
    CHECK(true);
}

#endif // DIC_HAVE_OPENCV
