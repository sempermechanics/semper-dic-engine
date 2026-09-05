// =====================================================================
// SUITE: ClPipelineParity — the GPU stages as the pipeline actually calls them
//
// tests/unit/test_cl_strain.cpp and tests/unit/test_cl_hessian.cpp drive the
// dispatch functions directly, at any geometry they like. This file closes
// the other half: that run_full_field, wired end to end, produces the SAME
// packed output whether its GPU stages ran or not.
//
// It exists because the dispatch functions are deliberately policy-free while
// the size thresholds live in the callers — so the callers are code that no
// parity test would otherwise execute. The geometry below is chosen to clear
// both of them (a 512x512 reference at step 7 is 5329 grid points, above the
// 4000-point floor and the 3600-point strain threshold), and the two runs
// differ only in SEMPER_OPENCL_DISABLE.
//
// The comparison is `==` on every float of the packed field, like the rest of
// the GPU parity work. A tolerance here would defeat the purpose.
//
// Skips itself, loudly, without a device that can run the stages. Under
// -DSEMPER_OPENCL=OFF the file reduces to one test, so the suite count does
// not change silently with the build flag.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <semper/pipeline.hpp>

#include <cstdio>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <vector>

#if defined(DIC_HAVE_OPENCV) && defined(SEMPER_OPENCL)

#include "gpu/cl_runtime.hpp"

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;

namespace {

constexpr int W = 512, H = 512;
constexpr int STEP = 7, SUBSET = 21, STRAIN_WIN = 21;
constexpr int GRID_PTS = (W / STEP) * (H / STEP);   // 5329

cv::Mat gray8_from_image(const Semper::Image &img) {
    cv::Mat m(img.height, img.width, CV_8UC1);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            float v = img.intensities[(size_t) y * img.width + x];
            if (v < 0.f) v = 0.f;
            if (v > 255.f) v = 255.f;
            m.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
        }
    return m;
}

void make_pair(cv::Mat &ref_gray, cv::Mat &def_gray) {
    dictest::SpeckleField field(/*seed=*/23, W, H, /*blob_count=*/1400);
    dictest::AffineDeformation def;
    def.u = 2.5f;
    def.v = -1.5f;
    def.ux = 0.004f;
    def.vy = -0.003f;
    def.cx = W / 2.0f;
    def.cy = H / 2.0f;
    ref_gray = gray8_from_image(dictest::make_reference_image(field, W, H));
    def_gray = gray8_from_image(dictest::make_deformed_image(field, W, H, def));
}

void set_disable(bool on) {
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", on ? "1" : "");
#else
    if (on) setenv("SEMPER_OPENCL_DISABLE", "1", 1);
    else unsetenv("SEMPER_OPENCL_DISABLE");
#endif
    Semper::gpu::reset_for_testing();
}

struct Run {
    int rc = -99;
    std::vector<float> out;
    float metrics[19] = {};
};

// One full solve. The reference cache is rebuilt per run on purpose: the GPU
// pre-pass reads the cached reference image and its gradients, so sharing a
// cache between the two runs would hide a difference in how they are built.
Run solve(const cv::Mat &ref_gray, const cv::Mat &def_gray) {
    Run r;
    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());
    FullFieldParams p;
    p.rect_x = 0;
    p.rect_y = 0;
    p.rect_w = W;
    p.rect_h = H;
    p.step = STEP;
    p.subset_size = SUBSET;
    p.strain_window = STRAIN_WIN;
    p.use_6x6_interpolator = false;

    r.out.assign((size_t) GRID_PTS * 8, 0.0f);
    r.metrics[16] = -1.0f;
    // Third argument is the ROI mask, not the reference: the reference lives
    // in the cache, which is why it is seeded above.
    r.rc = run_full_field(cache, def_gray, cv::Mat(), p, r.out.data(),
                          (int) r.out.size(), r.metrics, 19, nullptr);
    return r;
}

} // namespace

// The whole point of the file. Two identical solves, one with the device
// engaged for the Hessian pre-pass and the strain fit, one with it forced off
// — same bytes out.
TEST_CASE(ClPipelineParity, FullFieldOutputIsIdenticalWithAndWithoutGpu) {
    const auto &c = Semper::gpu::caps();
    // The pre-pass needs correctly-rounded fp32; the strain fit needs fp64.
    // Either one engaging is enough to make this test say something.
    if (!c.available || !(c.exact_fp32 || c.fp64)) {
        std::printf("  ClPipelineParity: skipped (no usable device) — %s\n",
                    c.available ? "device runs neither stage"
                                : c.unavailable_reason.c_str());
        set_disable(false);
        CHECK(true);
        return;
    }
    std::printf("  ClPipelineParity: '%s' fp64=%d exact_fp32=%d, %d grid points\n",
                c.device_name.c_str(), (int) c.fp64, (int) c.exact_fp32, GRID_PTS);

    cv::Mat ref_gray, def_gray;
    make_pair(ref_gray, def_gray);

    set_disable(true);
    const Run cpu = solve(ref_gray, def_gray);
    set_disable(false);
    // Two device solves. The first pays the one-time cost of the whole
    // backend -- dlopen, platform/device probe, clBuildProgram for every
    // kernel -- inside its pre-pass timing, because set_disable() dropped the
    // cached caps. The second is what a second frame in a live session costs,
    // and comparing the two also proves the device path is reproducible
    // across solves in one process, not merely equal to the CPU once.
    const Run gpu_cold = solve(ref_gray, def_gray);
    const Run gpu = solve(ref_gray, def_gray);

    REQUIRE(cpu.rc >= 0);
    CHECK(gpu.rc == cpu.rc);
    REQUIRE(cpu.out.size() == gpu.out.size());

    int bad = 0;
    for (size_t i = 0; i < cpu.out.size(); ++i) {
        if (cpu.out[i] == gpu.out[i]) continue;
        if (bad < 6)
            std::printf("     float %zu: cpu %.9g  gpu %.9g\n", i,
                        (double) cpu.out[i], (double) gpu.out[i]);
        ++bad;
    }

    // metrics[1] is the valid point count, [3]/[4] the Path A / Path B split,
    // [5..7] the simplex-rescue tally, [8] the mean ICGN iteration count,
    // [9] total wall time and [11] the pre-pass time in ms. The times are
    // printed, never asserted -- they are what says the GPU branch actually
    // fired, but they are far too noisy to gate on.
    std::printf("     valid: cpu %.0f, gpu %.0f | A/B split: cpu %.0f/%.0f, gpu %.0f/%.0f"
                " | rescues: cpu %.0f, gpu %.0f | mean iters: cpu %.6f, gpu %.6f\n"
                "     pre-pass: cpu %.2f ms, gpu %.2f ms cold / %.2f ms warm"
                " | total: cpu %.1f ms, gpu %.1f ms warm | %d float mismatches\n",
                (double) cpu.metrics[1], (double) gpu.metrics[1],
                (double) cpu.metrics[3], (double) cpu.metrics[4],
                (double) gpu.metrics[3], (double) gpu.metrics[4],
                (double) cpu.metrics[5], (double) gpu.metrics[5],
                (double) cpu.metrics[8], (double) gpu.metrics[8],
                (double) cpu.metrics[11], (double) gpu_cold.metrics[11],
                (double) gpu.metrics[11],
                (double) cpu.metrics[9], (double) gpu.metrics[9], bad);

    // A solve that found nothing would compare equal and prove nothing.
    CHECK(cpu.metrics[1] > 0.0f);
    CHECK(gpu.metrics[1] == cpu.metrics[1]);
    // A moved Path A / Path B split means propagation changed even if every
    // surviving point still matches -- docs/DETERMINISM.md:142-147.
    CHECK(gpu.metrics[3] == cpu.metrics[3]);
    CHECK(gpu.metrics[4] == cpu.metrics[4]);
    // Phase 4 makes these say something they did not before. The ICGN kernel
    // never runs Nelder-Mead, so Path A accepts a device answer only when the
    // CPU would not have gone on to a rescue; if that rule were wrong, the
    // rescue tally and the mean iteration count would move even where the
    // surviving points happened to agree. Both are derived from counters the
    // device path still has to feed by hand (full_field_path_a.cpp), which is
    // exactly why they are worth asserting rather than printing.
    CHECK(gpu.metrics[5] == cpu.metrics[5]);
    CHECK(gpu.metrics[6] == cpu.metrics[6]);
    CHECK(gpu.metrics[7] == cpu.metrics[7]);
    CHECK(gpu.metrics[8] == cpu.metrics[8]);
    CHECK(bad == 0);

    // Cold and warm device solves must agree with each other too -- a
    // difference here would mean the result depended on backend state
    // carried across solves.
    REQUIRE(gpu_cold.out.size() == gpu.out.size());
    int drift = 0;
    for (size_t i = 0; i < gpu.out.size(); ++i)
        if (gpu_cold.out[i] != gpu.out[i]) ++drift;
    CHECK(drift == 0);
}

#else // !(DIC_HAVE_OPENCV && SEMPER_OPENCL)

TEST_CASE(ClPipelineParity, PipelineGpuStagesCompiledOut) {
    std::printf("  ClPipelineParity: not built (needs OpenCV and SEMPER_OPENCL=ON)\n");
    CHECK(true);
}

#endif
