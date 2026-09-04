// =====================================================================
// SUITE: FullFieldGolden — end-to-end regression + determinism gate
//
// tests/integration/test_golden_corpus.cpp pins the SUBSET solver: it
// calls precompute_subset / calculate_deformation directly and never
// enters run_full_field. That leaves the whole orchestration layer --
// AKAZE seeding, the Delaunay mesh guess field, Path A, Path B's
// propagation, and the strain stage -- with no golden reference at all.
//
// This suite closes that gap, which the OpenCL work needs for two
// reasons: those are exactly the stages moving to the GPU, and Path B's
// propagation order is the one place where GPU parallelism and
// result-identity genuinely conflict.
//
// Values are compared at a tight but non-zero tolerance, for the same
// reason as the subset corpus: synthetic.h builds its images from
// several hundred std::exp terms per pixel, so the INPUT depends on the
// host libm and an exact bound would pin the fixture to one toolchain.
// The exact cross-ABI gate is the `determinism` CI job.
// See docs/DETERMINISM.md.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <semper/pipeline.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;

#if defined(DIC_HAVE_OPENCV)

namespace {

constexpr int W = 320;
constexpr int H = 320;
constexpr int STEP = 12;
constexpr int SUBSET = 21;
// The VSG window must span several grid steps or every point fails the
// valid_pts >= 3 gate in compute_vsg_strain and the whole field is dropped
// by the strain post-filter. radius = 48/2 = 24 px = 2 steps, giving a
// 13-point circular window. (step 12 / window 15 yields radius 7.5 px,
// smaller than one step: the window holds only its own centre.)
constexpr int STRAIN_WIN = 48;
constexpr int FLOATS_PER_POINT = 8; // x y u v exx eyy exy corr (Frozen)

struct Scenario {
    const char *name;
    float u, v, ux, uy, vx, vy;
};

// Translation is the common case; the affine case drives Path B harder by
// making a single global guess insufficient across the field.
const Scenario SCENARIOS[] = {
    {"translate", 3.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {"affine", 2.0f, -1.5f, 0.012f, -0.006f, 0.005f, 0.010f},
};

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

FullFieldParams default_params() {
    FullFieldParams p;
    p.rect_x = 0;
    p.rect_y = 0;
    p.rect_w = W;
    p.rect_h = H;
    p.step = STEP;
    p.subset_size = SUBSET;
    p.strain_window = STRAIN_WIN;
    p.use_6x6_interpolator = false;
    return p;
}

struct SolveOutput {
    int code = 0;
    std::vector<float> points; // code * FLOATS_PER_POINT
    // Only the COUNTING metrics are recorded. Slots 9..14 and 17/18 are
    // wall-clock timings and would differ on every run by construction.
    float counts[10] = {};
};

SolveOutput solve_scenario(const Scenario &sc) {
    dictest::SpeckleField field(/*seed=*/11, W, H, /*blob_count=*/600);
    dictest::AffineDeformation def;
    def.u = sc.u; def.v = sc.v;
    def.ux = sc.ux; def.uy = sc.uy;
    def.vx = sc.vx; def.vy = sc.vy;
    def.cx = W / 2.0f;
    def.cy = H / 2.0f;

    const Semper::Image ref = dictest::make_reference_image(field, W, H);
    const Semper::Image deformed = dictest::make_deformed_image(field, W, H, def);
    const cv::Mat ref_gray = gray8_from_image(ref);
    const cv::Mat def_gray = gray8_from_image(deformed);

    ReferenceCache cache;
    cache.set_from_gray(ref_gray, cv::Mat());

    const FullFieldParams p = default_params();
    const int grid_w = p.rect_w / p.step;
    const int grid_h = p.rect_h / p.step;
    const int capacity = grid_w * grid_h * FLOATS_PER_POINT;

    std::vector<float> out((size_t) std::max(1, capacity), 0.f);
    float metrics[17] = {};
    metrics[16] = -1.f;

    SolveOutput r;
    r.code = run_full_field(cache, def_gray, cv::Mat(), p, out.data(), capacity,
                            metrics, 17, nullptr);
    if (r.code > 0) {
        r.points.assign(out.begin(),
                        out.begin() + (size_t) r.code * FLOATS_PER_POINT);
    }
    const int count_slots[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 16};
    for (int i = 0; i < 10; ++i) r.counts[i] = metrics[count_slots[i]];
    return r;
}

std::string golden_path() {
    if (const char *env = std::getenv("SEMPER_FF_GOLDEN_FILE")) return env;
    return std::string(NATIVE_TESTS_SOURCE_DIR) + "/fixtures/full_field_golden.bin";
}

bool capture_mode() {
    const char *env = std::getenv("SEMPER_FF_GOLDEN_CAPTURE");
    return env && std::strcmp(env, "1") == 0;
}

bool write_golden(const std::string &path, const std::vector<SolveOutput> &runs) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n = static_cast<uint32_t>(runs.size());
    std::fwrite(&n, sizeof(n), 1, f);
    for (const auto &r : runs) {
        const int32_t code = r.code;
        std::fwrite(&code, sizeof(code), 1, f);
        std::fwrite(r.counts, sizeof(r.counts), 1, f);
        const uint32_t np = static_cast<uint32_t>(r.points.size());
        std::fwrite(&np, sizeof(np), 1, f);
        if (np) std::fwrite(r.points.data(), sizeof(float), np, f);
    }
    std::fclose(f);
    return true;
}

// Empty on any read failure — callers treat that as "not captured yet".
std::vector<SolveOutput> read_golden(const std::string &path) {
    std::vector<SolveOutput> runs;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return runs;
    uint32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1 || n == 0 || n > 64) {
        std::fclose(f);
        return runs;
    }
    for (uint32_t i = 0; i < n; ++i) {
        SolveOutput r;
        int32_t code = 0;
        uint32_t np = 0;
        if (std::fread(&code, sizeof(code), 1, f) != 1 ||
            std::fread(r.counts, sizeof(r.counts), 1, f) != 1 ||
            std::fread(&np, sizeof(np), 1, f) != 1) {
            runs.clear();
            std::fclose(f);
            return runs;
        }
        r.code = code;
        r.points.resize(np);
        if (np && std::fread(r.points.data(), sizeof(float), np, f) != np) {
            runs.clear();
            std::fclose(f);
            return runs;
        }
        runs.push_back(std::move(r));
    }
    std::fclose(f);
    return runs;
}

// x/y are integer grid coordinates in float form; u/v are pixels; the three
// strain components are dimensionless and several orders smaller, so they
// get their own bound. corr is a ZNSSD residual in [0, 2].
constexpr float TOL_COORD = 0.0f;   // grid geometry must be identical
constexpr float TOL_DISP = 1e-6f;
constexpr float TOL_STRAIN = 1e-7f;
constexpr float TOL_CORR = 1e-6f;

const float *tol_for_slot() {
    static const float t[FLOATS_PER_POINT] = {
        TOL_COORD, TOL_COORD, TOL_DISP, TOL_DISP,
        TOL_STRAIN, TOL_STRAIN, TOL_STRAIN, TOL_CORR};
    return t;
}

} // namespace

TEST_CASE(FullFieldGolden, CaptureOrCompare) {
#if defined(SEMPER_SANITIZER_BUILD) || defined(SEMPER_COVERAGE_BUILD)
    std::printf("  [full-field] skipped under sanitizer/coverage "
                "(numeric golden is a Release-host gate)\n");
    return;
#endif
    std::vector<SolveOutput> runs;
    for (const auto &sc : SCENARIOS) runs.push_back(solve_scenario(sc));

    if (capture_mode()) {
        REQUIRE(write_golden(golden_path(), runs));
        std::printf("  [full-field] captured %zu scenarios to %s\n",
                    runs.size(), golden_path().c_str());
        for (size_t i = 0; i < runs.size(); ++i)
            std::printf("    %-10s %d points  (pathA=%.0f pathB=%.0f "
                        "attempted=%.0f solved=%.0f seedmode=%.0f)\n",
                        SCENARIOS[i].name, runs[i].code,
                        (double) runs[i].counts[3], (double) runs[i].counts[4],
                        (double) runs[i].counts[0], (double) runs[i].counts[1],
                        (double) runs[i].counts[9]);
        return;
    }

    const auto golden = read_golden(golden_path());
    // "not captured yet" — run with SEMPER_FF_GOLDEN_CAPTURE=1 first
    REQUIRE(!golden.empty());
    REQUIRE(golden.size() == runs.size());

    const float *tol = tol_for_slot();
    for (size_t s = 0; s < runs.size(); ++s) {
        const auto &g = golden[s];
        const auto &c = runs[s];
        if (g.code != c.code) {
            std::printf("  [%s] point count changed %d -> %d\n",
                        SCENARIOS[s].name, g.code, c.code);
        }
        CHECK(g.code == c.code);
        if (g.code != c.code) continue;

        for (int i = 0; i < 10; ++i) {
            if (g.counts[i] != c.counts[i])
                std::printf("  [%s] metric slot %d changed %.0f -> %.0f\n",
                            SCENARIOS[s].name, i, (double) g.counts[i],
                            (double) c.counts[i]);
            CHECK(g.counts[i] == c.counts[i]);
        }

        REQUIRE(g.points.size() == c.points.size());
        int mismatches = 0;
        for (size_t k = 0; k < c.points.size(); ++k) {
            const int slot = static_cast<int>(k % FLOATS_PER_POINT);
            if (std::fabs(g.points[k] - c.points[k]) > tol[slot]) {
                if (mismatches < 5)
                    std::printf("  [%s] point %zu slot %d: %.9g -> %.9g\n",
                                SCENARIOS[s].name, k / FLOATS_PER_POINT, slot,
                                (double) g.points[k], (double) c.points[k]);
                ++mismatches;
            }
        }
        std::printf("  [%s] %d points, %d value mismatches\n",
                    SCENARIOS[s].name, c.code, mismatches);
        CHECK(mismatches == 0);
    }
}

// The full-field solve must return the same field every time it is run on
// the same inputs.
//
// This is NOT guaranteed by the subset-level Engine.RepeatSolve_BitIdentical
// test. Path B drains a shared std::priority_queue from several worker
// threads, and each point's initial guess is extrapolated from whichever
// parent won the compare_exchange for that cell -- so the propagation order,
// and with it the answer at hard points, can vary between runs of the same
// binary on the same data.
//
// A GPU implementation cannot be "bit-exact with the CPU" while the CPU has
// no fixed answer to be exact against, which is why this property has to
// hold before any of Path B moves to a kernel.
TEST_CASE(FullFieldGolden, RepeatSolve_IsDeterministic) {
    for (const auto &sc : SCENARIOS) {
        const SolveOutput a = solve_scenario(sc);
        const SolveOutput b = solve_scenario(sc);

        CHECK(a.code == b.code);
        if (a.code != b.code) {
            std::printf("  [%s] NON-DETERMINISTIC point count: %d vs %d\n",
                        sc.name, a.code, b.code);
            continue;
        }
        REQUIRE(a.points.size() == b.points.size());

        int differing = 0;
        for (size_t k = 0; k < a.points.size(); ++k)
            if (!(a.points[k] == b.points[k])) ++differing;

        if (differing)
            std::printf("  [%s] NON-DETERMINISTIC: %d of %zu floats differ "
                        "between two runs of the same binary\n",
                        sc.name, differing, a.points.size());
        CHECK(differing == 0);
    }
}

#endif // DIC_HAVE_OPENCV
