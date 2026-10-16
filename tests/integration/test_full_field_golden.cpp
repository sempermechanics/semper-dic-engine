// =====================================================================
// SUITE: FullFieldGolden — end-to-end regression + determinism gate
//
// tests/integration/test_golden_corpus.cpp pins the SUBSET solver: it
// calls precompute_subset / calculate_deformation directly and never
// enters run_full_field. That leaves the whole orchestration layer --
// anchor-lattice seeding, the Delaunay mesh guess field, Path A, Path B's
// propagation, and the strain stage -- with no golden reference at all.
// (Upstream this line read "AKAZE seeding"; that front-end was replaced by
// phase correlation plus an IC-GN anchor lattice in v0.2.2.)
//
// PROVENANCE. This file originated on
// claude/gpu-parallelization-displacement-strain-asx9w1 (07981e6). Two
// branches needed the same gate, so rather than write a second one this
// branch adopted that file and changed only what the anchor-lattice seeding
// requires -- both changes are marked DIVERGENCE below. The fixtures are NOT
// interchangeable: the seeding front-end differs, so the field differs.
// Whichever branch merges second recaptures the .bin once, deliberately,
// and records why.
//
// This suite closes that gap, which the OpenCL work needs for two
// reasons: those are exactly the stages moving to the GPU, and Path B's
// propagation order is the one place where GPU parallelism and
// result-identity genuinely conflict.
//
// The METRICS are compared exactly and the point VALUES are not, at bounds
// chosen to hold across toolchains rather than to pin the fixture to one.
// Two reasons, in order of size: IC-GN stops at ||delta_p|| < 1e-3, so hosts
// that differ only in FMA contraction stop one step apart; and synthetic.h
// builds its images from several hundred std::exp terms per pixel, so even
// the INPUT depends on the host libm. The numbers, and the measurement they
// come from, are at TOL_DISP below. This branch has no exact cross-ABI gate;
// the GPU branch adds one (a `determinism` CI job and docs/DETERMINISM.md),
// and when that lands this comment should point at it.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"
#include "framework/synthetic_cv.h"

#include <semper/pipeline.hpp>
#include <semper/semper_c.h>

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
// Ask for the whole metrics array. run_full_field fills only as much as it
// is given, so a short buffer here would silently zero the slots this
// fixture exists to watch. SEMPER_METRICS_LEN pins the length itself.
constexpr int kMetricsLen = SEMPER_METRICS_LEN;

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

// The metrics slots recorded in the fixture, and the bound each is compared
// at. All but two are integer-valued counts or a 0/1 flag and are compared
// exactly -- that exactness is what makes this suite a gate once the point
// values carry a portable tolerance (see TOL_DISP below).
//
// The two exceptions are both averages. Slot 8 is mean IC-GN iterations per
// point, which moves when a host takes one extra step to cross the same
// convergence threshold: 7.96579 on Ubuntu GCC 11.4 against 7.94737 on
// Windows MinGW, a difference of one iteration spread over 380 points.
// Slot 22 (mesh coverage) is a convex-hull area ratio accumulated in double
// precision and moves in its last bits across libm implementations.
constexpr int kCountSlots = 14;
constexpr int kCountMetric[kCountSlots] = {0,  1,  2,  3,  4,  5,  6,
                                           7,  8,  16, 19, 20, 21, 22};
constexpr float kCountTol[kCountSlots] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                          0.f, 0.1f, 0.f, 0.f, 0.f, 0.f,
                                          1e-4f};

struct SolveOutput {
    int code = 0;
    std::vector<float> points; // code * FLOATS_PER_POINT
    // Only the COUNTING metrics are recorded. Slots 9..14 and 17/18 are
    // wall-clock timings and would differ on every run by construction.
    //
    // DIVERGENCE from the upstream file: slots 19..22 are recorded too.
    // They are this branch's additions -- anchor points solved, points
    // dropped by the strain post-filter, the phase lock, and mesh coverage.
    // Without them a seeding change that shifted work between the three
    // solving paths, or that quietly lost the phase lock, would leave every
    // recorded number unchanged and pass.
    float counts[kCountSlots] = {};
};

// The speckle field and the reference image are the same for every scenario,
// and solve_scenario runs six times per suite (two here, four in the
// determinism case). Building them once is bit-exact -- the field is seeded and
// dictest::gray8 is deterministic -- and saves five 320x320x600 samplings.
struct RefFixture {
    dictest::SpeckleField field{/*seed=*/11, W, H, /*blob_count=*/600};
    cv::Mat gray = dictest::gray8(dictest::make_reference_image(field, W, H));
};

const RefFixture &ref_fixture() {
    static const RefFixture f;
    return f;
}

SolveOutput solve_scenario(const Scenario &sc) {
    const RefFixture &fix = ref_fixture();
    dictest::AffineDeformation def;
    def.u = sc.u; def.v = sc.v;
    def.ux = sc.ux; def.uy = sc.uy;
    def.vx = sc.vx; def.vy = sc.vy;
    def.cx = W / 2.0f;
    def.cy = H / 2.0f;

    const Semper::Image deformed = dictest::make_deformed_image(fix.field, W, H, def);
    const cv::Mat def_gray = dictest::gray8(deformed);

    ReferenceCache cache;
    cache.set_from_gray(fix.gray, cv::Mat());

    const FullFieldParams p = default_params();
    const int grid_w = p.rect_w / p.step;
    const int grid_h = p.rect_h / p.step;
    const int capacity = grid_w * grid_h * FLOATS_PER_POINT;

    std::vector<float> out((size_t) std::max(1, capacity), 0.f);
    float metrics[kMetricsLen] = {};
    metrics[16] = -1.f;

    SolveOutput r;
    r.code = run_full_field(cache, def_gray, cv::Mat(), p, out.data(), capacity,
                            metrics, kMetricsLen, nullptr);
    if (r.code > 0) {
        r.points.assign(out.begin(),
                        out.begin() + (size_t) r.code * FLOATS_PER_POINT);
    }
    for (int i = 0; i < kCountSlots; ++i)
        r.counts[i] = metrics[kCountMetric[i]];
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
//
// These bounds are TOOLCHAIN-PORTABLE, and were widened to that from an
// original 1e-6 px once the fixture was compared across hosts. IC-GN is a
// fixed-point iteration stopped at ||delta_p|| < 1e-3: two toolchains that
// differ only in FMA contraction take a slightly different path to the same
// basin and stop one step apart, so the disagreement is the size of a final
// IC-GN step, not of a rounding error. Measured between the Ubuntu GCC 11.4
// capture and Windows MinGW GCC on the same source: worst |du| 9.7e-4 px,
// |dv| 1.5e-3 px, strain 1.8e-5, corr 1.9e-5, over 772 points in two
// scenarios. The bounds below sit at roughly 3-5x that.
//
// What still makes this a gate is that the structure is compared exactly:
// the point count, the grid coordinates, and 13 of the 14 metrics slots are
// bit-exact across hosts and stay at tolerance zero. Perturbing the anchor
// stride by one fails on slots 3, 8, 19 and 22 before any value bound is
// consulted. A displacement regression of 5e-3 px is also an order of
// magnitude below anything the engine is specified to resolve.
constexpr float TOL_COORD = 0.0f;   // grid geometry must be identical
constexpr float TOL_DISP = 5e-3f;
constexpr float TOL_STRAIN = 1e-4f;
constexpr float TOL_CORR = 1e-4f;

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
            std::printf("    %-10s %d points  (anchor=%.0f pathA=%.0f "
                        "pathB=%.0f dropped=%.0f attempted=%.0f solved=%.0f "
                        "seedmode=%.0f lock=%.0f cov=%.3f)\n",
                        SCENARIOS[i].name, runs[i].code,
                        (double) runs[i].counts[10], (double) runs[i].counts[3],
                        (double) runs[i].counts[4], (double) runs[i].counts[11],
                        (double) runs[i].counts[0], (double) runs[i].counts[1],
                        (double) runs[i].counts[9], (double) runs[i].counts[12],
                        (double) runs[i].counts[13]);
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

        for (int i = 0; i < kCountSlots; ++i) {
            const float d = std::fabs(g.counts[i] - c.counts[i]);
            if (d > kCountTol[i])
                std::printf("  [%s] metrics[%d] changed %g -> %g\n",
                            SCENARIOS[s].name, kCountMetric[i],
                            (double) g.counts[i], (double) c.counts[i]);
            CHECK(d <= kCountTol[i]);
        }

        REQUIRE(g.points.size() == c.points.size());
        int mismatches = 0;
        float worst[FLOATS_PER_POINT] = {};
        for (size_t k = 0; k < c.points.size(); ++k) {
            const int slot = static_cast<int>(k % FLOATS_PER_POINT);
            const float dev = std::fabs(g.points[k] - c.points[k]);
            if (dev > worst[slot]) worst[slot] = dev;
            if (dev > tol[slot]) {
                if (mismatches < 5)
                    std::printf("  [%s] point %zu slot %d: %.9g -> %.9g\n",
                                SCENARIOS[s].name, k / FLOATS_PER_POINT, slot,
                                (double) g.points[k], (double) c.points[k]);
                ++mismatches;
            }
        }
        std::printf("  [%s] %d points, %d value mismatches "
                    "(worst: u %.3g v %.3g exx %.3g eyy %.3g exy %.3g corr %.3g)\n",
                    SCENARIOS[s].name, c.code, mismatches,
                    (double) worst[2], (double) worst[3], (double) worst[4],
                    (double) worst[5], (double) worst[6], (double) worst[7]);
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
