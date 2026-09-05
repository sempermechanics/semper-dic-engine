// Golden-corpus equivalence harness for Phase 3 (engine SIMD).
//
// EnginePipelineSmokeTest already checks ABSOLUTE correctness (median error
// vs analytic ground truth). This checks something different and necessary
// alongside it: RELATIVE equivalence — did a change move any point's
// convergence status, or shift a converged point's u/v/strains beyond a
// tight tolerance, relative to a captured "before" run? A change could pass
// the aggregate median-error check while still silently flipping which
// points converge or introducing a small systematic bias; this catches that.
//
// The whole test tree compiles -fno-fast-math -ffp-contract=off (see
// tests/CMakeLists.txt), so an identical binary rerun IS bit-exact, and so
// is the same source built at a different -march level. The diff below is
// still tolerance-based rather than byte-for-byte for one narrower reason:
// the corpus synthesizes its own input images from several hundred std::exp
// terms per pixel, so a different libm can move the last ulp of the INPUT.
// See the tolerance comment at TOL_DISPLACEMENT_PX below, and
// docs/DETERMINISM.md. The byte-exact gate is the `determinism` CI job.
//
// Sanitizer and coverage builds skip this suite: instrumentation and
// Debug -O0 change floating-point results (and which subsets initialize)
// enough to trip the relative check. The other suites still cover those
// builds.
//
// Usage:
//   Capture (before making a change):
//     SEMPER_GOLDEN_CAPTURE=1 ./dic_tests GoldenCorpus
//   Compare (after making a change):
//     ./dic_tests GoldenCorpus
//
// Golden file location: $SEMPER_GOLDEN_FILE, or tests/fixtures/golden_corpus.bin
// (next to the source tree, so it survives out-of-source builds) by default.
#include "framework/test_framework.h"
#include "framework/synthetic.h"
#include <semper/solver.hpp>
#include <semper/subset.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using Semper::Image;
using Semper::SubsetData;
using Semper::SubsetPrecomputer;
using Semper::OptimizationEngine;
using Semper::AnalysisResult;
using Semper::INIT_NO_SIMPLEX;

namespace {

    constexpr int W = 512, H = 512;
    constexpr int SUBSET_SIZE = 27;
    constexpr int GRID_LO = 60, GRID_HI = 452, GRID_STEP = 22; // 18x18 = 324 subsets/scenario

    // Two scenarios per interpolator mode: pure translation (the common case),
    // and a combined affine (rotation + skew + normal strain) that exercises
    // interpolation across a wider range of sub-pixel offsets and gradients.
    struct Scenario {
        const char *name;
        float u, v, ux, uy, vx, vy;
    };

    const Scenario SCENARIOS[] = {
        {"translate", 3.2f, -1.7f, 0.0f, 0.0f, 0.0f, 0.0f},
        {"affine", 2.1f, 1.4f, 0.015f, -0.008f, 0.006f, 0.012f},
    };

    struct GoldenPoint {
        int32_t x, y;
        int32_t status;
        float u, v, ux, uy, vx, vy;
        float correlation_score;
    };

    std::string golden_file_path() {
        if (const char *env = std::getenv("SEMPER_GOLDEN_FILE")) return env;
        return std::string(NATIVE_TESTS_SOURCE_DIR) + "/fixtures/golden_corpus.bin";
    }

    bool capture_mode() {
        const char *env = std::getenv("SEMPER_GOLDEN_CAPTURE");
        return env && std::strcmp(env, "1") == 0;
    }

    std::vector<GoldenPoint> run_corpus(bool use_6x6) {
        std::vector<GoldenPoint> out;
        for (const auto &scenario : SCENARIOS) {
            dictest::SpeckleField field(/*seed=*/42, W, H);
            dictest::AffineDeformation def;
            def.u = scenario.u;
            def.v = scenario.v;
            def.ux = scenario.ux;
            def.uy = scenario.uy;
            def.vx = scenario.vx;
            def.vy = scenario.vy;
            def.cx = W / 2.0f;
            def.cy = H / 2.0f;

            const Image ref = dictest::make_reference_image(field, W, H);
            const Image deformed = dictest::make_deformed_image(field, W, H, def);

            for (int y = GRID_LO; y <= GRID_HI; y += GRID_STEP) {
                for (int x = GRID_LO; x <= GRID_HI; x += GRID_STEP) {
                    SubsetData subset;
                    SubsetPrecomputer::precompute_subset(subset, ref, x, y, SUBSET_SIZE);
                    if (!subset.is_initialized) continue;

                    OptimizationEngine engine;
                    engine.use_6x6_interpolator = use_6x6;
                    AnalysisResult res = engine.calculate_deformation(
                        subset, deformed, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, INIT_NO_SIMPLEX);

                    out.push_back(GoldenPoint{
                        x, y, res.status,
                        res.u, res.v, res.ux, res.uy, res.vx, res.vy,
                        res.correlation_score,
                    });
                }
            }
        }
        return out;
    }

    bool write_golden(const std::string &path, const std::vector<GoldenPoint> &points) {
        FILE *f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        uint32_t count = static_cast<uint32_t>(points.size());
        std::fwrite(&count, sizeof(count), 1, f);
        std::fwrite(points.data(), sizeof(GoldenPoint), points.size(), f);
        std::fclose(f);
        return true;
    }

    // Empty on any read failure (missing/short/corrupt file) — callers must
    // treat that as "no golden captured yet", not "zero points differ".
    std::vector<GoldenPoint> read_golden(const std::string &path) {
        std::vector<GoldenPoint> points;
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) return points;
        uint32_t count = 0;
        if (std::fread(&count, sizeof(count), 1, f) != 1) {
            std::fclose(f);
            return points;
        }
        points.resize(count);
        size_t got = std::fread(points.data(), sizeof(GoldenPoint), count, f);
        std::fclose(f);
        if (got != count) points.clear();
        return points;
    }

    // Converged-point value tolerance.
    //
    // These used to be 1e-2 px / 1e-3 strain, sized to absorb -ffast-math
    // runner-to-runner noise. That noise is gone: the engine is now compiled
    // under strict IEEE FP with a pinned reduction order, and produces
    // bit-identical results across -march levels (see docs/DETERMINISM.md).
    //
    // They are tightened by four orders of magnitude rather than to exactly
    // zero for one specific reason. The corpus synthesizes its images in
    // tests/framework/synthetic.h by summing several hundred std::exp terms
    // per pixel, so the INPUT depends on the host libm; a different glibc
    // can move the last ulp of exp and shift every downstream value slightly.
    // Pinning to zero here would make the fixture toolchain-specific and
    // fail on any CI image but the one that captured it.
    //
    // The true bit-exactness gate is therefore not this tolerance but the
    // `determinism` CI job, which builds the same source at two -march
    // levels and byte-compares the captured corpora. This bound is tight
    // enough that any real change in engine arithmetic still trips it.
    constexpr float TOL_DISPLACEMENT_PX = 1e-6f;
    constexpr float TOL_STRAIN = 1e-7f;

    void compare_against_golden(bool use_6x6, const char *label, const std::string &suffix) {
        const auto path = golden_file_path() + suffix;
        const auto golden = read_golden(path);
        REQUIRE(!golden.empty()); // "no golden captured yet" — run with SEMPER_GOLDEN_CAPTURE=1 first
        const auto current = run_corpus(use_6x6);
        REQUIRE(current.size() == golden.size());

        int status_mismatches = 0;
        int value_mismatches = 0;
        for (size_t i = 0; i < current.size(); ++i) {
            const auto &g = golden[i];
            const auto &c = current[i];
            if (g.x != c.x || g.y != c.y) {
                std::printf("  [%s] grid mismatch at index %zu (%d,%d) vs (%d,%d) — "
                            "scenario/grid layout changed, golden is stale\n",
                            label, i, g.x, g.y, c.x, c.y);
                REQUIRE(false);
            }
            if (g.status != c.status) {
                ++status_mismatches;
                std::printf("  [%s] (%d,%d): status changed %d -> %d\n", label, g.x, g.y, g.status, c.status);
                continue;
            }
            if (g.status != 0) continue; // only compare values where both sides converged
            const bool ok =
                std::fabs(g.u - c.u) <= TOL_DISPLACEMENT_PX &&
                std::fabs(g.v - c.v) <= TOL_DISPLACEMENT_PX &&
                std::fabs(g.ux - c.ux) <= TOL_STRAIN &&
                std::fabs(g.uy - c.uy) <= TOL_STRAIN &&
                std::fabs(g.vx - c.vx) <= TOL_STRAIN &&
                std::fabs(g.vy - c.vy) <= TOL_STRAIN;
            if (!ok) {
                ++value_mismatches;
                std::printf("  [%s] (%d,%d): u %.6f->%.6f v %.6f->%.6f ux %.7f->%.7f "
                            "uy %.7f->%.7f vx %.7f->%.7f vy %.7f->%.7f\n",
                            label, g.x, g.y, g.u, c.u, g.v, c.v, g.ux, c.ux, g.uy, c.uy, g.vx, c.vx, g.vy, c.vy);
            }
        }

        std::printf("  [%s] %zu points: %d status mismatches, %d value mismatches (tol u/v=%.1e px, strain=%.1e)\n",
                    label, current.size(), status_mismatches, value_mismatches,
                    (double) TOL_DISPLACEMENT_PX, (double) TOL_STRAIN);
        CHECK(status_mismatches == 0);
        CHECK(value_mismatches == 0);
    }

} // namespace

TEST_CASE(GoldenCorpus, CaptureOrCompare_4x4Bicubic) {
#if defined(SEMPER_SANITIZER_BUILD) || defined(SEMPER_COVERAGE_BUILD)
    std::printf("  [4x4] skipped under sanitizer/coverage (numeric golden is a Release-host gate)\n");
    return;
#endif
    if (capture_mode()) {
        const auto points = run_corpus(/*use_6x6=*/false);
        const auto path = golden_file_path() + ".bicubic";
        REQUIRE(write_golden(path, points));
        std::printf("  [4x4] captured %zu golden points to %s\n", points.size(), path.c_str());
        return;
    }
    compare_against_golden(/*use_6x6=*/false, "4x4", ".bicubic");
}

TEST_CASE(GoldenCorpus, CaptureOrCompare_6x6Keys) {
#if defined(SEMPER_SANITIZER_BUILD) || defined(SEMPER_COVERAGE_BUILD)
    std::printf("  [6x6] skipped under sanitizer/coverage (numeric golden is a Release-host gate)\n");
    return;
#endif
    if (capture_mode()) {
        const auto points = run_corpus(/*use_6x6=*/true);
        const auto path = golden_file_path() + ".keys6x6";
        REQUIRE(write_golden(path, points));
        std::printf("  [6x6] captured %zu golden points to %s\n", points.size(), path.c_str());
        return;
    }
    compare_against_golden(/*use_6x6=*/true, "6x6", ".keys6x6");
}
