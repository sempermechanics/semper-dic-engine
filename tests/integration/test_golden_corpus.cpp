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
// -ffast-math is enabled for host tests (see tests/CMakeLists.txt), so
// bit-exact reproducibility is not something this build guarantees even
// for an IDENTICAL binary rerun — hence a tolerance-based diff, not a
// byte-for-byte one. Fixtures are captured from Ubuntu GCC Release.
// GitHub-hosted runners can still differ by a few thousandths of a
// pixel from a local container under -ffast-math, mostly on a handful
// of poorly conditioned 6x6 subsets. The tolerances below cover that
// noise while still failing on a status flip or a systematic field shift.
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

    // Converged-point value tolerance. Pixel-scale (u/v) vs strain-scale
    // (ux/uy/vx/vy) differ by orders of magnitude, so each gets its own bound
    // rather than one shared epsilon being too loose for one and too tight
    // for the other. Sized to absorb -ffast-math runner-to-runner noise
    // (observed up to ~5.5e-3 px / ~3e-4 strain on GitHub vs a local Ubuntu
    // container, on a few high-strain 6x6 subsets) while still catching a
    // real field shift.
    constexpr float TOL_DISPLACEMENT_PX = 1e-2f;
    constexpr float TOL_STRAIN = 1e-3f;

    // solve_icgn returns status 1 for two different things, and the fixture
    // records enough to tell them apart. A hard reject -- the 90%-valid-pixel
    // guard, the deactivation check, the minimum-gradient check -- returns the
    // sentinel score 2.0f. Exhausting the iteration budget returns the real
    // final ZNSSD, which for a nearly-converged point is tiny. Anything below
    // the sentinel therefore carries a real measurement.
    constexpr float REJECT_SENTINEL_SCORE = 2.0f;

    // A status flip that survives the value check is excused, but only up to a
    // point, and the useful quantity is the *imbalance*, not the count.
    //
    // Host arithmetic noise pushes a point across `delta_p.norm() < 1e-3` in
    // whichever direction its last bits happen to fall, so it produces flips in
    // both directions with no bias. A shrunken iteration budget can only ever
    // move a point from "converged" to "budget exhausted" -- never the reverse.
    // The signed difference therefore separates the two causes where a raw
    // count does not.
    //
    // Measured, by perturbing `kIcgnMaxIter` and rerunning both suites:
    //
    //   configuration           4x4 flips (net)   6x6 flips (net)
    //   Ubuntu GCC 11.4              0 (0)             2 (0)
    //   Windows MinGW GCC            0 (0)             1 (+1)
    //   Ubuntu, kIcgnMaxIter-2       2 (+2)            3 (+3)
    //   Windows, kIcgnMaxIter-2      2 (+2)            4 (+4)
    //   either host, kIcgnMaxIter-8   3 and 6 HARD mismatches -- fails loudly
    //
    // A net cap of 2 leaves a point of headroom over the worst clean host and
    // still fails the -2 perturbation through the 6x6 case on both hosts. A raw
    // count cannot do both: Ubuntu is already at 2 clean and goes to 3 perturbed.
    // The total cap is a backstop for churn in both directions at once. Every
    // perturbed flip measured was 0 -> 1, on both hosts, which is the
    // directional argument above holding up rather than being assumed.
    constexpr int MAX_FLIP_IMBALANCE = 2;
    constexpr int MAX_SLOW_CONVERGENCE_FLIPS = 6;

    bool values_agree(const GoldenPoint &g, const GoldenPoint &c) {
        return std::fabs(g.u - c.u) <= TOL_DISPLACEMENT_PX &&
               std::fabs(g.v - c.v) <= TOL_DISPLACEMENT_PX &&
               std::fabs(g.ux - c.ux) <= TOL_STRAIN &&
               std::fabs(g.uy - c.uy) <= TOL_STRAIN &&
               std::fabs(g.vx - c.vx) <= TOL_STRAIN &&
               std::fabs(g.vy - c.vy) <= TOL_STRAIN;
    }

    void compare_against_golden(bool use_6x6, const char *label, const std::string &suffix) {
        const auto path = golden_file_path() + suffix;
        const auto golden = read_golden(path);
        REQUIRE(!golden.empty()); // "no golden captured yet" — run with SEMPER_GOLDEN_CAPTURE=1 first
        const auto current = run_corpus(use_6x6);
        REQUIRE(current.size() == golden.size());

        int status_mismatches = 0;
        int value_mismatches = 0;
        int flips_to_exhausted = 0; // converged -> budget exhausted
        int flips_to_converged = 0; // the reverse
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
                // A point whose IC-GN is still creeping when the iteration
                // budget runs out lands on either side of
                // `delta_p.norm() < 0.001f` purely on the host's last-bit
                // arithmetic, and then reports "budget exhausted" instead of
                // "converged" while arriving at the same answer. Both hosts
                // measured so far show it: Ubuntu GCC 11.4 flips (368,192) and
                // (170,434) in opposite directions, Windows MinGW flips
                // (258,126), and in every case the two runs agree on all six
                // values to within 4e-3 px.
                //
                // A flip is therefore excused only when neither side is a hard
                // reject and the six values still agree. A flip into or out of
                // a hard reject, or one that moves the answer, is a real change
                // and still fails. The excused ones are counted by direction and
                // gated below.
                const bool both_measured =
                        g.correlation_score < REJECT_SENTINEL_SCORE &&
                        c.correlation_score < REJECT_SENTINEL_SCORE;
                if (both_measured && values_agree(g, c)) {
                    if (c.status != 0) ++flips_to_exhausted;
                    else ++flips_to_converged;
                    std::printf("  [%s] (%d,%d): status %d -> %d at unchanged values "
                                "(slow convergence at the iteration cap, corr %.2e -> %.2e)\n",
                                label, g.x, g.y, g.status, c.status,
                                (double) g.correlation_score, (double) c.correlation_score);
                    continue;
                }
                ++status_mismatches;
                std::printf("  [%s] (%d,%d): status changed %d -> %d (corr %.2e -> %.2e)\n",
                            label, g.x, g.y, g.status, c.status,
                            (double) g.correlation_score, (double) c.correlation_score);
                continue;
            }
            if (g.status != 0) continue; // only compare values where both sides converged
            if (!values_agree(g, c)) {
                ++value_mismatches;
                std::printf("  [%s] (%d,%d): u %.6f->%.6f v %.6f->%.6f ux %.7f->%.7f "
                            "uy %.7f->%.7f vx %.7f->%.7f vy %.7f->%.7f\n",
                            label, g.x, g.y, g.u, c.u, g.v, c.v, g.ux, c.ux, g.uy, c.uy, g.vx, c.vx, g.vy, c.vy);
            }
        }

        const int flips = flips_to_exhausted + flips_to_converged;
        const int imbalance = flips_to_exhausted - flips_to_converged;
        std::printf("  [%s] %zu points: %d status mismatches, %d value mismatches, "
                    "%d slow-convergence flips (%+d net, %d to exhausted, %d to "
                    "converged) (tol u/v=%.1e px, strain=%.1e)\n",
                    label, current.size(), status_mismatches, value_mismatches,
                    flips, imbalance, flips_to_exhausted, flips_to_converged,
                    (double) TOL_DISPLACEMENT_PX, (double) TOL_STRAIN);
        CHECK(status_mismatches == 0);
        CHECK(value_mismatches == 0);
        CHECK(std::abs(imbalance) <= MAX_FLIP_IMBALANCE);
        CHECK(flips <= MAX_SLOW_CONVERGENCE_FLIPS);
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
