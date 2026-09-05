// =====================================================================
// SUITE: ClParity — src/gpu/hessian_dispatch.cpp + src/gpu/kernels/hessian_prepass.cl
//
// GPU Phase 3's parity gate, and a continuation of the ClParity suite the
// strain kernel opened. The claim is the same one: the pre-pass kernel and
// Semper::SubsetPrecomputer::compute_hessian_only produce the SAME floats,
// bit for bit, for every grid point — including the points both refuse.
// Everything here compares with ==. A failure is a divergence to find, never
// a tolerance to widen (docs/DETERMINISM.md, docs/GPU_ACCELERATION.md §1).
//
// Phase 3a is what made this testable: the CPU reference used to invert the
// static Hessian with Eigen's PartialPivLU, whose fixed-size kernels are
// packed per -march and which a kernel cannot reproduce. It now shares
// semper_mat6_add_outer / semper_mat6_symmetrize / semper_inv6x6 with the
// device, verbatim.
//
// What is deliberately covered:
//
//   * interior points that invert cleanly;
//   * points whose subset hangs off the image, rejected before any
//     accumulation;
//   * points behind a stamped -5.0f "ghost wall", which fail the
//     valid_pixels >= n/2 test;
//   * points with a flat (rank-deficient) neighbourhood, rejected by the
//     det / cond_2x2 thresholds AFTER the Hessian is built — those must
//     agree on the raw H as well as on the rejection;
//   * pool slots the caller did not ask for, which must come back exactly
//     as they went in.
//
// One documented asymmetry, in data nothing reads: when a point never
// reaches the accumulation, the CPU returns a default-constructed
// CachedHessianData whose Eigen H and H_inv are UNINITIALISED while the GPU
// path leaves them zeroed. precompute_subset_fast re-runs the full
// precompute whenever valid is false, so neither is ever consumed. The
// comparison below therefore checks valid/mean/std everywhere and H/H_inv
// exactly where the CPU actually wrote them — computed from the image, not
// assumed.
//
// Like the rest of ClParity, every test must pass on a machine with no
// OpenCL: the device-dependent ones report themselves skipped rather than
// passing vacuously.
// =====================================================================
#include "framework/test_framework.h"
#include "framework/synthetic.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(SEMPER_OPENCL)

#include "gpu/cl_runtime.hpp"
#include "gpu/hessian_dispatch.hpp"

#include <semper/gpu/embedded_kernels.hpp>
#include <semper/subset.hpp>

using Semper::CachedHessianData;
using Semper::Image;
using Semper::SubsetPrecomputer;
using Semper::gpu::caps;
using Semper::gpu::compute_hessian_pool_gpu;
using Semper::gpu::reset_for_testing;

namespace {

// This stage is fp32 end to end, so it needs correctly-rounded divide and
// sqrt — NOT cl_khr_fp64. A device with one and not the other still runs it.
bool stage_available() { return caps().available && caps().exact_fp32; }

void report_skip(const char *what) {
    const auto &c = caps();
    if (!c.available)
        std::printf("  ClParity: skipped %s (no OpenCL device) — %s\n", what,
                    c.unavailable_reason.c_str());
    else
        std::printf("  ClParity: skipped %s (device '%s' lacks correctly-rounded fp32)\n",
                    what, c.device_name.c_str());
}

constexpr int W = 160, H = 140;

Image test_image() {
    dictest::SpeckleField field(4241, W, H);
    return dictest::make_reference_image(field, W, H);
}

// Did compute_hessian_only get as far as writing H for this point? Both
// early returns are reproduced here from the source rather than inferred
// from the output, so the comparison below cannot be fooled by a point
// whose real mean happens to be 0 and std happens to be 1.
bool reached_accumulation(const Image &img, int cx, int cy, int dim) {
    const int half = dim / 2, n = dim * dim;
    if (cx - half < 0 || cx + half >= img.width ||
        cy - half < 0 || cy + half >= img.height)
        return false;
    int valid_pixels = 0;
    for (int oy = -half; oy <= half; ++oy)
        for (int ox = -half; ox <= half; ++ox)
            if (img.intensities[(size_t) (cy + oy) * img.width + cx + ox] >= -5.0f)
                ++valid_pixels;
    return !(valid_pixels < n * 0.5f);
}

struct Geometry {
    int rect_x, rect_y, step, grid_w, grid_h, dim;
};

// The CPU pre-pass, in exactly the shape full_field_solver.cpp:248-260 runs
// it: one entry per grid point, only for the points `wanted` asks for.
std::vector<CachedHessianData> cpu_pool(const Image &img, const Geometry &g,
                                        const std::vector<unsigned char> &wanted) {
    std::vector<CachedHessianData> pool((size_t) g.grid_w * g.grid_h);
    for (int i = 0; i < g.grid_w * g.grid_h; ++i) {
        if (!wanted[(size_t) i]) continue;
        const int gx = i % g.grid_w, gy = i / g.grid_w;
        pool[(size_t) i] = SubsetPrecomputer::compute_hessian_only(
                img, g.rect_x + gx * g.step, g.rect_y + gy * g.step, g.dim);
    }
    return pool;
}

// Exact comparison, printing the first few disagreements so a failure says
// where rather than merely that.
int count_mismatches(const Image &img, const Geometry &g,
                     const std::vector<unsigned char> &wanted,
                     const std::vector<CachedHessianData> &cpu,
                     const std::vector<CachedHessianData> &gpu,
                     const char *label) {
    int bad = 0;
    for (int i = 0; i < g.grid_w * g.grid_h; ++i) {
        if (!wanted[(size_t) i]) continue;
        const CachedHessianData &a = cpu[(size_t) i], &b = gpu[(size_t) i];
        const char *why = nullptr;

        if (a.valid != b.valid) why = "valid";
        else if (a.mean_intensity != b.mean_intensity) why = "mean";
        else if (a.std_dev != b.std_dev) why = "std";

        const int gx = i % g.grid_w, gy = i / g.grid_w;
        if (!why && reached_accumulation(img, g.rect_x + gx * g.step,
                                         g.rect_y + gy * g.step, g.dim)) {
            for (int r = 0; r < 6 && !why; ++r)
                for (int c = 0; c < 6 && !why; ++c) {
                    if (a.H(r, c) != b.H(r, c)) why = "H";
                    // H_inv is only written on the accepted path; on the
                    // rejected one the CPU zeroes it and so does the kernel.
                    else if (a.H_inv(r, c) != b.H_inv(r, c)) why = "H_inv";
                }
        }

        if (!why) continue;
        if (bad < 6)
            std::printf("  [%s] idx %d (%d,%d): %s differs — cpu valid=%d mean=%.9g std=%.9g "
                        "H00=%.9g Hi00=%.9g | gpu valid=%d mean=%.9g std=%.9g H00=%.9g Hi00=%.9g\n",
                        label, i, gx, gy, why, (int) a.valid, (double) a.mean_intensity,
                        (double) a.std_dev, (double) a.H(0, 0), (double) a.H_inv(0, 0),
                        (int) b.valid, (double) b.mean_intensity, (double) b.std_dev,
                        (double) b.H(0, 0), (double) b.H_inv(0, 0));
        ++bad;
    }
    return bad;
}

int valid_count(const std::vector<CachedHessianData> &pool) {
    int k = 0;
    for (const CachedHessianData &d : pool)
        if (d.valid) ++k;
    return k;
}

bool run_gpu(const Image &img, const Geometry &g,
             const std::vector<unsigned char> &wanted,
             std::vector<CachedHessianData> *out) {
    out->assign((size_t) g.grid_w * g.grid_h, CachedHessianData());
    return compute_hessian_pool_gpu(img, g.rect_x, g.rect_y, g.step, g.grid_w,
                                    g.grid_h, g.dim, wanted.data(), out->data(),
                                    g.grid_w * g.grid_h);
}

} // namespace

// The gate itself, and the one test that runs everywhere. A machine with no
// device must get a clean false and an untouched pool — a dispatch that
// half-filled the pool before failing would corrupt the entries the caller
// then computes on the CPU.
TEST_CASE(ClParity, HessianDispatchDeclinesCleanlyWhenStageUnavailable) {
    Image img = test_image();
    const Geometry g{20, 20, 9, 8, 7, 21};
    const std::vector<unsigned char> wanted((size_t) g.grid_w * g.grid_h, 1u);

    std::vector<CachedHessianData> pool((size_t) g.grid_w * g.grid_h);
    for (CachedHessianData &d : pool) {
        d.mean_intensity = 7.0f;   // recognisable prior contents
        d.std_dev = 7.0f;
        d.valid = true;
    }

    const bool ran = compute_hessian_pool_gpu(img, g.rect_x, g.rect_y, g.step,
                                              g.grid_w, g.grid_h, g.dim,
                                              wanted.data(), pool.data(),
                                              g.grid_w * g.grid_h);
    if (!stage_available()) {
        CHECK(!ran);
        // Untouched, not merely "not garbage".
        CHECK(pool[0].mean_intensity == 7.0f);
        CHECK(pool[pool.size() - 1].std_dev == 7.0f);
        CHECK(pool[3].valid);
        report_skip("hessian parity (gate test still ran)");
    } else {
        CHECK(ran);
    }
}

// Null arguments and nonsense geometry must be refused rather than
// dereferenced or launched.
TEST_CASE(ClParity, HessianDispatchRejectsBadArguments) {
    Image img = test_image();
    std::vector<unsigned char> wanted(64, 1u);
    std::vector<CachedHessianData> pool(64);

    CHECK(!compute_hessian_pool_gpu(img, 20, 20, 9, 8, 8, 21, nullptr, pool.data(), 64));
    CHECK(!compute_hessian_pool_gpu(img, 20, 20, 9, 8, 8, 21, wanted.data(), nullptr, 64));
    // out_count smaller than the grid it was asked to fill.
    CHECK(!compute_hessian_pool_gpu(img, 20, 20, 9, 8, 8, 21, wanted.data(), pool.data(), 63));
    // Degenerate geometry.
    CHECK(!compute_hessian_pool_gpu(img, 20, 20, 9, 0, 8, 21, wanted.data(), pool.data(), 64));
    CHECK(!compute_hessian_pool_gpu(img, 20, 20, 9, 8, 8, 0, wanted.data(), pool.data(), 64));
}

// The env override is the documented way to force the CPU path (there is no
// FullFieldParams field for it — see docs/CONTRACT.md). It must shut this
// stage down too, not just the probe and the strain kernel.
TEST_CASE(ClParity, HessianDispatchHonoursEnvDisable) {
    reset_for_testing();
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "1");
#else
    setenv("SEMPER_OPENCL_DISABLE", "1", 1);
#endif
    Image img = test_image();
    const Geometry g{20, 20, 9, 8, 7, 21};
    const std::vector<unsigned char> wanted((size_t) g.grid_w * g.grid_h, 1u);
    std::vector<CachedHessianData> pool;
    const bool ran = run_gpu(img, g, wanted, &pool);
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "");
#else
    unsetenv("SEMPER_OPENCL_DISABLE");
#endif
    reset_for_testing();

    CHECK(!ran);
    CHECK(valid_count(pool) == 0);
}

// The embedded source must be self-contained, must keep contraction off,
// and must be free of any double: this stage is gated on exact_fp32 and has
// to build on a device with no cl_khr_fp64 at all. A stray double would
// make it fail to compile there, and since a program build failure must not
// flip caps().available, the symptom would be a silent CPU fallback.
TEST_CASE(ClParity, EmbeddedHessianKernelIsSelfContained) {
    const std::string src = Semper::gpu::kernels::HESSIAN_PREPASS;
    REQUIRE(!src.empty());
    const size_t entry = src.find("__kernel void semper_hessian_prepass");
    REQUIRE(entry != std::string::npos);
    CHECK(src.find("#include <semper/") == std::string::npos);
    CHECK(src.find("FP_CONTRACT OFF") != std::string::npos);
    // The shared routines the CPU reference also calls.
    CHECK(src.find("semper_mat6_add_outer") != std::string::npos);
    CHECK(src.find("semper_mat6_symmetrize") != std::string::npos);
    CHECK(src.find("semper_inv6x6") != std::string::npos);
    // The header's fp64 block is conditional...
    CHECK(src.find("SEMPER_HAS_FP64") != std::string::npos);
    // ...and this kernel is outside it.
    CHECK(src.find("double", entry) == std::string::npos);
}

// The main event: several geometries, every point requested.
TEST_CASE(ClParity, HessianMatchesCpuBitExactly) {
    if (!stage_available()) { report_skip("HessianMatchesCpuBitExactly"); return; }

    struct Case { Geometry g; const char *name; };
    const Case cases[] = {
        {{20, 20,  9, 12, 10, 21}, "12x10 step9 dim21"},
        {{15, 12,  7, 17, 15, 31}, "17x15 step7 dim31"},
        {{ 8,  8, 11,  9,  8, 15}, "9x8 step11 dim15"},
        // Runs off the right and bottom edges, so a band of points is
        // rejected before any accumulation happens.
        {{100, 90, 13, 10, 10, 25}, "10x10 step13 dim25 (edge overrun)"},
    };

    Image img = test_image();
    for (const Case &c : cases) {
        const int n = c.g.grid_w * c.g.grid_h;
        const std::vector<unsigned char> wanted((size_t) n, 1u);

        const std::vector<CachedHessianData> cpu = cpu_pool(img, c.g, wanted);
        std::vector<CachedHessianData> gpu;
        REQUIRE(run_gpu(img, c.g, wanted, &gpu));

        const int bad = count_mismatches(img, c.g, wanted, cpu, gpu, c.name);
        std::printf("  ClParity[%s]: %d points, %d valid, %d mismatch\n",
                    c.name, n, valid_count(cpu), bad);
        // A case where nothing inverted would pass while proving nothing.
        CHECK(valid_count(cpu) > 0);
        CHECK(bad == 0);
    }
}

// Parity on the points that are given up on, and on the two distinct ways
// of giving up: the ghost wall (rejected before the Hessian exists) and the
// det / cond_2x2 thresholds (rejected after it does, so the raw H must
// still match). A flat patch has zero gradients, hence a singular H.
TEST_CASE(ClParity, HessianRejectionCasesMatchExactly) {
    if (!stage_available()) { report_skip("HessianRejectionCasesMatchExactly"); return; }

    Image img = test_image();
    // Stamp a ghost wall and a flat region straight into the prepared
    // buffers: both paths read the same arrays, so this is a legitimate
    // input, and it is the only way to reach these branches from a speckle
    // field that clamps to [5, 250].
    for (int y = 30; y < 70; ++y)
        for (int x = 30; x < 70; ++x)
            img.intensities[(size_t) y * W + x] = -10.0f;
    for (int y = 90; y < 130; ++y)
        for (int x = 20; x < 60; ++x) {
            img.intensities[(size_t) y * W + x] = 100.0f;
            img.grad_x[(size_t) y * W + x] = 0.0f;
            img.grad_y[(size_t) y * W + x] = 0.0f;
        }

    const Geometry g{18, 18, 6, 20, 19, 21};
    const int n = g.grid_w * g.grid_h;
    const std::vector<unsigned char> wanted((size_t) n, 1u);

    const std::vector<CachedHessianData> cpu = cpu_pool(img, g, wanted);
    std::vector<CachedHessianData> gpu;
    REQUIRE(run_gpu(img, g, wanted, &gpu));

    const int good = valid_count(cpu);
    const int bad = count_mismatches(img, g, wanted, cpu, gpu, "rejection");
    std::printf("  ClParity[hessian rejection]: %d points, %d valid, %d mismatch\n",
                n, good, bad);
    // If everything (or nothing) was rejected the test is not testing what
    // it claims to.
    CHECK(good > 0);
    CHECK(good < n);
    CHECK(bad == 0);
}

// The pre-pass skips points whose result the solver already has. Those pool
// slots belong to the caller and must come back byte-for-byte as they went
// in — the CPU loop leaves them default-constructed, and a kernel that
// wrote its defaults over them would quietly discard solved points.
TEST_CASE(ClParity, HessianLeavesUnwantedPoolSlotsAlone) {
    if (!stage_available()) { report_skip("HessianLeavesUnwantedPoolSlotsAlone"); return; }

    Image img = test_image();
    const Geometry g{20, 20, 9, 12, 10, 21};
    const int n = g.grid_w * g.grid_h;

    std::vector<unsigned char> wanted((size_t) n, 1u);
    for (int i = 0; i < n; i += 3) wanted[(size_t) i] = 0u;

    std::vector<CachedHessianData> gpu((size_t) n);
    for (CachedHessianData &d : gpu) {
        d.mean_intensity = 3.5f;
        d.std_dev = 11.25f;
        d.valid = true;
        d.H.setConstant(2.0f);
        d.H_inv.setConstant(-4.0f);
    }
    REQUIRE(compute_hessian_pool_gpu(img, g.rect_x, g.rect_y, g.step, g.grid_w,
                                     g.grid_h, g.dim, wanted.data(), gpu.data(), n));

    int touched = 0;
    for (int i = 0; i < n; i += 3) {
        const CachedHessianData &d = gpu[(size_t) i];
        if (d.mean_intensity != 3.5f || d.std_dev != 11.25f || !d.valid ||
            d.H(2, 4) != 2.0f || d.H_inv(5, 1) != -4.0f)
            ++touched;
    }
    CHECK(touched == 0);

    // And the requested ones did get written, or the check above is vacuous.
    const std::vector<CachedHessianData> cpu = cpu_pool(img, g, wanted);
    CHECK(count_mismatches(img, g, wanted, cpu, gpu, "unwanted") == 0);
}

// Same input twice must give the same bytes. A kernel whose result depended
// on work-group scheduling would still pass the parity tests intermittently.
TEST_CASE(ClParity, HessianIsRunToRunReproducible) {
    if (!stage_available()) { report_skip("HessianIsRunToRunReproducible"); return; }

    Image img = test_image();
    const Geometry g{15, 12, 7, 17, 15, 31};
    const int n = g.grid_w * g.grid_h;
    const std::vector<unsigned char> wanted((size_t) n, 1u);

    std::vector<CachedHessianData> a, b;
    REQUIRE(run_gpu(img, g, wanted, &a));
    REQUIRE(run_gpu(img, g, wanted, &b));

    int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (a[(size_t) i].valid != b[(size_t) i].valid ||
            a[(size_t) i].mean_intensity != b[(size_t) i].mean_intensity ||
            a[(size_t) i].std_dev != b[(size_t) i].std_dev) { ++bad; continue; }
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 6; ++c)
                if (a[(size_t) i].H(r, c) != b[(size_t) i].H(r, c) ||
                    a[(size_t) i].H_inv(r, c) != b[(size_t) i].H_inv(r, c)) { ++bad; r = 6; break; }
    }
    CHECK(bad == 0);
}

#else // !SEMPER_OPENCL

TEST_CASE(ClParity, HessianKernelAbsentWithoutOpenCLBuild) {
    std::printf("  ClParity: hessian pre-pass compiled out (SEMPER_OPENCL=OFF)\n");
    CHECK(true);
}

#endif // SEMPER_OPENCL
