// =====================================================================
// SUITE: ClParity — src/gpu/image_prep_dispatch.cpp + src/gpu/kernels/image_grad.cl
//
// GPU Phase 6's parity gate. The claim under test is not "the gradients are
// close enough"; it is that semper_image_gradients and the interior loop of
// Semper::Image::prepare_data write the SAME float into every element of
// both planes — border pixels included. Everything here compares with ==.
// If a case ever fails, the fix is to find the divergence, never to reach
// for CHECK_NEAR. See docs/DETERMINISM.md and docs/GPU_ACCELERATION.md.
//
// The arithmetic is four float operations, so unlike Phases 2–4 the risk
// here is not the numerics. It is indexing and coverage, and the geometries
// below are chosen for that:
//
//   * a NON-SQUARE image, so a transposed row/column index cannot pass. A
//     square one would let grad_x and grad_y swap unnoticed;
//   * an image whose pixel count is prime, so the launch cannot be assumed
//     to divide evenly into work-groups;
//   * images smaller than the 2-pixel border in one or both dimensions,
//     where the CPU's interior loop body never executes and every pixel must
//     come out zero on both paths;
//   * the -10.0f ghost-wall and -5.0f void sentinels the pipeline injects
//     into intensities, so the stencil is exercised on negative operands and
//     on the large cancellations they produce next to a bright pixel.
//
// The border band is compared, not skipped. The CPU zero-fills and then
// overwrites the interior; the kernel writes the zeros explicitly. Comparing
// only the interior would hide a kernel that left the band as whatever the
// device allocator handed it.
//
// Like ClRuntime, every test must pass on a machine with no OpenCL: the
// device-dependent ones report themselves skipped rather than passing
// vacuously.
// =====================================================================
#include "framework/test_framework.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(SEMPER_OPENCL)

#include "gpu/cl_runtime.hpp"
#include "gpu/image_prep_dispatch.hpp"

#include <semper/gpu/embedded_kernels.hpp>
#include <semper/image.hpp>

using Semper::Image;
using Semper::gpu::caps;
using Semper::gpu::compute_image_gradients_gpu;
using Semper::gpu::reset_for_testing;

namespace {

// This stage is fp32 end to end — three adds and a divide by 12 — so it
// needs correctly-rounded divide, NOT cl_khr_fp64.
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

// A deterministic speckle with the pipeline's two sentinels stirred in. The
// stride constants are coprime with the widths used below so the pattern
// does not line up with row boundaries.
Image make_image(int w, int h, unsigned seed) {
    std::vector<unsigned char> raw((size_t) (w > 0 ? w : 1) * (size_t) (h > 0 ? h : 1), 0u);
    for (size_t i = 0; i < raw.size(); ++i)
        raw[i] = (unsigned char) ((i * 61u + (i / (size_t) (w > 0 ? w : 1)) * 173u + seed * 7u) & 0xFFu);
    Image img(w, h, raw.data());
    for (size_t i = 0; i < img.intensities.size(); i += 37) img.intensities[i] = -10.0f;
    for (size_t i = 5; i < img.intensities.size(); i += 53) img.intensities[i] = -5.0f;
    return img;
}

// Bit-level, not value-level: -0.0f == 0.0f compares equal as floats, and a
// gradient plane that came back with the wrong zero sign is a real
// difference in a buffer this engine promises is reproducible.
int bit_mismatches(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return -1;
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++bad;
    return bad;
}

// Runs one geometry through both paths and returns the number of differing
// floats across both planes, or -1 if the device declined.
int compare_geometry(int w, int h, unsigned seed) {
    Image cpu = make_image(w, h, seed);
    Image gpu = cpu;                       // same intensities, no gradients yet
    cpu.prepare_data(false);
    if (!compute_image_gradients_gpu(gpu)) return -1;

    // The dispatch must leave intensities alone and size both planes exactly
    // as prepare_data does, or the == below would be comparing the wrong
    // things and still passing.
    if (bit_mismatches(cpu.intensities, gpu.intensities) != 0) return -2;
    if (gpu.grad_x.size() != cpu.grad_x.size()) return -3;
    if (gpu.grad_y.size() != cpu.grad_y.size()) return -3;

    const int bx = bit_mismatches(cpu.grad_x, gpu.grad_x);
    const int by = bit_mismatches(cpu.grad_y, gpu.grad_y);
    return bx + by;
}

} // namespace

// --- Structural: these run everywhere, device or not --------------------

// A degenerate image is refused before the device is touched, and refusal
// means the caller's Image is exactly as it was — not half-filled.
TEST_CASE(ClParity, ImageGradDispatchRejectsBadArguments) {
    Image empty(0, 0, nullptr);
    CHECK(!compute_image_gradients_gpu(empty));
    CHECK(empty.grad_x.empty());
    CHECK(empty.grad_y.empty());

    // Well-formed dimensions, but intensities shorter than width*height —
    // the shape a caller that forgot to load pixels would present.
    Image truncated = make_image(16, 16, 3u);
    truncated.intensities.resize(100);
    CHECK(!compute_image_gradients_gpu(truncated));
    CHECK(truncated.grad_x.empty());
    CHECK(truncated.grad_y.empty());
}

// SEMPER_OPENCL_DISABLE=1 must take this stage down with the rest of the
// backend, so a suspected device-side difference can be bisected without a
// rebuild.
TEST_CASE(ClParity, ImageGradDispatchHonoursEnvDisable) {
    reset_for_testing();
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "1");
#else
    setenv("SEMPER_OPENCL_DISABLE", "1", 1);
#endif
    Image img = make_image(32, 24, 11u);
    const bool ran = compute_image_gradients_gpu(img);
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "");
#else
    unsetenv("SEMPER_OPENCL_DISABLE");
#endif
    reset_for_testing();

    CHECK(!ran);
    CHECK(img.grad_x.empty());
    CHECK(img.grad_y.empty());
}

// The embedded source must be self-contained, must keep contraction off, and
// must be free of any double: this stage is gated on exact_fp32 and has to
// build on a device with no cl_khr_fp64 at all. A stray double would make it
// fail to compile there, and since a program build failure must not flip
// caps().available, the symptom would be a silent CPU fallback.
TEST_CASE(ClParity, EmbeddedImageGradKernelIsSelfContained) {
    const std::string src = Semper::gpu::kernels::IMAGE_GRAD;
    REQUIRE(!src.empty());
    const size_t entry = src.find("__kernel void semper_image_gradients");
    REQUIRE(entry != std::string::npos);
    CHECK(src.find("#include <semper/") == std::string::npos);
    // Without this the compiler is free to fuse 8.0f * p1 + m2 into a mad and
    // the two paths part company in the last bit on every pixel.
    CHECK(src.find("FP_CONTRACT OFF") != std::string::npos);
    // The shared stencil, not a device-side retyping of it.
    CHECK(src.find("SEMPER_CANON_DERIV5") != std::string::npos);
    CHECK(src.find("SEMPER_GRAD_BORDER") != std::string::npos);
    // No cross-lane anything: each work-item owns one pixel.
    CHECK(src.find("barrier(") == std::string::npos);
    CHECK(src.find("__local") == std::string::npos);
    CHECK(src.find("double", entry) == std::string::npos);
}

// The macro's parenthesisation is the contract — see canonical_math.h. This
// pins the expanded text, so a "simplification" to 8*(p1-m1)-(p2-m2) shows up
// here rather than as a one-bit drift in a gradient plane.
TEST_CASE(ClParity, CanonicalGradientStencilKeepsItsShape) {
    const std::string src = Semper::gpu::kernels::IMAGE_GRAD;
    REQUIRE(!src.empty());
    CHECK(src.find("((-(p2) + 8.0f * (p1) - 8.0f * (m1) + (m2)) / 12.0f)") !=
          std::string::npos);
}

// --- Device parity ------------------------------------------------------

TEST_CASE(ClParity, ImageGradMatchesCpuBitExactly) {
    if (!stage_available()) { report_skip("ImageGradMatchesCpuBitExactly"); return; }

    struct Case { int w, h; const char *why; };
    // 131*67 = 8777 (prime), so the last case also covers a launch that does
    // not divide evenly into work-groups.
    const Case cases[] = {
            {  64,  64, "square, one work-group multiple" },
            { 128,  96, "wide" },
            {  96, 128, "tall — catches a transposed stride" },
            { 257, 129, "odd in both dimensions" },
            { 131,  67, "prime pixel count, non-square" },
    };
    for (const Case &c : cases) {
        const int bad = compare_geometry(c.w, c.h, (unsigned) (c.w + c.h));
        if (bad == -1) { report_skip("ImageGradMatchesCpuBitExactly"); return; }
        if (bad != 0)
            std::printf("  ClParity: %dx%d (%s) — %d differing floats\n",
                        c.w, c.h, c.why, bad);
        CHECK(bad == 0);
    }
}

// Images at or below the border width in one or both dimensions. The CPU's
// interior loop body never runs, so both planes must be entirely zero — and
// the kernel has to reach that answer through its bounds test rather than by
// reading off the end of the buffer.
TEST_CASE(ClParity, ImageGradHandlesImagesSmallerThanTheBorder) {
    if (!stage_available()) { report_skip("ImageGradHandlesImagesSmallerThanTheBorder"); return; }

    const int dims[][2] = {{1, 1}, {4, 4}, {5, 5}, {3, 64}, {64, 3}, {1, 40}, {40, 1}};
    for (const auto &d : dims) {
        const int bad = compare_geometry(d[0], d[1], 5u);
        if (bad == -1) { report_skip("ImageGradHandlesImagesSmallerThanTheBorder"); return; }
        if (bad != 0)
            std::printf("  ClParity: %dx%d — %d differing floats\n", d[0], d[1], bad);
        CHECK(bad == 0);
    }
    // 5x5 has exactly one interior pixel; below that the interior is empty.
    Image tiny = make_image(4, 4, 5u);
    REQUIRE(compute_image_gradients_gpu(tiny));
    for (size_t i = 0; i < tiny.grad_x.size(); ++i) {
        CHECK(tiny.grad_x[i] == 0.0f);
        CHECK(tiny.grad_y[i] == 0.0f);
    }
}

// Same input twice through the device must give the same bits. A kernel that
// read a neighbour it had not synchronised on would show up here and nowhere
// else — the CPU comparison above would pass on whichever answer arrived
// first if the race happened to be benign on that run.
TEST_CASE(ClParity, ImageGradIsRunToRunReproducible) {
    if (!stage_available()) { report_skip("ImageGradIsRunToRunReproducible"); return; }

    Image a = make_image(200, 137, 21u);
    Image b = a;
    if (!compute_image_gradients_gpu(a)) { report_skip("ImageGradIsRunToRunReproducible"); return; }
    REQUIRE(compute_image_gradients_gpu(b));
    CHECK(bit_mismatches(a.grad_x, b.grad_x) == 0);
    CHECK(bit_mismatches(a.grad_y, b.grad_y) == 0);
}

#else // !SEMPER_OPENCL

TEST_CASE(ClParity, ImageGradKernelAbsentWithoutOpenCLBuild) {
    std::printf("  ClParity: image gradients compiled out (SEMPER_OPENCL=OFF)\n");
    CHECK(true);
}

#endif // SEMPER_OPENCL
