// =====================================================================
// SUITE: ClParity — src/gpu/strain_dispatch.cpp + src/gpu/kernels/strain_vsg.cl
//
// GPU Phase 2's parity gate. The claim under test is not "the GPU is close
// enough"; it is that the kernel and Semper::StrainCalculator produce the
// SAME float, bit for bit, for every grid point including the ones both
// refuse to solve. Everything here compares with ==. If a case ever fails,
// the fix is to find the divergence — never to reach for CHECK_NEAR. See
// docs/DETERMINISM.md and docs/GPU_ACCELERATION.md.
//
// Bit-exactness is only meaningful if the interesting cases are actually
// covered, so the fields below deliberately include:
//
//   * interior points with full circular window support;
//   * points clipped by the field edge, which fail the 90% fill ratio;
//   * holes of invalid neighbours, which move the fill ratio without
//     moving the window;
//   * a rank-deficient window (all support collinear), which fails either
//     semper_inv3x3's determinant test or the 1e-12 rcond floor.
//
// Each of those must leave the -1000 sentinel on BOTH paths — agreeing on
// which points to give up on is as much a part of parity as agreeing on
// the values.
//
// Like ClRuntime, every test must pass on a machine with no OpenCL: the
// device-dependent ones report themselves skipped rather than passing
// vacuously.
// =====================================================================
#include "framework/test_framework.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(SEMPER_OPENCL)

#include "gpu/cl_runtime.hpp"
#include "gpu/strain_dispatch.hpp"

#include <semper/gpu/embedded_kernels.hpp>
#include <semper/strain.hpp>
#include <semper/tuning.hpp>

using Semper::DisplacementField;
using Semper::StrainCalculator;
using Semper::StrainField;
using Semper::gpu::caps;
using Semper::gpu::compute_vsg_strain_gpu;
using Semper::gpu::reset_for_testing;

namespace {

// This stage needs fp64 specifically, not merely a working device.
bool stage_available() { return caps().available && caps().fp64; }

void report_skip(const char *what) {
    const auto &c = caps();
    if (!c.available)
        std::printf("  ClParity: skipped %s (no OpenCL device) — %s\n", what,
                    c.unavailable_reason.c_str());
    else
        std::printf("  ClParity: skipped %s (device '%s' has no cl_khr_fp64)\n",
                    what, c.device_name.c_str());
}

// u = aX + bY + e·X², v = cX + dY + f·Y². The quadratic terms matter: a
// purely linear field is fit exactly by the plane, which would let a
// genuinely different-but-consistent solve still land on the same answer.
// With curvature the fit is a compromise, and the compromise is what
// actually exercises the normal equations.
DisplacementField make_field(int gw, int gh, int step) {
    DisplacementField f;
    f.width = gw;
    f.height = gh;
    f.step = step;
    f.u.resize((size_t) gw * gh);
    f.v.resize((size_t) gw * gh);
    f.valid.assign((size_t) gw * gh, true);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            const float X = (float) (x * step), Y = (float) (y * step);
            const size_t i = (size_t) y * gw + x;
            f.u[i] = 0.011f * X + 0.0037f * Y + 0.00004f * X * X;
            f.v[i] = -0.0061f * X + 0.0092f * Y - 0.00003f * Y * Y;
        }
    }
    return f;
}

// Compare the two paths exactly, printing the first few disagreements so a
// failure says where rather than merely that.
int count_mismatches(const StrainField &cpu, const StrainField &gpu, int n,
                     const char *label) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const bool same = cpu.exx[(size_t) i] == gpu.exx[(size_t) i] &&
                          cpu.eyy[(size_t) i] == gpu.eyy[(size_t) i] &&
                          cpu.exy[(size_t) i] == gpu.exy[(size_t) i];
        if (same) continue;
        if (bad < 6)
            std::printf("  [%s] idx %d: exx %.9g/%.9g  eyy %.9g/%.9g  exy %.9g/%.9g\n",
                        label, i, (double) cpu.exx[(size_t) i], (double) gpu.exx[(size_t) i],
                        (double) cpu.eyy[(size_t) i], (double) gpu.eyy[(size_t) i],
                        (double) cpu.exy[(size_t) i], (double) gpu.exy[(size_t) i]);
        ++bad;
    }
    return bad;
}

int sentinel_count(const StrainField &s, int n) {
    int k = 0;
    for (int i = 0; i < n; ++i)
        if (s.exx[(size_t) i] == Semper::tuning::kStrainUninitSentinel) ++k;
    return k;
}

} // namespace

// The gate itself, and the one test that runs everywhere. A machine with no
// device must get a clean false and an untouched output — a dispatch that
// half-filled *out before failing would corrupt the CPU result the caller
// then computes into it.
TEST_CASE(ClParity, StrainDispatchDeclinesCleanlyWhenStageUnavailable) {
    DisplacementField f = make_field(12, 10, 11);
    StrainField out;
    out.exx.assign(3, 7.0f);   // recognisable prior contents
    out.eyy.assign(3, 7.0f);
    out.exy.assign(3, 7.0f);

    const bool ran = compute_vsg_strain_gpu(f, 44, &out);
    if (!stage_available()) {
        CHECK(!ran);
        // Untouched, not merely "not garbage".
        REQUIRE(out.exx.size() == 3u);
        CHECK(out.exx[0] == 7.0f);
        CHECK(out.eyy[2] == 7.0f);
        report_skip("parity (gate test still ran)");
    } else {
        CHECK(ran);
        CHECK(out.exx.size() == (size_t) (f.width * f.height));
    }
}

// Null output must be refused rather than dereferenced.
TEST_CASE(ClParity, StrainDispatchRejectsNullOutput) {
    DisplacementField f = make_field(8, 8, 11);
    CHECK(!compute_vsg_strain_gpu(f, 44, nullptr));
}

// The env override is the documented way to force the CPU path (there is no
// FullFieldParams field for it — see docs/CONTRACT.md). It must shut this
// stage down too, not just the probe.
TEST_CASE(ClParity, StrainDispatchHonoursEnvDisable) {
    reset_for_testing();
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "1");
#else
    setenv("SEMPER_OPENCL_DISABLE", "1", 1);
#endif
    DisplacementField f = make_field(10, 10, 11);
    StrainField out;
    const bool ran = compute_vsg_strain_gpu(f, 44, &out);
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "");
#else
    unsetenv("SEMPER_OPENCL_DISABLE");
#endif
    reset_for_testing();

    CHECK(!ran);
    CHECK(out.exx.empty());
}

// The embedded source must be self-contained and must still carry the fp64
// guard: without it, this kernel failing to compile on a device with no
// cl_khr_fp64 would take the whole program build down with it, collapsing
// the per-stage gate into one all-or-nothing switch.
TEST_CASE(ClParity, EmbeddedStrainKernelIsSelfContained) {
    const std::string src = Semper::gpu::kernels::STRAIN_VSG;
    REQUIRE(!src.empty());
    CHECK(src.find("__kernel void semper_strain_vsg") != std::string::npos);
    CHECK(src.find("#include <semper/") == std::string::npos);
    CHECK(src.find("FP_CONTRACT OFF") != std::string::npos);
    // The shared routines the CPU reference also calls.
    CHECK(src.find("semper_inv3x3") != std::string::npos);
    CHECK(src.find("semper_mat3_vec3") != std::string::npos);
    CHECK(src.find("semper_mat3_l1_norm") != std::string::npos);
    // The per-stage fp64 guard.
    CHECK(src.find("SEMPER_HAS_FP64") != std::string::npos);
}

// The main event: a fully valid field, several window sizes and steps.
TEST_CASE(ClParity, StrainMatchesCpuBitExactly) {
    if (!stage_available()) { report_skip("StrainMatchesCpuBitExactly"); return; }

    struct Case { int gw, gh, step, window; const char *name; };
    // strain_window >= 2*step throughout, or every window degenerates to its
    // own centre and the whole field comes back empty on BOTH paths — which
    // would make this test pass while proving nothing.
    const Case cases[] = {
        {12, 10, 11, 44, "12x10 step11 w44"},
        {21, 17,  5, 25, "21x17 step5 w25"},
        {17, 13,  7, 36, "17x13 step7 w36"},
        { 5,  5, 11, 44, "5x5 step11 w44 (all edge)"},
    };

    for (const Case &c : cases) {
        DisplacementField f = make_field(c.gw, c.gh, c.step);
        const int n = c.gw * c.gh;

        const StrainField cpu = StrainCalculator::compute_vsg_strain(f, c.window);
        StrainField gpu;
        REQUIRE(compute_vsg_strain_gpu(f, c.window, &gpu));
        REQUIRE(gpu.exx.size() == (size_t) n);

        const int bad = count_mismatches(cpu, gpu, n, c.name);
        std::printf("  ClParity[%s]: %d points, %d sentinel, %d mismatch\n",
                    c.name, n, sentinel_count(cpu, n), bad);
        CHECK(bad == 0);
    }
}

// Parity on the points that are given up on. Holes in the valid mask push
// windows below the 90% fill ratio without changing their geometry, and an
// entirely collinear support is rank-deficient — the two paths must refuse
// exactly the same points, for the same reasons.
TEST_CASE(ClParity, StrainRejectionCasesMatchExactly) {
    if (!stage_available()) { report_skip("StrainRejectionCasesMatchExactly"); return; }

    const int gw = 15, gh = 13, step = 11, window = 44;
    const int n = gw * gh;

    DisplacementField f = make_field(gw, gh, step);
    // A scattered hole pattern, plus one solid block, plus a single row left
    // valid in isolation so its window support is collinear.
    for (int i = 0; i < n; i += 7) f.valid[(size_t) i] = false;
    for (int y = 4; y <= 6; ++y)
        for (int x = 4; x <= 7; ++x) f.valid[(size_t) y * gw + x] = false;
    for (int y = 0; y < gh; ++y)
        if (y != 11) f.valid[(size_t) y * gw + (gw - 1)] = false;

    const StrainField cpu = StrainCalculator::compute_vsg_strain(f, window);
    StrainField gpu;
    REQUIRE(compute_vsg_strain_gpu(f, window, &gpu));

    const int sent = sentinel_count(cpu, n);
    const int bad = count_mismatches(cpu, gpu, n, "rejection");
    std::printf("  ClParity[rejection]: %d points, %d sentinel, %d mismatch\n",
                n, sent, bad);
    // If nothing was rejected the test is not testing what it claims to.
    CHECK(sent > 0);
    CHECK(sent < n);
    CHECK(bad == 0);
}

// Same input twice must give the same bytes. A kernel whose result depends
// on work-group scheduling would still pass the parity tests intermittently.
TEST_CASE(ClParity, StrainIsRunToRunReproducible) {
    if (!stage_available()) { report_skip("StrainIsRunToRunReproducible"); return; }

    DisplacementField f = make_field(14, 12, 11);
    const int n = f.width * f.height;
    StrainField a, b;
    REQUIRE(compute_vsg_strain_gpu(f, 44, &a));
    REQUIRE(compute_vsg_strain_gpu(f, 44, &b));
    CHECK(count_mismatches(a, b, n, "repeat") == 0);
}

#else // !SEMPER_OPENCL

// Kept as a real test so the suite count does not silently change with the
// build flag — the same convention as ClRuntime.
TEST_CASE(ClParity, BackendCompiledOut) {
    std::printf("  ClParity: SEMPER_OPENCL=OFF, strain stays on the CPU\n");
    CHECK(true);
}

#endif // SEMPER_OPENCL
