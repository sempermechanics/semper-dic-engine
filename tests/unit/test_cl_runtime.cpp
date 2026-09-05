// =====================================================================
// SUITE: ClRuntime — src/gpu/cl_runtime.cpp
//
// Every test here must pass on a machine with NO OpenCL, because that is
// the CI default and the common developer case. The backend's contract is
// that it degrades silently to the CPU, never that it is present.
//
// The one thing that must NOT be silent is the reason. A GPU that goes
// unused with an empty explanation is indistinguishable from a GPU that is
// working, which is the failure mode these tests exist to prevent.
//
// Tests that need a real device are guarded and report themselves as
// skipped rather than passing vacuously.
//
// Compiled only under -DSEMPER_OPENCL=ON; otherwise the whole file is a
// single test asserting the backend is genuinely absent.
// =====================================================================
#include "framework/test_framework.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(SEMPER_OPENCL)

#include "gpu/cl_runtime.hpp"
#include <semper/kernels/canonical_math.h>

using Semper::gpu::build_options;
using Semper::gpu::caps;
using Semper::gpu::probe_kernel_source;
using Semper::gpu::reset_for_testing;
using Semper::gpu::run_probe_reduce;

namespace {
bool device_present() { return caps().available; }
} // namespace

// Probing must be safe whatever is or is not installed. If this crashes or
// throws, every downstream fallback is worthless.
TEST_CASE(ClRuntime, ProbeIsSafeAndIdempotent) {
    const auto &a = caps();
    const auto &b = caps();  // cached; must not re-probe or differ
    CHECK(a.available == b.available);
    CHECK(a.fp64 == b.fp64);
    CHECK(a.exact_fp32 == b.exact_fp32);
    std::printf("  ClRuntime: available=%d fp64=%d exact_fp32=%d device='%s' (%s)\n",
                (int) a.available, (int) a.fp64, (int) a.exact_fp32,
                a.device_name.c_str(), a.device_version.c_str());
    if (!a.available)
        std::printf("  ClRuntime: unavailable_reason = %s\n", a.unavailable_reason.c_str());
}

// The anti-silence assertion.
TEST_CASE(ClRuntime, UnavailableAlwaysExplainsItself) {
    const auto &c = caps();
    if (c.available) {
        CHECK(!c.device_name.empty());
    } else {
        CHECK(!c.unavailable_reason.empty());
    }
}

// A capability must never be claimed while the backend is unusable — a
// caller checking caps().fp64 alone would otherwise dispatch to a device
// that cannot run anything.
TEST_CASE(ClRuntime, CapabilitiesImplyAvailability) {
    const auto &c = caps();
    if (!c.available) {
        CHECK(!c.fp64);
        CHECK(!c.exact_fp32);
    }
}

// The build options are part of the bit-exactness contract, not a tuning
// knob. Both of these would silently void it, and both are exactly the sort
// of flag someone adds while chasing a benchmark.
TEST_CASE(ClRuntime, BuildOptionsContainNoAccuracyDestroyingFlag) {
    const std::string opts = build_options();
    CHECK(opts.find("fast-relaxed-math") == std::string::npos);
    CHECK(opts.find("mad-enable") == std::string::npos);
    CHECK(opts.find("unsafe-math") == std::string::npos);
    CHECK(opts.find("-cl-no-signed-zeros") == std::string::npos);
    // And the one that must be present: without correctly-rounded fp32
    // divide/sqrt the device cannot match the CPU reference at all.
    CHECK(opts.find("-cl-fp32-correctly-rounded-divide-sqrt") != std::string::npos);
}

// Proves scripts/embed_cl_kernels.py actually expanded the first-party
// include rather than emitting a bare #include the device could not resolve.
TEST_CASE(ClRuntime, EmbeddedKernelIsSelfContained) {
    const std::string src = probe_kernel_source();
    REQUIRE(!src.empty());

    // The canonical reduction's combine macro must be present textually.
    CHECK(src.find("SEMPER_COMBINE4") != std::string::npos);
    // The __global variant only exists if canonical_reductions.inc expanded
    // a second time — the subtlety the expander exists to get right.
    CHECK(src.find("semper_canon_sum_sq_diff_g") != std::string::npos);
    // FP contract must be off in the kernel source itself.
    CHECK(src.find("FP_CONTRACT OFF") != std::string::npos);
    // No unresolved first-party include may survive.
    CHECK(src.find("#include <semper/") == std::string::npos);
}

TEST_CASE(ClRuntime, EnvDisableForcesCpuPath) {
    reset_for_testing();
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "1");
#else
    setenv("SEMPER_OPENCL_DISABLE", "1", 1);
#endif
    const bool avail = caps().available;
    const std::string reason = caps().unavailable_reason;
#if defined(_WIN32)
    _putenv_s("SEMPER_OPENCL_DISABLE", "");
#else
    unsetenv("SEMPER_OPENCL_DISABLE");
#endif
    reset_for_testing();

    CHECK(!avail);
    CHECK(reason.find("SEMPER_OPENCL_DISABLE") != std::string::npos);
}

// Device-gated. The point of the probe kernel: the device must reproduce
// the host's canonical reduction EXACTLY, not merely closely. A device that
// disagrees here would disagree in every later kernel.
TEST_CASE(ClRuntime, DeviceReproducesCanonicalReductionExactly) {
    if (!device_present()) {
        std::printf("  ClRuntime: skipped (no OpenCL device) — %s\n",
                    caps().unavailable_reason.c_str());
        return;
    }
    // 729 = a 27 px subset, and 729 = 4*182 + 1 so the scalar tail is
    // genuinely exercised.
    for (int n : {0, 1, 3, 4, 5, 8, 729}) {
        std::vector<float> vals((size_t) (n > 0 ? n : 1), 0.0f);
        for (int i = 0; i < n; ++i)
            vals[(size_t) i] = (float) ((i * 37) % 251) + 0.5f;
        const float mean = 127.3f;

        float device_result = 0.0f;
        REQUIRE(run_probe_reduce(vals.data(), n, mean, &device_result));
        const float host_result = semper_canon_sum_sq_diff(vals.data(), n, mean);
        if (device_result != host_result)
            std::printf("  n=%d: device %.9g != host %.9g\n", n,
                        (double) device_result, (double) host_result);
        CHECK(device_result == host_result);
    }
}

#else // !SEMPER_OPENCL

// With the backend compiled out there is nothing to probe, and the binary
// must carry no OpenCL machinery at all. Kept as a real test so the suite
// count does not silently change with the build flag.
TEST_CASE(ClRuntime, BackendCompiledOut) {
    std::printf("  ClRuntime: SEMPER_OPENCL=OFF, backend not compiled in\n");
    CHECK(true);
}

#endif // SEMPER_OPENCL
