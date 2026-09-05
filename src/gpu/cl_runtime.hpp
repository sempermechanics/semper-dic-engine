#ifndef SEMPER_GPU_CL_RUNTIME_HPP
#define SEMPER_GPU_CL_RUNTIME_HPP

#include <string>

namespace Semper {
namespace gpu {

// What this machine can actually be trusted to compute.
//
// fp64 and exact_fp32 are SEPARATE, not one gate. A device may legitimately
// run the fp64 strain kernel but not the bit-exact ICGN, or the reverse, so
// dispatch decides per stage rather than switching the whole backend on or
// off. See docs/GPU_ACCELERATION.md.
struct ClCaps {
    bool available  = false;  // loader + platform + device + program all OK
    bool fp64       = false;  // cl_khr_fp64                 -> strain kernels
    bool exact_fp32 = false;  // correctly-rounded / and sqrt -> ICGN kernels

    std::string device_name;
    std::string driver_version;
    std::string device_version;

    // Why the backend is not being used. NEVER empty when available is
    // false: a silent fallback is the failure mode this field exists to
    // prevent, and a test asserts it is populated.
    std::string unavailable_reason;
};

// Probe once, lazily, and cache. Thread-safe. Never throws: every failure
// path sets unavailable_reason and leaves available false, so callers use
// the CPU path rather than handling an exception.
const ClCaps &caps();

// The exact option string handed to clBuildProgram. Exposed so a unit test
// can assert it contains no accuracy-destroying flag -- see
// docs/DETERMINISM.md. Pasting in "-cl-fast-relaxed-math" to chase a
// benchmark would silently void the bit-exactness contract, so it is
// checked rather than merely commented.
const char *build_options();

// The embedded probe kernel source, after include expansion. Exposed for
// the test that verifies the expansion actually ran.
const char *probe_kernel_source();

// Run the probe kernel: out = canonical sum of squared deviations of
// vals[0..n) about mean, computed ON THE DEVICE. Returns false when no
// device is available. Used by the parity test to confirm the device
// reproduces the host's canonical reduction bit-for-bit.
bool run_probe_reduce(const float *vals, int n, float mean, float *out);

// Drop the cached probe so the next caps() call re-runs it. Only for tests
// that manipulate SEMPER_OPENCL_DISABLE; not part of any shipping path.
void reset_for_testing();

} // namespace gpu
} // namespace Semper

#endif // SEMPER_GPU_CL_RUNTIME_HPP
