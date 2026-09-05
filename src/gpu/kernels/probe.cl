// Phase 1 probe kernel.
//
// No DIC work happens here. Its job is to prove, on the actual device, that
// the whole toolchain is sound before Phase 2 depends on it:
//
//   1. scripts/embed_cl_kernels.py resolved the canonical_math.h include
//      (and, inside it, both address-space expansions of
//      canonical_reductions.inc -- the _g variant used below only exists if
//      that worked);
//   2. the canonical math compiles as OpenCL C on this device's compiler,
//      not merely under clang offline;
//   3. the device agrees with the host on the canonical reduction, which is
//      the property every later kernel is built on.
//
// The host runs the identical input through semper_canon_sum_sq_diff and
// compares exactly. A device that disagrees here would disagree everywhere,
// and is refused rather than debugged one kernel at a time.

#pragma OPENCL FP_CONTRACT OFF

#include <semper/kernels/canonical_math.h>

// out[0] = the canonical sum of squared deviations over vals[0..n).
// Deliberately single-work-item: this is a correctness probe, and a
// parallel reduction would change the summation order and so test
// something other than what it claims to.
__kernel void semper_probe_reduce(__global const float *vals,
                                  const int n,
                                  const float mean,
                                  __global float *out) {
    if (get_global_id(0) != 0) return;
    out[0] = semper_canon_sum_sq_diff_g(vals, n, mean);
}
