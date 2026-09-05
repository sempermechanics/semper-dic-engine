#ifndef SEMPER_GPU_HESSIAN_DISPATCH_HPP
#define SEMPER_GPU_HESSIAN_DISPATCH_HPP

#include <semper/subset.hpp>

namespace Semper {
namespace gpu {

// GPU Phase 3: the static Hessian pre-pass, one work-item per grid point.
//
// Fills out[i] for every i where wanted[i] is non-zero, with exactly what
// SubsetPrecomputer::compute_hessian_only would have written there --
// bit-identical floats, not close ones. Entries where wanted[i] is zero are
// left untouched, matching the CPU pre-pass, which skips already-solved
// points and leaves their pool slot default-constructed.
//
// One documented divergence, in dead data only. When a point never reaches
// the Hessian accumulation -- its subset hangs off the image, or more than
// half its pixels are behind the -5.0f ghost wall -- the CPU returns a
// default-constructed CachedHessianData whose Eigen H and H_inv are
// UNINITIALISED, while this path leaves them zeroed. Nothing reads them:
// precompute_subset_fast re-runs the full precompute whenever valid is
// false. valid, mean_intensity and std_dev still match exactly on those
// points, and the parity test compares H/H_inv only where the CPU actually
// wrote them.
//
// Returns false, having touched nothing, when the stage cannot run: no
// device, no correctly-rounded fp32 divide/sqrt, a build or enqueue
// failure, or a nonsensical geometry. The caller then runs the CPU
// pre-pass. Never throws.
//
// Deliberately free of any size or performance policy: the parity tests
// drive it at whatever geometry they need, and the decision about when the
// device is worth using lives in the caller.
bool compute_hessian_pool_gpu(const Image &ref_img,
                              int rect_x, int rect_y, int step,
                              int grid_w, int grid_h, int dim,
                              const unsigned char *wanted,
                              CachedHessianData *out, int out_count);

} // namespace gpu
} // namespace Semper

#endif // SEMPER_GPU_HESSIAN_DISPATCH_HPP
