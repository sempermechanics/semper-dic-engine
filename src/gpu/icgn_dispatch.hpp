#ifndef SEMPER_GPU_ICGN_DISPATCH_HPP
#define SEMPER_GPU_ICGN_DISPATCH_HPP

#include <semper/image.hpp>
#include <semper/subset.hpp>

namespace Semper {
namespace gpu {

// GPU Phase 4: Path A ICGN, one work-item per subset.
//
// Runs ONE ICGN solve per point -- precisely what
// OptimizationEngine::calculate_deformation(..., INIT_NO_SIMPLEX) runs, and
// what Path A runs first for every point before deciding whether a rescue is
// needed. The results are bit-identical to the CPU's, not close to them;
// tests/unit/test_cl_icgn.cpp compares them with ==.
//
// The simplex rescue is NOT here. Nelder-Mead is a sequential search with a
// data-dependent trip count, so folding it into the same launch would make
// every lane wait on the few percent of points that need it. The caller runs
// the untouched CPU path for those, which is also what it must do for any
// point this returns kIcgnHostRequired for.

// status values. 0 and 1 are AnalysisResult::status verbatim.
constexpr int kIcgnHostRequired = -1;

// One point's inputs. `cached` is the pooled Hessian entry for this point --
// the same one precompute_subset_fast would be handed -- and must outlive
// the call. A null or invalid entry is legal and comes back as
// kIcgnHostRequired: the kernel deliberately does not carry a second copy of
// the full precompute_subset fallback.
struct IcgnGpuPoint {
    int cx = 0;
    int cy = 0;
    float guess[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // u v ux uy vx vy
    const CachedHessianData *cached = nullptr;
};

// One point's outputs, in AnalysisResult's field order so the caller can
// copy them across without reordering.
struct IcgnGpuResult {
    float p[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float score = 1.0f;
    int status = kIcgnHostRequired;
    int iters = 0;
    int invalid_ref_pixels = 0;
};

// Solves all n_pts points, writing out[0..n_pts). Points the device refused
// come back with status == kIcgnHostRequired and defaults elsewhere; the
// caller re-solves exactly those on the CPU.
//
// lm_alpha collapses the CPU's two damping conditions into one number: pass
// 0 for `lm_enabled == false` or a zero alpha, and the kernel takes the
// undamped branch. max_iter must be tuning::kIcgnMaxIter to match the CPU,
// which hardcodes it (optimization_engine.cpp:158); it is a parameter only
// so the kernel does not carry a copy of a host tuning constant.
//
// Returns false, having touched nothing, when the stage cannot run: no
// device, no correctly-rounded fp32 divide/sqrt, a build, allocation or
// enqueue failure, or a nonsensical geometry. The caller then runs the CPU
// path for every point. Never throws.
//
// Deliberately free of any size or performance policy -- the parity tests
// drive it at whatever geometry they need, and the decision about when the
// device is worth using lives in the caller, exactly as it does for the
// strain and Hessian stages.
bool solve_icgn_batch_gpu(const Image &ref_img, const Image &def_img,
                          int dim, bool use_keys6, int max_iter, float lm_alpha,
                          const IcgnGpuPoint *pts, int n_pts,
                          IcgnGpuResult *out);

} // namespace gpu
} // namespace Semper

#endif // SEMPER_GPU_ICGN_DISPATCH_HPP
