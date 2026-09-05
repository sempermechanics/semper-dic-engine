#ifndef SEMPER_GPU_STRAIN_DISPATCH_HPP
#define SEMPER_GPU_STRAIN_DISPATCH_HPP

#include <semper/strain.hpp>

namespace Semper {
namespace gpu {

// GPU Phase 2: the VSG plane fit, one work-item per grid point.
//
// Fills *out with results bit-identical to
// StrainCalculator::compute_vsg_strain for the same inputs -- identical as
// in operator==, not "within tolerance". tests/unit/test_cl_strain.cpp
// enforces that; see docs/DETERMINISM.md for why a tolerance here would be
// meaningless.
//
// Returns false, having written nothing, whenever the device cannot be
// trusted with this stage: no OpenCL, SEMPER_OPENCL_DISABLE=1, or no
// cl_khr_fp64 (the fit is fp64 throughout). The caller must then run the CPU
// path. Never throws.
bool compute_vsg_strain_gpu(const DisplacementField &disp, int window_pixels,
                            StrainField *out);

} // namespace gpu
} // namespace Semper

#endif // SEMPER_GPU_STRAIN_DISPATCH_HPP
