#ifndef SEMPER_GPU_IMAGE_PREP_DISPATCH_HPP
#define SEMPER_GPU_IMAGE_PREP_DISPATCH_HPP

#include <semper/image.hpp>

namespace Semper {
namespace gpu {

// GPU Phase 6: the image gradient stencil, one work-item per pixel.
//
// Fills img.grad_x and img.grad_y with exactly what img.prepare_data(false)
// would have written -- bit-identical floats, borders included, not close
// ones. Sizes both vectors to width*height first, so the post-condition is
// the same whichever path ran.
//
// Does NOT apply the DICe 7-tap blur. prepare_data's blur argument is false
// at every call site in the engine (reference_cache.cpp,
// full_field_solver.cpp) and true only in tests/unit/test_image.cpp, so a
// device blur would be a kernel with no production caller; a caller that
// wants the blur keeps using prepare_data(true). See
// docs/GPU_ACCELERATION.md.
//
// Returns false, having touched nothing, when the stage cannot run: no
// device, no correctly-rounded fp32 divide, a build or enqueue failure, or a
// degenerate image. The caller then runs prepare_data. Never throws.
//
// Deliberately free of any size or performance policy -- the parity tests
// drive it at whatever geometry they need, and the decision about when the
// device is worth using lives in the caller. On the hardware measured so far
// that decision is "never"; see the Phase 6 results block in
// docs/GPU_ACCELERATION.md for the numbers and the reason.
bool compute_image_gradients_gpu(Image &img);

} // namespace gpu
} // namespace Semper

#endif // SEMPER_GPU_IMAGE_PREP_DISPATCH_HPP
