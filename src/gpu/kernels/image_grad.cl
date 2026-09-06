// Phase 6: the image gradient stencil, one work-item per pixel.
//
// This is a transcription of the second half of Semper::Image::prepare_data
// (src/math/image_processor.cpp) and must produce bit-identical float output
// -- tests/unit/test_cl_image_grad.cpp compares the two buffers with ==,
// every pixel, never a tolerance.
//
// Two things carry that guarantee:
//
//   * SEMPER_CANON_DERIV5 from canonical_math.h, shared verbatim with the
//     host, so the parenthesisation cannot drift on one side only. The host
//     stopped spelling the expression out inline in the same commit that
//     added this file, for exactly that reason;
//   * #pragma OPENCL FP_CONTRACT OFF below. OpenCL C leaves contraction on by
//     default, and 8.0f * p1 + m2 is precisely the shape a compiler fuses
//     into a mad. The host is built -ffp-contract=off; without this pragma
//     the two would disagree in the last bit on every pixel where the fused
//     intermediate rounds differently.
//
// fp32 end to end, so the stage is gated on exact_fp32 (for the correctly
// rounded divide by 12) and NOT on cl_khr_fp64. It compiles and runs on a
// device with no double support at all.
//
// The border band is written, not skipped. The CPU path zero-fills both
// planes and then overwrites only the interior; if this kernel left the
// border untouched the comparison would have to be restricted to the
// interior, and a buffer that is only checked where it is interesting is a
// buffer that will eventually be wrong where it is not.

#pragma OPENCL FP_CONTRACT OFF

#include <semper/kernels/canonical_math.h>

// img -> grad_x, grad_y, all three width*height floats in row-major order,
// matching Image::intensities / grad_x / grad_y exactly.
//
// Launched with a global size of width*height. The bound check is not
// redundant padding insurance: it lets a caller round the launch up to a
// work-group multiple without a second code path, and costs one comparison
// against a memory-bound kernel.
__kernel void semper_image_gradients(__global const float *img,
                                     const int width,
                                     const int height,
                                     __global float *grad_x,
                                     __global float *grad_y) {
    const int idx = (int) get_global_id(0);
    const int total = width * height;
    if (idx >= total) return;

    const int x = idx % width;
    const int y = idx / width;
    const int b = SEMPER_GRAD_BORDER;

    // Also the whole-image case when width or height is <= 2*b: the interior
    // is empty and every pixel takes this branch, which is what the CPU
    // loops do when their bounds cross.
    if (x < b || x >= width - b || y < b || y >= height - b) {
        grad_x[idx] = 0.0f;
        grad_y[idx] = 0.0f;
        return;
    }

    // Argument order is m2, m1, p1, p2 -- see the macro's comment. The two
    // calls differ only in the stride: 1 for x, width for y.
    grad_x[idx] = SEMPER_CANON_DERIV5(img[idx - 2], img[idx - 1],
                                      img[idx + 1], img[idx + 2]);
    grad_y[idx] = SEMPER_CANON_DERIV5(img[idx - 2 * width], img[idx - width],
                                      img[idx + width], img[idx + 2 * width]);
}
