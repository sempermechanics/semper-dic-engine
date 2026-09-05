// Phase 3: the static Hessian pre-pass, one work-item per grid point.
//
// A transcription of Semper::SubsetPrecomputer::compute_hessian_only
// (src/math/subset_precomputer.cpp), not a reimplementation. The two must
// produce bit-identical float output; tests/unit/test_cl_hessian.cpp
// compares them with ==, never a tolerance. What carries that guarantee:
//
//   * the three-pass structure -- mean, then std dev, then the Hessian --
//     each over the same oy-outer / ox-inner scan, which fixes the order
//     every sum accumulates in;
//   * semper_mat6_add_outer / semper_mat6_symmetrize / semper_inv6x6 from
//     canonical_math.h, shared verbatim with the host, which is why the CPU
//     side had to stop using Eigen's PartialPivLU before this kernel could
//     exist (GPU Phase 3a, see docs/DETERMINISM.md);
//   * the -5.0f "ghost wall" intensity guard applied identically in all
//     three passes, so both sides drop the same pixels.
//
// Unlike strain_vsg.cl this is fp32 from end to end, so it is gated on
// caps().exact_fp32 -- correctly-rounded divide and sqrt -- rather than on
// cl_khr_fp64, and it builds on a device with no double support at all.
//
// The pre-pass computes only what CachedHessianData holds. It does NOT
// produce sdi_planes: those belong to precompute_subset_fast, which runs
// per solve inside Path A rather than in this pre-pass, and shipping
// dim*dim*6 floats per point back to the host would cost far more than
// recomputing them there.

#pragma OPENCL FP_CONTRACT OFF

#include <semper/kernels/canonical_math.h>

// H and H_inv are 36 floats per grid point, laid out row-major exactly as
// the host's float[36] is -- the host transposes into Eigen storage on
// arrival, and getting that wrong would be silent because the Hessian is
// symmetric while its Gauss-Jordan inverse is only nearly so.
__kernel void semper_hessian_prepass(__global const float *intensities,
                                     __global const float *grad_x,
                                     __global const float *grad_y,
                                     const int img_width,
                                     const int img_height,
                                     __global const uchar *wanted,
                                     const int grid_w,
                                     const int grid_h,
                                     const int rect_x,
                                     const int rect_y,
                                     const int step,
                                     const int dim,
                                     __global float *out_mean,
                                     __global float *out_std,
                                     __global float *out_H,
                                     __global float *out_H_inv,
                                     __global uchar *out_valid) {
    const int idx = get_global_id(0);
    if (idx >= grid_w * grid_h) return;

    // Every early-out leaves out_valid at 0 and the numeric outputs at the
    // defaults CachedHessianData carries, so a caller that ignores the flag
    // still reads defined memory rather than whatever the buffer held.
    out_valid[idx] = 0;
    out_mean[idx] = 0.0f;
    out_std[idx] = 1.0f;
    for (int i = 0; i < 36; ++i) {
        out_H[idx * 36 + i] = 0.0f;
        out_H_inv[idx * 36 + i] = 0.0f;
    }

    if (!wanted[idx]) return;

    const int gx = idx % grid_w;
    const int gy = idx / grid_w;
    const int cx = rect_x + gx * step;
    const int cy = rect_y + gy * step;

    const int n = dim * dim;
    // Named half_dim, not half: `half` is a reserved type name in OpenCL C
    // (the cl_khr_fp16 scalar), and declaring an int by that name is a
    // build error rather than a shadowing warning.
    const int half_dim = dim / 2;

    if (cx - half_dim < 0 || cx + half_dim >= img_width ||
        cy - half_dim < 0 || cy + half_dim >= img_height)
        return;

    // Pass 1: mean intensity.
    float sum = 0.0f;
    int valid_pixels = 0;
    for (int oy = -half_dim; oy <= half_dim; ++oy) {
        const int row = (cy + oy) * img_width + cx - half_dim;
        for (int ox = 0; ox < dim; ++ox) {
            const float iv = intensities[row + ox];
            if (iv < -5.0f) continue;
            sum += iv;
            valid_pixels++;
        }
    }

    if ((float) valid_pixels < (float) n * 0.5f) return;

    const float mean = sum / (float) valid_pixels;

    // Pass 2: std dev.
    float sum_sq = 0.0f;
    for (int oy = -half_dim; oy <= half_dim; ++oy) {
        const int row = (cy + oy) * img_width + cx - half_dim;
        for (int ox = 0; ox < dim; ++ox) {
            const float iv = intensities[row + ox];
            if (iv < -5.0f) continue;
            const float d = iv - mean;
            sum_sq += d * d;
        }
    }
    float std_dev = sqrt(sum_sq / (float) valid_pixels);
    if (std_dev < 1e-5f) std_dev = 1.0f;
    const float inv_std = 1.0f / std_dev;

    out_mean[idx] = mean;
    out_std[idx] = std_dev;

    // Pass 3: Hessian accumulation, upper triangle then mirrored.
    float H[36];
    for (int i = 0; i < 36; ++i) H[i] = 0.0f;

    for (int oy = -half_dim; oy <= half_dim; ++oy) {
        const int row = (cy + oy) * img_width + cx - half_dim;
        const float fy = (float) oy;
        for (int ox = -half_dim; ox <= half_dim; ++ox) {
            if (intensities[row + ox + half_dim] < -5.0f) continue;

            const float gxv = grad_x[row + ox + half_dim] * inv_std;
            const float gyv = grad_y[row + ox + half_dim] * inv_std;
            const float fx = (float) ox;

            const float sd[6] = {gxv, gyv, gxv * fx, gxv * fy,
                                 gyv * fx, gyv * fy};
            semper_mat6_add_outer(H, sd);
        }
    }
    semper_mat6_symmetrize(H);

    float H_inv[36];
    float det = 0.0f;
    const int invertible = semper_inv6x6(H, H_inv, &det);

    const float det_2x2 = H[0 * 6 + 0] * H[1 * 6 + 1] - H[1 * 6 + 0] * H[0 * 6 + 1];
    const float norm_2x2 = H[0 * 6 + 0] * H[0 * 6 + 0] + H[0 * 6 + 1] * H[0 * 6 + 1]
                         + H[1 * 6 + 0] * H[1 * 6 + 0] + H[1 * 6 + 1] * H[1 * 6 + 1];
    const float cond_2x2 = (fabs(det_2x2) > 1e-12f) ? (norm_2x2 / fabs(det_2x2))
                                                    : 1.0e13f;

    // The raw Hessian is published even when the point is rejected: the host
    // does the same, because the LM path wants H whether or not H_inv exists.
    for (int i = 0; i < 36; ++i) out_H[idx * 36 + i] = H[i];

    if (!invertible || fabs(det) < 1e-6f || cond_2x2 > 1.0e12f) return;

    for (int i = 0; i < 36; ++i) out_H_inv[idx * 36 + i] = H_inv[i];
    out_valid[idx] = 1;
}
