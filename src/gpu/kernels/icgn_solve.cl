// Phase 4: Path A ICGN, ONE WORK-ITEM PER SUBSET.
//
// A transcription of Semper::OptimizationEngine::solve_icgn
// (src/math/optimization_engine.cpp) fused with the part of
// SubsetPrecomputer::precompute_subset_fast it consumes. Both are
// reproduced here rather than uploaded: the subset data is 8 floats per
// pixel per point -- 14 KB for a 21 px subset -- against a reference image
// the device is already holding for the Phase 3 pre-pass.
//
// ONE WORK-ITEM PER SUBSET IS THE DESIGN, NOT A FIRST DRAFT.
// Every accumulation in solve_icgn -- def_sum, def_sum_sq on the partial
// path, dp_sum, the Hessian outer products, the two gradient sums in the
// sigma check -- runs in an order the CPU also runs in. A work-group-wide
// tree reduction would compute the same sums in a different order and give
// different bits, which is the one thing this backend may not do. Spreading
// a subset across lanes is therefore not an optimisation to try later; it
// is a different program. See docs/GPU_ACCELERATION.md.
//
// fp32 from end to end, so this gates on caps().exact_fp32 -- correctly
// rounded divide and sqrt -- and never on cl_khr_fp64. It builds on a
// device with no double support at all.
//
// WHAT THIS KERNEL DELIBERATELY DOES NOT DO
//
//   * The simplex rescue. Path A runs ICGN once, and only if that fails or
//     scores worse than tuning::kCorrSimplexTrigger does it run Nelder-Mead
//     and a second ICGN. This kernel produces the FIRST ICGN result for
//     every point and the host re-solves the rest through the untouched CPU
//     path. Nelder-Mead is a sequential search with a data-dependent trip
//     count; putting it in the same kernel would make the whole launch wait
//     on its slowest lane for a few percent of points.
//
//   * Points whose pooled Hessian is invalid, or whose subset hangs off the
//     reference image. precompute_subset_fast falls back to a full
//     precompute for those, and carrying a second copy of precompute_subset
//     here would double the surface that has to stay bit-exact for no gain.
//     They come back as SEMPER_ICGN_HOST and the host runs them.
//
// Scratch is __global and caller-allocated, one slice per work-item: a
// 21 px subset is 441 pixels and the planes below are 8 floats each, two
// orders of magnitude past a sane private allocation. That is also why
// canonical_math.h expands its reductions into _g variants at all.

#pragma OPENCL FP_CONTRACT OFF

#include <semper/kernels/canonical_math.h>

// out_status. 0 and 1 are AnalysisResult::status verbatim; the negative is
// this kernel handing the point back rather than guessing at it.
#define SEMPER_ICGN_OK      0
#define SEMPER_ICGN_FAILED  1
#define SEMPER_ICGN_HOST   (-1)

__kernel void semper_icgn_solve(
        // Reference image and its gradients -- the same three buffers the
        // Hessian pre-pass uploads.
        __global const float *ref_int,
        __global const float *ref_gx,
        __global const float *ref_gy,
        const int ref_w,
        const int ref_h,
        // Deformed image.
        __global const float *def_int,
        const int def_w,
        const int def_h,
        // Geometry and solver mode.
        const int dim,
        const int use_keys6,
        const int max_iter,
        // 0 disables LM damping exactly as lm_enabled == false does; the
        // host collapses those two CPU-side conditions into this one value.
        const float lm_alpha,
        // Per-point inputs. pt_H / pt_Hinv are ROW-major: the host holds
        // Eigen (column-major) storage and transposes on the way out, the
        // same transpose solve_icgn does at optimization_engine.cpp:128-130.
        __global const int   *pt_cx,
        __global const int   *pt_cy,
        __global const float *pt_mean,
        __global const float *pt_std,
        __global const uchar *pt_valid,
        __global const float *pt_H,
        __global const float *pt_Hinv,
        __global const float *pt_guess,     // u, v, ux, uy, vx, vy
        // Per-work-item scratch, n = dim*dim entries each unless noted.
        __global float *scr_def,
        __global float *scr_normref,
        __global float *scr_sdi,            // 6n
        __global uchar *scr_refvalid,
        // Outputs, one entry per point.
        __global float *out_p,              // 6 each
        __global float *out_score,
        __global int   *out_status,
        __global int   *out_iters,
        __global int   *out_invalid_ref,
        const int n_pts) {

    const int gid = get_global_id(0);
    if (gid >= n_pts) return;

    const int n = dim * dim;
    // Named half_dim because `half` is a reserved type name in OpenCL C --
    // the same trap hessian_prepass.cl documents.
    const int half_dim = dim / 2;

    // Defaults before any early-out. The host reads every slot it asked
    // for, so a refused point must still leave defined memory rather than
    // whatever the buffer happened to hold.
    for (int k = 0; k < 6; ++k) out_p[(size_t) gid * 6 + k] = 0.0f;
    out_score[gid] = 1.0f;
    out_status[gid] = SEMPER_ICGN_HOST;
    out_iters[gid] = 0;
    out_invalid_ref[gid] = 0;

    // precompute_subset_fast falls back to the full precompute when the
    // pool entry is invalid, and marks the subset uninitialised when it
    // hangs off the image. Both are the host's job.
    if (!pt_valid[gid]) return;

    const int cx = pt_cx[gid];
    const int cy = pt_cy[gid];
    if (cx - half_dim < 0 || cx + half_dim >= ref_w ||
        cy - half_dim < 0 || cy + half_dim >= ref_h) return;

    __global float *def_vals  = scr_def      + (size_t) gid * n;
    __global float *norm_ref  = scr_normref  + (size_t) gid * n;
    __global float *sdi       = scr_sdi      + (size_t) gid * n * 6;
    __global uchar *ref_valid = scr_refvalid + (size_t) gid * n;

    // ---- precompute_subset_fast, minus what solve_icgn never reads ----
    //
    // Same fused oy-outer / ox-inner scan, same cached mean and std_dev,
    // same inv_std applied to the gradients before the six steepest-descent
    // planes are formed. gx_vec / gy_vec are NOT staged: they are the raw
    // reference gradients, and the one place that reads them (the sigma
    // check) can index ref_gx / ref_gy directly for the same floats, which
    // is 2n fewer scratch entries per work-item.
    const float ref_mean = pt_mean[gid];
    const float ref_inv_std = 1.0f / pt_std[gid];

    int invalid_ref_count = 0;
    {
        int idx = 0;
        for (int oy = -half_dim; oy <= half_dim; ++oy) {
            const float fy = (float) oy;
            const int row_base = (cy + oy) * ref_w + (cx - half_dim);
            for (int ox = -half_dim; ox <= half_dim; ++ox) {
                const int s = row_base + (ox + half_dim);
                const float ri = ref_int[s];

                // build_ref_valid (solver.hpp:64-71): the -5.0f ghost wall.
                const uchar rv = (ri < -5.0f) ? (uchar) 0 : (uchar) 1;
                ref_valid[idx] = rv;
                if (!rv) ++invalid_ref_count;

                norm_ref[idx] = (ri - ref_mean) * ref_inv_std;

                const float gx = ref_gx[s] * ref_inv_std;
                const float gy = ref_gy[s] * ref_inv_std;
                const float fx = (float) ox;

                sdi[            idx] = gx;
                sdi[    n     + idx] = gy;
                sdi[2 * n     + idx] = gx * fx;
                sdi[3 * n     + idx] = gx * fy;
                sdi[4 * n     + idx] = gy * fx;
                sdi[5 * n     + idx] = gy * fy;
                ++idx;
            }
        }
    }

    // ---- solve_icgn ----
    const float gu  = pt_guess[(size_t) gid * 6 + 0];
    const float gv  = pt_guess[(size_t) gid * 6 + 1];
    const float gux = pt_guess[(size_t) gid * 6 + 2];
    const float guy = pt_guess[(size_t) gid * 6 + 3];
    const float gvx = pt_guess[(size_t) gid * 6 + 4];
    const float gvy = pt_guess[(size_t) gid * 6 + 5];

    float W[9] = {1.0f + gux, guy,        gu,
                  gvx,        1.0f + gvy, gv,
                  0.0f,       0.0f,       1.0f};

    float H_inv_rm[36];
    for (int i = 0; i < 36; ++i) H_inv_rm[i] = pt_Hinv[(size_t) gid * 36 + i];

    // The Newton-step matrix is built ONCE, before the loop, exactly as on
    // the CPU: alpha is a fixed scalar, so the damped inverse is valid for
    // every iteration of this call. Damping hits the two translation DOFs
    // only -- the four strain-gradient diagonals are left alone (DICe
    // computeUpdateFast).
    float H_solve[36];
    if (lm_alpha > 0.0f) {
        float H_damped[36];
        for (int i = 0; i < 36; ++i) H_damped[i] = pt_H[(size_t) gid * 36 + i];
        H_damped[0 * 6 + 0] += lm_alpha;
        H_damped[1 * 6 + 1] += lm_alpha;

        float H_damped_inv[36], det;
        const int ok = semper_inv6x6(H_damped, H_damped_inv, &det);
        if (!ok || fabs(det) < 1e-6f) {
            // Still singular after damping: fall back to the pre-inverted
            // H_inv so the point at least attempts a step.
            for (int i = 0; i < 36; ++i) H_solve[i] = H_inv_rm[i];
        } else {
            for (int i = 0; i < 36; ++i) H_solve[i] = H_damped_inv[i];
        }
    } else {
        for (int i = 0; i < 36; ++i) H_solve[i] = H_inv_rm[i];
    }

    float final_score = 1.0f;

    for (int iter = 0; iter < max_iter; ++iter) {
        float def_sum = 0.0f;
        int valid_pixels = 0;

        // 1. Affine warp and interpolation.
        //
        // The CPU batches four pixels at a time when all four clear the
        // guard, but interpolate_*_x4 does the identical per-lane
        // arithmetic in the identical order -- pinned with == by
        // Image.BatchOfFourMatchesScalarExactly -- and def_sum /
        // valid_pixels accumulate in the same i = 0..n-1 order either way.
        // A plain scalar loop here is therefore bit-identical to the
        // batched one, and does not have to reproduce which groups of four
        // happened to be batchable.
        {
            int idx = 0;
            for (int oy = -half_dim; oy <= half_dim; ++oy) {
                const float y = (float) oy;
                for (int ox = -half_dim; ox <= half_dim; ++ox) {
                    const float x = (float) ox;
                    const float final_x = (float) cx + W[0] * x + W[1] * y + W[2];
                    const float final_y = (float) cy + W[3] * x + W[4] * y + W[5];

                    // DICe parity: the explicit 4-pixel guard, with its
                    // rounding. Tighter than every interpolator's own
                    // demotion band, which is why the samplers' fallbacks
                    // are unreachable from here.
                    const int px = ((int) (final_x + 0.5f) == (int) (final_x))
                                           ? (int) (final_x) : (int) (final_x) + 1;
                    const int py = ((int) (final_y + 0.5f) == (int) (final_y))
                                           ? (int) (final_y) : (int) (final_y) + 1;
                    if (px < 4 || px >= def_w - 4 || py < 4 || py >= def_h - 4) {
                        def_vals[idx] = -1.0f;
                        ++idx;
                        continue;
                    }

                    const float val = semper_canon_sample_g(def_int, def_w, def_h,
                                                            final_x, final_y, use_keys6);
                    // DICe layer 2: the mask-aware drop. A pixel counts only
                    // if it is live in the reference mask AND inside the
                    // deformed image.
                    if (val > 0.0f && ref_valid[idx]) {
                        def_vals[idx] = val;
                        def_sum += val;
                        ++valid_pixels;
                    } else {
                        def_vals[idx] = -1.0f;
                    }
                    ++idx;
                }
            }
        }

        // Abort before the division: a subset dragged off the image would
        // otherwise inject a NaN straight into the Newton matrix. DICe
        // parity threshold, 90% survival.
        if (valid_pixels < (int) ((float) n * 0.90f)) {
            out_p[(size_t) gid * 6 + 0] = W[2];
            out_p[(size_t) gid * 6 + 1] = W[5];
            out_p[(size_t) gid * 6 + 2] = W[0] - 1.0f;
            out_p[(size_t) gid * 6 + 3] = W[1];
            out_p[(size_t) gid * 6 + 4] = W[3];
            out_p[(size_t) gid * 6 + 5] = W[4] - 1.0f;
            out_status[gid] = SEMPER_ICGN_FAILED;
            out_score[gid] = 2.0f;
            out_iters[gid] = iter;
            out_invalid_ref[gid] = invalid_ref_count;
            return;
        }

        const float def_mean = def_sum / (float) valid_pixels;
        float def_sum_sq = 0.0f;
        float dp_sum[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float error_sum_sq = 0.0f;

        if (valid_pixels == n) {
            // Fast path: every pixel survived, so both reductions run over
            // the whole array in the canonical 4-accumulator form -- the
            // same two the CPU reaches through Semper::simd, whose SIMD
            // width is static_asserted to 4 lanes for exactly this reason.
            def_sum_sq = semper_canon_sum_sq_diff_g(def_vals, n, def_mean);

            float def_std = sqrt(def_sum_sq / (float) valid_pixels);
            if (def_std < 1e-5f) def_std = 1.0f;
            const float inv_std = 1.0f / def_std;

            float dp_out[6];
            error_sum_sq = semper_canon_znssd_error_and_gradient_g(
                    def_vals, norm_ref, sdi, n, def_mean, inv_std, dp_out);
            for (int k = 0; k < 6; ++k) dp_sum[k] += dp_out[k];
        } else {
            // Partial path: the survivors are scattered, so every sum here
            // is sequential over i. That is not an oversight -- the CPU
            // cannot use the four-accumulator form here either, and the two
            // sides have to agree on the shape they actually run.
            float H_dynamic[36];
            for (int i = 0; i < 36; ++i) H_dynamic[i] = 0.0f;

            for (int i = 0; i < n; ++i) {
                if (def_vals[i] >= 0.0f) {
                    const float diff = def_vals[i] - def_mean;
                    def_sum_sq += diff * diff;
                }
            }
            float def_std = sqrt(def_sum_sq / (float) valid_pixels);
            if (def_std < 1e-5f) def_std = 1.0f;
            const float inv_std = 1.0f / def_std;

            for (int i = 0; i < n; ++i) {
                if (def_vals[i] >= 0.0f) {
                    const float norm_def = (def_vals[i] - def_mean) * inv_std;
                    const float diff = norm_ref[i] - norm_def;
                    error_sum_sq += diff * diff;
                    float sd[6];
                    for (int k = 0; k < 6; ++k) sd[k] = sdi[(size_t) k * n + i];
                    for (int k = 0; k < 6; ++k) dp_sum[k] += sd[k] * diff;
                    semper_mat6_add_outer(H_dynamic, sd);
                }
            }

            // add_outer fills the upper triangle only; mirror before
            // anything below reads a lower-triangle entry.
            semper_mat6_symmetrize(H_dynamic);

            // DICe layer 3: the 2x2 sub-block condition guard.
            const float det_2x2 = H_dynamic[0 * 6 + 0] * H_dynamic[1 * 6 + 1]
                                - H_dynamic[1 * 6 + 0] * H_dynamic[0 * 6 + 1];
            const float norm_2x2 = H_dynamic[0 * 6 + 0] * H_dynamic[0 * 6 + 0]
                                 + H_dynamic[0 * 6 + 1] * H_dynamic[0 * 6 + 1]
                                 + H_dynamic[1 * 6 + 0] * H_dynamic[1 * 6 + 0]
                                 + H_dynamic[1 * 6 + 1] * H_dynamic[1 * 6 + 1];
            const float cond_2x2 = (fabs(det_2x2) > 1e-12f)
                                           ? (norm_2x2 / fabs(det_2x2)) : 1.0e13f;

            if (cond_2x2 > 1.0e12f) {
                out_p[(size_t) gid * 6 + 0] = W[2];
                out_p[(size_t) gid * 6 + 1] = W[5];
                out_p[(size_t) gid * 6 + 2] = W[0] - 1.0f;
                out_p[(size_t) gid * 6 + 3] = W[1];
                out_p[(size_t) gid * 6 + 4] = W[3];
                out_p[(size_t) gid * 6 + 5] = W[4] - 1.0f;
                out_status[gid] = SEMPER_ICGN_FAILED;
                out_score[gid] = 2.0f;
                out_iters[gid] = iter;
                // This one CPU return site is a braced AnalysisResult that
                // never assigns invalid_ref_pixels, so it keeps the member's
                // default of 0 while the other four returns set the real
                // count. Matched deliberately -- a "fix" here would show up
                // as a parity failure, not as an improvement.
                out_invalid_ref[gid] = 0;
                return;
            }

            if (lm_alpha > 0.0f) {
                H_dynamic[0 * 6 + 0] += lm_alpha;
                H_dynamic[1 * 6 + 1] += lm_alpha;
            }
            float dyn_det;
            // Return value unused, as on the CPU: a singular dynamic
            // Hessian still writes H_solve, and the step it produces is
            // caught by the guards rather than by a check here.
            semper_inv6x6(H_dynamic, H_solve, &dyn_det);
        }

        final_score = error_sum_sq / (float) valid_pixels;

        float step[6];
        semper_mat6_vec6(H_solve, dp_sum, step);
        float delta_p[6];
        for (int k = 0; k < 6; ++k) delta_p[k] = -step[k];

        float dW[9] = {1.0f + delta_p[2], delta_p[3],        delta_p[0],
                       delta_p[4],        1.0f + delta_p[5], delta_p[1],
                       0.0f,              0.0f,              1.0f};

        float dW_inv[9], dW_det;
        if (semper_inv3x3f(dW, dW_inv, &dW_det)) {
            float W_next[9];
            semper_mat3_mul(W, dW_inv, W_next);
            for (int k = 0; k < 9; ++k) W[k] = W_next[k];
        }
        // A singular dW means the Newton step collapsed the warp; leaving W
        // untouched lets the iteration budget or the guards end the solve
        // rather than propagating inf/NaN through the field.

        if (semper_norm6(delta_p) < 0.001f) {
            // DICe post-match sigma (texture) check, over the RAW
            // un-normalised reference gradients of the surviving pixels.
            // Sequential on purpose: the survivors are scattered, and the
            // CPU sums them in this same k = 0..n-1 order.
            float sum_gx_sq = 0.0f, sum_gy_sq = 0.0f;
            int valid_survivors = 0;
            {
                int k = 0;
                for (int oy = -half_dim; oy <= half_dim; ++oy) {
                    const int row_base = (cy + oy) * ref_w + (cx - half_dim);
                    for (int ox = 0; ox < dim; ++ox) {
                        if (ref_valid[k] && def_vals[k] >= 0.0f) {
                            const float g_x = ref_gx[row_base + ox];
                            const float g_y = ref_gy[row_base + ox];
                            sum_gx_sq += g_x * g_x;
                            sum_gy_sq += g_y * g_y;
                            ++valid_survivors;
                        }
                        ++k;
                    }
                }
            }

            out_p[(size_t) gid * 6 + 0] = W[2];
            out_p[(size_t) gid * 6 + 1] = W[5];
            out_p[(size_t) gid * 6 + 2] = W[0] - 1.0f;
            out_p[(size_t) gid * 6 + 3] = W[1];
            out_p[(size_t) gid * 6 + 4] = W[3];
            out_p[(size_t) gid * 6 + 5] = W[4] - 1.0f;
            out_iters[gid] = iter + 1;
            out_invalid_ref[gid] = invalid_ref_count;

            // DICe 100% deactivation check.
            if (valid_survivors == 0) {
                out_status[gid] = SEMPER_ICGN_FAILED;
                out_score[gid] = 2.0f;
                return;
            }
            // DICe minimum-gradient check, the barcode killer: texture in
            // one direction only is not texture. min(), never the sum.
            const float sum_grad = (sum_gx_sq < sum_gy_sq) ? sum_gx_sq : sum_gy_sq;
            if (sum_grad <= 0.0f) {
                out_status[gid] = SEMPER_ICGN_FAILED;
                out_score[gid] = 2.0f;
                return;
            }

            out_status[gid] = SEMPER_ICGN_OK;
            out_score[gid] = final_score;
            return;
        }
    }

    // Iteration budget exhausted.
    out_p[(size_t) gid * 6 + 0] = W[2];
    out_p[(size_t) gid * 6 + 1] = W[5];
    out_p[(size_t) gid * 6 + 2] = W[0] - 1.0f;
    out_p[(size_t) gid * 6 + 3] = W[1];
    out_p[(size_t) gid * 6 + 4] = W[3];
    out_p[(size_t) gid * 6 + 5] = W[4] - 1.0f;
    out_status[gid] = SEMPER_ICGN_FAILED;
    out_score[gid] = final_score;
    out_iters[gid] = max_iter;
    out_invalid_ref[gid] = invalid_ref_count;
}
