// Phase 2: the VSG (virtual strain gauge) plane fit, one work-item per grid
// point.
//
// This is a transcription, not a reimplementation. Every line below has a
// counterpart in Semper::StrainCalculator::compute_vsg_strain
// (src/strain/strain_calculator.cpp), and the two must produce bit-identical
// float output -- tests/unit/test_cl_strain.cpp compares them with ==, never
// a tolerance. Three things carry that guarantee and none of them may be
// "cleaned up":
//
//   * the dy-outer / dx-inner window scan, which fixes the order the nine
//     AtA entries and the two right-hand sides accumulate in;
//   * semper_inv3x3 / semper_mat3_vec3 / semper_mat3_l1_norm from
//     canonical_math.h, shared verbatim with the host -- which is why the
//     CPU side had to stop using Eigen before this kernel could exist;
//   * the scalars the host precomputes (radius_sq, grid_rad,
//     d_radius_sq, expected_full_window_pts) rather than recomputing here,
//     so a float/int rounding difference cannot creep in between the two.
//
// The whole kernel is fp64 and therefore compiled only where cl_khr_fp64 is.
// On a device without it this file contributes no kernel at all and the
// program still builds, leaving the fp32 stages working -- see the
// SEMPER_HAS_FP64 comment in canonical_math.h. Dispatch checks caps().fp64
// before it ever reaches here.

#pragma OPENCL FP_CONTRACT OFF

#include <semper/kernels/canonical_math.h>

#if SEMPER_HAS_FP64

// valid[] is uchar, not the host's std::vector<bool>: that type has no
// contiguous storage to upload, so the caller unpacks it first.
__kernel void semper_strain_vsg(__global const float *u,
                                __global const float *v,
                                __global const uchar *valid,
                                const int width,
                                const int height,
                                const int step,
                                const int grid_rad,
                                const double d_radius_sq,
                                const int expected_full_window_pts,
                                const float sentinel,
                                __global float *exx,
                                __global float *eyy,
                                __global float *exy) {
    const int idx = get_global_id(0);
    if (idx >= width * height) return;

    // Every early-out below leaves the sentinel in place, exactly as the CPU
    // path leaves the value it pre-filled the arrays with.
    exx[idx] = sentinel;
    eyy[idx] = sentinel;
    exy[idx] = sentinel;

    if (!valid[idx]) return;

    const int x = idx % width;
    const int y = idx / width;

    double AtA[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double AtU[3] = {0.0, 0.0, 0.0};
    double AtV[3] = {0.0, 0.0, 0.0};
    int valid_pts = 0;

    for (int dy = -grid_rad; dy <= grid_rad; ++dy) {
        for (int dx = -grid_rad; dx <= grid_rad; ++dx) {
            const int nx = x + dx;
            const int ny = y + dy;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            const int nidx = ny * width + nx;
            if (!valid[nidx]) continue;

            const double d_phys_dx = (double) (dx * step);
            const double d_phys_dy = (double) (dy * step);

            if ((d_phys_dx * d_phys_dx + d_phys_dy * d_phys_dy) <= d_radius_sq) {
                const double d_u = (double) u[nidx];
                const double d_v = (double) v[nidx];

                const double a[3] = {1.0, d_phys_dx, d_phys_dy};

                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        AtA[r * 3 + c] += a[r] * a[c];
                    }
                    AtU[r] += a[r] * d_u;
                    AtV[r] += a[r] * d_v;
                }
                valid_pts++;
            }
        }
    }

    // 90% structural fill: a half-circle window at a field edge fits a plane
    // to a badly shifted centroid, and rcond alone does not catch it.
    const double fill_ratio =
            (double) valid_pts / (double) expected_full_window_pts;
    if (!(fill_ratio >= 0.90 && valid_pts >= 3)) return;

    // LAPACK GECON('1') reciprocal condition number.
    const double anorm = semper_mat3_l1_norm(AtA);

    double AtA_inv[9];
    double det;
    if (!semper_inv3x3(AtA, AtA_inv, &det)) return;

    const double inv_anorm = semper_mat3_l1_norm(AtA_inv);
    const double rcond =
            (anorm * inv_anorm > 0.0) ? (1.0 / (anorm * inv_anorm)) : 0.0;
    if (rcond < 1e-12) return;

    double Cu[3];
    double Cv[3];
    semper_mat3_vec3(AtA_inv, AtU, Cu);
    semper_mat3_vec3(AtA_inv, AtV, Cv);

    const double dudx = Cu[1];
    const double dudy = Cu[2];
    const double dvdx = Cv[1];
    const double dvdy = Cv[2];

    // Green-Lagrange, large deformation. Written out in the CPU path's exact
    // grouping -- the 0.5 factored out front, and the narrowing to float
    // last, once.
    exx[idx] = (float) (0.5 * (2.0 * dudx + dudx * dudx + dvdx * dvdx));
    eyy[idx] = (float) (0.5 * (2.0 * dvdy + dudy * dudy + dvdy * dvdy));
    exy[idx] = (float) (0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy));
}

#endif /* SEMPER_HAS_FP64 */
