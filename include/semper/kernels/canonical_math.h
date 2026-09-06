#ifndef SEMPER_CANONICAL_MATH_H
#define SEMPER_CANONICAL_MATH_H

/* =====================================================================
 * CANONICAL MATH — the single definition of every numerical kernel that
 * the CPU and the OpenCL backend must agree on bit-for-bit.
 *
 * This header is written in the common subset of C99 and OpenCL C so the
 * SAME TEXT compiles into both the host library and the .cl kernels.
 * There is deliberately no second copy: two hand-maintained versions
 * drifting apart is the failure mode this file exists to prevent.
 *
 * RULES FOR ANYTHING ADDED HERE
 *   1. No C++. No namespaces, templates, references, or overloads.
 *   2. Only + - * / and sqrt. No transcendentals, no native_* builtins,
 *      no fma() (contraction is switched off on both sides, so an
 *      explicit fma would change the result).
 *   3. Every reduction uses SEMPER_COMBINE4 (below). Never write a bare
 *      sequential accumulation over a long array.
 *   4. Loop bounds and operation order are part of the contract. Do not
 *      "tidy" a loop here without recapturing the golden fixtures.
 *
 * Compiled with -fno-fast-math -ffp-contract=off (host, see
 * CMakeLists.txt) and without -cl-fast-relaxed-math / -cl-mad-enable
 * plus `#pragma OPENCL FP_CONTRACT OFF` (device). See docs/DETERMINISM.md.
 * ===================================================================== */

/* ---------------------------------------------------------------------
 * CONTRACT ENFORCEMENT
 *
 * Everything below is written so that a specific sequence of floating
 * point operations happens in a specific order. -ffast-math licenses the
 * compiler to reassociate exactly those sequences, which silently
 * destroys the guarantee -- the code still computes "the sum", just not
 * the sum this file promises.
 *
 * That failure is invisible at runtime and only shows up as a CPU/GPU
 * mismatch much later, so it is made a build error here instead. Any
 * translation unit including this header must be compiled with
 * -fno-fast-math -ffp-contract=off (see CMakeLists.txt and
 * tests/CMakeLists.txt, which both maintain that source list).
 * ------------------------------------------------------------------- */
#if defined(__FAST_MATH__)
  #error "canonical_math.h requires -fno-fast-math: -ffast-math permits the \
reassociation this header exists to prevent. Add this translation unit to \
the strict-FP source list in CMakeLists.txt / tests/CMakeLists.txt."
#endif

/* Which language are we being compiled as?
 * A real OpenCL runtime compiler defines __OPENCL_VERSION__ (OpenCL spec
 * 6.10); clang in offline mode (-x cl) defines only __OPENCL_C_VERSION__.
 * Check both, or an offline syntax check silently takes the host branch
 * and every address-space bug goes unnoticed until device build time. */
#if defined(__OPENCL_VERSION__) || defined(__OPENCL_C_VERSION__)
  #define SEMPER_IS_OPENCL 1
#else
  #define SEMPER_IS_OPENCL 0
#endif

#if !SEMPER_IS_OPENCL
  #include <math.h>          /* OpenCL C has sqrt/fabs as builtins */
#endif

/* Is double precision available?
 *
 * The host always has it. A device does NOT: cl_khr_fp64 is an optional
 * OpenCL 1.2 extension, and Mali/Adreno parts largely lack it. Using
 * `double` without the pragma is a compile error there, which would fail
 * the whole clBuildProgram -- taking down the fp32 ICGN path too, even
 * though it has no fp64 dependency. That would collapse the per-stage
 * capability gate (see src/gpu/cl_runtime.hpp: fp64 and exact_fp32 are
 * deliberately separate flags) into one all-or-nothing switch.
 *
 * So the double-precision section below is compiled only when the device
 * actually supports it, and a device without it still builds and runs
 * every float kernel. Dispatch checks caps().fp64 before enqueuing
 * anything that needs semper_inv3x3. */
#if SEMPER_IS_OPENCL
  #if defined(cl_khr_fp64) || defined(__opencl_c_fp64) || defined(cl_amd_fp64)
    #if defined(cl_khr_fp64)
      #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif defined(cl_amd_fp64)
      #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif
    #define SEMPER_HAS_FP64 1
  #else
    #define SEMPER_HAS_FP64 0
  #endif
#else
  #define SEMPER_HAS_FP64 1
#endif

#ifndef SEMPER_INLINE
  #ifdef __cplusplus
    #define SEMPER_INLINE inline
  #else
    #define SEMPER_INLINE static inline
  #endif
#endif

/* ---------------------------------------------------------------------
 * THE CANONICAL REDUCTION
 *
 * Every long summation is accumulated into exactly FOUR accumulators,
 * stride-4 interleaved, and combined in this fixed pairwise order:
 *
 *     acc[k] += term(4*i + k)          k = 0..3
 *     result  = (acc0 + acc1) + (acc2 + acc3)
 *     result += term(j)                 j = 4*(n/4) .. n-1, in order
 *
 * Four is chosen because it is reproducible everywhere that matters: it
 * is one native SSE/NEON vector, two halves of an AVX2 vector, and four
 * scalar registers inside an OpenCL work-item. Without this the engine
 * produces different bits on AVX2 (8 lanes) than on NEON (4 lanes) --
 * which is exactly what it did before, and why the golden tolerances
 * had been widened for "runner FP noise".
 *
 * NOTE ON THE TAIL: the remainder is added to the COMBINED result, not
 * folded into acc0 before the combine. This matches the shape a SIMD
 * implementation naturally has (vector body -> lane reduce -> scalar
 * tail) and is therefore the cheaper of the two equivalent conventions
 * to honour on every backend. n = 729 for a 27 px subset, so the tail is
 * genuinely exercised (729 = 4*182 + 1).
 * ------------------------------------------------------------------- */
#define SEMPER_COMBINE4(a0, a1, a2, a3) (((a0) + (a1)) + ((a2) + (a3)))

/* Largest multiple of 4 not exceeding n. */
#define SEMPER_BODY4(n) ((n) & ~((int)3))

/* The three reductions are single-sourced in canonical_reductions.inc and
 * expanded once per address space. OpenCL C 1.2 has no generic address
 * space, so a `const float *` parameter cannot receive a `__global float *`;
 * the ICGN scratch buffers are far too large for private memory, so the
 * device genuinely needs __global variants. Host code uses the unsuffixed
 * names; kernels use the _g names for global pointers and the unsuffixed
 * ones for private arrays. */

#define SEMPER_AS
#define SEMPER_FN(name) name
#include "canonical_reductions.inc"
#undef SEMPER_AS
#undef SEMPER_FN

#if SEMPER_IS_OPENCL
#define SEMPER_AS __global
#define SEMPER_FN(name) name##_g
#include "canonical_reductions.inc"
#undef SEMPER_AS
#undef SEMPER_FN
#endif

/* ---------------------------------------------------------------------
 * INTERPOLATION WEIGHTS
 * Lifted verbatim from src/math/image_processor.cpp. The polynomial
 * forms and the order the six/four weights are written are part of the
 * contract -- rewriting one as a Horner form changes the low bit.
 * ------------------------------------------------------------------- */

/* Keys 4th-order, zone 0: |s| <= 1 */
SEMPER_INLINE float semper_keys_f0(float s) {
    return (4.0f / 3.0f) * s * s * s - (7.0f / 3.0f) * s * s + 1.0f;
}
/* Keys 4th-order, zone 1: 1 < |s| <= 2 */
SEMPER_INLINE float semper_keys_f1(float s) {
    return -(7.0f / 12.0f) * s * s * s + 3.0f * s * s - (59.0f / 12.0f) * s + 2.5f;
}
/* Keys 4th-order, zone 2: 2 < |s| <= 3 */
SEMPER_INLINE float semper_keys_f2(float s) {
    return (1.0f / 12.0f) * s * s * s - (2.0f / 3.0f) * s * s + (7.0f / 4.0f) * s - 1.5f;
}

/* The six Keys weights for a fractional offset d in [0,1). */
SEMPER_INLINE void semper_keys6_weights(float d, float w[6]) {
    w[0] = semper_keys_f2(d + 2.0f);
    w[1] = semper_keys_f1(d + 1.0f);
    w[2] = semper_keys_f0(d);
    w[3] = semper_keys_f0(1.0f - d);
    w[4] = semper_keys_f1(2.0f - d);
    w[5] = semper_keys_f2(3.0f - d);
}

/* The four Catmull-Rom (a = -0.5) bicubic weights. */
SEMPER_INLINE void semper_keys4_weights(float s, float w[4]) {
    const float s2 = s * s;
    const float s3 = s2 * s;
    w[0] = -0.5f * s3 + s2 - 0.5f * s;
    w[1] =  1.5f * s3 - 2.5f * s2 + 1.0f;
    w[2] = -1.5f * s3 + 2.0f * s2 + 0.5f * s;
    w[3] =  0.5f * s3 - 0.5f * s2;
}

/* The samplers are single-sourced the same way the reductions are, and
 * for the same reason: the ICGN kernel needs __global variants, and a
 * hand-maintained device copy of an interpolator would drift. */

#define SEMPER_AS
#define SEMPER_FN(name) name
#include "canonical_interp.inc"
#undef SEMPER_AS
#undef SEMPER_FN

#if SEMPER_IS_OPENCL
#define SEMPER_AS __global
#define SEMPER_FN(name) name##_g
#include "canonical_interp.inc"
#undef SEMPER_AS
#undef SEMPER_FN
#endif

/* ---------------------------------------------------------------------
 * IMAGE GRADIENT STENCIL
 *
 * The 5-point central difference Image::prepare_data applies to every
 * interior pixel, in both x and y. Written as a macro rather than a
 * function because the four samples come from a __global pointer in the
 * kernel and a std::vector in the host, and this is the one piece of
 * arithmetic small enough that an address-space duplication would cost
 * more than it explains.
 *
 * The parenthesisation IS the contract. It expands to
 *
 *     (((-p2 + 8*p1) - 8*m1) + m2) / 12
 *
 * left to right, exactly as the host expression in
 * src/math/image_processor.cpp parses. Rewriting it as
 * (8*(p1-m1) - (p2-m2))/12 is algebraically identical and numerically
 * is not: it changes which intermediate is rounded first. The
 * multiplications by 8 are exact (a power of two, and the operands are
 * bounded intensities), so the only rounding is the three adds and the
 * divide -- and the divide is why the OpenCL build asks for
 * -cl-fp32-correctly-rounded-divide-sqrt.
 *
 * All four arguments must be float. Offsets are named for their position
 * relative to the centre pixel: m2 = idx-2, p1 = idx+1, and so on. The
 * centre pixel itself does not appear -- a central difference does not
 * read it.
 * ------------------------------------------------------------------- */
#define SEMPER_CANON_DERIV5(m2, m1, p1, p2)     ((-(p2) + 8.0f * (p1) - 8.0f * (m1) + (m2)) / 12.0f)

/* Width of the band at each edge of the image where the stencil would
 * read out of bounds. Those pixels keep the 0.0f the CPU path zero-fills
 * them with, and the kernel writes the same zero rather than skipping
 * them, so the two buffers agree everywhere and not merely in the
 * interior. */
#define SEMPER_GRAD_BORDER 2

/* ---------------------------------------------------------------------
 * MATRIX INVERSES
 *
 * These replace Eigen's Matrix3d/Matrix<float,6,6> inverse() on the host
 * as well as providing the device implementation. Eigen's LU uses
 * internal blocking and pivoting we cannot call from a kernel, so rather
 * than have the GPU chase Eigen, BOTH sides use the definitions below.
 * ------------------------------------------------------------------- */

/* The double-precision half of this section exists only where fp64 does.
 * Everything from here to the matching #endif is skipped on a device
 * without cl_khr_fp64; semper_inv6x6 and the float routines below are
 * always available. */
#if SEMPER_HAS_FP64

/* Matches Eigen's default invertibility threshold for double. */
#define SEMPER_DET_EPS_D 2.220446049250313e-16

/* 3x3 cofactor inverse in double, for the strain VSG normal equations.
 * Row-major a[9] -> inv[9]. Returns 1 when invertible, 0 otherwise;
 * *det_out is always written. */
SEMPER_INLINE int semper_inv3x3(const double a[9], double inv[9], double *det_out) {
    const double c00 =  (a[4] * a[8] - a[5] * a[7]);
    const double c01 = -(a[3] * a[8] - a[5] * a[6]);
    const double c02 =  (a[3] * a[7] - a[4] * a[6]);

    const double det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    *det_out = det;
    if (!(fabs(det) > SEMPER_DET_EPS_D)) return 0;

    const double r = 1.0 / det;
    /* Adjugate is the transpose of the cofactor matrix. */
    inv[0] = c00 * r;
    inv[1] = -(a[1] * a[8] - a[2] * a[7]) * r;
    inv[2] =  (a[1] * a[5] - a[2] * a[4]) * r;
    inv[3] = c01 * r;
    inv[4] =  (a[0] * a[8] - a[2] * a[6]) * r;
    inv[5] = -(a[0] * a[5] - a[2] * a[3]) * r;
    inv[6] = c02 * r;
    inv[7] = -(a[0] * a[7] - a[1] * a[6]) * r;
    inv[8] =  (a[0] * a[4] - a[1] * a[3]) * r;
    return 1;
}

/* Row-major 3x3 times 3-vector in double: out = m * v.
 *
 * Accumulation is pinned left to right. This is the VSG plane-fit solve,
 * run on both sides of the CPU/GPU boundary; Eigen's fixed-size product
 * would vectorize it differently per ISA, which is one of the three
 * divergence causes docs/DETERMINISM.md documents. */
SEMPER_INLINE void semper_mat3_vec3(const double m[9], const double v[3],
                                    double out[3]) {
    int r;
    for (r = 0; r < 3; ++r)
        out[r] = ((m[r * 3 + 0] * v[0]) + (m[r * 3 + 1] * v[1]))
                 + (m[r * 3 + 2] * v[2]);
}

/* Max absolute column sum of a row-major 3x3 -- the L1 matrix norm, which
 * is what LAPACK's GECON('1') uses. Paired with the same norm of the
 * inverse it gives the reciprocal condition number the VSG fit gates on. */
SEMPER_INLINE double semper_mat3_l1_norm(const double m[9]) {
    double best = 0.0;
    int c;
    for (c = 0; c < 3; ++c) {
        const double s = fabs(m[0 * 3 + c]) + fabs(m[1 * 3 + c])
                         + fabs(m[2 * 3 + c]);
        if (s > best) best = s;
    }
    return best;
}

#endif /* SEMPER_HAS_FP64 */

/* 6x6 inverse in float by Gauss-Jordan with partial pivoting, for the
 * ICGN Hessian. Row-major a[36] -> inv[36]. *det_out receives the
 * product of the pivots times the permutation sign. Returns 1 when
 * invertible, 0 otherwise (inv is left untouched on failure).
 *
 * Pivot selection takes the strictly largest |value| at or below the
 * current row, so ties resolve to the LOWEST row index on every
 * backend -- an unspecified tie-break would be a bit-exactness hole. */
SEMPER_INLINE int semper_inv6x6(const float a[36], float inv[36], float *det_out) {
    float m[6][12];
    int r, c, k, piv;
    float det = 1.0f;

    for (r = 0; r < 6; ++r) {
        for (c = 0; c < 6; ++c) {
            m[r][c] = a[r * 6 + c];
            m[r][6 + c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    for (c = 0; c < 6; ++c) {
        piv = c;
        for (r = c + 1; r < 6; ++r)
            if (fabs(m[r][c]) > fabs(m[piv][c])) piv = r;

        if (piv != c) {
            for (k = 0; k < 12; ++k) {
                const float t = m[c][k];
                m[c][k] = m[piv][k];
                m[piv][k] = t;
            }
            det = -det;
        }

        const float p = m[c][c];
        det = det * p;
        if (!(fabs(p) > 0.0f)) { *det_out = 0.0f; return 0; }

        const float ip = 1.0f / p;
        for (k = 0; k < 12; ++k) m[c][k] = m[c][k] * ip;

        for (r = 0; r < 6; ++r) {
            if (r == c) continue;
            const float f = m[r][c];
            if (f == 0.0f) continue;
            for (k = 0; k < 12; ++k) m[r][k] = m[r][k] - f * m[c][k];
        }
    }

    *det_out = det;
    for (r = 0; r < 6; ++r)
        for (c = 0; c < 6; ++c)
            inv[r * 6 + c] = m[r][6 + c];
    return 1;
}


/* ---------------------------------------------------------------------
 * SMALL DENSE LINEAR ALGEBRA
 *
 * These replace the Eigen fixed-size float expressions inside the ICGN
 * iteration. Eigen packs a 6x6*6x1 product into two SSE registers or one
 * masked AVX register depending on the -march level, and the two
 * associate the dot products differently -- which made the solver return
 * different values on AVX2 than on SSE2 even though every input stage
 * was bit-identical. Writing the loops out fixes the order on every
 * backend, and is also the only form callable from an OpenCL kernel.
 *
 * ALL MATRICES HERE ARE ROW-MAJOR. Eigen defaults to column-major, so
 * callers holding Eigen storage must transpose on the way in and out --
 * do not hand .data() to these directly.
 * ------------------------------------------------------------------- */

/* out[6] = m[6][6] * v[6], each dot product accumulated in index order. */
SEMPER_INLINE void semper_mat6_vec6(const float m[36], const float v[6],
                                    float out[6]) {
    int r, c;
    for (r = 0; r < 6; ++r) {
        float s = 0.0f;
        for (c = 0; c < 6; ++c) s += m[r * 6 + c] * v[c];
        out[r] = s;
    }
}

/* sqrt(Sigma v[i]^2) over 6 elements, summed in index order. */
SEMPER_INLINE float semper_norm6(const float v[6]) {
    float s = 0.0f;
    int i;
    for (i = 0; i < 6; ++i) s += v[i] * v[i];
    return sqrt(s);
}

/* out = a * b for row-major 3x3 float. */
SEMPER_INLINE void semper_mat3_mul(const float a[9], const float b[9],
                                   float out[9]) {
    int r, c, k;
    for (r = 0; r < 3; ++r)
        for (c = 0; c < 3; ++c) {
            float s = 0.0f;
            for (k = 0; k < 3; ++k) s += a[r * 3 + k] * b[k * 3 + c];
            out[r * 3 + c] = s;
        }
}

/* Matches Eigen's default invertibility threshold for float. */
#define SEMPER_DET_EPS_F 1.1920929e-7f

/* 3x3 cofactor inverse in float -- the ICGN warp update dW. Same shape as
 * semper_inv3x3 but single precision, because the warp matrix is float
 * and promoting it here would change the solver's arithmetic. */
SEMPER_INLINE int semper_inv3x3f(const float a[9], float inv[9], float *det_out) {
    const float c00 =  (a[4] * a[8] - a[5] * a[7]);
    const float c01 = -(a[3] * a[8] - a[5] * a[6]);
    const float c02 =  (a[3] * a[7] - a[4] * a[6]);

    const float det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    *det_out = det;
    if (!(fabs(det) > SEMPER_DET_EPS_F)) return 0;

    const float r = 1.0f / det;
    inv[0] = c00 * r;
    inv[1] = -(a[1] * a[8] - a[2] * a[7]) * r;
    inv[2] =  (a[1] * a[5] - a[2] * a[4]) * r;
    inv[3] = c01 * r;
    inv[4] =  (a[0] * a[8] - a[2] * a[6]) * r;
    inv[5] = -(a[0] * a[5] - a[2] * a[3]) * r;
    inv[6] = c02 * r;
    inv[7] = -(a[0] * a[7] - a[1] * a[6]) * r;
    inv[8] =  (a[0] * a[4] - a[1] * a[3]) * r;
    return 1;
}

/* Accumulate the outer product s*s^T of a 6-vector into a row-major 6x6,
 * upper triangle only (21 multiplies rather than 36), then mirrored.
 * Mirrors the accumulation already written out in subset_precomputer.cpp. */
SEMPER_INLINE void semper_mat6_add_outer(float h[36], const float s[6]) {
    int r, c;
    for (r = 0; r < 6; ++r)
        for (c = r; c < 6; ++c)
            h[r * 6 + c] += s[r] * s[c];
}

SEMPER_INLINE void semper_mat6_symmetrize(float h[36]) {
    int r, c;
    for (r = 1; r < 6; ++r)
        for (c = 0; c < r; ++c)
            h[r * 6 + c] = h[c * 6 + r];
}

#endif /* SEMPER_CANONICAL_MATH_H */
