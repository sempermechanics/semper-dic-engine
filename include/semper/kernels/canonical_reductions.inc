/* =====================================================================
 * CANONICAL REDUCTIONS — included once per address space.
 *
 * Do not include this file directly; include canonical_math.h, which
 * expands it for each address space the target supports:
 *
 *   host / private :  SEMPER_AS empty,     names unsuffixed
 *   OpenCL global  :  SEMPER_AS __global,  names suffixed _g
 *
 * OpenCL C 1.2 has no generic address space, so a function taking
 * `const float *` cannot accept a `__global float *`. Rather than keep
 * two hand-written copies of the arithmetic -- the one thing this whole
 * design exists to avoid -- the bodies live here once and the header
 * re-includes them with SEMPER_AS / SEMPER_FN redefined.
 *
 * Every accumulation below follows the canonical 4-accumulator form
 * documented in canonical_math.h. Do not "simplify" one into a single
 * running sum: that is the bit-exactness contract, not a style choice.
 * ===================================================================== */

/* Sigma (vals[i] - mean)^2 -- reference for Semper::simd::sum_sq_diff. */
SEMPER_INLINE float SEMPER_FN(semper_canon_sum_sq_diff)(
        const SEMPER_AS float *vals, int n, float mean) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const int nb = SEMPER_BODY4(n);
    int i;
    for (i = 0; i < nb; i += 4) {
        const float d0 = vals[i    ] - mean;
        const float d1 = vals[i + 1] - mean;
        const float d2 = vals[i + 2] - mean;
        const float d3 = vals[i + 3] - mean;
        a0 += d0 * d0;
        a1 += d1 * d1;
        a2 += d2 * d2;
        a3 += d3 * d3;
    }
    float sum = SEMPER_COMBINE4(a0, a1, a2, a3);
    for (i = nb; i < n; ++i) {
        const float d = vals[i] - mean;
        sum += d * d;
    }
    return sum;
}

/* Sigma (ref[i] - (vals[i] - mean) * inv_std)^2 -- the ZNSSD residual.
 * Reference for Semper::simd::znssd_sum. */
SEMPER_INLINE float SEMPER_FN(semper_canon_znssd_sum)(
        const SEMPER_AS float *vals, const SEMPER_AS float *ref,
        int n, float mean, float inv_std) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const int nb = SEMPER_BODY4(n);
    int i;
    for (i = 0; i < nb; i += 4) {
        const float e0 = ref[i    ] - (vals[i    ] - mean) * inv_std;
        const float e1 = ref[i + 1] - (vals[i + 1] - mean) * inv_std;
        const float e2 = ref[i + 2] - (vals[i + 2] - mean) * inv_std;
        const float e3 = ref[i + 3] - (vals[i + 3] - mean) * inv_std;
        a0 += e0 * e0;
        a1 += e1 * e1;
        a2 += e2 * e2;
        a3 += e3 * e3;
    }
    float sum = SEMPER_COMBINE4(a0, a1, a2, a3);
    for (i = nb; i < n; ++i) {
        const float e = ref[i] - (vals[i] - mean) * inv_std;
        sum += e * e;
    }
    return sum;
}

/* Fused ZNSSD residual + 6-DOF steepest-descent gradient.
 *
 *   error  = Sigma diff^2,  diff = ref[i] - (vals[i]-mean)*inv_std
 *   dp[k] += Sigma sdi_plane_k[i] * diff        k = 0..5
 *
 * sdi_planes is the Structure-of-Arrays mirror of the steepest-descent
 * images: plane k occupies [k*n, (k+1)*n). Reference for
 * Semper::simd::znssd_error_and_gradient. All seven accumulations
 * independently follow the canonical 4-accumulator form. */
SEMPER_INLINE float SEMPER_FN(semper_canon_znssd_error_and_gradient)(
        const SEMPER_AS float *vals, const SEMPER_AS float *ref,
        const SEMPER_AS float *sdi_planes, int n,
        float mean, float inv_std, float dp_out[6]) {
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    float g[6][4];
    const int nb = SEMPER_BODY4(n);
    int i, k, L;

    for (k = 0; k < 6; ++k)
        for (L = 0; L < 4; ++L)
            g[k][L] = 0.0f;

    for (i = 0; i < nb; i += 4) {
        const float d0 = ref[i    ] - (vals[i    ] - mean) * inv_std;
        const float d1 = ref[i + 1] - (vals[i + 1] - mean) * inv_std;
        const float d2 = ref[i + 2] - (vals[i + 2] - mean) * inv_std;
        const float d3 = ref[i + 3] - (vals[i + 3] - mean) * inv_std;
        e0 += d0 * d0;
        e1 += d1 * d1;
        e2 += d2 * d2;
        e3 += d3 * d3;
        for (k = 0; k < 6; ++k) {
            const SEMPER_AS float *p = sdi_planes + (int)k * n;
            g[k][0] += p[i    ] * d0;
            g[k][1] += p[i + 1] * d1;
            g[k][2] += p[i + 2] * d2;
            g[k][3] += p[i + 3] * d3;
        }
    }

    float err = SEMPER_COMBINE4(e0, e1, e2, e3);
    for (k = 0; k < 6; ++k)
        dp_out[k] = SEMPER_COMBINE4(g[k][0], g[k][1], g[k][2], g[k][3]);

    for (i = nb; i < n; ++i) {
        const float d = ref[i] - (vals[i] - mean) * inv_std;
        err += d * d;
        for (k = 0; k < 6; ++k)
            dp_out[k] += sdi_planes[(int)k * n + i] * d;
    }
    return err;
}
