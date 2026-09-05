/* =====================================================================
 * CANONICAL IMAGE SAMPLING — included once per address space.
 *
 * Do not include this file directly; include canonical_math.h, which
 * expands it for each address space the target supports:
 *
 *   host / private :  SEMPER_AS empty,     names unsuffixed
 *   OpenCL global  :  SEMPER_AS __global,  names suffixed _g
 *
 * These are the exact samplers Image::interpolate_bilinear,
 * Image::interpolate_bicubic and Image::interpolate_keys_fourth call.
 * They were lifted out of src/math/image_processor.cpp when the ICGN
 * kernel needed them, because an interpolator is the single most
 * parity-critical routine in the engine: every ICGN iteration on every
 * point goes through it, and a one-ulp disagreement here does not stay
 * one ulp -- it changes the Newton step, which changes the next warp,
 * which changes which pixels are sampled at all.
 *
 * The demotion boundaries are part of the contract, not a detail. Each
 * sampler falls back to the next-lower-order one at its own margin, and
 * those margins differ (bilinear kills the pixel outright, bicubic
 * demotes at 1.0 / width-2.0, Keys at 2.5 / width-3.5). ICGN's own px/py
 * guard is tighter than all of them, so in practice the fallbacks fire
 * only from evaluate_znssd and the coarse search -- which is exactly why
 * they must be reproduced rather than assumed unreachable.
 *
 * The image is passed as a bare pointer plus width/height rather than as
 * a struct so the same text compiles against std::vector<float>::data()
 * on the host and a __global float* on the device.
 * ===================================================================== */

/* Bilinear 2x2. Returns 0 outside its hard boundary -- that zero is the
 * "kill this pixel" signal the ICGN mask-aware drop tests for (val > 0),
 * so it is a value, not an error code. */
SEMPER_INLINE float SEMPER_FN(semper_canon_sample_bilinear)(
        const SEMPER_AS float *img, int width, int height, float x, float y) {
    if (x < 0.0f || x >= (float) width - 1.5f ||
        y < 0.0f || y >= (float) height - 1.5f) {
        return 0.0f;
    }
    const int xi = (int) x;
    const int yi = (int) y;
    const float dx = x - (float) xi;
    const float dy = y - (float) yi;
    const SEMPER_AS float *row0 = img + (yi * width + xi);
    const SEMPER_AS float *row1 = img + ((yi + 1) * width + xi);
    const float val0 = row0[0] * (1.0f - dx) + row0[1] * dx;
    const float val1 = row1[0] * (1.0f - dx) + row1[1] * dx;
    return val0 * (1.0f - dy) + val1 * dy;
}

/* Catmull-Rom bicubic over a 4x4 neighbourhood, demoting to bilinear
 * within one pixel of the border. Row-major accumulation: each row is
 * summed left to right into row_val, then row_val * wy[j] is added to
 * the running total in row order. Both orders are contractual. */
SEMPER_INLINE float SEMPER_FN(semper_canon_sample_bicubic)(
        const SEMPER_AS float *img, int width, int height, float x, float y) {
    if (x < 1.0f || x >= (float) width - 2.0f ||
        y < 1.0f || y >= (float) height - 2.0f) {
        return SEMPER_FN(semper_canon_sample_bilinear)(img, width, height, x, y);
    }

    const int xi = (int) x;
    const int yi = (int) y;
    const float dx = x - (float) xi;
    const float dy = y - (float) yi;

    float wx[4], wy[4];
    semper_keys4_weights(dx, wx);
    semper_keys4_weights(dy, wy);

    float val = 0.0f;
    int j;
    for (j = -1; j <= 2; ++j) {
        const SEMPER_AS float *row_ptr = img + ((yi + j) * width + xi);
        float row_val = 0.0f;
        row_val += row_ptr[-1] * wx[0];
        row_val += row_ptr[ 0] * wx[1];
        row_val += row_ptr[ 1] * wx[2];
        row_val += row_ptr[ 2] * wx[3];
        val += row_val * wy[j + 1];
    }
    return val;
}

/* Keys 4th-order over a 6x6 neighbourhood -- the DICe-parity sampler,
 * selected by FullFieldParams::use_6x6_interpolator. Note the demotion
 * test uses <= on the low side and >= on the high side, matching DICe;
 * it is deliberately not symmetric with the bicubic one above. */
SEMPER_INLINE float SEMPER_FN(semper_canon_sample_keys6)(
        const SEMPER_AS float *img, int width, int height, float x, float y) {
    if (x <= 2.5f || x >= (float) width - 3.5f ||
        y <= 2.5f || y >= (float) height - 3.5f) {
        return SEMPER_FN(semper_canon_sample_bilinear)(img, width, height, x, y);
    }

    const int xi = (int) x;
    const int yi = (int) y;
    const float dx = x - (float) xi;
    const float dy = y - (float) yi;

    float wx[6], wy[6];
    semper_keys6_weights(dx, wx);
    semper_keys6_weights(dy, wy);

    float val = 0.0f;
    int j;
    for (j = -2; j <= 3; ++j) {
        const SEMPER_AS float *row_ptr = img + ((yi + j) * width + xi);
        float row_val = 0.0f;
        row_val += row_ptr[-2] * wx[0];
        row_val += row_ptr[-1] * wx[1];
        row_val += row_ptr[ 0] * wx[2];
        row_val += row_ptr[ 1] * wx[3];
        row_val += row_ptr[ 2] * wx[4];
        row_val += row_ptr[ 3] * wx[5];
        val += row_val * wy[j + 2];
    }
    return val;
}

/* Dispatch on the interpolator flag the pipeline carries around, so the
 * caller does not branch on it at every pixel in its own words. */
SEMPER_INLINE float SEMPER_FN(semper_canon_sample)(
        const SEMPER_AS float *img, int width, int height,
        float x, float y, int use_keys6) {
    return use_keys6 ? SEMPER_FN(semper_canon_sample_keys6)(img, width, height, x, y)
                     : SEMPER_FN(semper_canon_sample_bicubic)(img, width, height, x, y);
}
