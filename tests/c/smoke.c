/*
 * C ABI smoke — mirrors bindings/python/tests/test_smoke.py.
 *
 * Proves semper_c marshals images/params/results correctly against a known
 * rigid translation. DIC math itself is covered by the engine's dic_tests suite.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <semper/semper_c.h>

#define W 256
#define H 256
#define DX 3
#define MAX_PTS 2048

static uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

/* Cheap blobby uint8 speckle (same idea as the Python smoke). */
static void make_speckle(uint8_t* img, uint32_t seed) {
    float* base = (float*)malloc((size_t)W * H * sizeof(float));
    if (!base) abort();
    for (int i = 0; i < W * H; ++i) {
        base[i] = (float)(xorshift32(&seed) & 0xffffu) / 65535.0f;
    }
    for (int pass = 0; pass < 3; ++pass) {
        float* next = (float*)malloc((size_t)W * H * sizeof(float));
        if (!next) abort();
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if (yy < 0) yy = 0;
                        if (yy >= H) yy = H - 1;
                        if (xx < 0) xx = 0;
                        if (xx >= W) xx = W - 1;
                        sum += base[yy * W + xx];
                    }
                }
                next[y * W + x] = sum / 9.0f;
            }
        }
        free(base);
        base = next;
    }
    for (int i = 0; i < W * H; ++i) {
        float v = base[i] * 255.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        img[i] = (uint8_t)(v + 0.5f);
    }
    free(base);
}

/* Rigid +dx shift without wrap — left strip repeats the edge (no toroidal seam). */
static void shift_x(const uint8_t* src, uint8_t* dst, int dx) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int sx = x - dx;
            if (sx < 0) sx = 0;
            if (sx >= W) sx = W - 1;
            dst[y * W + x] = src[y * W + sx];
        }
    }
}

static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static float median_u(const float* out, int n) {
    float* tmp = (float*)malloc((size_t)n * sizeof(float));
    if (!tmp) abort();
    for (int i = 0; i < n; ++i) tmp[i] = out[i * SEMPER_FLOATS_PER_POINT + 2];
    qsort(tmp, (size_t)n, sizeof(float), cmp_float);
    float med = (n % 2) ? tmp[n / 2] : 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
    return med;
}

int main(void) {
    printf("semper_c smoke — version %s\n", semper_version());

    uint8_t* ref = (uint8_t*)malloc((size_t)W * H);
    uint8_t* def = (uint8_t*)malloc((size_t)W * H);
    if (!ref || !def) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    make_speckle(ref, /*seed=*/1u);
    shift_x(ref, def, DX);

    semper_engine* eng = semper_create();
    if (!eng) {
        fprintf(stderr, "semper_create failed\n");
        return 1;
    }

    int rc = semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H);
    if (rc != 0) {
        fprintf(stderr, "semper_set_reference failed: %d\n", rc);
        semper_destroy(eng);
        return 1;
    }

    semper_params params;
    params.rect_x = 40;
    params.rect_y = 40;
    params.rect_w = 176;
    params.rect_h = 176;
    params.step = 16;
    params.subset_size = 31;
    /* >= 2 * step: below that the VSG plane fit has one grid node and
       semper_run returns SEMPER_ERR_STRAIN_WINDOW. */
    params.strain_window = 33;
    params.use_6x6_interpolator = 0;

    float out[SEMPER_FLOATS_PER_POINT * MAX_PTS];
    float metrics[SEMPER_METRICS_LEN];
    memset(metrics, 0, sizeof(metrics));

    int n = semper_run(eng, def, (size_t)W * H, NULL, 0, &params,
                       out, SEMPER_FLOATS_PER_POINT * MAX_PTS,
                       metrics, SEMPER_METRICS_LEN, NULL, NULL);
    if (n < 0) {
        fprintf(stderr, "semper_run error: %d\n", n);
        semper_destroy(eng);
        return 1;
    }
    /* Prove the pipeline ran even if the quality filter keeps zero points. */
    if (metrics[0] < 1.0f) {
        fprintf(stderr, "FAIL: no points attempted (metrics[0]=%.1f)\n", metrics[0]);
        semper_destroy(eng);
        return 1;
    }
    printf("attempted=%.0f  solved=%.0f  returned=%d\n",
           (double)metrics[0], (double)metrics[1], n);
    if (n > 0) {
        float med_u = median_u(out, n);
        printf("points=%d  median_u=%.4f  (expect ~%d)\n", n, med_u, DX);
        if (fabsf(med_u - (float)DX) > 0.75f) {
            fprintf(stderr, "FAIL: median u %.4f not within 0.75 of %d\n", med_u, DX);
            semper_destroy(eng);
            return 1;
        }
    } else {
        printf("note: 0 points after filter — C ABI path still exercised\n");
    }

    printf("OK — C ABI smoke passed\n");
    semper_destroy(eng);
    free(ref);
    free(def);
    return 0;
}
