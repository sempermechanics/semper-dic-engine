/*
 * C ABI contract — the Frozen surface of include/semper/semper_c.h, asserted
 * without depending on DIC solve quality. Where smoke.c proves the happy path
 * runs, this pins the error codes, argument guards, capacity/metrics rules, the
 * struct layout, and the version format that downstream C/C#/Rust callers rely
 * on. Returns non-zero from main() on any failure (same convention as smoke.c).
 *
 * See docs/CONTRACT.md §A.2 / §A.4 for the normative statements behind each check.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <semper/semper_c.h>

/* ---- Compile-time Frozen shape. A break here is a build failure, not a
 *      silent runtime mis-read in a caller that was not recompiled. ---- */
_Static_assert(SEMPER_FLOATS_PER_POINT == 8, "packed point stride is Frozen at 8");
_Static_assert(SEMPER_METRICS_LEN == 24, "metrics length grew to 23 in v0.3.0 and 24 in v0.5.0");
_Static_assert(SEMPER_ERR_SEED == -5, "SEMPER_ERR_SEED is Frozen at -5 (v0.5.0)");
_Static_assert(SEMPER_ERR_ROI == -2, "SEMPER_ERR_ROI is Frozen at -2");
_Static_assert(SEMPER_ERR_INIT == -3, "SEMPER_ERR_INIT is Frozen at -3");
_Static_assert(SEMPER_ERR_CANCELLED == -99, "SEMPER_ERR_CANCELLED is Frozen at -99");
_Static_assert(SEMPER_ERR_STRAIN_WINDOW == -4, "SEMPER_ERR_STRAIN_WINDOW is Frozen at -4");

/* semper_params layout is Frozen: reordering/inserting a field silently
 * mis-reads every caller's parameters. Pin size and every field offset. */
_Static_assert(sizeof(semper_params) == 8 * sizeof(int32_t), "semper_params size is Frozen");
_Static_assert(offsetof(semper_params, rect_x) == 0, "rect_x offset Frozen");
_Static_assert(offsetof(semper_params, rect_y) == 4, "rect_y offset Frozen");
_Static_assert(offsetof(semper_params, rect_w) == 8, "rect_w offset Frozen");
_Static_assert(offsetof(semper_params, rect_h) == 12, "rect_h offset Frozen");
_Static_assert(offsetof(semper_params, step) == 16, "step offset Frozen");
_Static_assert(offsetof(semper_params, subset_size) == 20, "subset_size offset Frozen");
_Static_assert(offsetof(semper_params, strain_window) == 24, "strain_window offset Frozen");
_Static_assert(offsetof(semper_params, use_6x6_interpolator) == 28, "use_6x6 offset Frozen");

/* semper_incremental_info layout is Frozen from v0.4.0. */
_Static_assert(sizeof(semper_incremental_info) == 6 * sizeof(int32_t), "semper_incremental_info size is Frozen");
_Static_assert(offsetof(semper_incremental_info, reference_frame) == 0, "reference_frame offset Frozen");
_Static_assert(offsetof(semper_incremental_info, reference_updated) == 4, "reference_updated offset Frozen");
_Static_assert(offsetof(semper_incremental_info, tracked_points) == 8, "tracked_points offset Frozen");
_Static_assert(offsetof(semper_incremental_info, frame0_points) == 12, "frame0_points offset Frozen");
_Static_assert(offsetof(semper_incremental_info, taylor_fallbacks) == 16, "taylor_fallbacks offset Frozen");
_Static_assert(offsetof(semper_incremental_info, lost_this_frame) == 20, "lost_this_frame offset Frozen");

#define W 128
#define H 128

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "  FAILED %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

/* A gentle gradient — enough for a valid reference; quality is not the point. */
static void fill_gradient(uint8_t* img) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img[y * W + x] = (uint8_t)((x * 2 + y * 3) & 0xFF);
}

static semper_params valid_params(void) {
    semper_params p;
    p.rect_x = 0;
    p.rect_y = 0;
    p.rect_w = 112;
    p.rect_h = 112;
    p.step = 16;
    p.subset_size = 21;
    /* >= 2 * step, or the VSG window holds one grid node and semper_run
       returns SEMPER_ERR_STRAIN_WINDOW. At the old value of 5 this whole
       suite ran against an empty field. */
    p.strain_window = 33;
    p.use_6x6_interpolator = 0;
    return p;
}

/* ---- Argument guards on semper_run: all null cases → SEMPER_ERR_INIT ---- */
static void test_run_null_guards(uint8_t* def) {
    semper_params p = valid_params();
    float out[8 * 4];
    float metrics[SEMPER_METRICS_LEN];
    semper_engine* eng = semper_create();
    CHECK(eng != NULL);

    /* No reference set yet → INIT regardless of otherwise-valid args. */
    CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out, 8 * 4,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);

    CHECK(semper_run(NULL, def, (size_t)W * H, NULL, 0, &p, out, 8 * 4,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);
    CHECK(semper_run(eng, NULL, (size_t)W * H, NULL, 0, &p, out, 8 * 4,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);
    CHECK(semper_run(eng, def, 0, NULL, 0, &p, out, 8 * 4,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);
    CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, NULL, out, 8 * 4,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);
    CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, &p, NULL, 8 * 4,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);

    semper_destroy(eng);
}

static void test_set_reference_guards(uint8_t* ref) {
    semper_engine* eng = semper_create();
    CHECK(eng != NULL);
    CHECK(semper_set_reference(eng, NULL, 0, NULL, 0, W, H) == SEMPER_ERR_INIT);
    CHECK(semper_set_reference(NULL, ref, (size_t)W * H, NULL, 0, W, H) == SEMPER_ERR_INIT);
    /* Null mask is legal. */
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    semper_destroy(eng);
}

static void test_degenerate_roi(uint8_t* ref, uint8_t* def) {
    semper_engine* eng = semper_create();
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    semper_params p = valid_params();
    p.rect_w = p.step - 1;   /* gridW == 0 → ROI error */
    p.rect_h = p.step - 1;
    float out[8 * 4];
    float metrics[SEMPER_METRICS_LEN];
    int rc = semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out, 8 * 4,
                        metrics, SEMPER_METRICS_LEN, NULL, NULL);
    CHECK(rc == SEMPER_ERR_ROI);
    semper_destroy(eng);
}

/* ---- Parameter validation (v0.5.0): odd subset >= 3, rect clipped to the image ---- */
static void test_param_validation(uint8_t* ref, uint8_t* def) {
    semper_engine* eng = semper_create();
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    float out[8 * 64];
    float metrics[SEMPER_METRICS_LEN];

    /* An even or zero subset wrote past the subset buffers (heap overflow). */
    const int bad_subsets[] = {20, 0, -21};
    for (size_t i = 0; i < sizeof(bad_subsets) / sizeof(bad_subsets[0]); ++i) {
        semper_params p = valid_params();
        p.subset_size = bad_subsets[i];
        CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out, 8 * 64,
                         metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);
    }

    /* A rect far larger than the image used to size the grid from the rect and
       throw std::bad_alloc through this C entry point (process abort). It is
       clipped to the image now. */
    semper_params huge = valid_params();
    huge.rect_w = 0x7fffffff;
    huge.rect_h = 0x7fffffff;
    CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, &huge, out, 8 * 64,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) >= 0);

    /* A strain window far beyond the grid used to spin in the VSG window
       count for hours (INT_MAX: ~4e18 iterations). It is an argument error
       past 2 * (min(gridW, gridH) + 1) * step (v0.5.0). */
    semper_params wide = valid_params();
    wide.strain_window = 0x7fffffff;
    CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, &wide, out, 8 * 64,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_INIT);
    wide.strain_window = 2 * (valid_params().rect_w / valid_params().step + 1) * valid_params().step;
    CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, &wide, out, 8 * 64,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) >= 0);

    /* Nothing of the rect on the image: invalid ROI. */
    semper_params off = valid_params();
    off.rect_x = W + 1;
    CHECK(semper_run(eng, def, (size_t)W * H, NULL, 0, &off, out, 8 * 64,
                     metrics, SEMPER_METRICS_LEN, NULL, NULL) == SEMPER_ERR_ROI);

    semper_incremental* inc = semper_incremental_create();
    semper_params even = valid_params();
    even.subset_size = 20;
    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, NULL, 0, W, H, &even, 1) == SEMPER_ERR_INIT);
    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, NULL, 0, W, H, &huge, 1) == 0);
    semper_incremental_destroy(inc);

    semper_destroy(eng);
}

/* ---- strain_window too small for step -> SEMPER_ERR_STRAIN_WINDOW (v0.3.0) ---- */
static void test_strain_window_guard(uint8_t* ref, uint8_t* def) {
    semper_engine* eng = semper_create();
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    semper_params p = valid_params();
    p.strain_window = p.step;   /* one grid node in the VSG window */
    float out[8 * 4];
    float metrics[SEMPER_METRICS_LEN];
    int rc = semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out, 8 * 4,
                        metrics, SEMPER_METRICS_LEN, NULL, NULL);
    CHECK(rc == SEMPER_ERR_STRAIN_WINDOW);
    semper_destroy(eng);
}

/* ---- Capacity-drop rule (Frozen): never overflow; excess points dropped ---- */
static void test_capacity_rule(uint8_t* ref, uint8_t* def) {
    semper_engine* eng = semper_create();
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    semper_params p = valid_params();
    float metrics[SEMPER_METRICS_LEN];

    /* Room for 4 packed points → 0..4 returned, never more, never negative. */
    float out4[8 * 4];
    int n = semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out4, 8 * 4,
                       metrics, SEMPER_METRICS_LEN, NULL, NULL);
    CHECK(n >= 0 && n <= 4);

    /* Capacity 0 is a legitimate metrics-only query: 0 points, metrics filled. */
    float out0[1];
    memset(metrics, 0, sizeof(metrics));
    n = semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out0, 0,
                   metrics, SEMPER_METRICS_LEN, NULL, NULL);
    CHECK(n == 0);
    CHECK(metrics[0] >= 1.0f);   /* attempted > 0 — the solve still ran */

    /* Capacity too small for even one point returns 0 (not an error) — this is
     * the documented capacity-drop rule, not a rejection. */
    float out_tiny[4];
    n = semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out_tiny, 4,
                   metrics, SEMPER_METRICS_LEN, NULL, NULL);
    CHECK(n == 0);

    semper_destroy(eng);
}

/* ---- metrics_len rule (Frozen): >= 16 to fill; below that, nothing written ---- */
static void test_metrics_len_rule(uint8_t* ref, uint8_t* def) {
    semper_engine* eng = semper_create();
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    semper_params p = valid_params();
    float out[8 * 64];
    const float S = -777.0f;   /* sentinel: "not written" */

    /* metrics_len == 15: below the 16 minimum → buffer left untouched. */
    float m15[SEMPER_METRICS_LEN];
    for (int i = 0; i < SEMPER_METRICS_LEN; ++i) m15[i] = S;
    int n = semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out, 8 * 64,
                       m15, 15, NULL, NULL);
    CHECK(n >= 0);
    int untouched = 1;
    for (int i = 0; i < SEMPER_METRICS_LEN; ++i) if (m15[i] != S) untouched = 0;
    CHECK(untouched);

    /* metrics_len == 16: fills exactly 16 slots; slot 16 stays the sentinel. */
    float m16[SEMPER_METRICS_LEN];
    for (int i = 0; i < SEMPER_METRICS_LEN; ++i) m16[i] = S;
    n = semper_run(eng, def, (size_t)W * H, NULL, 0, &p, out, 8 * 64,
                   m16, 16, NULL, NULL);
    CHECK(n >= 0);
    CHECK(m16[0] != S);     /* slot 0 (attempted) written */
    CHECK(m16[15] != S);    /* slot 15 (convergence %) written */
    CHECK(m16[16] == S);    /* slot 16 NOT written when metrics_len == 16 */

    semper_destroy(eng);
}

/* ---- Per-engine cancel independence + stale-cancel clearing ---- */
static void test_cancel_independence(uint8_t* ref, uint8_t* def) {
    semper_engine* a = semper_create();
    semper_engine* b = semper_create();
    CHECK(semper_set_reference(a, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    CHECK(semper_set_reference(b, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    semper_params p = valid_params();
    float out[8 * 64];
    float metrics[SEMPER_METRICS_LEN];

    semper_cancel(a);   /* cancel engine a only */

    /* A solve on b must be unaffected by a's cancel. */
    int nb = semper_run(b, def, (size_t)W * H, NULL, 0, &p, out, 8 * 64,
                        metrics, SEMPER_METRICS_LEN, NULL, NULL);
    CHECK(nb != SEMPER_ERR_CANCELLED);
    CHECK(nb >= 0);

    /* And a's own next solve clears the stale cancel on entry — no instant -99. */
    int na = semper_run(a, def, (size_t)W * H, NULL, 0, &p, out, 8 * 64,
                        metrics, SEMPER_METRICS_LEN, NULL, NULL);
    CHECK(na != SEMPER_ERR_CANCELLED);
    CHECK(na >= 0);

    semper_destroy(a);
    semper_destroy(b);
}

/* ---- semper_version() is non-NULL and MAJOR.MINOR.PATCH ---- */
static void test_version_format(void) {
    const char* v = semper_version();
    CHECK(v != NULL);
    if (v == NULL) return;
    int dots = 0, digits = 0, other = 0;
    for (const char* c = v; *c; ++c) {
        if (*c == '.') dots++;
        else if (*c >= '0' && *c <= '9') digits++;
        else other++;
    }
    CHECK(dots == 2);       /* MAJOR.MINOR.PATCH */
    CHECK(digits >= 3);     /* at least one digit per component */
    CHECK(other == 0);      /* nothing but digits and dots */
}

/* ---- Incremental session: guards and error codes (no solve quality) ---- */
static void test_incremental_guards(uint8_t* ref, uint8_t* def) {
    semper_params p = valid_params();
    float out[8 * 64];
    float metrics[SEMPER_METRICS_LEN];
    semper_incremental_info info;

    /* NULL handles are ignored / rejected, never dereferenced. */
    semper_incremental_destroy(NULL);
    semper_incremental_cancel(NULL);
    CHECK(semper_incremental_init(NULL, ref, (size_t)W * H, NULL, 0, W, H, &p, 1) == SEMPER_ERR_INIT);
    CHECK(semper_incremental_process(NULL, def, (size_t)W * H, out, 8 * 64, metrics,
                                     SEMPER_METRICS_LEN, &info, NULL, NULL) == SEMPER_ERR_INIT);

    semper_incremental* inc = semper_incremental_create();
    CHECK(inc != NULL);

    /* Before init. */
    CHECK(semper_incremental_process(inc, def, (size_t)W * H, out, 8 * 64, metrics,
                                     SEMPER_METRICS_LEN, &info, NULL, NULL) == SEMPER_ERR_INIT);

    /* Init argument guards and codes. */
    CHECK(semper_incremental_init(inc, NULL, 0, NULL, 0, W, H, &p, 1) == SEMPER_ERR_INIT);
    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, NULL, 0, W, H, NULL, 1) == SEMPER_ERR_INIT);
    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, NULL, 0, W, H, &p, -1) == SEMPER_ERR_INIT);
    semper_params bad = p;
    bad.rect_w = bad.step - 1;
    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, NULL, 0, W, H, &bad, 1) == SEMPER_ERR_ROI);
    bad = p;
    bad.strain_window = 5;
    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, NULL, 0, W, H, &bad, 1) == SEMPER_ERR_STRAIN_WINDOW);
    /* A failed init leaves the handle uninitialized. */
    CHECK(semper_incremental_process(inc, def, (size_t)W * H, out, 8 * 64, metrics,
                                     SEMPER_METRICS_LEN, &info, NULL, NULL) == SEMPER_ERR_INIT);

    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, NULL, 0, W, H, &p, 1) == 0);
    /* Process argument guards. */
    CHECK(semper_incremental_process(inc, NULL, 0, out, 8 * 64, metrics,
                                     SEMPER_METRICS_LEN, &info, NULL, NULL) == SEMPER_ERR_INIT);
    CHECK(semper_incremental_process(inc, def, (size_t)W * H, NULL, 0, metrics,
                                     SEMPER_METRICS_LEN, &info, NULL, NULL) == SEMPER_ERR_INIT);
    /* metrics and info are optional. */
    int rc = semper_incremental_process(inc, def, (size_t)W * H, out, 8 * 64, NULL, 0, NULL, NULL, NULL);
    CHECK(rc >= 0 || rc == SEMPER_ERR_ROI);
    semper_incremental_destroy(inc);
}

/* Mask bytes that do not decode are SEMPER_ERR_INIT on every entry point, not
 * "no mask": the solve used to run unmasked and report success. 37 bytes are
 * neither W*H nor W*H*4 raw pixels, nor an image file. */
static void test_undecodable_mask(uint8_t* ref, uint8_t* def) {
    uint8_t junk[37];
    memset(junk, 0xAB, sizeof(junk));
    semper_params p = valid_params();
    float out[8 * 64];

    semper_engine* eng = semper_create();
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, junk, sizeof(junk), W, H) == SEMPER_ERR_INIT);
    CHECK(semper_set_reference(eng, ref, (size_t)W * H, NULL, 0, W, H) == 0);
    CHECK(semper_run(eng, def, (size_t)W * H, junk, sizeof(junk), &p, out, 8 * 64,
                     NULL, 0, NULL, NULL) == SEMPER_ERR_INIT);
    semper_destroy(eng);

    semper_incremental* inc = semper_incremental_create();
    CHECK(semper_incremental_init(inc, ref, (size_t)W * H, junk, sizeof(junk), W, H, &p, 1) == SEMPER_ERR_INIT);
    semper_incremental_destroy(inc);
}

int main(void) {
    printf("semper_c contract — version %s\n", semper_version());

    uint8_t* ref = (uint8_t*)malloc((size_t)W * H);
    uint8_t* def = (uint8_t*)malloc((size_t)W * H);
    if (!ref || !def) { fprintf(stderr, "alloc failed\n"); return 2; }
    fill_gradient(ref);
    fill_gradient(def);   /* identical is fine; we assert codes, not displacement */

    test_run_null_guards(def);
    test_set_reference_guards(ref);
    test_degenerate_roi(ref, def);
    test_param_validation(ref, def);
    test_strain_window_guard(ref, def);
    test_capacity_rule(ref, def);
    test_metrics_len_rule(ref, def);
    test_cancel_independence(ref, def);
    test_version_format();
    test_incremental_guards(ref, def);
    test_undecodable_mask(ref, def);

    free(ref);
    free(def);

    if (g_failures == 0) {
        printf("OK — C ABI contract passed\n");
        return 0;
    }
    fprintf(stderr, "FAIL — %d contract assertion(s) failed\n", g_failures);
    return 1;
}
