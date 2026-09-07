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
_Static_assert(SEMPER_METRICS_LEN == 23, "metrics length grew to 23 in v0.3.0");
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
    test_strain_window_guard(ref, def);
    test_capacity_rule(ref, def);
    test_metrics_len_rule(ref, def);
    test_cancel_independence(ref, def);
    test_version_format();

    free(ref);
    free(def);

    if (g_failures == 0) {
        printf("OK — C ABI contract passed\n");
        return 0;
    }
    fprintf(stderr, "FAIL — %d contract assertion(s) failed\n", g_failures);
    return 1;
}
