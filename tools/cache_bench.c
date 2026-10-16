/* Fixed-reference cost per frame, before and after the reference cache.
 *
 * Loads a libsemper_c build by path, so the same binary times two engine
 * versions on identical inputs:
 *
 *   cc -O2 -o cache_bench tools/cache_bench.c -ldl -lm
 *   ./cache_bench old/libsemper_c.so tests/fixtures/dice
 *   ./cache_bench new/libsemper_c.so tests/fixtures/dice
 *
 * Two workloads, each one reference and a run of deformed frames, as the app
 * runs a batch:
 *   OHT  DICe's oht_cfrp_00 against 01/03/06/11, cycled (400x1040, step 7,
 *        subset 27, Keys 6x6). Encoded TIFFs, decoded by the engine.
 *   12MP a 4000x3000 synthetic speckle with five small affine frames
 *        (3000x2000 ROI, step 10, subset 21). Raw 8-bit pixels.
 * Prints wall time and the engine's own seeding (slot 10) and Hessian
 * pre-pass (slot 11) times for the first frame and the mean of the rest.
 * Linux/macOS (dlopen). Used for docs/INCREMENTAL_BENCHMARK.md.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { int32_t rect_x, rect_y, rect_w, rect_h, step, subset_size, strain_window, use6; } params_t;
typedef void* (*create_f)(void);
typedef void (*destroy_f)(void*);
typedef int (*setref_f)(void*, const uint8_t*, size_t, const uint8_t*, size_t, int, int);
typedef int (*run_f)(void*, const uint8_t*, size_t, const uint8_t*, size_t, const params_t*,
                     float*, int, float*, int, void*, void*);
typedef const char* (*version_f)(void);

static create_f s_create;
static destroy_f s_destroy;
static setref_f s_setref;
static run_f s_run;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint8_t* read_file(const char* dir, const char* name, size_t* len) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* b = malloc(*len);
    if (fread(b, 1, *len, f) != *len) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    return b;
}

typedef struct { double wall, seed, prepass; int rc; } sample_t;

static void report(const char* name, const sample_t* s, int n) {
    double w = 0, sd = 0, pp = 0;
    for (int i = 1; i < n; ++i) { w += s[i].wall; sd += s[i].seed; pp += s[i].prepass; }
    const int m = n - 1;
    printf("  %-5s frame 1: %8.1f ms (seeding %6.1f, pre-pass %6.1f, %d pts)   "
           "frames 2..%d: %8.1f ms (seeding %6.1f, pre-pass %6.1f)\n",
           name, s[0].wall, s[0].seed, s[0].prepass, s[0].rc, n, w / m, sd / m, pp / m);
}

static sample_t solve(void* eng, const uint8_t* def, size_t len, const params_t* p, float* out, int cap) {
    float met[23] = {0};
    sample_t s;
    const double t0 = now_ms();
    s.rc = s_run(eng, def, len, NULL, 0, p, out, cap, met, 23, NULL, NULL);
    s.wall = now_ms() - t0;
    s.seed = met[10];
    s.prepass = met[11];
    return s;
}

/* ---- 12 MP synthetic speckle: Gaussian blobs bucketed on a 16 px grid ---- */
#define SW 4000
#define SH 3000
#define CELL 16
static int ncx, ncy, nblob;
static float *bx, *by, *ba;
static int *cstart, *cidx;
static uint32_t rng = 2024;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) / 16777216.0f; }

static void make_speckle(void) {
    ncx = (SW + 64) / CELL + 1; ncy = (SH + 64) / CELL + 1;
    nblob = (int)(0.02 * (SW + 64) * (SH + 64));
    bx = malloc(sizeof(float) * nblob); by = malloc(sizeof(float) * nblob); ba = malloc(sizeof(float) * nblob);
    int* cell = malloc(sizeof(int) * nblob);
    int* cnt = calloc((size_t)ncx * ncy, sizeof(int));
    for (int i = 0; i < nblob; ++i) {
        bx[i] = frand() * (SW + 64) - 32; by[i] = frand() * (SH + 64) - 32; ba[i] = 0.5f + 0.5f * frand();
        int cx = (int)((bx[i] + 32) / CELL), cy = (int)((by[i] + 32) / CELL);
        cell[i] = cy * ncx + cx; cnt[cell[i]]++;
    }
    cstart = malloc(sizeof(int) * ((size_t)ncx * ncy + 1));
    cstart[0] = 0;
    for (int c = 0; c < ncx * ncy; ++c) cstart[c + 1] = cstart[c] + cnt[c];
    int* fill = malloc(sizeof(int) * (size_t)ncx * ncy);
    memcpy(fill, cstart, sizeof(int) * (size_t)ncx * ncy);
    cidx = malloc(sizeof(int) * nblob);
    for (int i = 0; i < nblob; ++i) cidx[fill[cell[i]]++] = i;
    free(cell); free(cnt); free(fill);
}

static float speckle(float x, float y) {
    const int cx = (int)((x + 32) / CELL), cy = (int)((y + 32) / CELL);
    float s = 0;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const int X = cx + dx, Y = cy + dy;
        if (X < 0 || Y < 0 || X >= ncx || Y >= ncy) continue;
        const int c = Y * ncx + X;
        for (int k = cstart[c]; k < cstart[c + 1]; ++k) {
            const int i = cidx[k];
            const float ddx = x - bx[i], ddy = y - by[i];
            s += ba[i] * expf(-(ddx * ddx + ddy * ddy) / (2 * 2.5f * 2.5f));
        }
    }
    float v = 20 + 180 * s;
    return v > 255 ? 255 : v;
}

/* deformed(x) = ref(F^-1 (x - c - t) + c), F = diag(1 + e, 1 - 0.3 e) */
static void render(uint8_t* img, double e, double tx, double ty) {
    const double sx = 1 + e, sy = 1 - 0.3 * e, cx = SW / 2.0, cy = SH / 2.0;
    for (int y = 0; y < SH; ++y)
        for (int x = 0; x < SW; ++x) {
            const double X = (x - cx - tx) / sx + cx, Y = (y - cy - ty) / sy + cy;
            img[(size_t)y * SW + x] = (uint8_t)lrintf(speckle((float)X, (float)Y));
        }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <libsemper_c.so> <dice fixtures dir> [oht cycles]\n", argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    s_create = (create_f)dlsym(h, "semper_create");
    s_destroy = (destroy_f)dlsym(h, "semper_destroy");
    s_setref = (setref_f)dlsym(h, "semper_set_reference");
    s_run = (run_f)dlsym(h, "semper_run");
    version_f ver = (version_f)dlsym(h, "semper_version");
    const int cycles = argc > 3 ? atoi(argv[3]) : 5;
    printf("cache_bench: engine %s (%s)\n", ver ? ver() : "?", argv[1]);

    /* ---- OHT ---- */
    {
        const char* names[] = {"oht_cfrp_01.tiff", "oht_cfrp_03.tiff", "oht_cfrp_06.tiff", "oht_cfrp_11.tiff"};
        size_t rlen, dlen[4];
        uint8_t* ref = read_file(argv[2], "oht_cfrp_00.tiff", &rlen);
        uint8_t* def[4];
        for (int i = 0; i < 4; ++i) def[i] = read_file(argv[2], names[i], &dlen[i]);
        const params_t p = {47, 47, 308, 938, 7, 27, 21, 1};
        const int cap = (p.rect_w / p.step + 1) * (p.rect_h / p.step + 1) * 8;
        float* out = malloc(sizeof(float) * cap);
        void* eng = s_create();
        if (s_setref(eng, ref, rlen, NULL, 0, 0, 0) != 0) { fprintf(stderr, "OHT set_reference failed\n"); return 1; }
        const int n = 4 * cycles;
        sample_t* s = malloc(sizeof(sample_t) * n);
        for (int i = 0; i < n; ++i) s[i] = solve(eng, def[i % 4], dlen[i % 4], &p, out, cap);
        report("OHT", s, n);
        s_destroy(eng);
    }

    /* ---- 12 MP ---- */
    {
        make_speckle();
        uint8_t* ref = malloc((size_t)SW * SH);
        uint8_t* def = malloc((size_t)SW * SH);
        render(ref, 0, 0, 0);
        const params_t p = {500, 500, 3000, 2000, 10, 21, 21, 0};
        const int cap = (p.rect_w / p.step + 1) * (p.rect_h / p.step + 1) * 8;
        float* out = malloc(sizeof(float) * cap);
        void* eng = s_create();
        if (s_setref(eng, ref, (size_t)SW * SH, NULL, 0, SW, SH) != 0) { fprintf(stderr, "12MP set_reference failed\n"); return 1; }
        sample_t s[5];
        for (int f = 0; f < 5; ++f) {
            render(def, 0.002 * (f + 1), 0.7 * (f + 1), -0.4 * (f + 1));
            s[f] = solve(eng, def, (size_t)SW * SH, &p, out, cap);
        }
        report("12MP", s, 5);
        s_destroy(eng);
    }
    return 0;
}
