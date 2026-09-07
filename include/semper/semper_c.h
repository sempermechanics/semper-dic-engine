#ifndef SEMPER_C_H
#define SEMPER_C_H

/*
 * Semper DIC engine — stable C ABI.
 *
 * The C++ API (include/semper/pipeline.hpp) is source-stable but not ABI-stable
 * across compilers/versions. This header is the ABI-stable surface for embedding
 * the engine from other languages and toolchains (the Python bindings use the C++
 * core directly; C/C#/Rust/etc. use this).
 *
 * Contract (see docs/CONTRACT.md): the return codes and the packed
 * output layout below are FROZEN — 8 floats per point, [x,y,u,v,exx,eyy,exy,corr].
 *
 * Threading: an engine handle is NOT internally synchronized; use one handle per
 * thread, or serialize calls to a handle. semper_cancel() is the exception — it is
 * safe to call from another thread while semper_run() is in flight on the handle.
 */

#include <stddef.h>
#include <stdint.h>

/* Default-visibility exports — adapters/c builds with -fvisibility=hidden. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(SEMPER_C_BUILD)
#    define SEMPER_C_API __declspec(dllexport)
#  else
#    define SEMPER_C_API __declspec(dllimport)
#  endif
#else
#  define SEMPER_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine instance. Create with semper_create, free with semper_destroy. */
typedef struct semper_engine semper_engine;

/* Solve inputs — mirrors Semper::pipeline::FullFieldParams (Frozen field meanings). */
typedef struct semper_params {
    int32_t rect_x, rect_y, rect_w, rect_h; /* ROI in pixels */
    int32_t step;                           /* grid spacing */
    int32_t subset_size;                    /* correlation subset (px) */
    int32_t strain_window;                  /* VSG strain window */
    int32_t use_6x6_interpolator;           /* 0/1 interpolation-kernel toggle */
} semper_params;

/* Return codes (Frozen), shared with run_full_field. */
#define SEMPER_ERR_ROI       (-2)  /* invalid ROI */
#define SEMPER_ERR_INIT      (-3)  /* init / argument failure (incl. no reference set) */
/* strain_window too small for step: the VSG plane fit would reject every point.
   Added in v0.3.0; this case previously returned 0 points with a success code. */
#define SEMPER_ERR_STRAIN_WINDOW (-4)
#define SEMPER_ERR_CANCELLED (-99) /* cancelled mid-solve */
/* >= 0 is the number of valid output points written. */

/* Progress callback: percentage in 0..100; `user` is the pointer passed to semper_run. */
typedef void (*semper_progress_cb)(int percentage, void* user);

/* Number of floats per output point in the packed result buffer (Frozen = 8). */
#define SEMPER_FLOATS_PER_POINT 8
/*
 * Preferred metrics buffer length. The slot *layout* is Frozen; this constant is
 * the Additive-only growth point and rises as telemetry is appended (17 -> 23 in
 * v0.3.0, which added slots 19-22). 16 is still the minimum honored, and the
 * engine writes only the prefix the caller's metrics_len allows, so a caller
 * compiled against an older value keeps working without a recompile.
 */
#define SEMPER_METRICS_LEN 23

/* Create / destroy an engine instance. Returns NULL on allocation failure. */
SEMPER_C_API semper_engine* semper_create(void);
SEMPER_C_API void           semper_destroy(semper_engine* eng);

/*
 * Cache the reference image (call once before a batch of runs against it).
 * `img`/`mask` are encoded bytes (PNG/JPEG) or, when expected_w/expected_h > 0, a
 * raw RGBA/ALPHA_8 buffer. `mask`/`mask_len` may be NULL/0 for no ROI mask.
 * Returns 0 on success, SEMPER_ERR_INIT on decode failure.
 */
SEMPER_C_API int semper_set_reference(semper_engine* eng,
                         const uint8_t* img, size_t img_len,
                         const uint8_t* mask, size_t mask_len,
                         int expected_w, int expected_h);

/*
 * Run the full-field solve for one deformed image against the cached reference.
 * `out` receives up to `out_capacity` floats (8 per point; excess points dropped).
 * `metrics` (may be NULL) receives up to `metrics_len` telemetry floats.
 * `cb` (may be NULL) is invoked with progress on the calling thread.
 * Returns the point count (>= 0) or a SEMPER_ERR_* code.
 */
SEMPER_C_API int semper_run(semper_engine* eng,
               const uint8_t* def, size_t def_len,
               const uint8_t* mask, size_t mask_len,
               const semper_params* params,
               float* out, int out_capacity,
               float* metrics, int metrics_len,
               semper_progress_cb cb, void* user);

/* Request cancellation of the run in flight on `eng`. Thread-safe. */
SEMPER_C_API void semper_cancel(semper_engine* eng);

/* Semantic version string, "MAJOR.MINOR.PATCH" (e.g. "0.2.0"). Never NULL. */
SEMPER_C_API const char* semper_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SEMPER_C_H */
