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
/* Nothing could seed the solve (no anchor converged, no global-motion
   lock, and the fallback search failed) — e.g. a frame unrelated to the
   reference. Added in v0.5.0; this case previously returned an undocumented -1. */
#define SEMPER_ERR_SEED      (-5)
#define SEMPER_ERR_CANCELLED (-99) /* cancelled mid-solve */
/* >= 0 is the number of valid output points written. */

/* Progress callback: percentage in 0..100; `user` is the pointer passed to semper_run. */
typedef void (*semper_progress_cb)(int percentage, void* user);

/* Number of floats per output point in the packed result buffer (Frozen = 8). */
#define SEMPER_FLOATS_PER_POINT 8
/*
 * Preferred metrics buffer length. The slot *layout* is Frozen; this constant is
 * the Additive-only growth point and rises as telemetry is appended (17 -> 23 in
 * v0.3.0, which added slots 19-22; 24 in v0.5.0, slot 23 = points dropped by
 * the displacement median test). 16 is still the minimum honored, and the
 * engine writes only the prefix the caller's metrics_len allows, so a caller
 * compiled against an older value keeps working without a recompile.
 */
#define SEMPER_METRICS_LEN 24

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

/* ---------------------------------------------------------------------------
 * Incremental DIC (added in v0.4.0)
 *
 * A session over a frame sequence that moves its reference to the latest
 * solved frame every `update_interval` frames (0 = never: a fixed reference,
 * identical to semper_run per frame). Results are always the TOTAL
 * displacement from the first image, at the first image's grid points, in the
 * same packed layout as semper_run: 8 floats per point
 * [x0, y0, u, v, exx, eyy, exy, corr], strain being Green-Lagrange of the
 * total displacement. A point the solve cannot reach on a frame is followed
 * on estimated motion but not reported until it is measured (and, after a
 * long coast, confirmed) again; one that cannot be followed at all is dropped
 * for good.
 *
 * The metrics array describes the solve against the current reference;
 * frame-0 bookkeeping is in semper_incremental_info. Threading as for
 * semper_engine: one call at a time per handle; semper_incremental_cancel is
 * safe from another thread. A cancelled or failed frame leaves the session as
 * it was, so the same frame can be submitted again.
 * ------------------------------------------------------------------------- */
typedef struct semper_incremental semper_incremental;

/* Per-frame bookkeeping (layout Frozen: six int32, 24 bytes). */
typedef struct semper_incremental_info {
    int32_t reference_frame;   /* frame solved against (0 = first image; frames count from 1) */
    int32_t reference_updated; /* 1 if this frame became the reference for the next one */
    int32_t tracked_points;    /* first-image grid points still followed, reported or not */
    int32_t frame0_points;     /* first-image grid points that passed the ROI/mask scan */
    int32_t taylor_fallbacks;  /* points sampled by first-order expansion near an edge */
    int32_t lost_this_frame;   /* points dropped for good at this frame's reference update */
} semper_incremental_info;

SEMPER_C_API semper_incremental* semper_incremental_create(void);
SEMPER_C_API void                semper_incremental_destroy(semper_incremental* inc);

/*
 * Start a sequence at the first image. `ref`/`mask` are decoded as for
 * semper_set_reference; `params` describes the first image's grid (as for
 * semper_run). Returns 0, SEMPER_ERR_ROI, SEMPER_ERR_INIT (decode failure,
 * update_interval < 0) or SEMPER_ERR_STRAIN_WINDOW.
 */
SEMPER_C_API int semper_incremental_init(semper_incremental* inc,
                         const uint8_t* ref, size_t ref_len,
                         const uint8_t* mask, size_t mask_len,
                         int expected_w, int expected_h,
                         const semper_params* params,
                         int32_t update_interval);

/*
 * Solve the next frame. `out` holds up to `out_capacity` floats; the most a
 * frame can return is (rect_w / step) * (rect_h / step) points. `metrics`,
 * `info` and `cb` may be NULL. Returns the point count (>= 0) or a
 * SEMPER_ERR_* code (SEMPER_ERR_INIT also before a successful init).
 */
SEMPER_C_API int semper_incremental_process(semper_incremental* inc,
                         const uint8_t* def, size_t def_len,
                         float* out, int out_capacity,
                         float* metrics, int metrics_len,
                         semper_incremental_info* info,
                         semper_progress_cb cb, void* user);

/* Request cancellation of the frame in flight on `inc`. Thread-safe. */
SEMPER_C_API void semper_incremental_cancel(semper_incremental* inc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SEMPER_C_H */
