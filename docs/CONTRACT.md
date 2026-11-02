# Engine API Contract

This is the API and stability contract of the engine, written for contributors.
A private Android app links the engine through the JNI adapter in
`adapters/android/`; it lives in another repository and cannot be seen from an
engine pull request. **The one rule:** anything marked **Frozen** or **Stable**
is load-bearing for that app, so extend it additively or change it only in a
major release coordinated with the app.

You do not need the app to contribute. Build and test against `tests/` as
usual, respect the surface below, and declare your change's
[tier](#a1-stability-tiers) in the PR. Tests enforce part of this contract
([§A.6](#a6-how-improvements-reach-downstream-apps)); this document is the rest.

*Since X.Y* marks surface or behavior that earlier tags lack. 0.7.0 is the next
release and is not yet tagged. Version history lives in
[CHANGELOG.md](../CHANGELOG.md).

---

## A.1 Stability tiers

| Tier | What it covers | Rule for contributors |
|---|---|---|
| **Frozen** | Binary and data formats: output packing, metrics layout, error codes, the `semper_params` and `semper_incremental_info` layouts, and the *meaning* of existing `FullFieldParams` fields | Never change in place. A change is a **major** bump and needs a coordinated change in the downstream app. |
| **Stable** | Public function **signatures** in `include/semper/{pipeline,incremental,io,cancel,metrics,version}.hpp` and `include/semper/semper_c.h` | Extend additively (new overloads, new functions). Existing signatures, parameter order, defaults and semantics change only with a **major** bump. |
| **Additive-only** | Deliberate growth points: trailing `FullFieldParams` fields, unused metrics slots | Append only, each with a default that reproduces prior behavior. Never reorder or repurpose an existing entry. |
| **Internal** | Everything in `src/`, algorithm internals, private headers, and every other header under `include/semper/` (`assert`, `guarded_call`, `image`, `seeding`, `simd`, `solver`, `strain`, `subset`, `tuning` constants, `types`): living there does not make a header Stable | Change freely. The golden tests guard behavior, not this contract. |

**Semantic versioning** (`include/semper/version.hpp`):

| Bump | When |
|---|---|
| major | Any Frozen or Stable break |
| minor | An additive Stable or Additive-only change, or a new capability |
| patch | An Internal-only improvement (accuracy, speed, robustness) with no surface change |

Downstream apps pin an exact engine tag and adopt only minor and patch bumps
automatically.

**Build-plumbing carve-out.** Export and visibility machinery (the
`SEMPER_C_API` macro, `-fvisibility=hidden`, the `dllexport`/`dllimport`
choice) is build plumbing, not public surface. It changes *how* the documented
symbols are emitted, never *which* symbols exist or their signatures, so such
changes ship as **patch** (0.1.1 → 0.1.4 added `SEMPER_C_API` to every entry
point). The golden symbol list (§A.6) guards the exported set against drift.

---

## A.2 Public API reference (the app-facing surface)

The Android app reaches the engine only through the JNI adapter, which calls
the C++ API below (§A.3 maps each export). The C ABI serves other consumers.
Keep these signatures in sync with the headers; they are authoritative.

### `pipeline::run_full_field` — the solve · *Stable; behavior Frozen via golden test*

```cpp
// include/semper/pipeline.hpp
int run_full_field(ReferenceCache& cache,
                   const cv::Mat& def_gray,
                   const cv::Mat& roi_mask,
                   const FullFieldParams& params,
                   float* output_ptr, int output_capacity,
                   float* metrics, int metrics_len,
                   ProgressCallback on_progress = nullptr);

// Additive overload: per-solve cancel for the SDK and Python.
int run_full_field(..., CancelToken& cancel, ProgressCallback on_progress = nullptr);
```

**Returns** *(Frozen)*: the number of valid output points (`>= 0`), or a
negative code from the table in [§A.4](#a4-frozen-data-formats).

**Input checks.** The call rejects bad inputs before it solves:

| Input | Result |
|---|---|
| No cached reference, `output_ptr == nullptr`, or empty `def_gray` | `-3` |
| `def_gray` not `CV_8UC1`, or not the size of the cached reference | `-3` |
| Non-empty `roi_mask` not `CV_8UC1` (any size is fine; it is resized) | `-3` |
| `step <= 0` (it would divide by zero forming the grid; this clarifies the meaning of `-2`, it is not a new code) | `-2` |
| `subset_size` even or below 3 | `-3` |
| `rect_w` or `rect_h` not positive; a rect with nothing on the image, or narrower than one step once clipped | `-2` |
| `strain_window` above `2 * (min(gridW, gridH) + 1) * step` | `-3` |
| `strain_window` too small for `step` (§A.4) | `-4` |
| No grid node passes the boundary and mask scan | `-2` |

- The rect is clipped to the reference image before the grid is formed. The
  near edge moves by whole steps, so every node keeps its position
  `rect_x + i * step`.
- In the strain-window bound, `gridW = rect_w / step` and
  `gridH = rect_h / step` of the clipped rect. Above it, the VSG circle reaches
  more than a step beyond the grid's short side from every node. No node can
  then meet the strain fit's 90% window-coverage rule, and the field would be
  empty. This is a rejection, not a clamp: a clamped window would solve a
  different strain fit.
- A negative `output_capacity` holds nothing: no point is written. (The C ABI
  returns `SEMPER_ERR_INIT` instead.)

**Writes** *(Frozen)*: at most `output_capacity` floats, **8 per point**
([§A.4](#a4-frozen-data-formats)). Points beyond capacity are **dropped, never
overflow**.

**Metrics** *(Frozen layout)*: when `metrics != nullptr && metrics_len >= 16`,
the call fills the `metrics_prefix_len(metrics_len)` leading slots.
`kMetricsLen` (24) is preferred (`include/semper/metrics.hpp`). Slot indices are
Frozen.

**Exceptions** never cross the C ABI or JNI: each entry point catches what
escapes and returns `-3` (`NULL` from the C constructors). The C++ functions
may still throw `std::bad_alloc` and OpenCV exceptions.

**Cancellation**: the legacy overload polls the process-wide token; the
`CancelToken&` overload polls only its own. See *Cancellation* below.

**You may** make it faster or more accurate (Internal). **You may not** change
return-code meanings, the 8-float packing, the capacity-drop rule, or metric
slot indices without a major bump.

### `pipeline::ReferenceCache` — cached reference state · *Stable*

```cpp
struct ReferenceCache {                 // non-copyable, non-movable
    ReferenceCache();
    ~ReferenceCache();
    void reset();
    void set_from_gray(const cv::Mat& gray_in, const cv::Mat& roi_mask);
    void adopt(std::unique_ptr<Image> prepared, cv::Mat gray_in,
               const cv::Mat& roi_mask);  // Internal, since 0.4.0
    std::string debug_dir;              // "" disables debug export
    // other members are Internal
};
```

- It must remain **non-copyable and non-movable**.
- `set_from_gray` with an image or mask that is not `CV_8UC1` leaves the cache
  empty, so the next solve returns `-3`.
- `set_from_gray` and `adopt` install a reference only once it is complete
  *(since 0.7.0)*. If one throws, the cache is left empty (`set_from_gray`) or
  as it was (`adopt`).
- `adopt` hands an already-prepared image to the cache. It exists for
  `IncrementalSession` and is **Internal** despite living in a Stable header.
- Each solve reads `debug_dir` and clears it, so a value arms one solve.

**Derived-state cache** *(Internal)*. The cache also keeps work that depends
only on the reference and the grid: the node validity scan, the Hessian pool
and the reference's AKAZE keypoints.

- It is reused while `rect`, `step`, `subset_size` and the ROI mask are
  unchanged.
- `reset`, `set_from_gray` and `adopt` drop it. `adopt` keeps one item: the
  AKAZE detection scale the outgoing reference settled on, so incremental DIC
  does not climb the scale pyramid again at every update.
- The scan, the Hessian pool and the keypoints give bit-identical outputs with
  or without the cache.
- The AKAZE starting scale does not. A solve starts its global-motion search at
  the scale an earlier completed solve against the same reference settled on.
  It can take its model from a finer scale than a cold solve would, so its
  field can differ from a cold solve's in the last bits (the seeds differ).
- A cancelled or failed solve leaves that scale as it was *(since 0.7.0)*.

### `pipeline::IncrementalSession` — updated-reference sequences · *Stable (added v0.4.0)*

```cpp
// include/semper/incremental.hpp
struct IncrementalParams {
    FullFieldParams field;     // the frame-0 grid, exactly as for run_full_field
    int update_interval = 1;   // 0 = fixed reference, N = move the reference every N frames
                               // (sooner when a frame reports 5% fewer points, N > 1;
                               // never onto a frame reporting under half, any N)
};

struct IncrementalFrameInfo {
    int  reference_frame;      // frame this one was solved against (0 = first image)
    bool reference_updated;    // this frame became the next reference
    int  tracked_points;       // frame-0 nodes still followed, reported or not
    int  frame0_points;        // frame-0 nodes after the boundary + mask scan
    int  taylor_fallbacks;     // nodes sampled by first-order expansion
    int  lost_this_frame;      // nodes that stopped being followed on this frame
};

class IncrementalSession {     // move-only
public:
    int  init(const cv::Mat& ref_gray, const cv::Mat& roi_mask, const IncrementalParams&);
    int  process_frame(const cv::Mat& def_gray, float* output, int output_capacity,
                       float* metrics, int metrics_len,
                       ProgressCallback on_progress = nullptr,
                       IncrementalFrameInfo* info = nullptr);
    int  process_frame(..., CancelToken& cancel, ProgressCallback = nullptr,
                       IncrementalFrameInfo* = nullptr);
    void reset();
    bool initialized() const;
    int  frames_processed() const;
    int  max_points() const;   // frame-0 grid node count
    void set_debug_dir(std::string dir);
};
```

- **Output** *(Frozen)*: the same 8-float packing and capacity rule as
  `run_full_field`. `x, y` are **frame-0 grid coordinates**, `u, v` the
  **total** displacement from the first image, and the strains are
  Green-Lagrange strain of that total.
- **Reported nodes**: only measured nodes are reported. A node the solve cannot
  reach on a frame (a gap it cannot correlate across) is followed on estimated
  motion, unreported, and is reported again once it is measured. After a long
  coast, it is reported again only once its own earlier subset confirms where
  it is ([MATHEMATICS.md](MATHEMATICS.md) §9.5). A frame-0 node that cannot be
  followed at all is dropped and never returns.
- **`update_interval = 0`** *(Frozen)*: every frame is bit-identical to
  `run_full_field` against the first image. With any interval, the first frame
  is too.
- **Returns** *(Frozen)*: `init` returns `0`, `-2`, `-3` or `-4`. `-3` covers
  an empty image, an image or mask that is not `CV_8UC1`, a negative interval,
  a `subset_size` that is not odd and at least 3, and a strain window above the
  upper bound (on the frame-0 grid). The rect is validated and clipped exactly
  as `run_full_field` does it. `process_frame` returns the same codes as
  `run_full_field`; `-3` also covers "not initialized" and a frame whose size
  differs from the first image.
- **Errors and cancellation** leave the session exactly as it was, so the same
  frame can be submitted again. That includes an exception thrown out of
  `process_frame` *(since 0.7.0)*: a reference update builds its next state in
  full before installing any of it. A failed frame is not counted, so after one
  the session's frame indices run one behind the caller's.
- **Update floor** *(since 0.7.0)*: the reference never moves to a frame that
  reports under half of a base count. The reference stays, and an update that
  was due moves to the next frame. The base count is the first of these that
  exists:
  1. the count of the first frame on the current reference (N > 1 only);
  2. what the reference reported when it was solved;
  3. while the reference is still the first image, its node count after the
     boundary and mask scan (`frame0_points`).
- **Nothing left to follow** *(since 0.7.0)*: when no node of the current
  reference's grid can be solved (the tracked material has moved into the
  border margin), the nodes are dropped. The frame returns 0 points with zeroed
  metrics, as when every node has been lost.
- **Metrics** describe the increment solve against the current reference. The
  frame-0 bookkeeping is in `IncrementalFrameInfo`, not in new metric slots.

### `pipeline::FullFieldParams` — solve inputs · *Frozen fields + Additive-only tail*

```cpp
struct FullFieldParams {
    int rect_x = 0, rect_y = 0, rect_w = 0, rect_h = 0;
    int step = 0;
    int subset_size = 0;
    int strain_window = 0;
    bool use_6x6_interpolator = false;
};
```

The meaning of each existing field is Frozen. New tunables must be
**appended**, each with a default that reproduces current behavior.

### `pipeline::ProgressCallback` · *Stable*

```cpp
using ProgressCallback = std::function<void(int percentage)>;   // 0..100
```

- Intermediate ticks come from a thread the solve owns, not the caller's.
- The final `100` is sent only when the solve succeeds. It comes on the calling
  thread after that ticker has been joined, so calls never overlap.
- The same holds for the C ABI's `semper_progress_cb`.
- A callback that throws is logged and ignored.

### Cancellation — `include/semper/cancel.hpp` · *Stable*

```cpp
constexpr int kCancelled       = -99;   // Frozen value
constexpr int kBadStrainWindow = -4;    // Frozen value
constexpr int kSeedFailed      = -5;    // Frozen value, since 0.5.0

void request_cancel();
void clear_cancel();
bool cancel_requested();

class CancelToken {                     // per-instance cancel (SDK / Python); non-copyable, non-movable
public:
    void request() noexcept;  void clear() noexcept;  bool requested() const noexcept;
    void begin() noexcept;    void end() noexcept;    void enter() noexcept;   // since 0.7.0
};

class CancelScope {                     // since 0.7.0: begin() on construction, end() on exit
public:
    explicit CancelScope(CancelToken& token) noexcept;
};
```

- **The free functions stay**: JNI `setCancelRequested` maps to them. They act
  on one process-wide token, the one every solve started without a token polls.
- **Per-token solves**: the `CancelToken&` overloads clear and poll only that
  token. Solves with their own tokens can run concurrently and cancel
  independently, and `request_cancel()` does not reach them.
- **Polling**: a solve polls its token inside its point loops, so a cancel
  takes effect within a point or two.
- **Entry**: a solve clears its token on entry, so a stale cancel never aborts
  the next solve.
- **Cancel window** *(since 0.7.0)*: a caller that does work before the solve
  (every adapter decodes the frame first) opens the window at its own entry
  with `CancelScope`. The solve then keeps a request made since, instead of
  clearing it on entry.
- **A seen request stays in force** *(since 0.7.0)*: clearing the flag
  mid-solve (JNI `setCancelRequested(false)`) prepares the next solve. It does
  not un-cancel this one, which returns `-99`.
- `detail::global_cancel_token()` is Internal. `detail::ScopedCancelToken` was
  removed in 0.5.0.

### Image I/O — `include/semper/io.hpp` · *Stable*

```cpp
cv::Mat decode_gray(const uint8_t* data, size_t len,
                    int expected_width = 0, int expected_height = 0);
cv::Mat decode_bgr (const uint8_t* data, size_t len);
void    image_dimensions(const uint8_t* data, size_t len, int& out_w, int& out_h);
cv::Mat to_gray8(const cv::Mat& img);                       // since 0.5.0
cv::Mat to_mask8(const cv::Mat& mask);                      // since 0.7.0
cv::Mat decode_mask(const uint8_t* data, size_t len,
                    int expected_width = 0, int expected_height = 0);  // since 0.7.0
```

An empty `cv::Mat` means failure (`image_dimensions` returns zeros). Do not
switch to throwing.

- **Raw or encoded**: with `expected_width/height > 0`, a buffer of exactly
  `w*h*4` (RGBA) or `w*h` (ALPHA_8) bytes is raw, unless it starts with a PNG,
  JPEG, TIFF, BMP, WebP or PNM signature and decodes as one.
- **Depth**: an encoded image of any depth other than 8-bit is mapped linearly
  onto 0..255 (`to_gray8`). The range mapped is the image's 0.1th to 99.9th
  percentile *(since 0.7.0; min..max before)*, so a hot pixel does not set the
  scale; values outside it saturate. A flat image, or one with an infinite
  value, becomes 128. ZNSSD is invariant to that affine map, so only
  quantization changes for frames that already decoded correctly.
- **Masks** *(since 0.7.0)*: the engine excludes a pixel whose 8-bit mask
  value is below 128. An 8-bit mask passes through unchanged (0 out, 255 in).
  A mask of any other depth (16-bit, signed, float, a NumPy bool array) is
  boolean: nonzero is included. The C ABI and JNI decode masks with
  `decode_mask`. Python treats bool and non-uint8 arrays as boolean, and a
  uint8 mask holding only 0 and 1 as boolean too.
- **Size** *(since 0.7.0)*: an image of more than 2^28 pixels is refused (empty
  `Mat`). Where the header gives the size (PNG, JPEG, BMP, TIFF), it is refused
  before decoding: the engine spends about 13 bytes a pixel, and a few
  kilobytes of compressed input could declare enough pixels to exhaust memory.
- **Dimensions** *(since 0.7.0)*: `image_dimensions` reads PNG, JPEG, BMP and
  TIFF headers without decoding; other formats are decoded, in gray.

### C ABI — `include/semper/semper_c.h` · *Stable (ABI)*

The C ABI has the same Frozen return codes and 8-float packing as the C++ API.

```c
typedef struct semper_engine semper_engine;                /* opaque */
typedef struct semper_params {                             /* Frozen: eight int32, 32 bytes */
    int32_t rect_x, rect_y, rect_w, rect_h;
    int32_t step, subset_size, strain_window;
    int32_t use_6x6_interpolator;                          /* 0/1 */
} semper_params;
typedef void (*semper_progress_cb)(int percentage, void* user);

semper_engine* semper_create(void);                        /* NULL on failure */
void semper_destroy(semper_engine*);
int  semper_set_reference(semper_engine*, const uint8_t* img, size_t img_len,
                          const uint8_t* mask, size_t mask_len,
                          int expected_w, int expected_h);
int  semper_run(semper_engine*, const uint8_t* def, size_t def_len,
                const uint8_t* mask, size_t mask_len, const semper_params*,
                float* out, int out_capacity, float* metrics, int metrics_len,
                semper_progress_cb cb, void* user);
void semper_cancel(semper_engine*);                        /* thread-safe */
const char* semper_version(void);                          /* "MAJOR.MINOR.PATCH", never NULL */
```

The incremental session (since 0.4.0):

```c
typedef struct semper_incremental semper_incremental;      /* opaque */
typedef struct semper_incremental_info {                    /* Frozen: six int32, 24 bytes */
    int32_t reference_frame, reference_updated, tracked_points,
            frame0_points, taylor_fallbacks, lost_this_frame;
} semper_incremental_info;

semper_incremental* semper_incremental_create(void);
void semper_incremental_destroy(semper_incremental*);
int  semper_incremental_init(semper_incremental*, const uint8_t* ref, size_t ref_len,
                             const uint8_t* mask, size_t mask_len,
                             int expected_w, int expected_h,
                             const semper_params*, int32_t update_interval);
int  semper_incremental_process(semper_incremental*, const uint8_t* def, size_t def_len,
                                float* out, int out_capacity,
                                float* metrics, int metrics_len,
                                semper_incremental_info* info,   /* may be NULL */
                                semper_progress_cb cb, void* user);
void semper_incremental_cancel(semper_incremental*);           /* thread-safe */
```

**Layout.** `update_interval` is a separate argument so the `semper_params`
layout stays Frozen. `sizeof` and every field offset of `semper_params` and
`semper_incremental_info` are Frozen. `tests/c/contract.c` asserts them at
compile time, so a layout change is a build failure, not a silent parameter
mis-read in a downstream caller.

**Exports.** These eleven functions are the **only** exported symbols. CI
compares every defined dynamic symbol of `libsemper_c`, of any type, with the
golden list (§A.6) and fails if the set changes. On Linux and Android a linker
version script (`adapters/c/semper_c.map`) keeps template instantiations from
system headers out of the export table.

**Visibility.** Every entry point is annotated `SEMPER_C_API`. The shared
library is built with `-fvisibility=hidden`; on Windows a DLL exports only what
is marked `__declspec(dllexport)`, which `SEMPER_C_API` expands to while
building. So only annotated symbols are exported, and a symbol that loses its
annotation silently vanishes from the ABI.

- **Consumers must NOT define `SEMPER_C_BUILD`.** It is set `PRIVATE` while
  building `libsemper_c` (`adapters/c/CMakeLists.txt`) and selects `dllexport`
  on Windows. A consumer that defines it gets `dllexport` instead of
  `dllimport` and fails to link. Just `#include <semper/semper_c.h>`: the macro
  resolves to `dllimport` (Windows) or default visibility (ELF/Mach-O).

**Behavior.**

- A handle is not internally synchronized: use one handle per thread, or
  serialize calls. `semper_cancel` and `semper_incremental_cancel` are safe
  from any thread.
- A call on a handle while another is in flight on it (from another thread, or
  from the first call's progress callback) returns `SEMPER_ERR_INIT` and does
  nothing *(since 0.7.0)*. The two cancel functions are exempt.
- Mask bytes that do not decode return `SEMPER_ERR_INIT`; they are never
  treated as "no mask".
- `semper_set_reference` and `semper_incremental_init` release what the handle
  held first, so a failed call leaves no reference or session behind
  *(since 0.7.0)*.
- A negative `out_capacity` is `SEMPER_ERR_INIT` *(since 0.7.0)*.
- `info` is written only when `semper_incremental_process` returns `>= 0`
  *(since 0.7.0)*.
- A `semper_cancel` issued at any point of `semper_run`, while `def` is still
  decoding included, stops it *(since 0.7.0)*.

---

## A.3 The JNI mapping (informative)

JNI exports live in `adapters/android/jni/SemperJNI.cpp` under the package
`com.sempermechanics.semper.SemperNativeLib`. All nine exports are also
forwarded from the old `com.indicvision.semper` package until the apps have
moved. CI checks every exported `Java_*` and `JNI_*` symbol against
`tests/jni/jni_symbols.txt`.

| JNI export | Public engine calls used |
|---|---|
| `JNI_OnLoad` | `cv::setNumThreads(1)` |
| `setDebugOutputDir(String?)` | `ReferenceCache::debug_dir` |
| `setCancelRequested(bool)` | `request_cancel` / `clear_cancel` |
| `getImageDimensions(byte[])` | `io::image_dimensions` |
| `getPreviewFromBytes(byte[], int)` | `io::decode_bgr` |
| `initializeReference(byte[], byte[]?, int, int)` | `io::decode_gray`, `io::decode_mask`, `ReferenceCache::set_from_gray` / `reset` |
| `computeFullFieldDirect(…)` | `CancelScope`, `io::decode_gray`, `io::decode_mask`, `run_full_field` |
| `initializeIncremental(byte[], byte[]?, int w, int h, rectX, rectY, rectW, rectH, step, subset, strainWindow, bool use6x6, int updateInterval)` (since 0.4.0) | `io::decode_gray`, `io::decode_mask`, `IncrementalSession::init`; resets the fixed-reference cache |
| `computeIncrementalFrame(byte[], ByteBuffer, callback, float[]?, int[]? info)` (since 0.4.0) | `CancelScope`, `io::decode_gray`, `IncrementalSession::set_debug_dir` / `process_frame`; `info` (length ≥ 6) receives the `IncrementalFrameInfo` fields in declaration order, the bool as 0/1 |
| `releaseIncremental()` (since 0.4.0) | destroys the session |

- **Threads**: `JNI_OnLoad` sets OpenCV's own thread pool to one thread. The
  solver's parallel regions and Path B's workers use
  `std::thread::hardware_concurrency()` threads.
- **One lock**: the fixed-reference and incremental paths share one lock and do
  not run at the same time. `initializeIncremental` drops the cached reference.
  `initializeReference` closes any open incremental session, so call it again
  before going back to `computeFullFieldDirect`. `setCancelRequested` takes no
  lock, so it reaches a running solve.
- **Cancel window** *(since 0.7.0)*: it opens once a solve export holds the
  lock (`CancelScope`). A `setCancelRequested(true)` from then on, during the
  frame decode included, stops that solve. One that arrived while the previous
  solve held the lock is cleared.
- **Re-entry** *(since 0.7.0)*: a locking export called from inside
  `onProgressUpdate` returns its error value (`-3`, or nothing for the `void`
  ones) instead of waiting for the lock the solve calling it holds.
- **Failed reference** *(since 0.7.0)*: `initializeReference` releases the
  previous reference first. After a failed call there is none, and the next
  `computeFullFieldDirect` returns `-3`.
- **Output arrays** *(since 0.7.0)*: `computeFullFieldDirect` and
  `computeIncrementalFrame` write the metrics and info arrays only when they
  return `>= 0`.
- **Metrics copy-back**: the adapter copies the largest prefix the Kotlin array
  holds, by `metrics_prefix_len` (since 0.4.1; before, at most 19 slots).

---

## A.4 Frozen data formats

### Output buffer — packed `float32`, **8 per point**

```
index:  0   1   2   3     4     5     6      7
field:  x   y   u   v    exx   eyy   exy   corr
```

| Field | Meaning |
|---|---|
| `x, y` | Node position, px (`rect_x + i * step`, `rect_y + j * step`). Incremental: the frame-0 grid position. |
| `u, v` | Displacement, px. Incremental: the total from the first image. |
| `exx, eyy, exy` | Green-Lagrange strain from the VSG fit. `exy` is the tensor component, not the engineering shear strain (`γxy = 2 · exy`). |
| `corr` | Correlation score: the per-pixel mean ZNSSD ([MATHEMATICS.md](MATHEMATICS.md)). |

**Capacity rule** *(Frozen)*: the caller passes the capacity in floats.
Packing stops at the first point that does not fit, so points beyond capacity
are dropped, never overflow. A capacity below one point returns 0 points, not
an error; a capacity of 0 is a valid metrics-only query.

### Metrics buffer — **24 × `float32`**

Slot indices are Frozen; new telemetry appends. `metrics_len == 16` is the
minimum honored, and below it nothing is written. **24 is preferred**
(`kMetricsLen`, `SEMPER_METRICS_LEN`).

The writer copies the largest prefix the caller's buffer can hold: 16, 17, 19,
20, 21, 22, 23 or 24 slots (an 18-slot buffer gets 17, because slots 17 and 18
came in together). `metrics_prefix_len` is the one table; the solver and the
JNI copy-back both use it. An existing caller that still passes 17 or 23 keeps
working unchanged.

| Slot | Meaning | Added |
|---|---|---|
| 0 | Total attempted: grid nodes that passed the boundary and mask scan | |
| 1 | Total solved: points written to the output | |
| 2 | Total rejected: slot 0 − slot 1 | |
| 3 | Solved via mesh (Path A) | |
| 4 | Solved via flood fill (Path B) | |
| 5 | Simplex rescues | |
| 6 | Simplex rescues that succeeded | |
| 7 | Simplex rescues that failed | |
| 8 | Mean IC-GN iterations per solved point | |
| 9 | Total wall time (ms) | |
| 10 | Seeding time (ms) | |
| 11 | Hessian pre-pass time (ms) | |
| 12 | Mesh setup time (ms) | |
| 13 | Strain calculation time (ms) | |
| 14 | Throughput over the whole solve (points/ms) | |
| 15 | Convergence: slot 1 / slot 0, in % | |
| 16 | Seeding route: `2` = full mesh, `1` = sparse mesh, `0` = Path C fallback | |
| 17 | Total simplex-rescue time (ms), all paths | 0.2.2 |
| 18 | Total IC-GN time (ms), all paths | 0.2.2 |
| 19 | Solved via the anchor lattice | 0.3.0 |
| 20 | Converged, then discarded by the VSG strain post-filter | 0.3.0 |
| 21 | Global-motion lock: `1` = locked, `0` = no lock | 0.3.0 |
| 22 | Mesh coverage: convex hull of the accepted anchors / ROI area, `0..1` | 0.3.0 |
| 23 | Converged, then rejected by the displacement median test before packing | 0.5.0 |

**Notes on individual slots.**

- **Slots 3 and 4** count Path A and Path B only. Points the anchor lattice
  solves are in slot 19, appended in 0.3.0 rather than repurposing either
  Frozen slot.
- **Slot 10** is "seeding time (ms)" since 0.2.2 (before: "AKAZE + RANSAC time
  (ms)"). Since 0.6.0 it measures AKAZE matching plus the anchor lattice
  (phase correlation plus the lattice before). Callers that display it should
  label it "seeding time".
- **Slot 11** reads about `0` on the second and later solves against one
  reference with the same grid and mask: the Hessian pre-pass is cached in
  `ReferenceCache` (§A.2). That cache changes no other count and no output
  float; the cached AKAZE starting scale can (§A.2).
- **Slots 8, 17 and 18** include the anchor lattice's own IC-GN phase since
  0.3.0. Slots 17 and 18 include Path C's seed search since 0.7.0.
- **Slot 20** separates a point that converged and was then discarded (its
  strain fit was rank-deficient) from one that never converged.
- **Slot 21** reports whether the global motion model locked. Since 0.6.0 that
  is an AKAZE homography with at least `kAkazeMinInliers` RANSAC inliers;
  before, it was a phase-correlation peak. The two fail in different places:
  AKAZE locks at any rotation and under large stretch, but needs enough texture
  at some scale to match. The same image pair can therefore read differently
  in the two versions. A `0` means the lattice searched for its own offsets.
- **Slots 21 and 22 are the seeding health signals.** Treat a `0` in slot 21,
  or a slot 22 below roughly 0.5, as "these displacements are probably fine
  near the anchors and unreliable away from them", not as an error.
  Background: [SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md) §4.5.

**Point accounting.** With slots 19, 20 and 23 present, and an output buffer
large enough to hold the whole field:

```
metrics[3] + metrics[4] + metrics[19] == metrics[1] + metrics[20] + metrics[23]
```

Every point some path solved is in the output buffer, was rejected as a
displacement outlier, or was dropped by the strain post-filter. A caller
reading a 23-slot buffer can recover slot 23 as the difference.
`FullField.MetricsIdentityHoldsWithOutliers` asserts the identity;
`FullField.MetricsPointCountsAccountForEverySolvedPoint` asserts its 23-slot
form on a field without outliers.

**The identity does not survive truncation, by design.** Under the capacity
rule, packing stops as soon as the next point would not fit. The points never
reached were already counted by their solving path into slots 3, 4 and 19, but
land in neither slot 1 nor slot 20. The left side then exceeds the right by
exactly the number left unpacked. Test for that case with
`(metrics[1] + 1) * 8 > output_capacity` (no room for one more point); the
identity is guaranteed only when that is false.

**Incremental sessions.** Metrics from `IncrementalSession::process_frame`
describe the solve against the current reference on that reference's own grid.
The counts (slots 0–4, 19, 20) are of that grid, not of the frame-0 output. On
the first frame against the first image they match the output exactly. After
the reference has moved, the number of points written is the return value, and
slot 1 is the increment solve's count.

### Return / error codes — Frozen

| Value | C++ name | C ABI name | Meaning | Added |
|---|---|---|---|---|
| `>= 0` | | | number of valid output points | |
| `-2` | | `SEMPER_ERR_ROI` | invalid ROI | |
| `-3` | | `SEMPER_ERR_INIT` | init / argument failure | |
| `-4` | `kBadStrainWindow` | `SEMPER_ERR_STRAIN_WINDOW` | `strain_window` too small for `step` | 0.3.0 |
| `-5` | `kSeedFailed` | `SEMPER_ERR_SEED` | no seed: no anchor converged, the global motion model did not lock, and the fallback search failed | 0.5.0 |
| `-99` | `kCancelled` | `SEMPER_ERR_CANCELLED` | cancelled mid-solve | |

Existing values never change. `-4` and `-5` were appended as minor changes:
callers that branch on `n < 0` need no edit, and no earlier value moved.

**The strain-window rule** behind `-4`: the VSG plane fit needs 3 grid nodes
inside the window. Below that, the strain post-filter discards every point,
however well it correlated. The rule is `strain_window >= 2 * step`
(`StrainCalculator::vsg_window_node_count` is the exact test). Before 0.3.0
this case returned 0 points with a success code.

---

## A.5 Change checklist

- **Free to change (patch):** correlation math, IC-GN, seeding, SIMD,
  threading, `tuning.hpp`, anything under `src/`. The golden tests must still
  pass. `GoldenCorpus` covers the subset solver, not `run_full_field`, so a
  seeding change can move the full field without moving that golden;
  `FullFieldGolden` covers `run_full_field`. See [TESTING.md](TESTING.md).
- **Additive only (minor):** a trailing `FullFieldParams` field, a new metrics
  slot, a new overload or function, a new SDK entry point.
- **Do not without major + app coordination:** change any §A.2 signature;
  change the 8-float packing; change a metric slot's meaning; change
  return-code values; make `decode_*` throw; make `ReferenceCache` copyable or
  movable.

Every PR that touches `include/semper/*` must state its tier (**patch / minor /
major**) and update this document.

---

## A.6 How improvements reach downstream apps

1. Land the change behind the same public signatures, or additively. Update
   `SEMPER_VERSION_MAJOR` / `_MINOR` / `_PATCH` in `include/semper/version.hpp`
   per §A.1, and keep `project(VERSION)` in `CMakeLists.txt` and `version` in
   `bindings/python/pyproject.toml` in sync with it.
2. Tag a release. Downstream apps bump their **pinned submodule tag**, review
   the diff, and run their own contract test through the real entry point.
3. Ship. Source compatibility of `include/semper/*` is what matters for static
   linkers; `semper_c.h` also guarantees ABI stability for external consumers.

### Consuming the C SDK

With `-DSEMPER_BUILD_C_SDK=ON`, the install step provides a CMake package, so
downstreams need no hard-coded paths: `find_package(Semper)` provides the
target `Semper::semper_c` ([README.md](../README.md) has the build commands).
The package enforces §A.1 mechanically:

- The shared library carries `VERSION = <full>` and `SOVERSION = <MAJOR>`. The
  runtime `SONAME` bump on a major release stops an old binary from silently
  loading an incompatible `.so`.
- The installed `SemperConfigVersion.cmake` is written with
  `COMPATIBILITY SameMajorVersion`. A consumer that pinned
  `find_package(Semper 0.x)` refuses to configure against a future `1.x`.

### The automated backstop

These checks are the machine-enforced half of this document. A change that
breaks one fails CI.

| Check | CI job | What it pins |
|---|---|---|
| `tests/c/contract.c` | `c-sdk-smoke` (and `adapters-sanitized`) | Every Frozen return code, the 8-float packing, the capacity-drop rule, the metrics-length rule, `sizeof`/`offsetof` of `semper_params` and `semper_incremental_info`, the argument guards (fixed and incremental), the strain-window guard, undecodable masks on every entry point, one call at a time per handle, cancels that stay on their own handle, and the `MAJOR.MINOR.PATCH` shape of `semper_version()` |
| `tests/c/abi_symbols.txt` | `c-sdk-smoke` | The exported symbol set, diffed against `nm -D --defined-only` of the built `libsemper_c`: an accidental export or a dropped `SEMPER_C_API` annotation fails |
| `tests/jni/jni_symbols.txt` | `android-ndk` | The exported `Java_*` and `JNI_*` symbols of both packages |
| `dic_tests` | host jobs | Host-side behavior: the solver itself |

[TESTING.md](TESTING.md) describes each suite and how to run it.
