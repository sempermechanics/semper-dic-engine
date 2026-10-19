# Engine API Contract

**Audience:** contributors to this repository.

A private Android application links this engine through the JNI adapter in
`adapters/android/`. That app lives in a different repository and **cannot** be
seen from an engine pull request, so this document is the contract that keeps
the two in sync.

> **The rule in one sentence:** anything marked **Frozen** or **Stable** below is
> load-bearing for a downstream app; change it carelessly and you break a shipping
> product whose code you can't see. The automated backstop is the host test suite
> (`dic_tests`) plus the C ABI contract binary (`tests/c/contract.c`, run in the
> `c-sdk-smoke` CI job) and its golden exported-symbol list (`tests/c/abi_symbols.txt`);
> this document is the human one. See [§A.6](#a6-how-improvements-reach-downstream-apps).

You do **not** need the app to contribute. Build and test against `tests/` as
usual. Respect the surface described here, and declare your change's
[tier](#a1-stability-tiers) in the PR.

---

## A.1 Stability tiers

| Tier | What it covers | Rule for contributors |
|---|---|---|
| **Frozen** | Binary/data formats: output packing, metrics layout, error codes, and the *meaning* of existing `FullFieldParams` fields | Never change in place. A change here is a **major** version bump and requires a coordinated change in the downstream app. |
| **Stable** | Public function **signatures** in `include/semper/{pipeline,incremental,io,cancel,metrics,version}.hpp` and `include/semper/semper_c.h` | May be **extended additively** (new overloads / new functions). Existing signatures, parameter order, defaults, and semantics must not change without a **major** bump. |
| **Additive-only** | Deliberate growth points: trailing `FullFieldParams` fields, unused metrics slots | Append only, each with a default that reproduces prior behavior. Never reorder or repurpose an existing entry. |
| **Internal** | Everything in `src/`, `tuning.hpp` constants, `types.hpp`, `seeding.hpp`, algorithm internals, private headers. Living under `include/semper/` does **not** make a header Stable — only the six named above and `semper_c.h` are (`guarded_call.hpp`, added v0.5.0, is Internal). | Change freely. Behavior is guarded by the golden test, not by this contract. |

**Semantic versioning** (`include/semper/version.hpp` / `SEMPER_VERSION`):

- **major** — any Frozen or Stable break.
- **minor** — additive Stable / Additive-only change, or a new capability.
- **patch** — Internal-only improvement (accuracy, speed, robustness) with no surface change.

Downstream apps pin an **exact** engine tag and only auto-adopt minor/patch bumps.

> **v0.5.0 is a minor release.** Additive: `kSeedFailed` / `SEMPER_ERR_SEED`
> (`-5`), metrics slot 23 and `kMetricsLen` / `SEMPER_METRICS_LEN` = 24,
> `metrics.hpp` with `metrics_prefix_len`, `io::to_gray8`. Every Frozen value
> and every existing signature is unchanged. What callers can observe besides:
> inputs that used to be accepted and mis-solved are now refused (`-3` for an
> even subset, a frame that does not match the reference, or a strain window
> too large for the grid to ever fit, `-2` for a rect
> with nothing on the image), exceptions no longer cross the C ABI or JNI, a
> `CancelToken` solve no longer listens to `request_cancel()`, non-8-bit
> images keep their contrast, and the solved field moves (IC-GN validity and
> conditioning, MATHEMATICS §6.6; goldens re-captured).

> **Build-plumbing carve-out.** Export/visibility machinery — the `SEMPER_C_API`
> macro, `-fvisibility=hidden`, `dllexport`/`dllimport` selection — is treated as
> build plumbing, not public surface: it changes *how* the documented symbols are
> emitted, never *which* symbols exist or their signatures. Such changes ship as
> **patch** (e.g. `0.1.1` → `0.1.4`, which added `SEMPER_C_API` to every entry
> point). The golden symbol list in §A.6 is what guards the set of exported symbols
> against accidental drift.

---

## A.2 Public API reference (the app-facing surface)

These are the **only** engine symbols the Android app calls (through the JNI
adapter). Keep the signatures here in sync with the code — they are authoritative.

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

// Additive overload — per-solve cancel for SDK / Python:
int run_full_field(..., CancelToken& token, ProgressCallback on_progress = nullptr);
```

- **Returns** *(Frozen)*: `>= 0` → number of valid output points. `-2` → invalid ROI.
  `-3` → init/argument failure. `-4` (`kBadStrainWindow`) → strain window too
  small for the step. `-5` (`kSeedFailed`, v0.5.0) → nothing could seed the
  solve. `-99` (`kCancelled`) → cancelled mid-solve.
  A non-positive `params.step` is treated as an invalid ROI (`-2`) — it would
  otherwise divide-by-zero when forming the grid. This is a clarification of the
  existing `-2` meaning, not a new code.
- **Image types** *(v0.5.0)*: `def_gray` must be `CV_8UC1` and the size of
  the cached reference, and a non-empty `roi_mask` `CV_8UC1` (any size; it is
  resized), or the call returns `-3`. A frame of another size used to skip
  phase correlation and then solve anyway, returning a plausible field of
  nonsense. `set_from_gray` with a non-`CV_8UC1` image or mask leaves the
  cache empty (so the next solve is `-3`), and `IncrementalSession::init`
  returns `-3` for either.
- **Parameter validation** *(v0.5.0)*: `subset_size` must be odd and at least
  3, or the call returns `-3` (an even size overran the subset buffers). The
  rect is clipped to the reference image before the grid is formed; the near
  edge moves by whole steps, so every node keeps its position
  `rect_x + i * step`. A rect with nothing on the image, or narrower than one
  step once clipped, is `-2`. Until v0.5.0 an oversized rect sized the grid
  from the rect and could throw `std::bad_alloc` out of the call.
- **Strain window upper bound** *(v0.5.0)*: `strain_window` may be at most
  `2 * (min(gridW, gridH) + 1) * step` px, with `gridW = rect_w / step` and
  `gridH = rect_h / step` of the clipped rect; above that the call returns
  `-3` (`IncrementalSession::init` likewise, on the frame-0 grid). Past that
  size the VSG circle reaches more than a step beyond the grid's short side
  from every node, so no node can meet the strain fit's 90% window-coverage
  rule and the field would come back empty — the same outcome as before,
  except that it used to take `(strain_window / step)^2` work per node first
  (hours at `INT_MAX`). The lower bound is unchanged: too small for the step is
  still `-4`. A rejection, not a clamp: a clamped window would solve a
  different strain fit than the one asked for.
- **Exceptions** never cross the C ABI or JNI: each entry point catches what
  escapes and returns `-3` (`NULL` from the C constructors). The C++ functions
  may still throw `std::bad_alloc` and OpenCV exceptions.
- **Writes** *(Frozen)*: at most `output_capacity` floats, **8 per point** (see
  [A.4](#a4-frozen-data-formats)). Points beyond capacity are **dropped, never
  overflow**.
- **Metrics** *(Frozen layout)*: when `metrics != nullptr && metrics_len >= 16`,
  fill the `metrics_prefix_len(metrics_len)` leading slots; `kMetricsLen` (24)
  preferred (`include/semper/metrics.hpp`). Slot indices are frozen.
- **Cancellation**: the legacy overload clears the process-global cancel flag on
  entry and polls it inside point loops. The `CancelToken&` overload clears and
  polls only that token, so solves with their own tokens can run concurrently
  and cancel independently, and `request_cancel()` does not reach them. (Until
  v0.5.0 the token was bound into one process-global pointer for the duration
  of the solve, so two concurrent solves polled whichever was bound last.)
- **You may** make it faster or more accurate (Internal). **You may not** change
  return-code meanings, 8-float packing, the capacity-drop rule, or metric slot
  indices without a major bump.

### `pipeline::ReferenceCache` — cached reference state · *Stable*

```cpp
struct ReferenceCache {                 // non-copyable, non-movable
    ReferenceCache();
    void reset();
    void set_from_gray(const cv::Mat& gray_in, const cv::Mat& roi_mask);
    std::string debug_dir;              // "" disables debug export
    // internal members are INTERNAL
};
```

Must remain **non-copyable / non-movable**.

`ReferenceCache` also keeps work that depends only on the reference and the
grid: the node validity scan, the Hessian pool and the reference side of phase
correlation. That cache is **Internal**. It is reused while `rect`, `step`,
`subset_size` and the ROI mask are unchanged, and dropped by `reset`,
`set_from_gray` and `adopt`. Outputs are bit-identical with or without it.
`adopt(...)` (added v0.4.0) hands an already-prepared image to the cache; it
exists for `IncrementalSession` and is **Internal** despite living in a Stable
header.

### `pipeline::IncrementalSession` — updated-reference sequences · *Stable (added v0.4.0)*

```cpp
// include/semper/incremental.hpp
struct IncrementalParams {
    FullFieldParams field;     // the frame-0 grid, exactly as for run_full_field
    int update_interval = 1;   // 0 = fixed reference, N = move the reference every N frames
                               // (sooner when a frame reports 5% fewer points, N > 1)
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
  `run_full_field`. `x, y` are **frame-0 grid coordinates**, `u, v` the **total**
  displacement from the first image, and the strains are Green-Lagrange strain
  of that total. Only measured nodes are reported. A node the solve cannot
  reach on a frame (a gap it cannot correlate across) is followed on estimated
  motion, unreported, and is reported again once it is measured; after a long
  coast, only once its own earlier subset confirms where it is (MATHEMATICS
  §9.5). A frame-0 node that cannot be followed at all is dropped and never
  returns.
- **`update_interval = 0`** *(Frozen)*: every frame is bit-identical to
  `run_full_field` against the first image. With any interval, the first frame
  is too.
- **Returns** *(Frozen)*: `init` returns `0`, `-2`, `-3` (empty image, negative
  interval, a `subset_size` that is not odd and at least 3) or `-4`. The rect is
  validated and clipped exactly as `run_full_field` does. `process_frame` returns the same codes as `run_full_field`;
  `-3` also covers "not initialized" and a frame whose size differs from the
  first image.
- **Errors and cancellation** leave the session exactly as it was, so the same
  frame can be submitted again.
- **Metrics** describe the increment solve against the current reference. The
  frame-0 bookkeeping is in `IncrementalFrameInfo`, not in new metric slots.

### `pipeline::FullFieldParams` — solve inputs · *Frozen fields + Additive-only tail*

```cpp
struct FullFieldParams {
    int rect_x, rect_y, rect_w, rect_h;
    int step;
    int subset_size;
    int strain_window;
    bool use_6x6_interpolator;
};
```

New tunables must be **appended** with a default that reproduces current behavior.

### `pipeline::ProgressCallback` · *Stable*

```cpp
using ProgressCallback = std::function<void(int percentage)>;   // 0..100
```

### Cancellation — `include/semper/cancel.hpp` · *Stable*

```cpp
constexpr int kCancelled = -99;   // Frozen value
void request_cancel();
void clear_cancel();
bool cancel_requested();
// Plus CancelToken for per-instance cancel (SDK / Python)
```

The free functions stay — JNI `setCancelRequested` maps to them. They act on
one process-wide token, the one every solve started without a token polls.
`detail::ScopedCancelToken` (never documented here) was removed in v0.5.0.

### Image I/O — `include/semper/io.hpp` · *Stable*

```cpp
cv::Mat decode_gray(const uint8_t* data, size_t len, int expected_w = 0, int expected_h = 0);
cv::Mat decode_bgr (const uint8_t* data, size_t len);
void    image_dimensions(const uint8_t* data, size_t len, int& out_w, int& out_h);
cv::Mat to_gray8(const cv::Mat& img);                         // added v0.5.0
```

Empty `cv::Mat` = failure; do not switch to throwing.

- **Depth** *(v0.5.0)*: an encoded image of any depth other than 8-bit is
  mapped linearly from its own min..max onto 0..255 (`to_gray8`); a flat one
  becomes 128. Before, `convertTo(CV_8U)` saturated it — a 12-bit camera frame
  came out nearly all white. ZNSSD is invariant to that affine map, so only
  quantization changes for frames that already decoded correctly.
- **Raw or encoded** *(v0.5.0)*: with `expected_w/h`, a buffer of exactly
  `w*h*4` or `w*h` bytes is still decoded as an image when it starts with a
  PNG, JPEG, TIFF, BMP, WebP or PNM signature and decodes as one; otherwise it
  is raw, as before.

### C ABI — `include/semper/semper_c.h` · *Stable (ABI)*

`semper_create` / `destroy`, `semper_set_reference`, `semper_run`, `semper_cancel`,
`semper_version`. Same Frozen return codes and 8-float packing.

Added in v0.4.0 (minor), the incremental session:

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

`semper_params` is unchanged; the interval is a separate argument so its layout
stays Frozen. These eleven functions are the **only** exported symbols; the
golden list in §A.6 fails CI if that set changes.

Every entry point is annotated `SEMPER_C_API`. The shared library is built with
`-fvisibility=hidden` (and `/W…` on MSVC), so only the annotated symbols are
exported — a symbol that loses its annotation silently vanishes from the ABI.

- **Consumers must NOT define `SEMPER_C_BUILD`.** It is set `PRIVATE` while
  building `libsemper_c` (`adapters/c/CMakeLists.txt`) and selects `dllexport` on
  Windows. A consumer that defines it gets `dllexport` instead of `dllimport` and
  fails to link. Just `#include <semper/semper_c.h>` — the macro resolves to
  `dllimport` (Windows) or default visibility (ELF/Mach-O) automatically.
- `sizeof(semper_params)` and the offset of each of its eight fields are Frozen;
  `tests/c/contract.c` asserts them at compile time so a struct-layout change is a
  build failure, not a silent parameter mis-read in a downstream caller.

---

## A.3 The JNI mapping (informative)

JNI exports live in `adapters/android/jni/SemperJNI.cpp` under the package
`com.sempermechanics.semper.SemperNativeLib`, with forwarders for the old
`com.indicvision.semper` package until the apps have moved. All OpenMP work must stay on one pinned
thread on Android.

| JNI export | Public engine calls used |
|---|---|
| `JNI_OnLoad` | `cv::setNumThreads(1)` |
| `setDebugOutputDir(String?)` | `ReferenceCache::debug_dir` |
| `setCancelRequested(bool)` | `request_cancel` / `clear_cancel` |
| `getImageDimensions(byte[])` | `io::image_dimensions` |
| `getPreviewFromBytes(byte[], int)` | `io::decode_bgr` |
| `initializeReference(byte[], byte[]?, int, int)` | `io::decode_gray`, `ReferenceCache::set_from_gray` / `reset` |
| `computeFullFieldDirect(…)` | `io::decode_gray`, `run_full_field` |
| `initializeIncremental(byte[], byte[]?, int w, int h, rectX, rectY, rectW, rectH, step, subset, strainWindow, bool use6x6, int updateInterval)` (v0.4.0) | `io::decode_gray`, `IncrementalSession::init`; resets the fixed-reference cache |
| `computeIncrementalFrame(byte[], ByteBuffer, callback, float[]?, int[]? info)` (v0.4.0) | `io::decode_gray`, `IncrementalSession::process_frame`; `info` (length ≥ 6) receives the `IncrementalFrameInfo` fields in declaration order, the bool as 0/1 |
| `releaseIncremental()` (v0.4.0) | destroys the session |

The fixed-reference and incremental paths share one lock and do not run at the
same time: `initializeIncremental` drops the cached reference, and
`initializeReference` closes any open incremental session, so call it again
before going back to `computeFullFieldDirect`.

---

## A.4 Frozen data formats

### Output buffer — packed `float32`, **8 per point**

```
index:  0   1   2   3     4     5     6      7
field:  x   y   u   v    exx   eyy   exy   corr
```

### Metrics buffer — **24 × `float32`**

Slot indices are Frozen; new telemetry appends. `metrics_len == 16` is the
minimum honored; **24 preferred** (`kMetricsLen`, `SEMPER_METRICS_LEN`). The
writer copies the largest prefix the caller's buffer can hold (16, 17, 19, 20,
21, 22, 23 or 24 slots; `metrics_prefix_len` is the one table, used by the
solver and the JNI copy-back alike), so an existing caller that still passes
17 or 23 keeps working unchanged. The JNI adapter honours this from v0.4.1;
before that it copied at most 19 slots, so slots 19–22 never reached Kotlin.

**Slot 10 semantic change (v0.2.2).** The index, type and layout are unchanged
and remain Frozen. Its *meaning* changed from "AKAZE + RANSAC time (ms)" to
**"seeding time (ms)"** — phase correlation plus the anchor lattice — when the
descriptor seeding front-end was replaced. Callers that display this value
should relabel it; nothing about the buffer's shape moved. Slot 16 keeps
`2 = full mesh / 1 = sparse mesh / 0 = Path C fallback`.

**Slots 3 and 4 narrowed (v0.3.0).** Their labels — "Solved via Mesh" (Path A)
and "Solved via Flood Fill" (Path B) — are unchanged and still Frozen, but they
no longer account for every solved point. The anchor lattice introduced in
v0.2.2 is a third solving path, and from v0.2.2 to v0.3.0 the points it solved
reached slot 1 ("Total Solved") without appearing in slot 3 or slot 4. Rather
than repurpose either Frozen slot, v0.3.0 **appends** the missing terms:

| Slot | Meaning | Added |
|---|---|---|
| 17 | Total simplex-rescue time (ms), all paths | v0.2.2 |
| 18 | Total IC-GN time (ms), all paths | v0.2.2 |
| 19 | Solved via the anchor lattice | v0.3.0 |
| 20 | Converged, then discarded by the VSG strain post-filter | v0.3.0 |
| 21 | Phase-correlation lock: `1` = locked, `0` = no lock | v0.3.0 |
| 22 | Mesh coverage: convex hull of the accepted anchors / ROI area, `0..1` | v0.3.0 |
| 23 | Converged, then rejected by the displacement median test before packing | v0.5.0 |

With slots 19, 20 and 23 present the point accounting closes exactly, as
long as the output buffer was large enough to hold the whole field:

```
metrics[3] + metrics[4] + metrics[19] == metrics[1] + metrics[20] + metrics[23]
```

That is, every point some path solved is either in the output buffer, was
rejected as a displacement outlier, or was dropped by the strain
post-filter. Until v0.5.0 the identity had no slot 23 and failed whenever the
median test rejected anything (`FullField.MetricsIdentityHoldsWithOutliers`);
a caller reading a 23-slot buffer can still recover slot 23 as the
difference. Slot 20 was previously invisible: a caller
seeing slot 1 fall short of slots 3 + 4 had no way to distinguish a point that
never converged from one that converged and was then discarded because its
strain fit was rank-deficient.
`FullField.MetricsPointCountsAccountForEverySolvedPoint` asserts the identity
above.

**The identity does not survive truncation, by design.** Under the capacity
rule (§A.2) packing stops as soon as the next point would not fit, and the
points never reached were already counted by their solving path into slots 3,
4 and 19 while landing in neither slot 1 nor slot 20 — so the left side
exceeds the right by exactly the number left unpacked. The caller's test for
that case is `(metrics[1] + 1) * 8 > output_capacity`, i.e. the buffer had no
room for one more point; the identity is only guaranteed when that is false.
`FullField.MetricsPointCountsAccountForEverySolvedPoint` passes a full-size
buffer for that reason.

**Slots 21 and 22 are the seeding health signals.** The anchor lattice estimates
translation only, so past roughly 5° of rotation the phase lock fails and mesh
coverage collapses — 0.45 at 5° and 0.15 at 15° against the old descriptor
front-end's 0.96 (`docs/SEEDING_BENCHMARK.md` §4.5). Both numbers were already
computed inside the seeder and then discarded, so a field solved off a
15%-coverage mesh was reported exactly like a healthy one. Treat a `0` in slot
21, or a slot 22 below roughly 0.5, as "these displacements are probably fine
near the anchors and unreliable away from them" rather than as an error.

Slots 17 and 18 also changed value in v0.3.0 without changing meaning: the
anchor lattice runs a full IC-GN phase of its own, and its time was being
dropped on the floor rather than added to these totals. Slot 8 ("Mean ICGN
Iterations") was understated for the same reason — it divided a sum that
excluded the anchor phase by a count that included it.

**Prepass time on a reused reference (v0.4.0).** Slot values for the same
inputs are unchanged, but the Hessian pre-pass time reads about `0` on the
second and later frames against one reference with the same grid and mask:
that work is now cached in `ReferenceCache` (§A.2). The other slots, and every
output float, are bit-identical.

**Incremental sessions.** Metrics from `IncrementalSession::process_frame`
describe the solve against the current reference on that reference's own
grid, so the counts (slots 0–4, 19, 20) are of that grid, not of the frame-0
output. On the first frame against the first image they match the output
exactly; after the reference has moved, the number of points written is the
return value, and slot 1 is the increment solve's count.

### Return / error codes — Frozen

Existing values never change. `-4` was **appended** in v0.3.0, which is an
additive (minor) change: callers that branch on `n < 0` need no edit, and no
previously-returned value moved. What did change is the *behaviour* of one
input: a `strain_window` too small for `step` used to return `0` points with a
success code — the VSG plane fit needs 3 grid nodes inside the window, and
below that every point is discarded by the strain post-filter no matter how
well it correlated. The rule is `strain_window >= 2 * step`
(`StrainCalculator::vsg_window_node_count` is the exact test). Every example
and smoke test in this repo was on the wrong side of it before v0.3.0.

| Value | Meaning |
|---|---|
| `>= 0` | number of valid output points |
| `-2` | invalid ROI |
| `-3` | init / argument failure |
| `-4` | `strain_window` too small for `step` (**added v0.3.0**) |
| `-5` | no seed: no anchor converged, phase correlation did not lock, and the fallback search failed (**added v0.5.0**; this case returned an undocumented `-1` before) |
| `-99` (`kCancelled`) | cancelled mid-solve |

---

## A.5 Change checklist

- **Free to change (patch):** correlation math, ICGN, seeding, SIMD, threading,
  `tuning.hpp`, anything under `src/` — golden test must still pass. Note that
  `GoldenCorpus` covers the engine, not `run_full_field`; a seeding change can
  move the full field without moving that golden. See docs/TESTING.md.
- **Additive only (minor):** trailing `FullFieldParams` field, new metrics slot,
  new overload/function, new SDK entry point.
- **Do not without major + app coordination:** change any §A.2 signature; change
  8-float packing; change a metric slot's meaning; change return-code values;
  make `decode_*` throw; make `ReferenceCache` copyable/movable.

Every PR that touches `include/semper/*` must state its tier (**patch / minor /
major**) and update this document.

---

## A.6 How improvements reach downstream apps

1. Land the change behind the same public signatures (or additively). Update
   `SEMPER_VERSION` per §A.1.
2. Tag a release. Downstream apps bump their **pinned submodule tag**, review the
   diff, and run their own contract test through the real entry point.
3. Ship. Source compatibility of `include/semper/*` is what matters for static
   linkers; `semper_c.h` additionally guarantees ABI stability for external
   consumers.

### Consuming the C SDK

Building with `-DSEMPER_BUILD_C_SDK=ON` installs a CMake package so downstreams do
not hard-code paths:

```cmake
find_package(Semper REQUIRED)      # provides the Semper:: namespace
target_link_libraries(my_app PRIVATE Semper::semper_c)
```

The shared library carries `VERSION = <full>` and `SOVERSION = <MAJOR>`, and the
installed `SemperConfigVersion.cmake` is written `COMPATIBILITY SameMajorVersion`.
Together these enforce §A.1 mechanically: a consumer that pinned `find_package(Semper 0.x)`
will refuse to configure against a future `1.x`, and the runtime `SONAME` bump on a
major release stops an old binary from silently loading an incompatible `.so`.

### The automated backstop

Two checks run in the `c-sdk-smoke` CI job and are the machine-enforced half of this
document:

- **`tests/c/contract.c`** — asserts every Frozen return code, the 8-float packing,
  the capacity-drop rule, the metrics-length rule, `sizeof`/`offsetof` of
  `semper_params`, and the `MAJOR.MINOR.PATCH` shape of `semper_version()`. A change
  that breaks any of them fails the build.
- **`tests/c/abi_symbols.txt`** — the golden list of exported symbols, diffed against
  `nm -D` of the built `libsemper_c`. An accidental export or a dropped `SEMPER_C_API`
  annotation fails the build.

Host-side behavior (the solver itself) is covered by `dic_tests` (see
[TESTING.md](TESTING.md)).
