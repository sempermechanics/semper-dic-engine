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
| **Stable** | Public function **signatures** in `include/semper/{pipeline,io,cancel,version}.hpp` and `include/semper/semper_c.h` | May be **extended additively** (new overloads / new functions). Existing signatures, parameter order, defaults, and semantics must not change without a **major** bump. |
| **Additive-only** | Deliberate growth points: trailing `FullFieldParams` fields, unused metrics slots | Append only, each with a default that reproduces prior behavior. Never reorder or repurpose an existing entry. |
| **Internal** | Everything in `src/`, `tuning.hpp` constants, `types.hpp`, `seeding.hpp`, algorithm internals, private headers. Living under `include/semper/` does **not** make a header Stable — only the four named above and `semper_c.h` are. | Change freely. Behavior is guarded by the golden test, not by this contract. |

**Semantic versioning** (`include/semper/version.hpp` / `SEMPER_VERSION`):

- **major** — any Frozen or Stable break.
- **minor** — additive Stable / Additive-only change, or a new capability.
- **patch** — Internal-only improvement (accuracy, speed, robustness) with no surface change.

Downstream apps pin an **exact** engine tag and only auto-adopt minor/patch bumps.

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
  `-3` → init/argument failure. `-99` (`kCancelled`) → cancelled mid-solve.
  A non-positive `params.step` is treated as an invalid ROI (`-2`) — it would
  otherwise divide-by-zero when forming the grid. This is a clarification of the
  existing `-2` meaning, not a new code.
- **Writes** *(Frozen)*: at most `output_capacity` floats, **8 per point** (see
  [A.4](#a4-frozen-data-formats)). Points beyond capacity are **dropped, never
  overflow**.
- **Metrics** *(Frozen layout)*: when `metrics != nullptr && metrics_len >= 16`,
  fill telemetry; 23 slots preferred. Slot indices are frozen.
- **Cancellation**: the legacy overload clears the process-global cancel flag on
  entry and polls it inside point loops. The `CancelToken&` overload binds that
  token for the duration of the solve.
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

The free functions stay — JNI `setCancelRequested` maps to them.

### Image I/O — `include/semper/io.hpp` · *Stable*

```cpp
cv::Mat decode_gray(const uint8_t* data, size_t len, int expected_w = 0, int expected_h = 0);
cv::Mat decode_bgr (const uint8_t* data, size_t len);
void    image_dimensions(const uint8_t* data, size_t len, int& out_w, int& out_h);
```

Empty `cv::Mat` = failure; do not switch to throwing.

### C ABI — `include/semper/semper_c.h` · *Stable (ABI)*

`semper_create` / `destroy`, `semper_set_reference`, `semper_run`, `semper_cancel`,
`semper_version`. Same Frozen return codes and 8-float packing. These six are the
**only** exported symbols; the golden list in §A.6 fails CI if that set changes.

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
`com.indicvision.semper.SemperNativeLib`. All OpenMP work must stay on one pinned
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

---

## A.4 Frozen data formats

### Output buffer — packed `float32`, **8 per point**

```
index:  0   1   2   3     4     5     6      7
field:  x   y   u   v    exx   eyy   exy   corr
```

### Metrics buffer — **23 × `float32`**

Slot indices are Frozen; new telemetry appends. `metrics_len == 16` is the
minimum honored; **23 preferred**. The writer copies the largest prefix the
caller's buffer can hold (16, 17, 19, 20, 21, 22 or 23 slots), so an existing
caller that still passes 17 keeps working unchanged.

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

With slot 19 and slot 20 present the point accounting closes exactly, as
long as the output buffer was large enough to hold the whole field:

```
metrics[3] + metrics[4] + metrics[19] == metrics[1] + metrics[20]
```

That is, every point some path solved is either in the output buffer or was
dropped by the strain post-filter. Slot 20 was previously invisible: a caller
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
