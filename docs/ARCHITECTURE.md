# Engine architecture

API-style reference for the Semper Digital Image Correlation (DIC) engine and
its Android JNI adapter — module by module, with the invariants each one
depends on. Written for developers modifying, extending, or debugging the
pipeline.

For the derivations behind the code see [MATHEMATICS.md](MATHEMATICS.md); for
how it's verified, [TESTING.md](TESTING.md); for the Frozen/Stable surface,
[CONTRACT.md](CONTRACT.md).

```
Downstream Android app (private)
  SemperNativeLib.kt
       |
       v  adapters/android (JNI marshalling only)
this repository
  src/io + pipeline   (decode, AKAZE seeding/anchors/mesh/RGDIC, OpenMP)
       |
       v
  src/math + strain
  Image -> SubsetPrecomputer -> OptimizationEngine -> StrainCalculator
  Public API: include/semper/*.hpp  (no jni.h in src/; android/log.h only
  in src/util/log.hpp, behind SEMPER_ANDROID_LOG)
```

**Data flow for one analysis:** decode images → `Image` (float intensities +
gradients) → Hessian pre-pass → **anchor-lattice seeding** (an AKAZE + RANSAC
homography for the global motion, then IC-GN on a lattice of grid nodes seeded
from it) → Delaunay mesh →
for each remaining grid point: `SubsetPrecomputer` builds a `SubsetData` →
`OptimizationEngine.calculate_deformation` solves the 6-DOF warp →
displacement field → `StrainCalculator` → results buffer back through the
caller's adapter (JNI, C ABI or Python).

Seeding runs **after** the Hessian pre-pass on purpose: the anchors sit on grid
nodes, so they reuse the pooled Hessians and a node that converges well enough
is a final result for that node. Path A then skips it and Path B takes it as a
boundary seed. See [SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md) for why this
replaced AKAZE keypoints as mesh vertices (v0.2.2).

AKAZE came back in v0.6.0 in a different role: its matches no longer become
mesh vertices, they only fit the global model. Phase correlation (v0.2.2 to
v0.5.0) gave translation, and a log-polar similarity added rotation and scale,
but it had a 180° ambiguity and read anisotropic stretch as rotation. A
homography fitted to AKAZE matches covers rotation at any angle, scale,
stretch and shear. Matching runs at the coarsest of 0.25x, 0.5x and 1x that
gives enough RANSAC inliers. Each anchor starts from the displacement and
gradients the homography puts at it; a homography refitted to the converged
anchors then seeds the nodes the mesh does not reach. Without a model the
lattice searches for its own offsets, as v0.5.0 did without a phase lock.

"Converges well enough" is decided in two steps, not one. The lattice solve
records every node's outcome and writes nothing to the field; the universal
median test then runs over the whole lattice; and only then does
`publish_anchor_results` mark the survivors solved. A node that correlates
confidently onto the wrong speckle blob is dropped by the median test before
Path B can flood fill from it.

**Work reused across frames.** `ReferenceCache` keeps what depends only on
the reference and the grid: the node validity scan, the Hessian pool and the
reference's AKAZE keypoints and descriptors at each scale tried, plus the
scale the last solve settled on, which is tried first. The
second and later frames against one reference, with the same `rect`, `step`,
`subset_size` and mask, skip those steps. The cache is internal, and outputs
are bit-identical with or without it (`FullField.RefCache_*`). Steepest-descent
images are not cached: at `dim² × 6` floats per node they would cost too much
memory for a whole grid on a phone.

Package layout and contributor rules: [`README.md`](../README.md).

---

## Module: `include/semper/types.hpp`

Shared value types. All math is 32-bit `float` (`scalar_t`).

### `struct AnalysisResult`
The engine's answer for one subset.

| Field | Type | Meaning |
|---|---|---|
| `u, v` | float | Displacement of the subset center (px) |
| `ux, uy, vx, vy` | float | Displacement gradients (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y) |
| `status` | int | `0` = converged, non-zero = failed/aborted |
| `correlation_score` | float | Final ZNSSD / valid_pixels. **Lower is better**; `2.0` = worst |
| `iters` | int | ICGN iterations consumed |
| `invalid_ref_pixels` | int | Reference pixels masked by the ROI ("Ghost Wall") |

### `struct SubsetData`
Everything precomputed about one reference subset. Invariants:

- `x_offsets/y_offsets` (and `_f` float mirrors): relative coords in `[-dim/2, +dim/2]`, row-major, length `n = dim²`.
  **`dim` must be odd and at least 3** — the offsets are built symmetrically
  around the center pixel, so the subset holds `(2·(dim/2)+1)²` pixels, and for
  an even `dim` that is more than the `dim²` its buffers hold. The engine
  enforces this rather than trusting the caller: `run_full_field` and
  `IncrementalSession::init` return `-3` for any other `subset_size`
  (`validate_params`, which also clips the rect to the image), and the three
  `SubsetPrecomputer` functions return an uninitialized subset for one. The app
  also only offers odd sizes (see [Parameters from the app](#parameters-from-the-app)).
- `norm_ref_intensities[i] = (ref_intensities[i] − mean_intensity) / std_dev`.
- `H` = Σ sd·sdᵀ (6×6 Gauss-Newton Hessian), `H_inv` = its inverse.
- `steepest_descent_images[i]` = `[gx, gy, gx·x, gx·y, gy·x, gy·y] / std_dev` (Eigen 6×1, AoS).
- **`sdi_planes`** = SoA mirror of the above: plane *k* occupies `[k·n, (k+1)·n)`.
  **Must stay in sync with `steepest_descent_images`** — the SIMD fast path reads
  the planes, the masked slow path reads the Eigen vectors.
  (Guarded by `SubsetPrecomputer.SdiPlanesMirrorSteepestDescentImages` test.)
- `is_initialized == false` ⇒ subset touched the image boundary or lacked texture; do not solve it.

---

## Module: `include/semper/image.hpp` — class `Image`

Owns intensities + precomputed gradients for one image.

```cpp
Image(int_t w, int_t h, const uint8_t* raw_pixels);   // grayscale, row-major
void prepare_data(bool apply_dice_blur);              // gradients (+ optional 7-tap blur)
```

**Must call `prepare_data` before any gradient/interpolation consumer.**
Gradients are 4th-order central differences (exact for polynomials ≤ deg 4),
valid in the interior `[2, dim−3]`; the border ring is zero.

### Interpolation ladder

| Method | Kernel | Valid region | Out-of-range behavior |
|---|---|---|---|
| `interpolate_keys_fourth(x,y)` | 6×6 Keys 4th-order | `(2.5, w−3.5)` | demotes → bicubic boundary rule |
| `interpolate_bicubic(x,y)` | 4×4 Catmull-Rom | `[1, w−2)` | demotes → bilinear |
| `interpolate_bilinear(x,y)` | 2×2 | `[0, w−1.5)` | returns `0.0f` (kills pixel) |

All three are *interpolating* kernels: they reproduce pixel values exactly at
integer coordinates and reproduce linear ramps exactly at sub-pixel positions.
IC-GN and the Simplex never call them where bilinear demotion could happen:
their 4 px guard (`sample_warped_subset`) is stricter than every demotion
boundary, and a sample's validity is decided by that guard and the reference
mask, never by its value. A deformed intensity of `0` is black, not missing.
(Until v0.5.0 IC-GN dropped every sample not `> 0`, so saturated black speckle
and cubic undershoot next to it were treated as off-image.)

---

## Module: `include/semper/subset.hpp`

Static factory for `SubsetData`. Three entry points, one implementation:
`precompute_subset` is `compute_hessian_only` followed by
`precompute_subset_fast`, so the two routes give the same bytes (guarded,
bit for bit, by `FastPathMatchesFullPrecompute`; v0.5.0 — it was a third copy
of the arithmetic before).

```cpp
// Full precompute: compute_hessian_only, then precompute_subset_fast from it.
static void precompute_subset(SubsetData&, const Image& ref, int cx, int cy, int dim);

// Stats + Hessian only — for the batch pipeline's Hessian pool.
static CachedHessianData compute_hessian_only(const Image& ref, int cx, int cy, int dim);

// Rebuilds per-pixel data but reuses pooled Hessian/stats (~65% cheaper).
static void precompute_subset_fast(SubsetData&, const Image& ref,
                                   int cx, int cy, int dim,
                                   const CachedHessianData& cached);
```

Failure modes (→ `is_initialized = false` / `cached.valid = false`):
`dim` not odd and ≥ 3; subset crosses the image boundary; > 50% masked pixels;
ill-conditioned Hessian (`cond(H₂ₓ₂) > 1e12`, or the 6×6 rcond test
`hessian::well_conditioned`). `precompute_subset_fast` with an invalid pool
entry gives an uninitialized subset (no recomputation: it would fail again).

Every solve in the pipeline goes through `solve_node` (pool entry → subset →
`calculate_deformation` → Ghost-Wall rejection) on an engine from
`make_engine` (LM damping `tuning::kLmAlpha`, the caller's interpolator):
anchor lattice, Paths A, B and C, and incremental re-anchoring.

---

## Module: `include/semper/solver.hpp`

Per-thread solver object (owns scratch buffers — **do not share across OMP threads**).

```cpp
AnalysisResult calculate_deformation(const SubsetData& subset, const Image& def_img,
                                     float guess_u, float guess_v,
                                     float guess_ux, float guess_uy,
                                     float guess_vx, float guess_vy,
                                     InitializationMode init_mode);
```

### Initialization modes

| Mode | Pipeline |
|---|---|
| `INIT_AUTO_SEARCH` | coarse ±15 px SSD grid search → translation-only Simplex → full ICGN |
| `INIT_NO_SEARCH` | ICGN from the given 6-DOF guess; Simplex rescue if ICGN fails or score > 0.4 |
| `INIT_NO_SIMPLEX` | pure ICGN from the given guess — no rescue (fastest, needs good seeds) |

### Warp convention (the sign contract everything depends on)

The shape function maps reference offsets `(x, y)` from subset center `c` to
deformed-image coordinates:

```
X_def = cx + (1+ux)·x + uy·y + u
Y_def = cy + vx·x + (1+vy)·y + v
```

i.e. a material point at `c + p` displaces by `t + [[ux,uy],[vx,vy]]·p`.
Synthetic test data must be generated with this exact convention
(see `tests/framework/synthetic.h`).

### Solver internals

- **ICGN** (inverse-compositional Gauss-Newton): the Hessian is constant per
  subset (precomputed); each iteration interpolates the deformed subset,
  normalizes (ZNSSD → invariant to affine intensity changes `a·I + b`),
  accumulates the gradient via `simd::znssd_error_and_gradient`, then updates
  `W ← W · ΔW⁻¹`. Converges when `‖Δp‖ < 0.001`, max 50 iterations.
- **Sampling:** `sample_warped_subset` is the one place the deformed image is
  read through a warp, for IC-GN and the Simplex alike; it fills values and a
  per-pixel validity mask (inside the 4 px guard and not in the Ghost Wall).
- **Guards:** ≥ 90% of warped pixels must be valid; post-convergence
  texture check (sum of squared gradients of surviving pixels) rejects
  texture-less matches. An iteration that loses pixels steps with its own
  masked Hessian, and only if that Hessian passes the same 6×6 conditioning
  test as the reference one (`hessian::well_conditioned`: 1-norm rcond with the
  gradient rows scaled by the half-width, `tuning::kHessianMinRcond`); the
  next full iteration is back on the reference Hessian. A step with a
  non-invertible `ΔW`, or one that would make `W` non-finite (checked on the
  bits, since `-ffast-math` folds `isfinite`), ends the solve as a failure on
  the last good warp.
- **Levenberg-Marquardt option** (`lm_enabled`, `lm_alpha`): adds `α` to
  `H(0,0)`/`H(1,1)` only (translation DOFs), DICe-compatible. Off in a bare
  `OptimizationEngine`, but `make_engine` turns it on for every pipeline
  solve (anchors, Paths A, B and C, incremental re-anchoring).
- **Simplex** (Nelder-Mead on ZNSSD): derivative-free rescue for bad seeds;
  2-DOF or 6-DOF.

---

## Module: `include/semper/simd.hpp`

Portable SIMD hot loops via **OpenCV universal intrinsics** — one codepath
compiled to NEON on ARM and SSE on x86, scalar fallback elsewhere. Header-only;
no OpenCV linkage required.

```cpp
namespace Semper::simd {
  // Σ (vals[i] − mean)²
  float sum_sq_diff(const float* vals, size_t n, float mean);

  // Σ (ref[i] − (vals[i] − mean)·inv_std)²
  float znssd_sum(const float* vals, const float* ref, size_t n,
                  float mean, float inv_std);

  // Fused: returns the ZNSSD error AND accumulates the 6 gradient
  // projections dp[k] = Σ sdi_plane_k[i]·diff over SoA planes.
  float znssd_error_and_gradient(const float* vals, const float* ref,
                                 const float* sdi_planes, size_t n,
                                 float mean, float inv_std, float dp_out[6]);
}
```

Contracts (all test-guarded):
- Exact for any `n` (full vector blocks + scalar tail).
- `znssd_error_and_gradient`'s error term ≡ `znssd_sum` for identical input.
- `sdi_planes` layout: plane `k` at `[k·n, (k+1)·n)` — produced by `SubsetPrecomputer`.

---

## Module: `include/semper/strain.hpp`

```cpp
struct DisplacementField { int width, height, step;       // grid dims + px spacing
                           std::vector<float> u, v;        // px displacements
                           std::vector<bool> valid; };

static StrainField compute_vsg_strain (const DisplacementField&, int window_pixels);
static int reject_displacement_outliers(DisplacementField&);   // clears valid[], returns count
```

`pack_full_field_output` runs `reject_displacement_outliers` on the solved grid
before `compute_vsg_strain`. It is a normalized median test (MATHEMATICS §9.4)
on u and v over each point's 5x5 neighbours, repeated until nothing new fails.
Rejected points feed no strain window and are not written to the output, so they
count in metrics slot 2 (rejected = attempted − solved) like any other drop.

Outputs **Green-Lagrange strain**:
`exx = ∂u/∂x + ½((∂u/∂x)² + (∂v/∂x)²)`, etc.

| | VSG |
|---|---|
| Method | least-squares plane fit over a circular window |
| Exact for | any linear displacement field |
| Rejection | < 90% window fill or `rcond < 1e-12` → sentinel `−1000.0f` |
| Displacement outliers | normalized median test, 5x5, ε 0.1 px, threshold 3, repeated (`tuning::kOutlier*`) → point dropped before the fit |

**Consumer warning:** VSG failures are marked with `−1000.0f`. Check before
rendering/statistics.

---

## Module: `include/semper/incremental.hpp` — class `IncrementalSession`

Multi-frame DIC that can move its reference forward. The maths is in
[MATHEMATICS.md §9.5](MATHEMATICS.md#95-incremental-updated-reference-dic);
the surface is in [CONTRACT.md](CONTRACT.md).

```
init(frame 0)        -> validate params, ReferenceCache::set_from_gray (the
                        frame-0 grid scan runs with the first solve)
process_frame(t):
  k == 0             -> run_full_field_impl into the caller's buffer (bit-identical)
  k  > 0             -> run_full_field_impl on reference k's grid into scratch
                        -> node_field_from_capture (solved, ZNSSD >= 0, median test)
                        -> sample_increment at X0 + U(X0,k) for every tracked node
                           - nothing in reach: fit_increment (coast, unreported)
                           - coasted far: match_anchor before it is reported again
                        -> frame-0 ResultGrid -> pack_full_field_output (VSG, packing)
  every N frames,    -> make_anchor for clean nodes near a gap (from this frame)
  or on a 5% drop       warp_material_mask, ReferenceCache::adopt(prepared image)
```

- **Hand-off without rework.** `run_full_field_impl` takes an optional
  `FieldCapture`. It returns the solved grid and, on a frame that will become
  the next reference, the prepared deformed `Image` and its gray `cv::Mat`.
  `ReferenceCache::adopt` takes them over and stamps the mask, so the new
  reference is never decoded or prepared twice, and at most two `Image`s are
  alive at once.
- **State.** Per frame-0 node: the base displacement `U(X0, k)`, an `alive`
  flag and the distance it has coasted. Anchors (a ~1.5 KB grey patch each)
  only for nodes near a gap or coasting. Also the reference index, the frames
  since the last update, the points reported on the first frame against the
  reference, and the reference's mask and ROI.
- **Errors and cancel.** Every state change happens after a successful solve,
  so a negative return leaves the session as it was.
- **Between updates (N > 1)** every frame after the first one on a reference
  reuses that reference's cached grid scan, Hessian pool and AKAZE keypoints.
- **On an update** the keypoints AKAZE found on the deformed frame become the
  new reference's (those inside its padded ROI), and the AKAZE scale carries
  over, so a reference is never detected twice.
- **Metrics** are the increment solve's; frame-0 counts are in
  `IncrementalFrameInfo`.

Tunables: `tuning::kIncTaylorMaxCells`, `kIncMaskMaxSamplesPerPx`,
`kIncMaskExtrapCells`, `kIncCoastRadiusCells`, `kIncCoastMinNodes`,
`kIncCoastMaxTravelSubsets`, `kIncCoastTrustedTravelSubsets`,
`kIncReanchorMaxShiftFraction`, `kIncReanchorMinShiftPx`, `kIncAnchorMarginPx`,
`kIncEarlyUpdateDrop`, `kIncUpdateMinReported`.

---

## Module: `adapters/android/jni/SemperJNI.cpp`

JNI surface (see `SemperNativeLib.kt` for the Kotlin declarations).
Key behaviors:

- `initializeReference(...)` caches the decoded reference image in
  `g_cache` (guarded by `g_cache.mutex`) so batch runs decode once. A mask
  that does not decode clears the cache rather than being ignored.
- `computeFullFieldDirect(...)` runs the full pipeline (OpenMP for the
  pre-pass, `static`; the anchor lattice, `dynamic, 4`; Path A,
  `dynamic, 32`; Path B on its own `std::thread` workers), reliability-guided seed propagation, and
  progress callbacks marshalled through a `NewGlobalRef` + `AttachCurrentThread`.
- Results are written into a direct `ByteBuffer` (no copy back through JNI).
- `initializeIncremental(...)` / `computeIncrementalFrame(...)` /
  `releaseIncremental()` (v0.4.0) drive one `IncrementalSession` under the same
  mutex. `initializeIncremental` drops the fixed-reference cache, since only one
  of the two paths runs at a time. `computeIncrementalFrame` fills an optional
  `int[6]` with the frame info.
- `setCancelRequested(...)` sets the cooperative cancel flag. It takes **no
  lock** on purpose: `computeFullFieldDirect` holds the reference-cache mutex
  for the whole solve, so a cancel that waited for it could never arrive while
  the solve it means to stop is still running.

Threading contract: **one `OptimizationEngine` + one `SubsetData` per OMP
thread**; `SubsetData` buffers are reused across grid points via the
`precompute_subset_fast` pool.

### Cancellation — `include/semper/cancel.hpp`

A `CancelToken` per solve, passed down to every phase (`SolveContext::cancel`,
and to `solve_anchor_seeds`) and polled inside the point loops, so a cancel
lands within a point or two instead of at the end of the frame. One relaxed
atomic load per point is nothing against an ICGN solve. The process-wide
functions (`request_cancel` / `clear_cancel` / `cancel_requested`) act on the
token that the overloads without one pass down; nothing is shared between
solves that bring their own.

Where it is polled, and why there:

| Site | Shape |
|---|---|
| Hessian pre-pass, anchor lattice, Path A mesh execution | `if (cancel.requested()) continue;` — OpenMP forbids breaking out of a parallel `for`, so a cancel skips the remaining iterations |
| Path B workers | the flag in the condition-variable predicate, then `return` |
| Path B's CV wait | bounded (`wait_for`, 20 ms) — a worker waiting for the next round has no one to notify it of a cancel, so it re-checks on a timer |
| After Path A, after Path B | `return kCancelled` — a cancelled field is partial, so strain and packing are never run on it |

`run_full_field` returns `kCancelled` (**-99**), which is deliberately the same
value as `AnalysisRunCodes.ERROR_CANCELLED` on the Kotlin side: the app already
treats that code as "cancelled, discard", distinct from the `-2` / `-3` engine
failures that raise a dialog.

The flag is **sticky** — it survives the solve it stopped — so every run clears
it before starting. Both `AnalysisViewModel.cancelRequested` and
`VsgStudyRunner.cancelRequested` forward to it from their setters, and both
assign `false` at the top of a run.

### Parameters from the app

`StaticAnalysisActivity` ("Advanced parameters" in the setup wizard) is the
only producer of the three solve parameters. Each has a slider plus an
editable numeric field; both write the same slider value, which is what
`currentSubsetSize()` / `currentStepSize()` / `currentStrainWindow()` read.

| Parameter | JNI argument | Default | Range | Step |
|---|---|---|---|---|
| Subset size | `subsetSize` → `dim` | 41 | 15–101 | 2 (**odd only**) |
| Step size | `stepSize` → grid spacing | 5 | 1–30 | 1 |
| Strain window | `strainWindow` → `window_pixels` | 15 | 5–51 | 2 (**odd only**) |

Typed input is clamped to the range and snapped onto the slider's step grid
by `snapToSlider`, so the odd-`dim` invariant above holds for hand-entered
values too. Fields commit on IME "Done", on focus loss, and before the solve
starts — a pending edit can never reach the engine uncommitted.

---

## Package map

| Path | Role |
|---|---|
| `include/semper/` | Public headers (`types`, `image`, `subset`, `solver`, `strain`, `simd`, `pipeline`, `incremental`, `io`, `seeding`, `cancel`, `metrics`, `version`, `tuning`, `assert`, `guarded_call`, `semper_c.h`; tiers in CONTRACT.md) |
| `src/math/` | Image / SubsetPrecomputer / OptimizationEngine |
| `src/strain/` | StrainCalculator |
| `src/io/` | Platform-agnostic OpenCV decode |
| `src/pipeline/` | Anchor seeding (AKAZE homography + lattice) + full-field Path A/B/C + OpenMP; incremental session (composition, mask warp) |
| `adapters/android/` | JNI only → `libsemper_core.so` |
| `adapters/c/` | Stable C ABI → `libsemper_c` |
| `bindings/python/` | pybind11 module |
| `tests/` | Host unit / integration / DICe / perf + C smoke |

CMake targets: `semper_math`, `semper_pipeline`, `semper` (interface), `semper_android` (`OUTPUT_NAME semper_core`), `semper_c`, `_semper` (Python).

---

## Dependencies (git submodules, built from source)

Both libraries are **git submodules**, not vendored binaries. Run
`git submodule update --init --recursive` after cloning.

`.gitmodules` records only `path` and `url` — there is no `branch` or tag entry.
What pins the build is the **commit SHA the superproject records**, which is
what `git submodule update` checks out. Those SHAs currently sit exactly on the
upstream release tags below, so `git submodule status` prints the tag name, but
that is a property of the commits chosen, not a constraint the repo enforces:
bumping a submodule to an arbitrary commit would be accepted silently. Verify
with `git submodule status` rather than trusting this table.

| Dependency | Path | Recorded commit | Tag at that commit | How it's used |
|---|---|---|---|---|
| **Eigen** | `third_party/eigen` | `bc3b3987` | 5.0.1 | header-only; added via `include_directories` |
| **OpenCV** | `third_party/opencv` | `0654a42e` | 4.14.0 | **compiled from source** in the engine build |

**OpenCV modules compiled:** `core,imgproc,imgcodecs,features2d,calib3d,flann`
(`SEMPER_OPENCV_REQUIRED_MODULES` in `cmake/SemperOpenCV.cmake`, the modules
`semper_pipeline` links). OpenCV's `BUILD_LIST` is always that set plus
`SEMPER_OPENCV_BUILD_LIST`, a comma-separated cache option (empty by default)
for extra modules; it cannot remove a required one. Through v0.5.x
`SEMPER_OPENCV_BUILD_LIST` was the whole list, so build trees configured then
cache `core,imgproc,imgcodecs`; the merge keeps such a cache from dropping the
AKAZE modules. Configure also fails if a required module ends up without a
target (e.g. a cached `BUILD_opencv_calib3d=OFF`), instead of failing at link
time. `features2d`, `flann` and `calib3d` were dropped in v0.2.2 along with the
descriptor seeding front-end that used them (~29% of the aarch64
`libsemper_c.so`), and came back in v0.6.0 for the AKAZE global model: the
stripped x86-64 Python extension grows from 12.30 MB to 14.32 MB (+16%). Only
`src/pipeline/akaze_features.cpp` and `full_field_anchors.cpp` include their
headers; nothing under `include/semper/` does.

**OpenCV threads.** AKAZE, the matcher and RANSAC run inside a one-stripe
`cv::parallel_for_`, so OpenCV runs their own parallel loops inline and never
wakes its thread pool for them. On a few hundred pixels the pool saves under
1 ms, and its workers would still be spinning when the engine's OpenMP region
starts (2–3× the wall time of a small-ROI frame on a 20-thread host). The
Android adapter pins OpenCV to one thread in `JNI_OnLoad`.

**Sparse OpenCV checkout:** the full OpenCV repo includes `doc/`, `samples/`,
`data/`, and `apps/` that this project never builds. After submodule init, run
`scripts/sparse-opencv.sh` (or `scripts/sparse-opencv.ps1` on Windows) to drop
those trees from the worktree (~100+ MB). CMake only needs `modules/`,
`include/`, `3rdparty/`, `cmake/`, and the top-level `CMakeLists.txt`.

**OpenCV from-source integration** (`CMakeLists.txt`):

- `add_subdirectory(third_party/opencv …)` builds OpenCV as part of this
  CMake project. A curated `BUILD_LIST` (see above) compiles only the modules
  the engine uses — `core, imgproc, imgcodecs, features2d, calib3d, flann` — which keeps the
  build to a few minutes per ABI instead of tens.
- `BUILD_SHARED_LIBS OFF` → OpenCV is **statically linked** into
  `libsemper_core.so` / `libsemper_c` (which is why the Android `.so` is
  ~10–22 MB per ABI). We link the in-tree targets directly:
  `opencv_core opencv_imgproc opencv_imgcodecs opencv_features2d
  opencv_calib3d opencv_flann`.
- OpenCV's bundled 3rd-party image codecs (`zlib/png/jpeg/tiff/webp`) are built
  in-tree, so `cv::imdecode` / `cv::imwrite` work with **no system libraries**.
- OpenCV's in-tree targets don't export their header dirs, so the module
  `include/` paths and the generated-header dir (`opencv_modules.hpp`,
  `cvconfig.h` in `CMAKE_BINARY_DIR`) are added explicitly.
- `OPENCV_PYTHON_SKIP_DETECTION ON` disables OpenCV's host-Python detection for
  the C++/JNI build (the optional `bindings/python` path uses pybind11
  separately).
- OpenCV compiles with its **own** default flags: `add_subdirectory` runs before
  the engine's aggressive `-O3 -ffast-math` flags are set, so those apply
  to the DIC code only, not to OpenCV.

The host test build ([TESTING.md](TESTING.md)) consumes OpenCV's universal-intrinsics
header from the submodule source; the two generated headers it needs are
committed under `tests/shim/opencv2/` so host builds need no OpenCV
configure.

## Build & ABI strategy

- Downstream Android apps configure with `-DSEMPER_ANDROID=ON` (typical ABI
  **arm64-v8a**; debug often adds **x86_64** for emulators).
- OpenCV is compiled once per ABI into the shared lib; NDK `.cxx/` caches the
  from-source build across incremental compiles.
- SIMD portability comes from `include/semper/simd.hpp`; there is **no**
  architecture-conditional code left in the engine (`#if __aarch64__` was
  removed — do not reintroduce it; extend the kernels instead).
- Production flags: `-O3 -ffast-math`, plus OpenMP on host when CMake finds
  it. Without it (MSVC, Apple Clang without libomp) the `#pragma omp` regions
  run on the calling thread: thread indices go through `util::omp_thread()`
  (`src/util/omp.hpp`), never `omp.h` directly. CI's `serial-build` job
  builds the suite that way (`DIC_OPENMP=OFF`, warnings fatal) and runs it.
  Android also uses
  `-flto` on the pipeline/JNI targets. Note `-ffast-math` disables NaN
  semantics: Inf/NaN checks read the bits (`util::finite_bits`,
  `src/util/finite.hpp`), never `std::isfinite` or `!(x > eps)`, and the
  code uses explicit sentinels (`-1.0f`, `-1000.0f`) instead of
  `std::isnan`. Preserve this convention. Host shared-lib builds intentionally
  omit `-flto` (MinGW LTO + DLL is unstable).
- 16 KB page alignment: `-Wl,-z,max-page-size=16384` on the Android shared lib for
  Android 15+ devices.

### Platform support, and what CI actually gates

| platform | status | gated by |
|---|---|---|
| Linux x86_64 (GCC, Clang) | supported | every CI job |
| Android arm64-v8a | shipping target | `android-ndk` compiles the JNI adapter (no device run); the app builds the APK |
| Windows x86_64 (MinGW-w64) | **works, but ungated** | nothing |
| aarch64 Linux cross | measurement only | nothing (see the toolchain header) |
| MSVC | headers declare `dllexport`/`dllimport`, never built | nothing |

Windows is a first-class *development* platform here — the engine, the C SDK
and the full test suite all build and pass under MinGW-w64 — but no workflow
builds it, so nothing stops a Windows break from landing. That is not
hypothetical: `tests/integration/test_seeding_bench.cpp` shipped with an
unguarded `<sys/resource.h>`, which broke not the benchmark but the whole
`dic_tests` binary, and it survived review and a green CI run because CI is
Linux.

A Windows job is the real fix and is not currently justified: it would have to
build the vendored OpenCV submodule on a Windows runner, which is the slowest
part of the build and would dominate CI time for a platform nothing ships on.
Until that changes, the rule is procedural: **a change that touches system
headers, process/OS APIs or paths must be compiled on both hosts before
merge**, and cross-host numeric results are covered separately by the
tolerances documented in [TESTING.md](TESTING.md).