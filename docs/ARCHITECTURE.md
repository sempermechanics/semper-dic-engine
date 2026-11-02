# Engine architecture

A module-by-module reference for developers who change the Semper Digital
Image Correlation (DIC) engine: what each module does, the invariants it
relies on, and how the modules connect. Derivations are in
[MATHEMATICS.md](MATHEMATICS.md); the public API, return codes and Frozen
formats in [CONTRACT.md](CONTRACT.md); tests and CI in
[TESTING.md](TESTING.md); build commands and layout in
[README.md](../README.md); history in [CHANGELOG.md](../CHANGELOG.md).

## Data flow

1. Decode: the adapter makes an 8-bit gray image and mask (`io.hpp`).
2. Reference: `ReferenceCache::set_from_gray` builds an `Image` (float
   intensities and gradients) and stamps the Ghost Wall into masked pixels.
3. Scan: `run_full_field` validates, prepares the deformed `Image`, and skips
   grid nodes whose subset touches the border reserve or the mask.
4. Hessian pre-pass: one `CachedHessianData` per node.
5. Seeding: an AKAZE + RANSAC homography gives the global motion; IC-GN solves
   a lattice of about 256 nodes from it; a median test drops blunders.
6. Mesh guess: a Delaunay mesh over the kept anchors guesses each covered node;
   the homography covers the rest.
7. Solve: Path C seeds when nothing else did, Path A solves the guessed nodes,
   Path B flood-fills the rest.
8. Pack: displacement median test, VSG strain, 8 floats per point, metrics.

Every grid-node solve runs `solve_node` (pool entry → `SubsetData` →
`calculate_deformation` → Ghost Wall rejection) on a `make_engine` engine: LM
damping (`tuning::kLmAlpha`) and the caller's interpolator. So does
incremental re-anchoring (`match_anchor`), without `solve_node`.

## Module: `include/semper/types.hpp`

All math is 32-bit `float` (`scalar_t`).

### `struct AnalysisResult`

| Field | Meaning |
|---|---|
| `u, v` | Displacement of the subset centre (px) |
| `ux, uy, vx, vy` | ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y |
| `status` | `0` converged, else failed |
| `correlation_score` | ZNSSD / valid pixels; lower is better, `2.0` on a rejection |
| `iters` | IC-GN iterations of the final run |
| `invalid_ref_pixels` | Reference pixels in the Ghost Wall |
| `trigger_iters` | Iterations of the run that triggered a Simplex rescue (else `iters`) |

### `struct SubsetData`

- `dim` is odd and ≥ 3: the subset holds `(2·(dim/2)+1)²` pixels, more than
  its `dim²` buffers for an even `dim`. `validate_params` returns `-3` for
  other sizes; `SubsetPrecomputer` leaves such a subset uninitialized. The app
  offers only odd sizes ([Parameters from the app](#parameters-from-the-app)).
- `dim` defaults to `0`; buffers are resized only when it changes.
- `x_offsets` / `y_offsets` (and `_f` mirrors): `[-dim/2, +dim/2]`, row-major,
  `n = dim²`.
- `norm_ref_intensities[i] = (ref_intensities[i] − mean_intensity) / std_dev`.
- `H` = Σ sd·sdᵀ, kept raw for LM damping; `H_inv` its inverse.
- `steepest_descent_images[i]` = `[gx, gy, gx·x, gx·y, gy·x, gy·y] / std_dev`
  (Eigen, AoS); `gx_vec` / `gy_vec` keep raw gradients for the texture check.
- `sdi_planes`: SoA mirror, plane *k* at `[k·n, (k+1)·n)`. Keep it in sync:
  the SIMD path reads the planes, the masked path the Eigen vectors
  (`SubsetPrecomputer.SdiPlanesMirrorSteepestDescentImages`).
- `is_initialized == false`: do not solve it.

## Module: `include/semper/image.hpp` — class `Image`

```cpp
Image(int_t w, int_t h, const uint8_t* raw_pixels);   // 8-bit gray, row-major
void prepare_data(bool apply_dice_blur);              // gradients (+ optional 7×7 Gaussian blur)
```

The constructor throws (`std::length_error`, `std::invalid_argument`) if
`w·h` overflows `int` or the pixels are null. Call `prepare_data` before use
(the engine passes `false`). Gradients are 4th-order central differences,
exact up to degree 4, for `x` in `[2, w−3]` and `y` in `[2, h−3]`; the border
is zero.

### Interpolation ladder

| Method | Kernel | Valid in x (y with h) | Outside |
|---|---|---|---|
| `interpolate_keys_fourth` | 6×6 Keys 4th-order | `(2.5, w−3.5)` | bilinear |
| `interpolate_bicubic` | 4×4 Catmull-Rom | `[1, w−2)` | bilinear |
| `interpolate_bilinear` | 2×2 | `[0, w−1.5)` | `0.0f` |

Each test reads "inside, else demote", so a NaN coordinate is rejected, never
cast to an index; the `_x4` variants skip it and rely on the caller's guard.
All three reproduce pixel values at integers and linear ramps exactly.

IC-GN, the Simplex and the coarse search sample only inside a 4 px guard
(`inside_guard`), stricter than every demotion boundary. Validity comes from
that guard and the mask, never the value: `0` is black, not missing.

## Module: `include/semper/subset.hpp`

`compute_hessian_only` fills one pool entry (`CachedHessianData`: stats, `H`,
`H_inv`) without heap allocation. `precompute_subset_fast` rebuilds the
per-pixel data from an entry (~65% cheaper). `precompute_subset` runs both, so
both routes give the same bytes (`FastPathMatchesFullPrecompute`).

A subset fails when `dim` is not odd and ≥ 3, it crosses the image edge,
under 50% of it is unmasked, or its Hessian is ill-conditioned:
`cond(H₂ₓ₂) > 1e12`, or `hessian::well_conditioned` fails (1-norm rcond,
gradient rows scaled by the half-width, `kHessianMinRcond`). An invalid entry
gives an uninitialized subset without recomputing. The pool is keyed by node,
not `dim`, so `precompute_subset_fast` checks `dim` again.

## Module: `include/semper/solver.hpp`

`OptimizationEngine::calculate_deformation(subset, def_img, 6-DOF guess,
InitializationMode)` solves one subset. It owns scratch buffers: never share
one across threads.

### Initialization modes

| Mode | What runs | Used by |
|---|---|---|
| `INIT_AUTO_SEARCH` | If the guess is `(0, 0)`, a ZNSSD scan of ±15 px in 2 px steps over every 9th pixel (valid samples only); then a translation-only Simplex and IC-GN. No rescue. | Path C; anchor lattice without a model |
| `INIT_NO_SEARCH` | IC-GN; if it fails or scores above `kCorrSimplexTrigger` (0.4), a 6-DOF Simplex and IC-GN again, keeping the better run. | Path A in a triangle; Path B; `match_anchor` |
| `INIT_NO_SIMPLEX` | IC-GN only. | Anchor lattice with a model; Path A outside triangles |

### Warp convention

The sign contract everything depends on. Offsets `(x, y)` from the subset
centre `c` map to:

```
X_def = cx + (1+ux)·x + uy·y + u
Y_def = cy + vx·x + (1+vy)·y + v
```

A material point at `c + p` moves by `t + [[ux,uy],[vx,vy]]·p`. Synthetic test
data must use this convention (`tests/framework/synthetic.h`). Positions are
summed in double and rounded to float once.

### Solver internals

- IC-GN keeps the subset's Hessian fixed. Each iteration samples, normalizes
  (ZNSSD, invariant to `a·I + b`), accumulates with
  `simd::znssd_error_and_gradient` and updates `W ← W · ΔW⁻¹`, until
  `‖Δp‖ < 0.001` or `kIcgnMaxIter` (50) iterations.
- `sample_warped_subset` is the only reader of the deformed image through a
  warp. Its validity mask means inside the 4 px guard and off the Ghost Wall.
- Rejected: under `kMinValidPixelFraction` (90%) valid pixels; an iteration
  that lost pixels and whose masked Hessian fails the 2×2 or 6×6 test; a singular
  `ΔW` or non-finite `W` (keeps the last good warp); after convergence, a zero
  gradient sum in either axis or a non-finite score. The next full iteration
  after a masked one returns to the reference Hessian.
- LM (`lm_enabled`, `lm_alpha`) adds `α` to `H(0,0)` and `H(1,1)` only, as in
  DICe. Off in a bare engine, on via `make_engine`.
- Simplex: Nelder-Mead on ZNSSD, 2- or 6-DOF, ≤ 100 iterations. A warp
  sampling under 90% scores `2.0`; a final score above `kSimplexFailScore`
  (0.1) fails.

## Module: `include/semper/simd.hpp`

Header-only loops on OpenCV universal intrinsics (NEON, SSE or scalar).

| Function | Computes |
|---|---|
| `sum_sq_diff(vals, n, mean)` | Σ (vals[i] − mean)² |
| `znssd_sum(vals, ref, n, mean, inv_std)` | Σ (ref[i] − (vals[i] − mean)·inv_std)² |
| `znssd_error_and_gradient(…, sdi_planes, …, dp_out)` | That error, plus `dp[k] = Σ sdi_plane_k[i]·diff` |

Tested contracts: any `n` (vector blocks, then a scalar tail); the fused error
equals `znssd_sum`. Multiply, then add, never `cv::v_fma`: NEON fuses it and
SSE2 cannot, so arm64 and x86-64 would disagree.

## Module: `include/semper/strain.hpp`

`StrainCalculator` works on a `DisplacementField` (`width`, `height`, `step`,
`u`, `v`, `valid`). `pack_full_field_output` runs
`reject_displacement_outliers`, then `compute_vsg_strain`. Before solving,
`run_full_field` returns `kBadStrainWindow` (`-4`) if `vsg_window_node_count`
is under 3.

| | |
|---|---|
| Outlier test | Normalized median test on u and v ([MATHEMATICS §9.4](MATHEMATICS.md)): 5×5, ε 0.1 px, threshold 3, ≥ 3 neighbours, until stable or 5 passes (`kOutlier*`) |
| VSG fit | Least-squares plane over a circular window; exact for linear fields |
| VSG rejection | < 90% window fill, < 3 points or `rcond < 1e-12` → `−1000.0f` |
| Output | Green-Lagrange, e.g. `exx = ∂u/∂x + ½((∂u/∂x)² + (∂v/∂x)²)`; `exy` is the tensor component |

Outliers feed no window, are not written, and count in metrics slots 2 and 23.
A field the fit cannot read (zero step, short vectors) is all `−1000.0f`;
consumers must check for it.

## Module: `include/semper/io.hpp`

The adapters decode; the engine accepts only 8-bit single-channel images and
masks. Rules: [CONTRACT.md](CONTRACT.md) ("Image I/O"). The engine relies on:

- A mask value below 128 excludes a pixel, in the grid scan, the Ghost Wall
  and the AKAZE match filter alike.
- `to_gray8` maps a non-8-bit image linearly from its 0.1th–99.9th percentile
  onto 0..255; `to_mask8` makes a non-8-bit mask boolean. `decode_gray` and
  `decode_mask` differ only in which they call.

## Module: `include/semper/pipeline.hpp` — `run_full_field`

`full_field_solver.cpp` sequences the phases.

### Phases

| Phase | Function (file) | Notes |
|---|---|---|
| Checks | `validate_params` (`full_field_solver.cpp`) | Clips the rect to the image without moving the lattice |
| Grid scan | `run_full_field_impl` | Marks `solved` (skipped) nodes within `subset_size/2 + 19` px of the border (4 px guard + 15 px reserve) or with a masked subset pixel |
| Hessian pre-pass | `compute_hessian_only` | OpenMP `static` |
| Seeding | `solve_anchor_seeds`, then `publish_anchor_results` (`full_field_anchors.cpp`, `akaze_features.cpp`) | Publishing is a separate, serial step |
| Mesh guess | `build_mesh_guess_field` (`full_field_mesh.cpp`) | |
| Path C | `run_path_c` (`full_field_path_c.cpp`) | Only without a seed |
| Path A | `run_path_a` (`full_field_path_a.cpp`) | OpenMP `dynamic, 32` |
| Path B | `run_path_b` (`full_field_path_b.cpp`) | Own `std::thread` workers |
| Pack | `pack_full_field_output`, `fill_engine_metrics` (`full_field_solver_stats.cpp`) | Layouts: CONTRACT §A.4 |
| Debug export | `full_field_debug_export.cpp` | When `cache.debug_dir` is set (one solve) |

Progress: a ticker thread reports 10–90%; the calling thread sends `100` after
it joins.

### Seeding: AKAZE model and anchor lattice

The global model: AKAZE matches the two frames over the ROI padded by
`kAkazePadPx / scale` (40 px at full scale), filtered by Lowe's ratio (0.75)
and the mask, then a RANSAC homography (3 px at the detection scale). The
coarsest of 0.25×, 0.5×, 1× with 25 inliers (`kAkazeGoodInliers`) wins, else
the best with 10 (`kAkazeMinInliers`). The search starts where the last
completed solve on this reference settled, trying coarser scales only if
nothing from there fits.

The lattice: about `kAnchorTarget` (256) nodes, per-axis stride, last row and
column included (`plan_anchor_lattice`), solved by IC-GN on the pool (OpenMP
`dynamic, 4`). An anchor starts from the homography at its position if the
model moves an ROI corner over `kSimilarityMinCornerPx` (4 px) beyond its
centre translation; otherwise from that translation.

1. ZNSSD ≤ `kAnchorAcceptScore` (0.40) makes a vertex candidate; ≤
   `kCorrAccept` (0.15) also makes it publishable.
2. `mark_kept_anchors` keeps a vertex within `kAnchorMedianTol` (1 px) of the
   median of its vertex 8-neighbours, on the residual from the model; with
   fewer than three such neighbours it is not kept.
3. `publish_anchor_results` marks kept, publishable anchors solved (Path A
   skips them, Path B seeds from them). As this follows the median test, a
   confident match on the wrong speckle never seeds the flood fill.

A model whose lattice is not `FULL`, or keeps under `kSimilarityKeepFraction`
(90%), is rejected and the lattice re-solved from the translation. With ≥ 8
kept anchors (`kAnchorRefitMinPoints`) the homography is refitted to them
(RANSAC 1 px); a rejected model without a refit seeds nothing outside the
mesh. The mesh is `FULL` with ≥ 10 kept vertices, ≥ 60% of attempts kept and
≥ 30% hull coverage; `SPARSE` with ≥ 3 vertices and ≥ 5%; else `NONE`.

Seeding follows the pre-pass so anchors reuse the pool. Anchors, not
keypoints, are the vertices: guess gradients err by vertex error over edge
length, about 1 px for keypoints against 0.01 px for anchors
([SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md)). `phase_correlate.cpp`, the
front end replaced in 0.6.0, remains for `seeding.hpp` but is unused.

### Mesh, Path A, Path B, Path C

- Mesh guess: `cv::Subdiv2D` triangulates the kept anchors; a node in a
  triangle takes its affine map (`kMeshInTriangle`). With a `FULL` mesh, nodes
  within 3 steps take the nearest triangle's (`kMeshExtrapolated`). The rest
  take the homography (`kMeshHomography`); the field is then smoothed over 5×5.
- Path A solves every guessed, unsolved node and accepts converged ones with
  ZNSSD ≤ `kCorrAccept`. Outside triangles it leaves failures to Path B.
- Path B floods in rounds. A serial claim pops the `kPathBBatchNodes` (32)
  lowest-ZNSSD nodes (ties by row, then column) and gives each unclaimed
  4-neighbour to the first that touches it, guessed by first-order expansion.
  Workers solve in parallel; a serial commit applies rounds in order, two in
  flight, so the field does not depend on timing or thread count.
- Path B seeds: solved nodes next to unsolved ones; else up to 5 kept anchors
  (smallest whole-pixel displacement, then nearest the centre); else, or if
  those fail, the Path C seed.
- Path C runs when the mesh is `NONE`, no kept anchor is publishable and under
  3 vertices were kept. It tries up to `kPathCMaxCandidates` (9) nodes nearest
  the centre, starting from the global shift. If all fail, the solve returns
  `kSeedFailed` without an AKAZE lock, or goes on to Path B with one.

### Reference cache

`ReferenceCache` holds the prepared `Image`, its gray `cv::Mat`, `debug_dir`,
`precompute` and a `mutex`, which JNI holds around a solve; it does not make
the methods thread-safe.

- `set_from_gray` resets first, refuses non-8-bit input or over `INT_MAX`
  pixels, and installs the image only once built: refusal or throw leaves the
  cache empty.
- `adopt` / `internal::adopt_reference` (incremental DIC) takes a prepared
  image. All throwing steps run first, so a throw keeps the outgoing reference.
  It carries over the AKAZE starting scale and features.
- The Ghost Wall stamps `kGhostWallValue` (−10) into masked pixels after
  `prepare_data`, so edge gradients use real intensities. Subset code skips
  pixels below `kGhostWallThreshold` (−5); `reject_if_ghosted` fails a subset
  over 5% of them.

| `RefPrecompute` part | Keyed on | Notes |
|---|---|---|
| Grid scan + Hessian pool | Clipped rect, `step`, `subset_size`, mask (exact) | Valid only after a complete, uncancelled pre-pass |
| AKAZE keypoints, per scale | ROI rect | Detected once per scale |
| `akaze_first_level` | — | Moved only by a completed solve |

The scan, pool and keypoints are bit-identical with or without the cache
(`FullField.RefCache_*`). The starting scale is not: a solve may take its model
from a finer scale than a cold one, the only way a solve depends on earlier
ones (CONTRACT, "Reference cache"). Steepest-descent images are not cached:
`dim² × 6` floats per node is too much for a phone.

### Errors and threads

- OpenMP regions and Path B use `std::thread::hardware_concurrency()` workers,
  each with its own engine and `SubsetData`. Both overwrite all they read, so
  the worker cannot change a result.
- An exception must not leave an OpenMP region: each catches into a
  `RegionErrors`, skips the remaining iterations and rethrows after the join.
  A throwing Path B worker stops the flood; it rethrows once all join. The
  adapters' `guarded_call` maps it to their error value.

## Module: `include/semper/cancel.hpp` — cancellation

API and token semantics are in CONTRACT. The mechanism:

- Window: each solve entry calls `CancelToken::enter`, which clears an old
  request unless an adapter opened a `CancelScope`. JNI and Python open it once
  they hold their lock; the C ABI before decoding.
- Latch: `run_full_field_impl` polls through `SolveCancel`; once a poll sees a
  request, the solve keeps seeing it even if the token is cleared (JNI
  `setCancelRequested(false)` is lock-free). Otherwise a pre-pass could mark a
  holed pool complete, or Paths A and B return a partial field as success.
- A poll is one relaxed load per point; a late one costs a point, not a result.

| Poll site | Shape |
|---|---|
| Pre-pass, anchor lattice, Path A | `continue` (no `break` out of an OpenMP `for`) |
| After pre-pass, seeding, Path A, Path B; Path C loop | `return kCancelled`: no strain or packing on a partial field |
| Path B seeds and workers | `return`; the flag is in the CV predicate |
| Path B's CV wait | `wait_for` `kCancelPollMs` (20 ms): nobody notifies a parked worker |

A cancelled solve leaves the pool stale and the AKAZE scale unmoved.
`kCancelled` (`-99`) equals Kotlin's `AnalysisRunCodes.ERROR_CANCELLED`, which
the app discards silently, unlike the `-2` / `-3` dialogs. In the app,
`AnalysisViewModel.cancelRequested` and `VsgStudyRunner.cancelRequested`
forward to `setCancelRequested` and assign `false` at the top of a run.

## Module: `include/semper/incremental.hpp` — class `IncrementalSession`

Maths: [MATHEMATICS.md §9.5](MATHEMATICS.md#95-incremental-updated-reference-dic).

```
init(frame 0)      -> validate_params, material mask, set_from_gray
process_frame(t):
  reference 0      -> run_full_field_impl into the caller's buffer (bit-identical)
  reference k > 0  -> run_full_field_impl on reference k's grid into scratch
                      -> node_field_from_capture (solved, ZNSSD >= 0, median test)
                      -> sample_increment at X0 + U(X0,k) per tracked node
                         (no sample: fit_increment, coast; coasted far: match_anchor)
                      -> frame-0 ResultGrid -> pack_full_field_output
  update due       -> make_anchor near gaps; prepare_reference (warp_material_mask,
                      new grid rect); adopt_reference; commit
```

- Updates: every `N` frames, or early (N > 1) when a frame reports
  `kIncEarlyUpdateDrop` (5%) fewer points than the reference's first frame.
  Never onto a frame under `kIncUpdateMinReported` (50%) of that count (else
  of the reference's own count, else of the frame-0 nodes); the update waits.
- Hand-off: `FieldCapture` (filled after Path B, before packing) also keeps,
  for N > 0, the deformed `Image`, its gray and AKAZE features, as any frame
  may become the reference. Nothing is prepared twice; at most two full-frame
  `Image`s exist.
- State per frame-0 node: `U(X0, k)`, `alive`, coasted distance. Anchors exist
  only near gaps or while coasting: a gray patch of side
  `subset_size + 2·kIncAnchorMarginPx` (3.7 KB at subset 41).
- Coasting: a node with no solved node within `kIncTaylorMaxCells` moves,
  unreported, on a weighted plane through solved increments within
  `kIncCoastRadiusCells`. It is dropped past `kIncCoastMaxTravelSubsets`; past
  `kIncCoastTrustedTravelSubsets` it is reported only once `match_anchor`
  confirms it.
- Errors: an update builds its state on copies; `adopt_reference` is the last
  throwing step; the commit only moves and swaps. A negative return or
  exception leaves the session unchanged. Failed frames, `kSeedFailed`
  included, are not counted, so indices then lag the caller's by one.
- A moved reference with no solvable node (`-2`) drops the nodes; the frame
  returns 0 points and zeroed metrics.
- Frames between updates reuse the reference's scan, pool and keypoints. At an
  update the deformed frame's AKAZE features replace a fresh detection at their
  scale if they cover ≥ `kAkazeReuseCover` (90%) of the new padded ROI.

Other tunables: `kIncMaskMaxSamplesPerPx`, `kIncMaskExtrapCells`,
`kIncCoastMinNodes`, `kIncReanchorMaxShiftFraction`, `kIncReanchorMinShiftPx`.

## Module: `adapters/android/jni/SemperJNI.cpp`

Exports and behaviour: CONTRACT §A.3. Internals:

- `g_cache` and at most one `g_session` share `g_cache.mutex`, held for a
  whole solve; `setCancelRequested` takes no lock, so it can stop that solve.
- `decode_optional`: an array that does not decode is `-3`, never "no mask";
  masks use `decode_mask`.
- Results go into the direct `ByteBuffer`, bounded by its capacity.
- `ProgressBridge` holds a `NewGlobalRef` and attaches the progress thread
  (`AttachCurrentThread`). A thread-local flag makes a locking export called
  from the callback fail instead of deadlocking.
- Every export runs through `guarded_call`; no JNI call runs with a Java
  exception pending.

### Parameters from the app

`StaticAnalysisActivity` ("Advanced parameters" in the setup wizard) alone
sets the solve parameters. A slider and a field write one value, read by
`currentSubsetSize()` / `currentStepSize()` / `currentStrainWindow()`.

| Parameter | JNI argument | Default | Range | Step |
|---|---|---|---|---|
| Subset size | `subsetSize` → `dim` | 41 | 15–101 | 2 (odd only) |
| Step size | `stepSize` → grid spacing | 5 | 1–30 | 1 |
| Strain window | `strainWindow` → `window_pixels` | 15 | 5–51 | 2 (odd only) |

`snapToSlider` clamps and snaps typed input, so `dim` stays odd. Fields commit
on IME "Done", on focus loss and before the solve starts.

## Module: `adapters/c/` and `bindings/python/`

Behaviour is in CONTRACT. Each C ABI handle owns a cache or session and a
`CancelToken`; a `busy` flag (`HandleCall`) admits one call per handle, cancel
excepted. Python's `run_interruptible` solves on a worker thread while the
caller waits without the GIL and checks signals every 50 ms, so Ctrl-C
cancels. An exception from `progress` cancels the solve and is raised after.

## Package map

Layout: README. In `src/`: `math/` (`Image`, `SubsetPrecomputer`,
`OptimizationEngine`, `hessian.hpp`), `strain/`, `io/` (`image_codec.cpp`),
`pipeline/` (the phases plus `reference_cache.cpp`, `cancel.cpp`,
`incremental_*.cpp`, `phase_correlate.cpp`) and `util/` (`finite.hpp`,
`omp.hpp`, `log.hpp`, `median.hpp`).

`src/` never includes `jni.h`; `android/log.h` appears only in
`src/util/log.hpp`, behind `SEMPER_ANDROID_LOG`. The JNI adapter includes only
`include/`, so it can move to the app repository.

## Dependencies

| Dependency | Path | Commit | Release | Use |
|---|---|---|---|---|
| Eigen | `third_party/eigen` | `bc3b3987` | 5.0.1 | Header-only, `SYSTEM` include |
| OpenCV | `third_party/opencv` | `0654a42e` | 4.14.0 | Built from source, static |

- Pinning: the recorded SHA pins the build. `.gitmodules` has `path`, `url`
  and (OpenCV) `ignore = untracked`, no branch or tag, so any bump goes in
  silently. `git submodule status` shows a tag only if the clone has tags.
- Modules: `core imgproc imgcodecs features2d calib3d flann`
  (`SEMPER_OPENCV_REQUIRED_MODULES`; keep `cmake/SemperOpenCV.cmake` and
  `CMakeLists.txt` in sync). `SEMPER_OPENCV_BUILD_LIST` only adds; a missing
  required target fails configure. `features2d`, `calib3d` and `flann` serve
  only AKAZE (`akaze_features.cpp`, `full_field_anchors.cpp`), never
  `include/semper/`. Rationale and size: CHANGELOG 0.6.0, 0.6.1.
- Build: `add_subdirectory(… EXCLUDE_FROM_ALL)`, `BUILD_SHARED_LIBS OFF`,
  bundled codecs. In-tree targets export no headers, so their `include/` dirs
  and `CMAKE_BINARY_DIR` (`opencv_modules.hpp`, `cvconfig.h`) are added
  explicitly. Engine flags are per target.
- Sparse checkout: `scripts/sparse-opencv.sh` (`.ps1` on Windows) keeps
  `modules include 3rdparty cmake platforms hal`; it and
  `cmake/SemperOpenCV.cmake` plant the stub `doc/` and `data/` CMake files
  OpenCV always adds.

### OpenCV threads

AKAZE detection, matching and RANSAC run inside a one-stripe
`cv::parallel_for_` (`without_opencv_pool`), so OpenCV runs their loops inline
and never wakes its pool. The pool saves under 1 ms on a small ROI, but its
spinning workers slowed the next OpenMP region, up to 2–3× a small-ROI frame's
wall time on a host. The Android adapter pins OpenCV to one thread in
`JNI_OnLoad`.

## Build & ABI strategy

- Android: `-DSEMPER_ANDROID=ON`, typically `arm64-v8a` (debug adds
  `x86_64`). `SEMPER_FORCE_RELEASE` keeps OpenCV off `-O0` under AGP Debug.
  OpenCV compiles once per ABI, cached in the NDK's `.cxx/`. The library links
  with `-Wl,-z,max-page-size=16384` for Android 15+ 16 KB pages.
- SIMD goes only through `simd.hpp`; never reintroduce `#if __aarch64__`.

### Compiler flags

Non-MSVC; MSVC keeps its defaults.

| Target | Flags |
|---|---|
| `semper_math` | `-O3 -ffast-math -fno-finite-math-only -ffp-contract=off -fPIC -fvisibility=hidden` |
| `semper_pipeline` | Same plus `-fexceptions`; Android adds `-flto -fopenmp`, host links OpenMP when found |
| `semper_android` | As the Android pipeline; links `-s -flto -Wl,-z,max-page-size=16384 -fopenmp -static-openmp -static-libstdc++` |
| `semper_c` | `-O3 -fPIC -fvisibility=hidden`; `semper_c.map` limits ELF exports |

- `-ffp-contract=off`: a multiply-add rounds alike on aarch64, where compilers
  fuse it, and on baseline x86-64.
- `-fno-finite-math-only`: NaN and Inf checks survive `-ffast-math`. The code
  still checks finiteness on the bits (`util::finite_bits`) and uses sentinels
  (`-1.0f` no correlation, `-1000.0f` failed strain), never NaN. Keep it so.
- Host shared libraries omit `-flto` (a MinGW DLL from an LTO archive fails to
  load). `SEMPER_SANITIZER` drops LTO and adds `float-cast-overflow` to
  `undefined`.

### OpenMP

Host builds link OpenMP when found; `SEMPER_REQUIRE_OPENMP=ON` (macOS wheels)
fails configure otherwise. Without it (MSVC, Apple Clang without libomp) every
`#pragma omp` region runs on the calling thread, so thread indices come from
`util::omp_thread()`, never `omp.h`. CI's `serial-build` job tests that build.

### Platform support and CI gates

What each CI job gates: [TESTING.md](TESTING.md) ("CI jobs").

| Platform | Status |
|---|---|
| Linux x86-64 (GCC; Clang 18 for TSan) | Supported, CI-gated |
| Android arm64-v8a | Shipping; CI compiles it (no device run), the app builds the APK |
| macOS; Windows with MSVC | Python wheels from the manual `wheels` workflow; not gated |
| Windows, MinGW-w64 | Engine, C SDK and tests build and pass; not gated |
| aarch64 Linux cross | Size measurement only (`cmake/toolchains/aarch64-linux-gnu.cmake`) |

No Windows job exists: it would build OpenCV on a Windows runner, the slowest
step, for a platform nothing ships on. An unguarded `<sys/resource.h>` in
`tests/integration/test_seeding_bench.cpp` once broke the Windows build past
review and green CI. So compile changes to system headers, OS APIs or paths on
both hosts before merge.
