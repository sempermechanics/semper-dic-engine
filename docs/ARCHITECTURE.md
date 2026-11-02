# Engine architecture

This is a module-by-module reference for developers who change the Semper
Digital Image Correlation (DIC) engine: what each module does, the invariants
it relies on, and how the modules connect. It covers the C++ core, the
pipeline, the adapters (Android JNI, C ABI, Python) and the build.

Related documents: derivations in [MATHEMATICS.md](MATHEMATICS.md), the
public surface and Frozen formats in [CONTRACT.md](CONTRACT.md), tests in
[TESTING.md](TESTING.md), history in [CHANGELOG.md](../CHANGELOG.md), build
commands and CMake options in [README.md](../README.md).

## Data flow

One analysis, reference image to packed result:

1. Decode: the adapter turns the caller's bytes or array into 8-bit gray and
   an 8-bit mask (`io.hpp`).
2. Reference: `ReferenceCache::set_from_gray` builds an `Image` (float
   intensities and gradients) and stamps the Ghost Wall into masked pixels.
3. Validate and scan: `run_full_field` checks the parameters, prepares the
   deformed `Image`, and marks each grid node whose subset touches the border
   reserve or the mask as skipped.
4. Hessian pre-pass: one `CachedHessianData` per grid node (OpenMP).
5. Seeding: an AKAZE + RANSAC homography gives the global motion. IC-GN then
   solves a lattice of about 256 grid nodes from it, and a median test drops
   blunders. The survivors become mesh vertices; those that also clear the
   result gate become final results.
6. Mesh guess: a Delaunay mesh over the kept anchors gives each covered node
   a 6-DOF initial guess. The refitted homography covers the rest.
7. Solve: Path C finds a seed when seeding found none. Path A solves the
   mesh-guided nodes. Path B flood-fills the remainder from solved neighbours.
8. Pack: displacement median test, VSG strain, then 8 floats per point and
   the metrics array, back through the adapter.

Every grid-node solve (anchor lattice, Paths A, B and C) goes through
`solve_node`: pool entry → `SubsetData` →
`OptimizationEngine::calculate_deformation` → Ghost Wall rejection. These
solves and incremental re-anchoring (`match_anchor`) all use an engine from
`make_engine`: LM damping `tuning::kLmAlpha` and the caller's interpolator.

---

## Module: `include/semper/types.hpp`

Shared value types. All math is 32-bit `float` (`scalar_t`).

### `struct AnalysisResult`

The engine's answer for one subset.

| Field | Type | Meaning |
|---|---|---|
| `u, v` | float | Displacement of the subset centre (px) |
| `ux, uy, vx, vy` | float | Displacement gradients (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y) |
| `status` | int | `0` = converged, non-zero = failed |
| `correlation_score` | float | Final ZNSSD / valid pixels. Lower is better; `2.0` on a rejection |
| `iters` | int | IC-GN iterations of the final run |
| `invalid_ref_pixels` | int | Reference pixels masked by the ROI (Ghost Wall) |
| `trigger_iters` | int | Iterations of the IC-GN run that triggered a Simplex rescue (else `iters`); the rescue telemetry classifies on it |

### `struct SubsetData`

Everything precomputed about one reference subset. Invariants:

- `x_offsets` / `y_offsets` (and their `_f` float mirrors) hold offsets in
  `[-dim/2, +dim/2]`, row-major, length `n = dim²`.
- `dim` must be odd and at least 3. The subset holds `(2·(dim/2)+1)²` pixels;
  for an even `dim` that exceeds the `dim²` its buffers hold.
- The engine enforces this. `validate_params` makes `run_full_field` and
  `IncrementalSession::init` return `-3` for any other `subset_size`, and the
  three `SubsetPrecomputer` functions return an uninitialized subset for one.
  The app offers only odd sizes ([Parameters from the app](#parameters-from-the-app)).
- `dim` defaults to `0`. Buffers are resized only when `dim` changes, so a
  stale `dim` would skip the allocation.
- `norm_ref_intensities[i] = (ref_intensities[i] − mean_intensity) / std_dev`.
- `H` = Σ sd·sdᵀ (6×6 Gauss-Newton Hessian, kept raw for LM damping);
  `H_inv` is its inverse.
- `steepest_descent_images[i]` = `[gx, gy, gx·x, gx·y, gy·x, gy·y] / std_dev`
  (Eigen 6×1, AoS). `gx_vec` / `gy_vec` keep the raw gradients for the texture
  check.
- `sdi_planes` is the SoA mirror of the above: plane *k* occupies
  `[k·n, (k+1)·n)`. It must stay in sync with `steepest_descent_images`: the
  SIMD fast path reads the planes, the masked slow path reads the Eigen
  vectors (test `SubsetPrecomputer.SdiPlanesMirrorSteepestDescentImages`).
- `is_initialized == false` means `dim` was invalid, or the subset left the
  image, was mostly masked, or had an ill-conditioned Hessian. Do not solve
  it.

---

## Module: `include/semper/image.hpp` — class `Image`

Owns the intensities and precomputed gradients of one image.

```cpp
Image(int_t w, int_t h, const uint8_t* raw_pixels);   // 8-bit gray, row-major
void prepare_data(bool apply_dice_blur);              // gradients (+ optional 7×7 Gaussian blur)
```

The constructor throws `std::length_error` when `w·h` does not fit an `int`,
and `std::invalid_argument` for null pixels on a non-empty image. Call
`prepare_data` before any consumer reads the image; the engine always passes
`false`. Gradients are 4th-order central differences (exact for polynomials
up to degree 4), set for `x` in `[2, w−3]` and `y` in `[2, h−3]`; the
two-pixel border ring is zero.

### Interpolation ladder

| Method | Kernel | Valid region | Outside it |
|---|---|---|---|
| `interpolate_keys_fourth(x,y)` | 6×6 Keys 4th-order | `(2.5, w−3.5)` | demotes to bilinear |
| `interpolate_bicubic(x,y)` | 4×4 Catmull-Rom | `[1, w−2)` | demotes to bilinear |
| `interpolate_bilinear(x,y)` | 2×2 | `[0, w−1.5)` | returns `0.0f` |

The same bounds apply to `y` with `h`. Each test is written "inside, else
demote or reject", so a NaN coordinate is rejected rather than cast to an
index. The `_x4` batch variants skip the test: callers must already hold a
tighter guard.

All three kernels interpolate: they reproduce pixel values at integer
coordinates and linear ramps at sub-pixel positions. IC-GN, the Simplex and
the coarse search sample only inside a 4 px guard (`inside_guard`), which is
stricter than every demotion boundary. A sample's validity comes from that
guard and the reference mask, never from its value: a deformed intensity of
`0` is black, not missing.

---

## Module: `include/semper/subset.hpp`

Static factory for `SubsetData`. Three entry points share one
implementation: `precompute_subset` is `compute_hessian_only` followed by
`precompute_subset_fast`, so both routes give the same bytes (test
`FastPathMatchesFullPrecompute`).

```cpp
// Full precompute, for callers without a Hessian pool (tests, re-anchoring).
static void precompute_subset(SubsetData&, const Image& ref, int cx, int cy, int dim);

// Stats + Hessian only, for the pipeline's Hessian pool. No heap allocation.
static CachedHessianData compute_hessian_only(const Image& ref, int cx, int cy, int dim);

// Rebuilds the per-pixel data from a pool entry (~65% cheaper).
static void precompute_subset_fast(SubsetData&, const Image& ref,
                                   int cx, int cy, int dim,
                                   const CachedHessianData& cached);
```

Failure modes (`is_initialized = false` / `cached.valid = false`):

- `dim` not odd and ≥ 3;
- the subset crosses the image boundary;
- fewer than 50% of its pixels are unmasked;
- an ill-conditioned Hessian: `cond(H₂ₓ₂) > 1e12`, or the 6×6 test
  `hessian::well_conditioned` fails (1-norm rcond with the gradient rows
  scaled by the half-width, `tuning::kHessianMinRcond`).

`precompute_subset_fast` with an invalid pool entry returns an uninitialized
subset without recomputing it. The pool is keyed on the grid node, not on
`dim`, so it checks `dim` again.

---

## Module: `include/semper/solver.hpp`

`OptimizationEngine` is a per-thread solver. It owns scratch buffers and
timing counters, so never share one between threads.

```cpp
AnalysisResult calculate_deformation(const SubsetData& subset, const Image& def_img,
                                     float guess_u, float guess_v,
                                     float guess_ux, float guess_uy,
                                     float guess_vx, float guess_vy,
                                     InitializationMode init_mode);
```

### Initialization modes

| Mode | What runs | Used by |
|---|---|---|
| `INIT_AUTO_SEARCH` | If the guess is `(0, 0)`: a ZNSSD scan of ±15 px in 2 px steps over every 9th subset pixel. Then a translation-only Simplex, then IC-GN from its translation. No rescue after. | Path C; anchor lattice without a global model |
| `INIT_NO_SEARCH` | IC-GN from the 6-DOF guess. If it fails or scores above `tuning::kCorrSimplexTrigger` (0.4): a 6-DOF Simplex, then IC-GN again, keeping the better of the two IC-GN runs. | Path A inside the mesh; Path B |
| `INIT_NO_SIMPLEX` | IC-GN from the guess only. | Anchor lattice with a model; Path A outside the mesh (extrapolated or homography guess) |

The coarse scan leaves out Ghost Wall and off-image samples. An offset that
keeps under half the sample, or samples a flat patch, is not a candidate.

### Warp convention

The sign contract everything depends on. The shape function maps reference
offsets `(x, y)` from the subset centre `c` to deformed-image coordinates:

```
X_def = cx + (1+ux)·x + uy·y + u
Y_def = cy + vx·x + (1+vy)·y + v
```

A material point at `c + p` therefore moves by `t + [[ux,uy],[vx,vy]]·p`.
Synthetic test data must use this exact convention
(`tests/framework/synthetic.h`). Sample positions are summed in double and
rounded to float once.

### Solver internals

- **IC-GN** (inverse-compositional Gauss-Newton). The Hessian is fixed per
  subset. Each iteration samples the deformed subset, normalizes it (ZNSSD,
  invariant to `a·I + b`), accumulates the gradient with
  `simd::znssd_error_and_gradient`, and updates `W ← W · ΔW⁻¹`. It converges
  when `‖Δp‖ < 0.001`, within `tuning::kIcgnMaxIter` (50) iterations.
- **Sampling.** `sample_warped_subset` is the one place the deformed image is
  read through a warp, for IC-GN and the Simplex alike. It fills the values
  and a per-pixel validity mask (inside the 4 px guard and not in the Ghost
  Wall).
- **Guards.** At least `tuning::kMinValidPixelFraction` (90%) of the warped
  pixels must be valid. An iteration that loses pixels steps with its own
  masked Hessian; if that Hessian fails the 2×2 or 6×6 conditioning test,
  the solve fails. The next full iteration is back on the reference Hessian.
  A non-invertible `ΔW`, or a step that would make `W` non-finite, ends the
  solve as a failure on the last good warp.
- **Texture check.** After convergence, the surviving pixels must have a
  non-zero sum of squared gradients in both axes, and the score must be
  finite.
- **Levenberg-Marquardt** (`lm_enabled`, `lm_alpha`). Adds `α` to `H(0,0)`
  and `H(1,1)` only (the translation DOFs), as DICe does. It is off in a bare
  `OptimizationEngine`; `make_engine` turns it on (`tuning::kLmAlpha`) for
  every pipeline solve.
- **Simplex** (Nelder-Mead on ZNSSD), 2-DOF or 6-DOF, at most 100
  iterations. It is a derivative-free rescue for bad seeds. A warp that
  samples under 90% of the subset scores `2.0`; a final score above
  `tuning::kSimplexFailScore` (0.1) marks it failed.

---

## Module: `include/semper/simd.hpp`

Portable SIMD hot loops built on OpenCV universal intrinsics: one code path,
compiled to NEON on ARM and SSE on x86, with a scalar fallback elsewhere.
Header-only; it needs OpenCV's headers, not its libraries.

```cpp
namespace Semper::simd {
  // Σ (vals[i] − mean)²
  float sum_sq_diff(const float* vals, size_t n, float mean);

  // Σ (ref[i] − (vals[i] − mean)·inv_std)²
  float znssd_sum(const float* vals, const float* ref, size_t n,
                  float mean, float inv_std);

  // Fused: returns the ZNSSD error and accumulates the 6 gradient
  // projections dp[k] = Σ sdi_plane_k[i]·diff over the SoA planes.
  float znssd_error_and_gradient(const float* vals, const float* ref,
                                 const float* sdi_planes, size_t n,
                                 float mean, float inv_std, float dp_out[6]);
}
```

Contracts (all test-guarded):

- Any `n` works: full vector blocks, then a scalar tail.
- The error term of `znssd_error_and_gradient` equals `znssd_sum` on the
  same input.
- `sdi_planes` layout: plane `k` at `[k·n, (k+1)·n)`, produced by
  `SubsetPrecomputer`.
- Multiply, then add; never `cv::v_fma`. NEON fuses `v_fma` into one
  rounding and SSE2 cannot, so it would make arm64 and x86-64 disagree.

---

## Module: `include/semper/strain.hpp`

```cpp
struct DisplacementField { int width, height, step;       // grid dims + px spacing
                           std::vector<float> u, v;        // px displacements
                           std::vector<bool> valid; };

static StrainField compute_vsg_strain(const DisplacementField&, int window_pixels);
static int vsg_window_node_count(int step, int window_pixels);
static int reject_displacement_outliers(DisplacementField&);   // clears valid[], returns count
```

`pack_full_field_output` runs `reject_displacement_outliers` on the solved
grid, then `compute_vsg_strain`. The outlier test is a normalized median test
on u and v ([MATHEMATICS §9.4](MATHEMATICS.md)); it repeats until no new point
fails, at most `tuning::kOutlierMaxPasses` (5) passes. Rejected points feed no
strain window and are not written. They count in metrics slot 2 (rejected =
attempted − solved) and slot 23.

`run_full_field` calls `vsg_window_node_count` before solving: fewer than 3
nodes in a window means every point would be rejected, so it returns
`kBadStrainWindow` (`-4`).

The output is Green-Lagrange strain, for example
`exx = ∂u/∂x + ½((∂u/∂x)² + (∂v/∂x)²)`; `exy` is the tensor component.

| | VSG |
|---|---|
| Method | Least-squares plane fit over a circular window |
| Exact for | Any linear displacement field |
| Rejection | < 90% window fill, < 3 points, or `rcond < 1e-12` → sentinel `−1000.0f` |
| Displacement outliers | Normalized median test, 5×5, ε 0.1 px, threshold 3, ≥ 3 neighbours, ≤ 5 passes (`tuning::kOutlier*`) → point dropped before the fit |

A field the fit cannot read (zero step, short vectors) comes back all
sentinel. Consumers must check for `−1000.0f` before rendering or computing
statistics.

---

## Module: `include/semper/io.hpp`

The adapters decode; the solver does not. The engine reads only 8-bit
single-channel images and masks, and `run_full_field`, `set_from_gray` and
`IncrementalSession` refuse anything else. The functions and their exact
rules are in [CONTRACT.md](CONTRACT.md) ("Image I/O"). What matters when
changing the engine:

- A pixel is excluded when its 8-bit mask value is below 128. The grid scan,
  the Ghost Wall and the AKAZE match filter all apply this rule.
- Images and masks convert differently. `to_gray8` maps a non-8-bit image
  linearly from its 0.1th–99.9th percentile onto 0..255. `to_mask8` makes a
  non-8-bit mask boolean (nonzero included). `decode_gray` and `decode_mask`
  differ only in which one they call.
- The C ABI and JNI decode masks with `decode_mask`. Python converts arrays
  itself: bool and non-uint8 masks are boolean, and so is a uint8 mask that
  holds only 0 and 1.

---

## Module: `include/semper/pipeline.hpp` — `run_full_field`

`run_full_field` orchestrates one solve. `full_field_solver.cpp` sequences
the phases; each phase lives in its own file under `src/pipeline/`, behind the
internal header `full_field_internal.hpp`.

### Phases

| Step | Function (file) | Notes |
|---|---|---|
| Checks | `run_full_field_impl`, `validate_params` (`full_field_solver.cpp`) | Null output or no reference, a deformed frame not 8-bit or not the reference's size, a non-8-bit mask, bad parameters, a strain window too small for the step (codes in [CONTRACT.md](CONTRACT.md)). Clips the rect to the image without moving the lattice. |
| Grid scan | `run_full_field_impl` | Skips a node within `subset_size/2 + 19` px of the border (4 px guard + 15 px deformation reserve) or with any subset pixel masked. Skipped nodes are marked `solved`, so every phase ignores them. |
| Hessian pre-pass | `compute_hessian_only` | OpenMP `schedule(static)`. |
| Seeding | `solve_anchor_seeds` (`full_field_anchors.cpp`, `akaze_features.cpp`) | Records outcomes; writes nothing to the field. |
| Publish | `publish_anchor_results` | Serial, so anchor numbering is deterministic. |
| Mesh guess | `build_mesh_guess_field` (`full_field_mesh.cpp`) | |
| Path C | `run_path_c` (`full_field_path_c.cpp`) | Only without a seed. |
| Path A | `run_path_a` (`full_field_path_a.cpp`) | OpenMP `schedule(dynamic, 32)`. |
| Path B | `run_path_b` (`full_field_path_b.cpp`) | Its own `std::thread` workers. |
| Pack | `pack_full_field_output`, `fill_engine_metrics` (`full_field_solver_stats.cpp`) | Layouts in [CONTRACT.md](CONTRACT.md) §A.4. |
| Debug export | `export_full_field_debug_suite` (`full_field_debug_export.cpp`) | Only when `cache.debug_dir` is set; it arms one solve. |

A progress thread reports 10–90% from the solved-point count. The final
`100` comes from the calling thread after that thread is joined.

### Seeding: AKAZE model and anchor lattice

Seeding has two layers.

- **L0, global model.** AKAZE features are matched between the reference and
  the deformed frame over the ROI padded by `kAkazePadPx / scale` (40 px at
  full scale). Matches pass Lowe's ratio test (0.75) and the mask, then a
  RANSAC homography (3 px at the detection scale). The search uses the
  coarsest of 0.25×, 0.5×, 1× with 25 inliers (`kAkazeGoodInliers`), else
  the scale with the most inliers if it has 10 (`kAkazeMinInliers`). It
  starts at the scale an earlier completed solve on this reference settled
  on, and tries the coarser scales only if nothing from there up fits.
- **L1, lattice.** A lattice of about `kAnchorTarget` (256) grid nodes, with
  a per-axis stride and the last row and column always included
  (`plan_anchor_lattice`). Each anchor is solved with IC-GN on the Hessian
  pool (OpenMP `schedule(dynamic, 4)`). It starts from the homography's
  displacement and gradients at the anchor when the model moves an ROI
  corner more than `kSimilarityMinCornerPx` (4 px) beyond its centre
  translation, else from that translation alone.

A homography covers rotation at any angle, scale, stretch and shear, which the
phase-correlation front end it replaced did not ([CHANGELOG](../CHANGELOG.md)
0.6.0). The phase-correlation functions stay in `seeding.hpp`
(`phase_correlate.cpp`); no pipeline phase calls them.

An anchor is accepted in three steps:

1. Its ZNSSD clears `kAnchorAcceptScore` (0.40): a candidate mesh vertex.
   Clearing `kCorrAccept` (0.15) as well makes it publishable as a result.
2. `mark_kept_anchors` runs the universal median test on the residual from
   the model: within `kAnchorMedianTol` (1 px) of the median of its vertex
   8-neighbours. A vertex with fewer than three vertex neighbours is not kept.
3. `publish_anchor_results` marks nodes that are both publishable and kept as
   solved. Path A then skips them and Path B takes them as boundary seeds.
   Because this runs after the median test, a confident match on the wrong
   speckle blob never seeds the flood fill.

When the model's lattice is not `FULL`, or keeps under
`kSimilarityKeepFraction` (90%) of its anchors, the lattice is solved again
from the translation alone (`model_rejected`). With at least
`kAnchorRefitMinPoints` (8) kept anchors, the homography is refitted to them
(RANSAC 1 px) and seeds the nodes the mesh does not reach. A rejected model
without a refit seeds no node outside the mesh; the flood fill covers them.

Mesh quality: `FULL` needs ≥ 10 kept vertices, ≥ 60% of the attempted
anchors kept, and ≥ 30% convex-hull coverage of the ROI. `SPARSE` needs ≥ 3
vertices and ≥ 5% coverage. Anything less is `NONE`.

Seeding runs after the Hessian pre-pass on purpose. The anchors sit on grid
nodes, so they reuse the pooled Hessians, and a kept anchor is the final
result for its node. Anchors, not keypoints, are the mesh vertices. A
triangle's guess gradients err by its vertex position error over its edge
length, and that error is about 1 px for keypoints against 0.01 px for
anchors ([SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md)).

### Mesh, Path A, Path B, Path C

- **Mesh guess.** `cv::Subdiv2D` triangulates the kept anchors. Each
  triangle's affine map gives the guess for the nodes inside it
  (`kMeshInTriangle`). With a `FULL` mesh, nodes within 3 steps of a triangle
  take its map (`kMeshExtrapolated`). Remaining nodes take the homography
  (`kMeshHomography`). The guess field is then smoothed over 5×5 neighbours.
- **Path A.** Solves every unsolved node with a guess. Inside the mesh it uses
  `INIT_NO_SEARCH`; outside it uses `INIT_NO_SIMPLEX` and leaves failures to
  Path B. A node is accepted when it converges with ZNSSD ≤ `kCorrAccept`.
- **Path B.** A reliability-guided flood fill in numbered rounds. A serial
  claim pops the `kPathBBatchNodes` (32) lowest-ZNSSD nodes (ties by row, then
  column) and hands each unclaimed 4-neighbour to the first that touches it,
  with the parent's first-order expansion as its guess. Workers solve claimed
  cells in parallel. A serial commit applies rounds in order, keeping two in
  flight. The field therefore does not depend on timing or thread count.
- **Path B seeds.** Solved nodes on the boundary of the unsolved region. If
  there are none: up to 5 kept anchors (smallest whole-pixel displacement,
  then nearest the grid centre). The Path C seed is used when there is no
  such anchor, or when they all fail.
- **Path C.** Runs when the mesh is `NONE`, no kept anchor cleared the
  result gate, and fewer than 3 vertices were kept. It tries up to
  `kPathCMaxCandidates` (9) nodes nearest the grid centre with
  `INIT_AUTO_SEARCH`, starting from the global shift. If all fail, the solve
  returns `kSeedFailed` (`-5`) without an AKAZE lock, and continues to Path B
  with one.

### Reference cache

`ReferenceCache` holds the prepared reference `Image` (with the Ghost Wall),
its gray `cv::Mat`, `width` / `height`, `debug_dir`, a `mutex` and
`precompute` (`RefPrecompute`). The mutex does not make the methods
thread-safe: the JNI adapter holds it around a whole solve.

| Member function | Behaviour |
|---|---|
| `set_from_gray` | Resets first. Refuses a non-8-bit image or mask, or more than `INT_MAX` pixels, leaving the cache empty (the next solve returns `-3`). Builds the image aside and installs it once complete; if it throws, the cache stays empty. |
| `adopt` / `internal::adopt_reference` | Takes an already-prepared image (incremental DIC). Stamps the Ghost Wall before changing anything; if it throws, the cache keeps the outgoing reference unchanged (strong guarantee). Carries over the AKAZE starting scale and the handed-over features. |
| `reset` | Drops the image and all precomputed state. |

The Ghost Wall stamps `kGhostWallValue` (−10) into masked reference pixels
after `prepare_data`, so gradients at the mask edge come from real
intensities. Subset routines skip pixels below `kGhostWallThreshold` (−5). A
subset with over 5% of its pixels in the Ghost Wall is rejected
(`reject_if_ghosted`).

`RefPrecompute` keeps what depends only on the reference and the grid:

| Part | Keyed on | Reuse |
|---|---|---|
| Grid scan + Hessian pool | Clipped rect, `step`, `subset_size`, mask (exact match) | Marked valid only after a pre-pass that ran to the end without a cancel |
| AKAZE keypoints and descriptors, one entry per scale tried | ROI rect | Detected once per reference and scale |
| `akaze_first_level` | — | Moves only when a solve completes |

The grid scan, pool and keypoints give bit-identical results with or without
the cache (`FullField.RefCache_*`). The AKAZE starting scale does not: a
solve can take its model from a finer scale than a cold solve would. This is
the one way a solve depends on earlier ones ([CONTRACT.md](CONTRACT.md),
"Reference cache").

Steepest-descent images are not cached: at `dim² × 6` floats per node, a
whole grid would cost too much memory on a phone.

### Errors and threads

- **Thread count.** `std::thread::hardware_concurrency()`. Every OpenMP region
  uses `num_threads` of it, and Path B starts that many workers.
- **One engine per worker.** Each OpenMP thread and each Path B worker owns
  one `OptimizationEngine` and one `SubsetData`. `precompute_subset_fast` and
  `calculate_deformation` overwrite everything they read, so which worker
  solves a node cannot change its result.
- **Exceptions.** An exception must not leave an OpenMP region. Each region
  catches into a `RegionErrors`; the remaining iterations skip, and the
  first exception is rethrown after the join. A Path B worker that throws
  stops the flood, and the exception is rethrown once all workers have
  joined. The adapters' `guarded_call` turns it into their error value.

---

## Module: `include/semper/cancel.hpp` — cancellation

Cancellation is cooperative. Each solve polls one `CancelToken`, passed down
explicitly (`SolveContext::cancel`, `solve_anchor_seeds`). A poll costs one
relaxed atomic load per point. A late poll costs one more point, never a
wrong result.

| Piece | Role |
|---|---|
| `CancelToken::request` / `clear` / `requested` | The flag. |
| `CancelToken::enter` | Called at every solve entry (`run_full_field`, `IncrementalSession::process_frame`, both overloads). Clears a request left from before, unless a `CancelScope` opened the window for this call. |
| `CancelToken::begin` / `end`, `CancelScope` | An adapter call's window. `begin` clears any earlier request and opens the window; `end` closes it on exit. A request made after the window opens, during the decode included, reaches the solve. JNI and Python open it once they hold their lock; the C ABI opens it before decoding. |
| `SolveCancel` (internal) | The solve's view of the token. Once any poll sees a request, every later poll of that solve sees it, even if the token is cleared meanwhile. |
| `request_cancel` / `clear_cancel` / `cancel_requested` | Act on one process-wide token (`detail::global_cancel_token`), which the overloads without a token poll. A solve given its own token never reads it. |

The latch matters because JNI `setCancelRequested(false)` takes no lock and
can clear the token mid-solve. Without it, a pre-pass that skipped iterations
could mark a pool with holes complete, and Paths A and B could return a
partial field as a success.

Where the solve polls:

| Site | Shape |
|---|---|
| Hessian pre-pass, anchor lattice, Path A | `if (cancel.requested()) continue;`. OpenMP forbids breaking out of a parallel `for`, so a cancel skips the remaining iterations. |
| After the pre-pass, after seeding, after Path A, after Path B | `return kCancelled`. A cancelled field is partial, so strain and packing never run on it. |
| Path C candidate loop | `return kCancelled`. |
| Path B seed solves | `return`. |
| Path B workers | The flag is in the condition-variable predicate, then `return`. |
| Path B's CV wait | Bounded (`wait_for`, `tuning::kCancelPollMs` = 20 ms). A parked worker has no one to notify it of a cancel, so it re-checks on a timer. |

A cancelled solve leaves the Hessian pool marked stale and does not move the
AKAZE starting scale, so submitting the frame again gives the result it would
have given.

`kCancelled` is `-99`, the same value as `AnalysisRunCodes.ERROR_CANCELLED`
on the Kotlin side. The app treats that code as "cancelled, discard", distinct
from the `-2` / `-3` failures that raise a dialog. On the app side,
`AnalysisViewModel.cancelRequested` and `VsgStudyRunner.cancelRequested`
forward to `setCancelRequested` from their setters, and both assign `false` at
the top of a run.

---

## Module: `include/semper/incremental.hpp` — class `IncrementalSession`

Multi-frame DIC that can move its reference forward. The maths is in
[MATHEMATICS.md §9.5](MATHEMATICS.md#95-incremental-updated-reference-dic);
the surface and return codes are in [CONTRACT.md](CONTRACT.md).

```
init(frame 0)       -> validate_params, material mask, ReferenceCache::set_from_gray
                       (the frame-0 grid scan runs with the first solve)
process_frame(t):
  reference is 0    -> run_full_field_impl into the caller's buffer (bit-identical)
  reference is k>0  -> run_full_field_impl on reference k's grid into scratch
                       -> node_field_from_capture (solved, ZNSSD >= 0, median test)
                       -> sample_increment at X0 + U(X0,k) for every tracked node
                          - nothing in reach: fit_increment (coast, unreported)
                          - coasted far: match_anchor before it is reported again
                       -> frame-0 ResultGrid -> pack_full_field_output (VSG, packing)
  update due        -> make_anchor for clean nodes near a gap (on this frame)
                       prepare_reference: warp_material_mask, new grid rect
                       adopt_reference(prepared image, AKAZE features), then commit
```

- **When the reference moves.** Every `N` frames (`update_interval`), or
  early (N > 1) when a frame reports `kIncEarlyUpdateDrop` (5%) fewer points
  than the first frame on the current reference. Never onto a frame that
  reports under `kIncUpdateMinReported` (50%) of the count it is compared
  with: the first frame on this reference, else what the reference reported
  when solved, else the frame-0 node count. That update falls to the next
  frame.
- **Hand-off without rework.** `run_full_field_impl` takes an optional
  `FieldCapture`, filled after Path B and before packing. With N > 0 it also
  keeps the prepared deformed `Image`, its gray `cv::Mat` and its AKAZE
  features, since any frame may become the reference. The new reference is
  never decoded or prepared twice, and at most two full-frame `Image`s are
  alive at once.
- **State.** Per frame-0 node: the base displacement `U(X0, k)`, an `alive`
  flag and the distance it has coasted. Anchors only for nodes near a gap or
  coasting: a grey patch of side `subset_size + 2·kIncAnchorMarginPx`
  (3.7 KB at subset 41). Also the reference index, the frames since the last
  update, and the reference's grid, mask and point counts.
- **Coasting.** A node whose increment cannot be sampled (no solved node
  within `kIncTaylorMaxCells`) coasts on a weighted plane through the solved
  increments within `kIncCoastRadiusCells`. It is dropped once it would travel
  more than `kIncCoastMaxTravelSubsets` subsets. After more than
  `kIncCoastTrustedTravelSubsets` of travel, it is reported again only once
  `match_anchor` confirms it.
- **Errors, exceptions and cancel.** A reference update builds the next state
  in full on copies. `adopt_reference` is the last step that can throw; the
  commit after it only moves and swaps. A negative return or an exception
  therefore leaves the session as it was, and the same frame can be
  submitted again. Failed frames, `kSeedFailed` included, are not counted, so
  the session's frame indices then run one behind the caller's.
- **Nothing left to follow.** When no node of a moved reference's grid can be
  solved (`-2` from the solve), the nodes are dropped and the frame returns 0
  points with zeroed metrics.
- **Caching between updates.** Every frame after the first on a reference
  reuses that reference's grid scan, Hessian pool and AKAZE keypoints.
- **On an update.** The deformed frame's AKAZE features at the model's scale
  replace a fresh detection at that scale when their rect covers at least
  `kAkazeReuseCover` (90%) of the new padded ROI. The AKAZE starting scale
  carries over.
- **Metrics** describe the increment solve. Frame-0 counts are in
  `IncrementalFrameInfo`.

Tunables: `tuning::kIncTaylorMaxCells`, `kIncMaskMaxSamplesPerPx`,
`kIncMaskExtrapCells`, `kIncCoastRadiusCells`, `kIncCoastMinNodes`,
`kIncCoastMaxTravelSubsets`, `kIncCoastTrustedTravelSubsets`,
`kIncReanchorMaxShiftFraction`, `kIncReanchorMinShiftPx`, `kIncAnchorMarginPx`,
`kIncEarlyUpdateDrop`, `kIncUpdateMinReported`, `kAkazeReuseCover`.

---

## Module: `adapters/android/jni/SemperJNI.cpp`

JNI marshalling only. The Kotlin declarations are in the app's
`SemperNativeLib.kt`. The exports, the engine calls behind them and their
observable behaviour are in [CONTRACT.md](CONTRACT.md) §A.3. Internals:

- **State.** One `ReferenceCache g_cache` and at most one
  `IncrementalSession g_session`, both guarded by `g_cache.mutex`. A solve
  export holds the mutex for the whole solve, so one solve runs at a time and
  the fixed and incremental paths never overlap.
- **Cancel.** `setCancelRequested` takes no lock, on purpose: a cancel that
  waited for the mutex could never stop the solve that holds it. The solve
  exports open their `CancelScope` on the global token once they hold the
  lock.
- **Decoding.** `decode_optional` reads an optional byte array. An array that
  exists but does not decode is an error (`-3`), never "no mask"; masks go
  through `io::decode_mask`. `initializeReference` resets the cache before
  decoding, so a failure leaves no reference.
- **Output.** Results go straight into the direct `ByteBuffer`, bounded by its
  capacity in floats. Metrics and frame info are copied back only on success.
- **Progress.** `ProgressBridge` holds a `NewGlobalRef` on the callback and
  attaches the engine's progress thread with `AttachCurrentThread`. A
  thread-local flag makes a locking export called from inside the callback
  fail instead of deadlocking.
- **Exceptions.** Every export runs its body through `guarded_call`: an
  exception becomes `-3` (or `null` for the preview). No JNI call is made
  with a Java exception pending.

### Parameters from the app

`StaticAnalysisActivity` ("Advanced parameters" in the setup wizard) is the
only producer of the three solve parameters. Each has a slider and an
editable numeric field. Both write the same slider value, which
`currentSubsetSize()` / `currentStepSize()` / `currentStrainWindow()` read.

| Parameter | JNI argument | Default | Range | Step |
|---|---|---|---|---|
| Subset size | `subsetSize` → `dim` | 41 | 15–101 | 2 (odd only) |
| Step size | `stepSize` → grid spacing | 5 | 1–30 | 1 |
| Strain window | `strainWindow` → `window_pixels` | 15 | 5–51 | 2 (odd only) |

`snapToSlider` clamps typed input to the range and snaps it onto the
slider's step grid, so hand-entered values also keep `dim` odd. Fields commit
on IME "Done", on focus loss, and before the solve starts, so a pending edit
never reaches the engine uncommitted.

---

## Module: `adapters/c/` and `bindings/python/`

Both adapters translate and hold no algorithm. The C ABI's behaviour is in
[CONTRACT.md](CONTRACT.md) §A.2. Internals:

- **C ABI (`semper_c.cpp`).** Each handle owns its cache or session and its
  own `CancelToken`, so handles cancel independently. A `busy` flag
  (`HandleCall`) admits one call at a time per handle; the cancel functions
  bypass it. `semper_run` and `semper_incremental_process` open their
  `CancelScope` before decoding, and every entry point runs through
  `guarded_call`.
- **Python (`module.cpp`).** `Engine` and `IncrementalSession` each own a
  cache or session, a `CancelToken` and a mutex taken with the GIL released.
  `run_interruptible` runs the solve on a worker thread; the calling thread
  waits without the GIL and runs signal handlers every 50 ms, so Ctrl-C
  cancels the solve. An exception raised by `progress` cancels the solve and
  is raised once it stops.

---

## Package map

| Path | Role |
|---|---|
| `include/semper/` | Public headers: `types`, `image`, `subset`, `solver`, `strain`, `simd`, `pipeline`, `incremental`, `io`, `seeding`, `cancel`, `metrics`, `version`, `tuning`, `assert`, `guarded_call`, `semper_c.h`. Stability tiers in [CONTRACT.md](CONTRACT.md) §A.1. |
| `src/math/` | `Image`, `SubsetPrecomputer`, `OptimizationEngine`; `hessian.hpp` (conditioning and finiteness checks) |
| `src/strain/` | `StrainCalculator` |
| `src/io/` | `image_codec.cpp`: platform-agnostic OpenCV decode |
| `src/pipeline/` | `full_field_*.cpp` (orchestration, seeding, mesh, Paths A/B/C, packing, debug export), `akaze_features.cpp`, `reference_cache.cpp`, `cancel.cpp`, `incremental_session.cpp` + `incremental_math.cpp`, `phase_correlate.cpp` |
| `src/util/` | `finite.hpp` (`finite_bits`), `omp.hpp` (`omp_thread`), `log.hpp`, `median.hpp` |
| `adapters/android/` | JNI → `libsemper_core.so` |
| `adapters/c/` | Stable C ABI → `libsemper_c` |
| `bindings/python/` | pybind11 module `_semper` |
| `tests/` | Host unit, integration, DICe and perf suites, C smoke and contract ([TESTING.md](TESTING.md)) |

Layering rules: `src/` never includes `jni.h`. `android/log.h` appears only
in `src/util/log.hpp`, behind `SEMPER_ANDROID_LOG`. The JNI adapter includes
only `include/`, not `src/`, so it can be lifted into the app repository.
CMake targets are listed in [README.md](../README.md).

---

## Dependencies

Both libraries are git submodules built from source, not vendored binaries.
Run `git submodule update --init --recursive` after cloning.

| Dependency | Path | Recorded commit | Upstream release | How it is used |
|---|---|---|---|---|
| Eigen | `third_party/eigen` | `bc3b3987` | 5.0.1 | Header-only; a `SYSTEM` include directory of `semper_math` and `semper_pipeline` |
| OpenCV | `third_party/opencv` | `0654a42e` | 4.14.0 | Compiled from source in the engine build, linked statically |

### Pinning

`.gitmodules` records `path`, `url` and, for OpenCV, `ignore = untracked`
(for the sparse-checkout stubs). It has no `branch` or tag entry. The commit
SHA the superproject records pins the build, and a bump to any commit is
accepted silently.

Both recorded commits are the upstream release commits. `git submodule
status` shows the tag name only when the submodule clone has its tags; a
shallow clone shows the SHA. Check with `git submodule status` rather than
trusting this table.

### OpenCV modules

The engine links `core`, `imgproc`, `imgcodecs`, `features2d`, `calib3d` and
`flann` (`SEMPER_OPENCV_REQUIRED_MODULES` in `cmake/SemperOpenCV.cmake`;
keep it in sync with `CMakeLists.txt`). OpenCV's `BUILD_LIST` is that set
plus `SEMPER_OPENCV_BUILD_LIST`, a comma-separated cache option for extra
modules (empty by default) that cannot remove a required one.

Configure fails if a required module has no target
(`semper_check_opencv_modules`), for example after a cached
`BUILD_opencv_calib3d=OFF`. Why the list is merged:
[CHANGELOG](../CHANGELOG.md) 0.6.1.

`features2d`, `calib3d` and `flann` serve only the AKAZE front end. Only
`src/pipeline/akaze_features.cpp` and `full_field_anchors.cpp` include their
headers; nothing under `include/semper/` does. Their size cost is recorded in
[CHANGELOG](../CHANGELOG.md) 0.6.0 and
[SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md) §6.

### OpenCV threads

AKAZE detection, the matcher and RANSAC run inside a one-stripe
`cv::parallel_for_` (`without_opencv_pool`), so OpenCV runs their inner
parallel loops inline and never wakes its thread pool. On a small ROI the
pool saves under 1 ms, but its workers were still spinning when the engine's
OpenMP region started: up to 2–3× the wall time of a small-ROI frame on a
host. The Android adapter pins OpenCV to one thread in `JNI_OnLoad`.

### Sparse OpenCV checkout

After submodule init, `scripts/sparse-opencv.sh` (or
`scripts/sparse-opencv.ps1` on Windows) keeps only `modules`, `include`,
`3rdparty`, `cmake`, `platforms` and `hal` (Carotene lives there), plus the
top-level files. It plants no-op `doc/` and `data/` `CMakeLists.txt` stubs,
because OpenCV's CMake always adds those directories. `cmake/SemperOpenCV.cmake`
plants the same stubs when they are missing.

### OpenCV from-source integration

In `CMakeLists.txt` and `cmake/SemperOpenCV.cmake`:

- `add_subdirectory(third_party/opencv … EXCLUDE_FROM_ALL)` builds OpenCV
  inside this CMake project. The curated `BUILD_LIST` keeps the build to a
  few minutes per ABI instead of tens.
- `BUILD_SHARED_LIBS OFF`: OpenCV links statically into `libsemper_core.so`,
  `libsemper_c` and `_semper`. The engine links the in-tree targets
  `opencv_core opencv_imgproc opencv_imgcodecs opencv_features2d
  opencv_calib3d opencv_flann` directly.
- The bundled codecs (`zlib`, `png`, `jpeg`, `tiff`, `webp`) build in-tree,
  so `cv::imdecode` / `cv::imwrite` need no system libraries.
- OpenCV's in-tree targets do not export their header directories. The
  module `include/` paths and the generated-header directory
  (`opencv_modules.hpp`, `cvconfig.h` in `CMAKE_BINARY_DIR`) are added
  explicitly.
- `OPENCV_PYTHON_SKIP_DETECTION ON` disables OpenCV's host-Python detection.
  The Python bindings use pybind11 separately.
- On MSVC, `BUILD_WITH_STATIC_CRT OFF`: Python extensions use the dynamic CRT
  (`/MD`), and OpenCV's default `/MT` fails to link with them (LNK2038).
- The engine's flags are set per target (`target_compile_options`), so
  OpenCV compiles with its own defaults.

The host test build ([TESTING.md](TESTING.md)) reads OpenCV's
universal-intrinsics header from the submodule source. The two generated
headers it needs are committed under `tests/shim/opencv2/`, so it needs no
OpenCV configure.

---

## Build & ABI strategy

- Android: downstream apps configure with `-DSEMPER_ANDROID=ON` (typically
  ABI `arm64-v8a`; debug builds often add `x86_64` for emulators).
  `SEMPER_FORCE_RELEASE` (default ON) forces Release when AGP passes Debug,
  so OpenCV is not compiled at `-O0`.
- OpenCV compiles once per ABI into the shared library. The NDK `.cxx/`
  directory caches it across incremental builds.
- SIMD portability comes from `include/semper/simd.hpp`. The engine has no
  architecture-conditional code: do not reintroduce `#if __aarch64__`; extend
  the kernels instead.
- The Android shared library links with `-Wl,-z,max-page-size=16384`, the
  16 KB page alignment Android 15+ devices need.

### Compiler flags

Non-MSVC builds; MSVC keeps its defaults.

| Target | Flags |
|---|---|
| `semper_math` | `-O3 -ffast-math -fno-finite-math-only -ffp-contract=off -fPIC -fvisibility=hidden` |
| `semper_pipeline` | The same plus `-fexceptions`. Android adds `-flto -fopenmp`; host links `OpenMP::OpenMP_CXX` when CMake finds it. |
| `semper_android` | `-O3 -flto -fopenmp -fexceptions` and the same math flags; links with `-s -flto -Wl,-z,max-page-size=16384 -fopenmp -static-openmp -static-libstdc++` |
| `semper_c` | `-O3 -fPIC -fvisibility=hidden`; on Linux and Android the `semper_c.map` version script limits exports |

Why these flags:

- `-ffp-contract=off` makes a multiply-add round the same on every target.
  Compilers fuse it on aarch64 but not on baseline x86-64, so results
  differed between the two. The SIMD kernels avoid `v_fma` for the same
  reason.
- `-fno-finite-math-only` keeps NaN and Inf meaningful under `-ffast-math`,
  so the engine's checks for non-finite warps, scores and image values
  survive.
- The code still tests finiteness on the bits (`util::finite_bits`,
  `src/util/finite.hpp`), and marks missing values with explicit sentinels
  (`-1.0f` for "no correlation", `-1000.0f` for a failed strain fit), never
  NaN. Preserve this convention.
- Host shared-library builds omit `-flto`: a MinGW DLL built from an LTO
  static archive fails at load time ("32 bit pseudo relocation … out of
  range").
- A sanitizer build (`SEMPER_SANITIZER`) drops LTO, and with `undefined`
  adds `float-cast-overflow`.

### OpenMP

On host, the pipeline links OpenMP when CMake finds it, and
`SEMPER_REQUIRE_OPENMP=ON` makes configure fail otherwise (the macOS wheels
set it). Without OpenMP (MSVC, Apple Clang without libomp) every `#pragma omp`
region runs on the calling thread.

Thread indices therefore go through `util::omp_thread()` (`src/util/omp.hpp`),
never `omp.h` directly. CI's `serial-build` job builds the suite with
`DIC_OPENMP=OFF` and warnings fatal, checks that nothing uses `-fopenmp` or
an OpenMP runtime, and runs it.

### Platform support and CI gates

| Platform | Status | Gated by |
|---|---|---|
| Linux x86-64, GCC | Supported | The host jobs in `ci.yml`: `host-tests`, `serial-build`, ASan + UBSan, `c-sdk-smoke`, `adapters-sanitized`, `python-bindings` (coverage is report-only) |
| Linux x86-64, Clang 18 | Supported | `sanitizers` (TSan) |
| Android arm64-v8a | Shipping target | `android-ndk` compiles `libsemper_core.so` and checks its JNI exports (no device run). The host `JniAdapter` suite drives the adapter through a fake `JNIEnv`. The app builds the APK. |
| macOS (Apple Clang + libomp) | Python wheels | Nothing: the `wheels` workflow runs only on manual dispatch |
| Windows x86-64, MSVC | Python wheels | Nothing: the same manual `wheels` workflow |
| Windows x86-64, MinGW-w64 | Works, ungated | Nothing |
| aarch64 Linux cross | Size measurement only | Nothing (see `cmake/toolchains/aarch64-linux-gnu.cmake`) |

Windows is a development platform: the engine, the C SDK and the full test
suite build and pass under MinGW-w64. No workflow builds that, so nothing
stops a Windows break from landing. It has happened: an unguarded
`<sys/resource.h>` in `tests/integration/test_seeding_bench.cpp` broke the
whole `dic_tests` build on Windows and passed review and a green CI run.

A Windows CI job is not justified yet. It would have to build the OpenCV
submodule on a Windows runner, the slowest part of the build, for a platform
nothing ships on. Until that changes, the rule is procedural: a change that
touches system headers, process or OS APIs, or paths must be compiled on both
hosts before merge. Cross-host numeric results are covered by the tolerances
in [TESTING.md](TESTING.md).
