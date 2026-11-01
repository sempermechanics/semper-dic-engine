# Changelog

Versions follow [docs/CONTRACT.md](docs/CONTRACT.md) §A.1. Entries before
0.5.0 live in the git history (`git log --grep Version`).

## Unreleased

The next release is a **minor** one (0.7.0): `CancelScope` and
`CancelToken::begin/end/enter` (`cancel.hpp`) and `io::to_mask8` /
`io::decode_mask` (`io.hpp`) are additive Stable API. No Frozen value,
signature, slot or packing changed.

### Fixed — cancellation
- A cancel issued while an adapter was still decoding the frame (or, in
  Python and JNI, waiting for its lock or allocating) was cleared by the
  solve's entry and the solve ran to completion. The adapters open the cancel
  window at their own entry (`CancelScope`).
- A cancel requested and then cleared mid-solve (JNI `setCancelRequested`
  is lock-free) let the Hessian pre-pass mark a pool with holes complete: the
  solve returned the holes as a success and every later solve against the
  reference replayed them. A request any poll saw now stays in force for the
  rest of the solve (`-99`).
- A cancelled or failed solve no longer moves the cache's AKAZE starting scale.

### Fixed — solver
- Path B's seed sort used a comparator that was not a strict weak ordering
  (undefined behaviour in `std::sort`). When every anchor candidate failed the
  gate, Path C's seed was never tried and the solve returned 0 points.
- `ReferenceCache::set_from_gray` / `adopt` and an incremental reference
  update install their result only once complete; an exception (`bad_alloc`,
  `cv::Exception`) no longer leaves a half-prepared reference or session.
- Incremental: a first frame that solved nothing (or a handful of nodes) became
  the reference and every later frame reported 0 points; the floor now compares
  it with frame 0's node count. A reference whose grid has no solvable node
  drops the nodes and reports 0 instead of returning `-2` forever.
- NaN handling: the engine is built with `-fno-finite-math-only` beside
  `-ffast-math`, under whose finite-math assumption no NaN check was
  guaranteed to survive, and the interpolators accept only points inside their
  bounds, so a NaN coordinate is rejected instead of cast to an index.
  `phase_correlate` rejects a flat reference crop; `median_inplace` leaves NaNs
  out; `compute_vsg_strain` refuses a zero step or short vectors; `Image`
  refuses a null pixel pointer; AKAZE is not run on a crop too small at its
  detection scale, and a detector failure is no model rather than `-3`.
- Floating point is the same on arm64 and x86-64 where it can be: no FP
  contraction (`-ffp-contract=off`) and multiply-then-add instead of `v_fma`
  in the SIMD kernels. Baseline x86-64 results are unchanged.

### Fixed — image I/O
- Masks: a non-8-bit mask is boolean (nonzero included). An all-zero 16-bit
  or bool mask used to be stretched to 128 (everything included); a NumPy
  uint8 0/1 mask excluded every pixel. 8-bit masks keep the `< 128` rule.
- `to_gray8` maps a non-8-bit image from its 0.1th to 99.9th percentile, so a
  hot pixel no longer collapses the speckle to a few grey levels.
- Images over 2^28 pixels are refused, before decoding where the header gives
  the size (PNG, JPEG, BMP, TIFF); `image_dimensions` reads those headers
  instead of decoding the image in colour. Raw-buffer sizes are checked in
  64 bits.

### Fixed — adapters
- C ABI: one call at a time per handle (a nested call returns
  `SEMPER_ERR_INIT`); `semper_set_reference` / `semper_incremental_init`
  release the old state first; negative `out_capacity` is `SEMPER_ERR_INIT`;
  `info` written only on success; ELF exports limited to `semper_*`.
- JNI: no JNI call with an exception pending (`getPreviewFromBytes`,
  `getImageDimensions`); string chars released on every path; a locking export
  called from the progress callback fails instead of hanging; metrics and info
  arrays written only on success; `initializeReference` releases the old
  reference first; legacy-package forwarders for the three incremental exports.
- Python: Ctrl-C cancels a solve; an exception raised by `progress` cancels the
  solve and is raised; a non-callable `progress` is a `TypeError`; calling the
  object from its own progress callback raises instead of deadlocking; the GIL
  is released while the reference is prepared; output buffers beyond the
  solver's int capacity are refused and invalid parameters no longer allocate
  the whole grid; `info` is empty on an error; `DicError` survives pickling.

### CI
- `dic_tests` fails when a filter matches no test; `ctest` runs the suite and
  the C ABI binaries; UBSan halts on the first report and includes
  `float-cast-overflow`; the golden corpora run (without the numeric
  comparison) under the sanitizers.
- New: the C ABI and the Python module under ASan + UBSan; the host suite
  against the vendored OpenCV that ships (the system-OpenCV jobs paired 4.14
  headers with 4.6 libraries); the C++ example is built; every exported
  `semper_*` and `Java_*` symbol is checked against a golden list.
- Tests build with the libraries' flags (`-O3`; MSVC `/fp:precise`). One
  long-chain rms bound in `Incremental.Rotation90_200Frames_TracksTheTurningRoi`
  (N = 10, 4x4) moved from 0.20 to 0.25 px: 0.175 px at `-O2`, 0.208 px at
  `-O3`, with the same share within 0.5 px.
- Actions pinned to commit SHAs; read-only token; runs on `main` are not
  cancelled by the next push; pip installs pinned; fail-closed tool checks;
  wheels fail on a missing sparse checkout or an empty wheelhouse, and the
  macOS wheels find (and require) OpenMP.

### Docs
- MATHEMATICS: the reported score is the per-pixel mean ZNSSD (so
  `2(1 - ZNCC)` and the 0.15 gate hold for it); IC-GN and Simplex iteration
  limits (50, 100); the VSG fit is a plane with an explicit inverse and an
  rcond gate, not a quadratic. `semper_c.h`: `exy` is the tensor component.
  CONTRACT: the AKAZE starting scale is history-dependent; the JNI thread
  count sentence matched no code.

### Tests
- The Android JNI adapter runs under the sanitizers. The `JniAdapter` suite
  compiles `adapters/android/jni/SemperJNI.cpp` into `dic_tests` against the
  NDK's own `jni.h` and drives it through a fake `JNIEnv` / `JavaVM`
  (`tests/jni/`), comparing every result with the engine called directly, so
  the ASan + UBSan and TSan jobs now cover array pinning, the output-capacity
  bound, the metrics prefix, progress-thread attach/detach, cancel and
  exceptions at the boundary. CMake finds the header from `ANDROID_NDK_*` or
  `-DSEMPER_NDK_JNI_H`; `-DDIC_REQUIRE_JNI=ON` (set in CI) fails configure
  without one. Clean under ASan + UBSan (GCC 11.4) and TSan (clang 18).
- `Metamorphic`: five relations between pairs of full-field solves — rigid
  translation, a whole-pixel shift of the deformed image, swapped images
  (inverse map), a gain and offset on the deformed image, and a mirror in x.
- No output changed.

### Fixed — examples
- `examples/samples/oht_cfrp/expected.json` gave the image as 512×512 and a
  suggested `rect` of `[40, 40, 432, 432]`. The pair is 400×1040, so the rect
  ran 72 px past the right edge and stopped at y 472, short of most of DICe's
  grid (y 61–971). Now `image_size` `[400, 1040]` and `rect`
  `[40, 40, 320, 950]` (`[x, y, w, h]`), which lies inside the image and covers
  DICe's grid.

### Fixed — telemetry
- Metrics slots 17 and 18 (Simplex and IC-GN time, "all paths") now include
  Path C's seed search. A solve that fell back to Path C reported less time
  than it spent, by up to `tuning::kPathCMaxCandidates` IC-GN runs.

### Fixed — incremental
- The reference never moves, early or on schedule, to a frame that reports
  fewer than half the points of the first frame on the current reference
  (`tuning::kIncUpdateMinReported`), or, before that frame and always with
  N = 1, of what the reference itself reported. Such a frame is a failed
  solve: the reference stays and an update that was due falls to the next
  frame. The early update moved the reference onto any frame that reported
  5% fewer points, so a collapsed solve (1–2 points, 20–40 px off) became the
  reference and every other node was lost for good. IncBench S4, 2°/frame,
  σ = 2: N = 5 ends at 0.194 px / 94% where it locked onto a wrong solution
  (12.7 px / 0%), and N = 10 at 0.146 px / 94% where it lost the chain.
  Every run without such a frame is bit-identical. These figures were
  measured before the 0.6.x seeding and not re-run since.
- A frame that returns `kSeedFailed` leaves the session as it was, as every
  error does (CONTRACT): it is not counted, so the session's frame indices run
  one behind the caller's after it. Unchanged behaviour, now pinned by tests.

## 0.6.1

Patch: no surface change. The field moves only where 0.6.0 kept an anchor
the median test could not check; the build no longer depends on a fresh
CMake cache.

### Fixed — seeding
- An anchor the lattice median test cannot check (fewer than three converged
  neighbours) is no longer kept as a mesh vertex or published; Path A or B
  solves its node. It used to be kept on its ZNSSD alone. On the real PMMA
  3-point bend (material_testing `real_data_pmma_bending`), v0.6.0's
  translation-only fallback kept such anchors under the load nose on frames
  24–29. They were ~20 px off at ZNSSD 0.17–0.30, and Path A followed them
  onto the wrong speckle. The outlier filter then dropped up to 3,861 of the
  9,431 deflection-probe points, and the deflection there read up to 1.5%
  low. Now 91 points are missing on one frame. The deflection RMSE
  (0.0080 mm) and E from the graph (2.002 GPa) match v0.4.2 again, and steel
  tensile is unchanged.

### Build
- The OpenCV modules the engine links (`core,imgproc,imgcodecs,features2d,
  calib3d,flann`) are always compiled. `SEMPER_OPENCV_BUILD_LIST` now names
  extra modules only (empty by default) and is merged in. A build tree
  configured by v0.3.0–v0.5.x caches it as `core,imgproc,imgcodecs`, and
  with v0.6.0 that dropped `features2d`/`calib3d`/`flann` and failed the link
  (`-lopencv_features2d`). A required module that still has no target now
  fails at configure time.

## 0.6.0

Minor: a new capability, no surface change. Every signature, Frozen value,
slot index and the packing are unchanged; the field moves.

### Changed — seeding
- The anchor lattice is seeded from an **AKAZE + RANSAC homography** instead
  of phase correlation and the log-polar similarity. AKAZE runs over the
  ROI padded by 40 px, at the coarsest of 0.25x / 0.5x / 1x that gives 25
  inliers (else the best scale with 10). The lattice, its median test, the
  fallback to a translation-only lattice when the model's lattice is weak,
  and Paths A/B/C are as before; without a model the lattice searches for
  its own offsets, as v0.5.0 did without a phase lock.
- A homography has no 180° ambiguity and does not read stretch as rotation:
  rotation past 90° and large stretch now seed the mesh. Fixed reference,
  300 px ROI: 95–180° goes from 0% to 100% coverage; 15% stretch + 15° is up
  to 17x faster on large ROIs (SEEDING_BENCHMARK §11).
- The homography is refitted to the converged anchors and seeds the nodes
  the mesh does not reach, solved with plain IC-GN.
- `ReferenceCache` keeps the reference's AKAZE keypoints per scale and the
  scale the last solve settled on. `IncrementalSession` hands the deformed
  frame's keypoints to the next reference, so a reference is detected once.
- Metrics slot 21 reports the global-motion (AKAZE) lock; slot 10 is AKAZE
  plus the anchor lattice (CONTRACT §A.4).
- Incremental: rotation to 180° at 4.5°/frame with N = 5 on 60–80 px ROIs
  keeps every node (v0.5.0 lost all of them); elsewhere accuracy is identical
  and time per frame is 0.45–1.11x v0.5.0's, faster on most ROIs. The one
  cost: a cheap small-ROI frame pays ~1.5–3 ms for AKAZE, 1–10% on the
  tensile video's run total. Sample 14 unchanged (≤ 0.00008 px).

### Build
- OpenCV `features2d`, `calib3d` and `flann` are back in
  `SEMPER_OPENCV_BUILD_LIST`: the stripped x86-64 Python extension grows from
  12.30 MB to 14.32 MB (+16%).
- AKAZE runs without waking OpenCV's thread pool, whose spinning workers
  otherwise slowed the engine's next OpenMP region (host builds; Android pins
  OpenCV to one thread).
- OpenCV 4.13.0 → **4.14.0** (submodule `0654a42e`). Its `core` and
  `imgproc` grew upstream: the stripped x86-64 Python extension is 15.58 MB
  (14.32 MB on 4.13; v0.5.0 12.30 MB). Results and timing unchanged (all 171
  host tests, 16 Python tests). OpenCV 5.0 was not taken: it moves AKAZE out
  of the main repository into opencv_contrib.
- The deprecated `cv::Mat_ << ...` comma initialiser (a `-Werror` failure on
  4.14) is replaced by `cv::Matx`.
- Eigen stays at 5.0.1, the latest release. CI actions: `checkout` v7,
  `cache` v6, `setup-python` v7, `upload-artifact` v7 (Node 24 runtime).
  Python build requires `pybind11>=3.0` (built and tested with 3.1.0).

### Fixed
- The mesh guess field's 5x5 smoothing moved nodes on the grid or mesh edge
  by gradient x offset (about 5 px at a quarter turn), outside IC-GN's reach,
  so every outer-ring node went to a Simplex rescue under large rotation.
  u/v are now carried back to the node along the smoothed gradient; interior
  nodes are unchanged bit for bit. 45-90 deg: Simplex rescues 1-83 -> 0,
  field CPU 1.3-5x lower, accuracy unchanged. `FullFieldGolden` re-captured.

## 0.5.0

Minor: additive surface, Frozen values unchanged, numerics move.

### Added
- `kSeedFailed` / `SEMPER_ERR_SEED` / `semper.ERR_SEED` (`-5`): nothing could
  seed the solve. This case returned an undocumented `-1`.
- Metrics slot 23, points dropped by the displacement median test, which
  closes `3 + 4 + 19 == 1 + 20 + 23`; `kMetricsLen` / `SEMPER_METRICS_LEN` = 24.
- `include/semper/metrics.hpp`: `kMetricsLen`, `metrics_prefix_len()`, the one
  prefix table the solver and the JNI adapter share.
- `io::to_gray8`: any image to 8-bit gray, non-8-bit depths stretched from
  their own range.

### Fixed — memory and process safety
- An even or zero `subset_size` overran the subset buffers (heap overflow);
  `run_full_field` and `IncrementalSession::init` return `-3` for it.
- The rect is clipped to the image (lattice kept); an oversized one no longer
  throws `std::bad_alloc` through the C ABI / JNI. Nothing on the image: `-2`.
- `strain_window` above `2 * (min(gridW, gridH) + 1) * step` is `-3`: such a
  window can never meet the strain fit's 90% coverage rule (the field was
  empty anyway), and at `INT_MAX` the window count looped for hours.
- No exception crosses the C ABI or JNI (`guarded_call`, `-3`); exceptions in
  the OpenMP regions are caught and rethrown after the region instead of
  terminating the process.
- Cancel tokens are passed down explicitly: concurrent engines no longer share
  one global "active token" pointer (cross-cancellation, dangling pointer).
  The progress callback is no longer entered from two threads at once.
- JNI preview: `targetWidth <= 0`, zero height, null JNI results and padded
  Bitmap rows are handled.

### Fixed — numerics (the field moves)
- IC-GN decides pixel validity by position, not value: black (0) speckle and
  cubic undershoot are no longer treated as off-image. IC-GN and the Simplex
  share one sampler.
- A full iteration steps with the reference Hessian again after masked ones.
- Hessians are inverted only when well conditioned (scaled 1-norm rcond,
  `kHessianMinRcond`), at precompute, with LM damping and per masked
  iteration; a non-finite or non-invertible step ends the solve cleanly.
- Incremental re-anchoring confirms a node only on a converged, accepted match.
- One median (mean of the middle pair) for the anchor lattice and the
  outlier test; Path C uses the same engine configuration (LM) as the rest.
- Encoded 16-bit images keep their contrast; an encoded file whose length
  happens to equal a raw size is still decoded; a deformed frame of another
  size or type than the reference, or a non-8-bit mask, is `-3`; a blank frame
  no longer "locks" phase correlation.

### Changed
- Rescue telemetry classifies crash/timeout by the IC-GN run that triggered
  the rescue.
- `precompute_subset` is `compute_hessian_only` + `precompute_subset_fast`.
- Python: non-uint8 arrays are scaled by `to_gray8` (a [0, 1] float image was
  truncated to black); metrics are 24 floats.
- Goldens re-captured: `GoldenCorpus` 578 → 554 recorded subsets per
  interpolator (the 24 dropped were all failed solves), `FullFieldGolden`
  translate 392 → 391 points, with slot 23 now recorded.
