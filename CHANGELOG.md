# Changelog

Versions follow [docs/CONTRACT.md](docs/CONTRACT.md) §A.1. Entries before
0.5.0 live in the git history (`git log --grep Version`).

## Unreleased

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
