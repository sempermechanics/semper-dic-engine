# Changelog

Versions follow [docs/CONTRACT.md](docs/CONTRACT.md) §A.1. Entries before
0.5.0 live in the git history (`git log --grep Version`).

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
