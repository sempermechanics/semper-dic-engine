# Performance / quality baseline — `bd44af0`

## Summary

- **What:** the non-regression floor for tech-debt work, captured at `bd44af0`: subset-solve speed, pipeline quality floors and release compile flags.
- **Speed:** ≥ 4797 solves/s (≤ 0.208 ms per solve) in `Perf.SubsetSolveThroughput`. Later phases must reach ≥ 95% of that: ≥ 4557 solves/s.
- **Quality:** median |U|,|V| error < 0.25 px on translation, rotation and skew, and < 0.4 px on blurred input. Do not loosen these floors.
- **Flags:** keep `-O3 -ffast-math` and `-fopenmp`. Drop `-ffast-math` only after a measured A/B that still meets the speed gate and quality floors.
- **Host:** not recorded. SEEDING_BENCHMARK.md §7 found the floor out of reach on a slower host and compared against a base commit on the same host instead.

**Commit:** `bd44af027a42c1f03b928f2cafa975e75addc799` (2026-07-31)  
**Purpose:** Non-regression floor for tech-debt work (plan: robustness without regression).

## Speed (host Release, no sanitizers)

Captured via `dic_tests` (`Perf.SubsetSolveThroughput`). Gate for later phases: **≥ 95% of 4797 ⇒ ≥ 4557 solves/s**.

| Metric | Value |
|---|---|
| Subsets precomputed | 196 |
| Subsets solved | 162 |
| Solve rate | **≥ 4797 solves/s** |
| Time per solve | **≤ 0.208 ms** |

## Quality floors (do not loosen)

From `EnginePipelineSmokeTest` (emulator):

- translation / rotation / skew: median |U|,|V| error < **0.25 px**; coverage floors unchanged
- blurred deformed: median error < **0.4 px**; min coverage **0.10**
- DICe (when OpenCV available): `RMS_TOL = 0.005`, `MIN_COMPARED_FRACTION = 0.95`

## Compile flags to preserve on release pipeline

- `-O3 -ffast-math` on math and pipeline; `-fopenmp` on pipeline.
- Android also uses `-flto` on pipeline/JNI. Host shared-lib builds omit LTO.
- Do not drop `-ffast-math` without a measured A/B that still meets the 95% speed gate and quality floors.
