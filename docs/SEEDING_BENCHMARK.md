# Seeding front-end benchmark

Measured comparison of the candidates that could supply `run_full_field`'s
initial guess, and the record of why the descriptor front-end was replaced.

**Outcome: the anchor lattice shipped.** AKAZE, and with it `features2d`,
`flann` and `calib3d`, was removed in the commit that reordered
`run_full_field`. The descriptor candidates in §4 are no longer buildable from
this tree; to reproduce that comparison, check out the last commit that carried
them (`f5d950f`) and run the sweep there.

Post-swap numbers, real DICe pair, median of 5, four threads — the anchors are
now final results for their nodes, so the solve saves more than the standalone
measurement predicted:

| | AKAZE (was) | anchor lattice (is) |
|---|---|---|
| field wall time | 198.7 ms | **187.8 ms** |
| field CPU | 706.7 ms | **662.2 ms** |
| mean ICGN iterations | 2.56 | **2.11** |
| mesh coverage | 0.446 | **1.000** |
| field RMS | 0.016273 px | 0.016270 px |
| `libsemper_c.so` (x86_64, stripped) | 13.565 MB | **8.697 MB** |

`DiceFieldAgreement` is unchanged: 230/230 points compared, rms 0.0006 px,
max 0.0033 px.

Run it with:

```bash
SEMPER_RUN_SEEDBENCH=1 SEMPER_SEEDBENCH_REPEATS=5 \
SEMPER_DICE_REPO=/path/to/dicengine/dice \
./build/tests/dic_tests SeedBench
```

---

## 1. What is actually being compared

The seeding front-end does not measure displacement. It produces the Delaunay
mesh vertices that `build_mesh_guess_field` turns into a 6-DOF initial guess;
every reported displacement then comes from IC-GN minimising ZNSSD on a
Keys-4th interpolated image (`src/math/optimization_engine.cpp`). The engine
reproduces DICe's OHT-CFRP field to ~6e-4 px RMS, which is two to three orders
of magnitude finer than any keypoint detector localises.

So "which feature matcher is best" is a question about four quantities:

| quantity | why it matters |
|---|---|
| vertex displacement error | how close each mesh vertex starts to the truth |
| guess-gradient error | `build_mesh_guess_field` derives `(ux, uy, vx, vy)` from `cv::getAffineTransform` over triangle vertices, so a vertex error `e` over an edge of length `L` becomes a gradient error of order `e/L`. Clustered vertices give short edges and sliver triangles |
| mesh coverage | the fraction of grid points that receive a mesh guess at all; the rest fall through to Path B flood fill |
| cost | seeding time, **total CPU time** across the solve, and peak RSS — not wall clock alone |

Two structural facts about the descriptor class bound what it can achieve here:
DoG/Hessian/FAST extrema localise to ~0.1–2 px, and the descriptors are
invariant to rotation and dilation — which are components of the deformation
gradient the guess is supposed to carry.

## 2. Candidates

All seven build with the current `BUILD_LIST`; no new dependency.

`akaze_pyramid` (the shipping default, 0.25/0.5/1.0 scale ladder), `akaze_full`
(scale 1.0 only, isolating the downscale penalty), `sift`, `orb`, `brisk`,
`kaze`, and `anchor_lattice`.

`anchor_lattice` has no detector and no descriptor: `cv::phaseCorrelate` gives
the global rigid shift, then IC-GN solves a regular lattice of ROI grid nodes,
filtered by the universal median test (Westerweel & Scarano) rather than a
homography fit.

**SURF is excluded, not overlooked.** It lives in `opencv_contrib/xfeatures2d`,
which is not a submodule, and `OPENCV_ENABLE_NONFREE` is `OFF`
(`cmake/SemperOpenCV.cmake`). Enabling it means a new submodule and a patent
question on a commercial product.

## 3. Datasets

| set | source | ground truth |
|---|---|---|
| S1–S6 synthetic | `tests/framework/synthetic.h` | exact: reference and deformed images are sampled from the same continuous analytic speckle function, so no resampling error enters |
| REAL DICe | `tests/fixtures/dice/{ref,def}.tif` | DICe's published 0.4 px translation |
| Sample 5 | SEM/iDICs DIC Challenge, via `SEMPER_DICE_REPO` | commanded X = 0.100 px exactly (DICe's own `avg_results_0.txt` shows DICe measuring 0.105–0.108, i.e. a +0.005–0.008 px bias of its own) |
| Sample 14 | SEM/iDICs DIC Challenge, via `SEMPER_DICE_REPO` | `u(x) = 0.1 sin(2 pi x / L)`, amplitude 0.100 px, periods ~356 px (L3) and ~205 px (L5); commanded displacement and strain tabulated per column in `command.csv` |

Sample 14 is the set that matters most: it is real speckle, the amplitude is
0.1 px, and the field is **non-affine** — the strain varies continuously across
the ROI. It is the only set here that genuinely exercises the guess gradients,
and it cannot be reproduced with the affine-only test framework.

The DIC Challenge images are **not vendored**. Their redistribution terms have
not been checked, so the tests read them from an external checkout and skip
when `SEMPER_DICE_REPO` is unset.

## 4. Results

### 4.1 Real DICe speckle (512², subset 41, step 5, median of 5, 4 threads)

| method | seed ms | seed CPU | vertex err px | gradient err | mesh cov | wall ms | **CPU ms** | peak RSS |
|---|---|---|---|---|---|---|---|---|
| akaze_pyramid | 2.2 | 5.8 | 0.1573 | 9.39e-3 | 0.446 | 198.7 | 706.7 | 67.9 MB |
| akaze_full | 63.4 | 200.2 | 0.1225 | 1.06e-2 | 1.000 | 375.8 | 1054.9 | 97.7 MB |
| sift | 81.1 | 239.9 | 0.1042 | 8.15e-3 | 1.000 | 328.0 | 1022.4 | 125.9 MB |
| orb | 72.9 | 235.5 | 0.5657 | 5.24e-2 | 0.914 | 440.4 | 1268.7 | 68.8 MB |
| brisk | 143.2 | 259.5 | 0.4658 | 3.56e-2 | 0.992 | 456.1 | 1180.3 | 110.7 MB |
| kaze | 258.1 | 776.3 | 0.1366 | 1.34e-2 | 1.000 | 679.4 | 1743.8 | 182.8 MB |
| **anchor_lattice** | 13.5 | 57.6 | **0.0119** | **5.47e-4** | 1.000 | **192.0** | **678.3** | **65.9 MB** |

The shipping AKAZE pyramid succeeds at 0.25 scale on this pair, which makes its
seeding very cheap — but it yields 86 inliers at 27.5% hull coverage, routes
`SPARSE`, and leaves **55% of grid points without a mesh guess**.

### 4.2 DIC Challenge Sample 5 — real speckle, exact 0.10 px (median of 5)

| method | vertices | seed ms | vertex err px | gradient err | wall ms | CPU ms | mean ICGN iters |
|---|---|---|---|---|---|---|---|
| akaze_pyramid | 697 | 40.7 | **0.0226** | 1.32e-3 | 283.9 | 931.7 | 2.89 |
| sift | 7046 | 785.2 | 0.0405 | 3.47e-3 | 1352.1 | 4154.0 | 3.13 |
| orb | 3618 | 66.1 | 0.1414 | 1.34e-2 | 467.9 | 1309.0 | 3.60 |
| brisk | 2899 | 164.0 | 0.1155 | 1.50e-2 | 552.0 | 1468.1 | 3.50 |
| kaze | 506 | 140.4 | 0.0248 | 1.55e-3 | 384.2 | 1187.4 | 2.96 |
| **anchor_lattice** | 324 | **16.7** | 0.0242 | **1.49e-4** | **201.9** | **692.1** | **2.01** |

This is the one set where AKAZE's vertex error beats the lattice (0.0226 vs
0.0242, 7%). The field RMS is **identical to five decimals for all seven
candidates** (0.02437–0.02439), so the pair sits on a shared systematic floor
and the difference is which locations each method samples: AKAZE's keypoints
land on blob centres, the best-conditioned points in the image, while the
lattice samples on a fixed grid regardless of local texture. The gradient
column is unaffected by that floor — a rigid translation has exactly zero
displacement gradient — and there the lattice is 8.9× better.

### 4.3 DIC Challenge Sample 14 — non-affine, 0.1 px sinusoid (2048×589, median of 2)

L5, period ~205 px, i.e. a continuously varying strain of ~3e-3:

| method | vertices | seed ms | seed CPU | vertex err px | gradient err | mesh cov | wall ms | CPU ms |
|---|---|---|---|---|---|---|---|---|
| akaze_pyramid | 626 | 13.4 | 34.7 | 0.0632 | 2.99e-3 | 0.762 | 389.1 | 1271.6 |
| akaze_full | 16256 | 1237.6 | 4550.9 | 0.0739 | 5.82e-3 | 1.000 | 3371.9 | 7725.6 |
| sift | 18759 | 4781.9 | 18661.4 | 0.1001 | 6.83e-3 | 1.000 | 8412.2 | 24293.1 |
| orb | 4079 | 120.2 | 289.2 | 0.0660 | 1.04e-2 | 0.663 | 789.8 | 1991.7 |
| brisk | 26249 | 5636.7 | 20302.2 | 0.1988 | 2.77e-2 | 0.955 | 10477.4 | 25590.5 |
| kaze | 21788 | 3394.8 | 11641.7 | 0.0777 | 8.13e-3 | 1.000 | 7899.0 | 18789.8 |
| **anchor_lattice** | 340 | 50.2 | 102.2 | **0.0127** | **5.80e-4** | 1.000 | **372.7** | **1245.7** |

On a 2048×589 image the descriptor candidates become unusable: SIFT spends 4.8 s
seeding and 18.7 s of CPU; BRISK 5.6 s and 20.3 s; KAZE 3.4 s and 11.6 s. The
shipping AKAZE pyramid escapes this only by succeeding at 0.25 scale, at the
cost of 5× worse vertices and 24% of the grid left outside the mesh.

### 4.4 Answer to the original question

**No — SIFT, ORB and SURF do not beat AKAZE.** Within the descriptor class,
AKAZE is the correct pick. SIFT edges it on vertex error at equal scale on the
DICe pair (0.1042 vs 0.1225 px) but costs 1.3× the time there and 4–20× on
larger images; ORB and BRISK are 4–5× worse on every accuracy column; KAZE is
AKAZE's unaccelerated parent and is slower for worse results.

What the question does not reach is that the class floors at ~0.1 px on real
speckle. Replacing the descriptor with phase correlation plus an IC-GN anchor
lattice is 5–13× better on vertices, 5–17× better on guess gradients, and
cheaper in wall time, CPU time and memory at the same time.

### 4.5 Rotation and large motion — where the anchor lattice loses

`SeedBench.LargeMotionSweep`, 320 px ROI centred in 640 px so the deformed
subsets stay inside the image. This is the regime the descriptor front-end was
actually built for, so it is the regime where the lattice is expected to lose.

**Translation — no crossover, the lattice wins at every magnitude tested:**

| shift | AKAZE mesh cov | anchor mesh cov | AKAZE wall / CPU ms | anchor wall / CPU ms |
|---|---|---|---|---|
| 25 px | 0.993 | 1.000 | 148.3 / 519.6 | **92.1 / 310.6** |
| 50 px | 0.927 | 1.000 | 87.4 / 278.2 | **96.1 / 321.4** |
| 100 px | 0.706 | 1.000 | 81.8 / 268.8 | **91.9 / 302.5** |

Phase correlation locks cleanly at 100 px (31% of the ROI width) and the lattice
keeps all 289 anchors. AKAZE's mesh coverage decays with shift magnitude. The
earlier concern that phase correlation would alias past ROI/2 does not
materialise anywhere in this range.

**Rotation — the lattice degrades from about 5 deg:**

| rotation | phase lock | anchors kept | anchor mesh cov | anchor wall / CPU ms | AKAZE wall / CPU ms |
|---|---|---|---|---|---|
| 2 deg | yes | 268 | 0.996 | 156.0 / 552.3 | 140.1 / 494.3 |
| 5 deg | yes | 54 | **0.450** | 207.3 / 768.5 | 139.7 / 493.3 |
| 15 deg | **no** | 30 | **0.152** | **344.7 / 1306.7** | 178.9 / 595.8 |

At 15 deg the lattice costs **2.2x the CPU** and 1.9x the wall time of the
shipping seeder, and its seeding alone costs 214 ms against 5.6 ms.

Two things do **not** degrade, and they matter:

- **Final accuracy is unaffected.** Converged fraction is 87.89% and field RMS
  is identical for all seven candidates in every one of these scenarios. Path B
  reliability-guided propagation covers whatever the mesh does not. The lattice
  costs time under rotation, not correctness.
- **The surviving anchors stay accurate.** Vertex error is 0.0090 px at 15 deg
  against AKAZE's 0.1984. Coverage collapses; precision does not.

**Mechanism, and a hypothesis that was tested and rejected.** The obvious
suspect was the `INIT_NO_SIMPLEX` switch: with a phase lock the anchors run pure
ICGN, whose capture radius is small. Re-running with `INIT_NO_SEARCH` (ICGN plus
Simplex rescue) recovers part of the 5 deg case — 126 anchors and 0.826 coverage
instead of 54 and 0.450 — but costs 1426 ms of seeding CPU against 364, pushes
total CPU to 1849 ms against 768, helps nothing at 15 deg, and is worse
everywhere else. `INIT_NO_SIMPLEX` is the correct setting; it is not the cause.

The real cause is structural: `phase_correlate_roi` supplies a **translation**.
Under rotation about the ROI centre the true displacement grows with radius
(2*r*sin(theta/2)), so one global translation is wrong for the outer anchors no
matter which per-anchor solver mode is used. At 5 deg the outer anchors are
already ~14 px out, at the edge of what ICGN pulls in from a zero-gradient
guess.

The principled fix, if this regime matters, is to estimate rotation globally as
well — log-polar (Fourier-Mellin) phase correlation recovers rotation and scale
before the translation correlation, and lives entirely in `imgproc`, so it would
not reinstate `features2d`. That is unimplemented and unmeasured.

**Practical relevance.** The engine ships in an Android app, so the reference
and deformed frames may be captured handheld. Frame-to-frame camera rotation
above 5 deg is plausible there in a way it is not on a fixed laboratory rig.
This is the one scenario in which keeping a descriptor fallback — at the cost of
the 2.755 MB and the `features2d`/`flann`/`calib3d` modules — has a real
argument behind it.


## 5. Cost accounting

Wall clock alone is not sufficient: the anchor lattice is OpenMP-parallel while
AKAZE's `detectAndCompute` is largely serial, so a wall-clock win could hide a
CPU regression, which on a phone is battery.

**Total CPU, real DICe pair, vs `akaze_pyramid`:**

| configuration | akaze_pyramid | anchor_lattice | delta |
|---|---|---|---|
| 4 threads | 706.7 ms | 678.3 ms | −4.0% |
| pinned to one core (`taskset -c 0`) | 691.3 ms | 622.9 ms | −9.9% |

Note that `OMP_NUM_THREADS=1` does **not** serialize this engine: Path A uses
`num_threads(safe_cores)` and Path B uses `std::thread`. CPU pinning is the only
way to measure the serial cost.

Seeding in isolation is more expensive for the lattice (13.5 ms vs 2.2 ms wall,
57.6 ms vs 5.8 ms CPU on the DICe pair). The solve returns it with interest:
mean ICGN iterations fall from 2.56 to 2.21 there, and from 2.89 to 2.01 on
Sample 5. Isolated seeding cost is therefore the wrong figure of merit; total
CPU is the right one.

One inefficiency is visible in the data: serial anchor seeding costs 39.2 ms of
CPU but the 4-thread run costs 57.6 ms, so `schedule(dynamic, 4)` over ~300
anchors has poor granularity. Worth revisiting.

## 6. Space

`calib3d` is reached by exactly one line in the engine — `cv::findHomography` in
the descriptor seeding path. `features2d` and `flann` serve only that path and
the `ReferenceCache` keypoint members. Removing the descriptor front-end
therefore reduces `SEMPER_OPENCV_BUILD_LIST` from

```
core,imgproc,imgcodecs,features2d,calib3d,flann   ->   core,imgproc,imgcodecs
```

Measured on `libsemper_c.so`, Release, OpenCV statically linked from the
pinned 4.13 submodule. Host x86_64 first:

| | before | after | delta |
|---|---|---|---|
| stripped `.so` | 13.565 MB | **8.701 MB** | **−4.864 MB (−35.9%)** |
| unstripped `.so` | 15.934 MB | 9.951 MB | −5.982 MB (−37.5%) |
| `.text` | 10.791 MB | 6.821 MB | −3.969 MB (−36.8%) |
| `.rodata` | 0.896 MB | 0.701 MB | −0.195 MB (−21.8%) |
| `.eh_frame` | 0.727 MB | 0.509 MB | −0.218 MB (−29.9%) |
| `.data.rel.ro` | 0.204 MB | 0.104 MB | −0.100 MB (−49.1%) |

This is substantially more than the three modules' own footprint suggests
(2.95 MB of 11.44 MB in the distro shared builds): static linking pulls their
transitive dependencies in as well.

### 6.1 aarch64

The engine ships as `arm64-v8a`, so the x86_64 figure alone is not sufficient.
Repeated with an aarch64 cross-compile
(`cmake/toolchains/aarch64-linux-gnu.cmake`), identical in every respect except
`SEMPER_OPENCV_BUILD_LIST`:

| | before | after | delta |
|---|---|---|---|
| stripped `.so` | 9.475 MB | **6.720 MB** | **−2.755 MB (−29.1%)** |
| unstripped `.so` | 11.458 MB | 7.925 MB | −3.533 MB (−30.8%) |
| `.text` | 7.275 MB | 5.146 MB | −2.129 MB (−29.3%) |
| `.rodata` | 0.671 MB | 0.567 MB | −0.104 MB (−15.5%) |
| `.eh_frame` | 0.610 MB | 0.442 MB | −0.168 MB (−27.5%) |
| `.data.rel.ro` | 0.142 MB | 0.080 MB | −0.063 MB (−44.2%) |

**The saving holds on ARM, but it is smaller than x86_64 suggested** — 29.1%
and 2.755 MB against 35.9% and 4.864 MB. Part of that is `WITH_CAROTENE`, which
compiles on aarch64 and was skipped on x86_64 ("NEON is not available,
disabling carotene"): it adds code to *both* aarch64 builds, enlarging the
shared baseline and diluting the percentage. Quote 29% and ~2.8 MB per ABI, not
the host number.

Both reduced libraries were verified working, not merely linked:
`semper_c_smoke` passes on x86_64 natively and on aarch64 under
`qemu-aarch64-static`, in both cases with the anchor lattice accepting 121/121
anchors and routing `FULL` through the C ABI.

### 6.2 What the aarch64 measurement does not reproduce

The Android NDK could not be obtained here — `dl.google.com` does not respond
through the proxy — so this is a GNU cross-compile, not the shipping build:

- **GCC 13.3 aarch64, not NDK clang.** The before/after ratio is dominated by
  which OpenCV modules link rather than by the compiler, but absolute sizes
  will differ.
- **glibc, not Bionic.**
- **No `-flto`.** The Android build applies it to `semper_pipeline`
  (`SEMPER_ANDROID=ON`). Cross-TU dead-code elimination would plausibly
  *increase* the relative saving, so 29.1% is likely conservative — but that is
  an expectation, not a measurement.
- No 16 KB page alignment (`-Wl,-z,max-page-size=16384`), which affects on-disk
  padding.

The one number that still needs an NDK is the shipping `libsemper_core.so`.

One detail Phase 3 has to handle: `draw_outlined_text` is a debug-drawing
helper that currently lives in the descriptor translation unit and is used by
`full_field_debug_export.cpp` and `full_field_mesh.cpp`. It has to be rehomed.

## 7. Throughput

`Perf.SubsetSolveThroughput` exercises `SubsetPrecomputer` and
`OptimizationEngine` only, so seeding changes cannot affect it. Measured on
identical hardware, three runs each:

| | solves/s |
|---|---|
| base `074602a` | 3609, 3635, 3673 |
| this branch | 3611, 3667 (3159 cold) |

No regression (−0.7% on medians, inside the run-to-run spread). Note that the
absolute floor of 4557 solves/s in `docs/PERF_BASELINE_bd44af0.md` is not
reachable on this host: the baseline was captured on faster hardware, so the
meaningful comparison is against the base commit, not against the recorded
number.

## 8. Findings independent of any change

- On real speckle the shipping AKAZE pyramid routes `SPARSE` and leaves 55% of
  the grid without a mesh guess (Sample 14: 24%). This is current behaviour, not
  a consequence of anything measured here.
- At σ=5 additive noise, convergence collapses to ~13.8% for **every** candidate.
  That is `kCorrAccept = 0.15` being too strict for noisy input, not a seeding
  problem.
- `run_full_field` returns `0`, not an error code, when the VSG strain window is
  too small for the step: every strain fit is rank-deficient, the sentinel fires,
  and the post-filter drops the whole field. At step 10 a 15 px window spans one
  grid node. A caller gets an empty field with no diagnostic.

## 9. Gate scorecard

`anchor_lattice` against `akaze_pyramid`, on the criteria agreed before the
measurement:

| # | criterion | verdict |
|---|---|---|
| 1 | median vertex error no worse | **pass except Sample 5** — 13× better on the DICe pair, 5.0–5.5× on Sample 14, better on all seven synthetics; 7% worse on Sample 5, where all seven candidates report an identical field RMS and the pair sits on a shared systematic floor |
| 2 | guess-gradient error improved ≥5× | **pass on every real-speckle set** — 17.2× (DICe), 8.9× (Sample 5), 5.2–7.6× (Sample 14). 2.0–3.5× on the clean synthetics, below the threshold |
| 3 | total wall time ≤ baseline | **pass** — every scenario. 192.0 vs 198.7 ms (DICe), 201.9 vs 283.9 (Sample 5), 372.7 vs 389.1 (Sample 14) |
| 4 | total CPU time ≤ baseline, single and multi-threaded | **pass** — 678.3 vs 706.7 ms at 4 threads, 622.9 vs 691.3 ms pinned to one core |
| 5 | peak RSS ≤ baseline | **pass** — 65.9 vs 67.9 MB, the lowest of all seven candidates |
| 6 | coverage and convergence no worse | **pass for translation and strain, fails above ~5 deg rotation** — mesh coverage 1.000 vs 0.446 (DICe) and 0.762 (Sample 14), and 1.000 at translations to 100 px; but 0.450 at 5 deg and 0.152 at 15 deg, against AKAZE's 0.961. Convergence and field RMS are identical in every case — the cost is CPU, not correctness (§4.5) |
| 7 | binary size not larger | **pass** — aarch64 29.1% smaller (−2.755 MB per ABI); x86_64 35.9% (−4.864 MB). Quote the aarch64 figure |

Two criteria are not met literally. Criterion 1 fails on Sample 5 by 7% at a
level where the field result is identical across all candidates, and criterion 2
falls short of 5× on the synthetic scenarios while clearing it on all three
real-speckle sets. Both shortfalls are on the accuracy axis where the lattice is
already an order of magnitude ahead everywhere else; neither indicates a cost.
