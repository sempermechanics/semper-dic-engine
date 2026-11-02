# Seeding front-end benchmark

Measurements behind `run_full_field`'s initial guess: which front-end supplies
the mesh vertices, and which global model seeds them.

## Summary

- **Compared:** six descriptor front-ends (the AKAZE pyramid that shipped
  before v0.2.2, AKAZE at full scale, SIFT, ORB, BRISK, KAZE) against an
  anchor lattice: phase correlation for the global shift, then IC-GN on a
  regular lattice of grid nodes.
- **Answer:** AKAZE is the right pick among descriptors; SIFT, ORB and SURF
  do not beat it. The anchor lattice beats the whole class: 5–13× lower vertex
  error (7% higher on Sample 5, at a floor all candidates share), 5–17× lower
  guess-gradient error, and lower wall time, CPU time and peak RSS (§4, §5, §9).
- **v0.2.2 shipped the lattice.** On the real DICe pair, mesh coverage rose
  from 0.446 to 1.000 and field CPU fell from 706.7 to 662.2 ms, with field RMS
  0.016273 → 0.016270 px. Dropping `features2d`, `flann` and `calib3d` saved
  2.755 MB (29.1%) on aarch64 (§6.1).
- **v0.4.0 fixed rotation.** The translation-only lattice lost coverage under
  rotation (0.450 at 5°, 0.152 at 15°). A log-polar rotation and scale
  estimate restored full coverage from 2° to 30° and cut field CPU by up to
  6.4× (§4.5, §4.5.1).
- **v0.6.0 is current.** An AKAZE + RANSAC homography replaced phase
  correlation and the log-polar estimate; the lattice is unchanged (§11).
  Coverage at 95–180° rose from 0% to 100%, and stretch plus rotation runs up
  to 17× faster on a 320 px ROI (1463 → 83 ms). Small-ROI video totals cost
  1–10% more. The three OpenCV modules are back, so §6 no longer applies.
- **Recommendation:** keep the anchor lattice seeded from the AKAZE homography.
  §1–§10 are the v0.2.2–v0.5.x record, kept as measured.

## Running it

```bash
SEMPER_RUN_SEEDBENCH=1 SEMPER_SEEDBENCH_REPEATS=5 \
SEMPER_DICE_REPO=/path/to/dicengine/dice \
./build/tests/dic_tests SeedBench
```

The descriptor candidates of §4 are no longer in this tree. To reproduce that
comparison, check out `f5d950f`, the last commit that carried them, and run the
sweep there.

## Outcome of the v0.2.2 swap

> Historical: the state after v0.2.2. v0.6.0 changed the global model (§11) and restored the three OpenCV modules.

The commit that reordered `run_full_field` removed AKAZE and, with it,
`features2d`, `flann` and `calib3d`. The anchors are final results for their
nodes, so the solve saves more than the standalone measurement predicted.

Real DICe pair, median of 5, four threads:

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

## 1. What is actually being compared

The seeding front-end does not measure displacement. It produces the Delaunay
mesh vertices that `build_mesh_guess_field` turns into a 6-DOF initial guess.
Every reported displacement comes from IC-GN minimising ZNSSD on a Keys-4th
interpolated image (`src/math/optimization_engine.cpp`). The engine reproduces
DICe's OHT-CFRP field to ~6e-4 px RMS, two to three orders of magnitude finer
than any keypoint detector localises.

So the comparison rests on four quantities:

| quantity | why it matters |
|---|---|
| vertex displacement error | how close each mesh vertex starts to the truth |
| guess-gradient error | `build_mesh_guess_field` derives `(ux, uy, vx, vy)` from `cv::getAffineTransform` over triangle vertices, so a vertex error `e` over an edge of length `L` becomes a gradient error of order `e/L`. Clustered vertices give short edges and sliver triangles |
| mesh coverage | the fraction of grid points that receive a mesh guess at all; the rest fall through to Path B flood fill |
| cost | seeding time, **total CPU time** across the solve, and peak RSS — not wall clock alone |

Two properties of the descriptor class limit it here. DoG/Hessian/FAST extrema
localise to ~0.1–2 px. The descriptors are invariant to rotation and dilation,
which are components of the deformation gradient the guess should carry.

## 2. Candidates

> Historical: the candidate set at v0.2.x, when `akaze_pyramid` shipped.

All seven built with the `BUILD_LIST` of the time; none needed a new
dependency.

| candidate | what it is |
|---|---|
| `akaze_pyramid` | the shipping default, 0.25/0.5/1.0 scale ladder |
| `akaze_full` | scale 1.0 only, isolating the downscale penalty |
| `sift`, `orb`, `brisk`, `kaze` | `features2d` detector-descriptors |
| `anchor_lattice` | no detector and no descriptor: `cv::phaseCorrelate` gives the global rigid shift, then IC-GN solves a regular lattice of ROI grid nodes, filtered by the universal median test (Westerweel & Scarano) rather than a homography fit |

SURF is excluded, not overlooked. It lives in `opencv_contrib/xfeatures2d`,
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

Sample 14 matters most. It is real speckle at 0.1 px amplitude, and the field
is non-affine: the strain varies continuously across the ROI. It is the only
set that exercises the guess gradients, and the affine-only test framework
cannot reproduce it.

The DIC Challenge images are not vendored, because their redistribution terms
have not been checked. The tests read them from an external checkout and skip
when `SEMPER_DICE_REPO` is unset.

## 4. Results

> Historical: §4.1–4.5 were measured at v0.2.x, when `akaze_pyramid` was the shipping seeder. §4.5.1 and §11 cover the later global models.

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

The AKAZE pyramid succeeds at 0.25 scale on this pair, so its seeding is very
cheap. But it yields 86 inliers at 27.5% hull coverage, routes `SPARSE`, and
leaves 55% of grid points without a mesh guess.

### 4.2 DIC Challenge Sample 5 — real speckle, exact 0.10 px (median of 5)

| method | vertices | seed ms | vertex err px | gradient err | wall ms | CPU ms | mean ICGN iters |
|---|---|---|---|---|---|---|---|
| akaze_pyramid | 697 | 40.7 | **0.0226** | 1.32e-3 | 283.9 | 931.7 | 2.89 |
| sift | 7046 | 785.2 | 0.0405 | 3.47e-3 | 1352.1 | 4154.0 | 3.13 |
| orb | 3618 | 66.1 | 0.1414 | 1.34e-2 | 467.9 | 1309.0 | 3.60 |
| brisk | 2899 | 164.0 | 0.1155 | 1.50e-2 | 552.0 | 1468.1 | 3.50 |
| kaze | 506 | 140.4 | 0.0248 | 1.55e-3 | 384.2 | 1187.4 | 2.96 |
| **anchor_lattice** | 324 | **16.7** | 0.0242 | **1.49e-4** | **201.9** | **692.1** | **2.01** |

This is the one set where AKAZE's vertex error beats the lattice: 0.0226 vs
0.0242 px, 7%. Field RMS is identical to five decimals for all seven
candidates (0.02437–0.02439), so the pair sits on a shared systematic floor.
The difference is where each method samples. AKAZE's keypoints land on blob
centres, the best-conditioned points in the image; the lattice samples a fixed
grid regardless of local texture. The floor does not affect the gradient
column, because a rigid translation has exactly zero displacement gradient;
there the lattice is 8.9× better.

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

At 2048×589 the descriptor candidates become unusable: SIFT spends 4.8 s
seeding and 18.7 s of CPU, BRISK 5.6 s and 20.3 s, KAZE 3.4 s and 11.6 s. The
AKAZE pyramid avoids this only by succeeding at 0.25 scale, at the cost of 5×
worse vertices and 24% of the grid left outside the mesh.

### 4.4 Answer to the original question

The question was whether SIFT, ORB or SURF beat AKAZE. They do not; within the
descriptor class AKAZE is the correct pick.

- SIFT edges it on vertex error at equal scale on the DICe pair (0.1042 vs
  0.1225 px), but costs 1.3× the time there and 4–20× on larger images.
- ORB and BRISK are 4–5× worse on every accuracy column.
- KAZE is AKAZE's unaccelerated parent, and is slower for worse results.

The class floors at ~0.1 px on real speckle. Phase correlation plus an IC-GN
anchor lattice is 5–13× better on vertices, 5–17× better on guess gradients,
and cheaper in wall time, CPU time and memory at the same time.

### 4.5 Rotation and large motion — where the anchor lattice loses

> Historical: the translation-only lattice, before the log-polar estimate of §4.5.1. v0.6.0 later replaced that estimate (§11).

`SeedBench.LargeMotionSweep`: 320 px ROI centred in 640 px, so the deformed
subsets stay inside the image. The descriptor front-end was built for this
regime, so this is where the lattice was expected to lose.

Translation shows no crossover. The lattice wins at every magnitude tested:

| shift | AKAZE mesh cov | anchor mesh cov | AKAZE wall / CPU ms | anchor wall / CPU ms |
|---|---|---|---|---|
| 25 px | 0.993 | 1.000 | 148.3 / 519.6 | **92.1 / 310.6** |
| 50 px | 0.927 | 1.000 | 87.4 / 278.2 | **96.1 / 321.4** |
| 100 px | 0.706 | 1.000 | 81.8 / 268.8 | **91.9 / 302.5** |

Phase correlation locks cleanly at 100 px (31% of the ROI width), and the
lattice keeps all 289 anchors. AKAZE's mesh coverage decays with shift
magnitude. The concern that phase correlation would alias past ROI/2 does not
materialise in this range.

Rotation degraded the translation-only lattice from about 5 deg (as first
measured; §4.5.1 fixes it):

| rotation | phase lock | anchors kept | anchor mesh cov | anchor wall / CPU ms | AKAZE wall / CPU ms |
|---|---|---|---|---|---|
| 2 deg | yes | 268 | 0.996 | 156.0 / 552.3 | 140.1 / 494.3 |
| 5 deg | yes | 54 | **0.450** | 207.3 / 768.5 | 139.7 / 493.3 |
| 15 deg | **no** | 30 | **0.152** | **344.7 / 1306.7** | 178.9 / 595.8 |

At 15 deg the lattice costs 2.2× the CPU and 1.9× the wall time of the AKAZE
seeder, and its seeding alone costs 214 ms against 5.6 ms.

Two things do not degrade:

- **Final accuracy.** Converged fraction is 87.89% and field RMS is identical
  for all seven candidates in every one of these scenarios. Path B
  reliability-guided propagation covers whatever the mesh does not. Rotation
  costs the lattice time, not correctness.
- **The surviving anchors.** Vertex error is 0.0090 px at 15 deg against
  AKAZE's 0.1984. Coverage collapses; precision does not.

**Not the solver mode.** With a phase lock the anchors run pure IC-GN
(`INIT_NO_SIMPLEX`), whose capture radius is small. `INIT_NO_SEARCH` (IC-GN
plus Simplex rescue) recovers part of the 5 deg case, 126 anchors and 0.826
coverage instead of 54 and 0.450, but costs 1426 ms of seeding CPU against 364
and 1849 ms total CPU against 768. It does not help at 15 deg and is worse
everywhere else, so `INIT_NO_SIMPLEX` stays.

**The cause is structural.** `phase_correlate_roi` supplies a translation.
Under rotation about the ROI centre the true displacement grows with radius
(2*r*sin(theta/2)), so one global translation is wrong for the outer anchors.
At 5 deg they are already ~14 px out, at the edge of what IC-GN pulls in from a
zero-gradient guess.

Handheld capture in the Android app makes rotation above 5 deg plausible. This
was the one real argument for keeping a descriptor fallback, at the cost of the
2.755 MB and the `features2d`/`flann`/`calib3d` modules. §4.5.1 removed it.

#### 4.5.1 Fixed: log-polar (Fourier-Mellin) rotation and scale

> Historical: the global model from v0.4.0 to v0.5.x. v0.6.0 replaced it with the AKAZE homography (§11).

L0 estimates rotation and scale as well as translation, still with `imgproc`
only (`src/pipeline/phase_correlate.cpp`; derivation in
[MATHEMATICS.md](MATHEMATICS.md) §8.1.1):

1. Phase correlation of the log-polar resampled magnitude spectra gives
   `theta` and `s`. The magnitude spectrum does not see translation, so this
   needs no prior shift estimate.
2. The deformed image is resampled with that rotation and scale undone about
   the ROI centre, and the ordinary translation correlation runs on the result.
3. Each anchor is seeded with the full similarity: the displacement the
   rotation puts at that node, and gradients `sR - I`.
4. The universal median test runs on the residual from the similarity, not on
   the raw displacement.

Step 4 is half the fix. At 15 deg, lattice neighbours 20 px apart differ by
5 px. On the lattice edge every neighbour lies on one side, so the raw
neighbourhood median is offset by that much and each edge anchor fails the
1 px `kAnchorMedianTol`, however accurate. Better seeds alone would still have
rejected the outer ring and shrunk the hull to about 0.77 of the ROI.

The similarity is used only when all three conditions hold:

1. it moves the ROI corners more than `kSimilarityMinCornerPx` = 4 px beyond
   the bare translation;
2. the rotation-corrected translation peak beats the uncorrected one;
3. the lattice it seeds routes `FULL` and keeps at least
   `kSimilarityKeepFraction` = 90% of its anchors.

Otherwise every seed is the translation-only value, bit for bit; when
condition 3 fails, the lattice is re-solved from the translation alone.
`FullFieldGolden` passes unchanged: its affine scenario implies 2.8 px at the
corners and stays on the translation path, which
`SeedBench.LogPolarRotationScale` asserts directly.

**Why condition 3 exists.** IncBench
([INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md)) showed the failure
without it. Under strong uniaxial stretch the log-polar correlation still
returns a peak that a similarity cannot describe: −5.8 deg × 0.92 for a 45%
stretch, about 90 deg once noise is added. The stretch has already weakened the
translation peak, so condition 2 passes. The lattice that model seeds can hold
a self-consistent cluster of wrong anchors that neither the ZNSSD gate nor the
residual median test rejects. Noise-free, the field went from 0.014 to
0.215 px rms; with noise, frames were lost outright (27 against 14 dropped in
the ring-mask scenario). Counting kept anchors cannot pick the better lattice,
since the bad one keeps more (123 against 86); only a lattice that keeps nearly
everything shows that the model fits. `SeedBench.SimilarityFallsBackUnderStretch`
pinned that frame until v0.6.0 (now `SeedBench.StretchSeedsWithoutBlunders`,
§11): 0.2150 px with the condition disabled, 0.0137 px with it.

**Measured** on Ubuntu 22.04 (WSL2), GCC 11.4, `taskset -c 0-3`,
`SEMPER_SEEDBENCH_REPEATS=11`, both columns on the same host. "Before" is
`045a500` with the same sweep rows added. The AKAZE column of §4.5 cannot be
re-measured, because that code is gone.

| scenario | anchors kept | mesh cov (frac) | hull cov | seed CPU ms | field CPU ms | mean ICGN iters | field RMS px |
|---|---|---|---|---|---|---|---|
| 2 deg | 268 → **289** | 0.996 → **1.000** | 0.955 → 0.969 | 84.0 → **37.7** | 396.8 → **343.5** | 3.36 → 2.70 | 0.02188 → 0.02188 |
| 5 deg | 54 → **289** | 0.450 → **1.000** | 0.443 → 0.969 | 290.1 → **34.9** | 629.6 → **355.7** | 5.57 → 2.83 | 0.02251 → 0.02251 |
| 10 deg | 99 → **289** | 0.521 → **1.000** | 0.514 → 0.969 | 666.7 → **31.9** | 1001.0 → **369.1** | 4.91 → 3.05 | 0.02042 → 0.02041 |
| 15 deg | 30 → **289** | 0.152 → **1.000** | 0.146 → 0.969 | 753.3 → **34.5** | 1084.5 → **383.6** | 5.81 → 3.26 | 0.01938 → 0.01937 |
| 30 deg | 4 → **289** | 0.059 → **1.000** | 0.055 → 0.969 | 804.7 → **35.8** | 2652.8 → **415.3** | 8.99 → 3.57 | 0.02510 → 0.02511 |
| 5% zoom | 189 → **289** | 0.906 → **1.000** | 0.816 → 0.969 | 162.7 → **35.6** | 490.5 → **338.7** | 4.20 → 2.74 | 0.02190 → 0.02190 |
| 10 deg + 5% + (20, −12) px | 61 → **289** | 0.239 → **1.000** | 0.227 → 0.969 | 726.5 → **38.0** | 1048.9 → **375.3** | 5.33 → 3.10 | 0.02170 → 0.02169 |
| 25 px shift | 289 → 289 | 1.000 → 1.000 | 0.969 → 0.969 | 32.5 → 34.3 | 204.1 → 207.6 | 1.31 → 1.31 | 0.00008 → 0.00008 |
| 50 px shift | 289 → 289 | 1.000 → 1.000 | 0.969 → 0.969 | 32.5 → 34.5 | 199.5 → 206.3 | 1.31 → 1.31 | 0.00006 → 0.00006 |
| 100 px shift | 289 → 289 | 1.000 → 1.000 | 0.969 → 0.969 | 33.0 → 35.1 | 204.9 → 209.4 | 1.32 → 1.32 | 0.00006 → 0.00006 |
| real DICe pair, 0.4 px | 289 → 289 | 1.000 → 1.000 | 0.975 → 0.975 | 32.7 → 34.4 | 468.7 → 483.6 | 2.25 → 2.25 | 0.01627 → 0.01627 |

"mesh cov (frac)" is the fraction of grid points with a mesh guess, what the
earlier tables call mesh coverage. "hull cov" is the anchors' convex hull as a
fraction of the ROI area; 0.969 is the full lattice, whose outermost nodes sit
one step inside the ROI edge. Every rotated scenario now locks (`lock` = `sim`
in the sweep's table) and routes `FULL`. The estimate reads 4.99, 14.94 and
9.96 deg × 1.049 on the 5 deg, 15 deg and combined rows.

What this gains:

- **Rotation no longer costs.** Seeding CPU is flat at 32–38 ms from 2 to
  30 deg, against 84–805 ms before. Field CPU falls 2.8× at 15 deg and 6.4× at
  30 deg. Mean ICGN iterations fall too, because Path A now starts from a mesh
  that covers the whole ROI and carries the rotation's gradients.
- **Against AKAZE, by inference only.** On the old host the lattice cost
  1.12×, 1.56× and 2.19× AKAZE's CPU at 2, 5 and 15 deg. This host's
  before/after ratios (0.87, 0.56, 0.35) put it at about 0.97×, 0.88× and
  0.78×, at or below the descriptor seeder throughout. That combines two
  hosts, so it is an estimate, not a measurement.
- **Accuracy is unchanged.** Converged fraction (87.89%) and field RMS agree
  to the fifth decimal in every scenario. Path B had been covering for the
  missing mesh; the fix removes the time cost.

What it costs:

- **Pure translation pays for the estimate.** One N² DFT plus a 256×128
  correlation adds 1.3–1.6 ms of serial time per solve: the L0 phase goes from
  3.0–3.3 to 4.3–4.4 ms on the 320 px ROI, and from 4.5 to 6.2 ms on the
  400 px DICe ROI. Field CPU on the translation rows and the DICe pair is up
  1.7–3.4%, more than the estimate itself. But the solve that follows does
  identical work (same seeds, iterations and RMS), and the same rows spread
  199.5–204.9 ms across repeats of unchanged code. Strictly, 0 deg regresses
  by 1.3–1.6 ms; from 2 deg up, CPU is lower.
- **Limits.**
  - The magnitude spectrum repeats every 180 deg, so rotation is recovered in
    (−90, 90) deg. Scale is accepted in (0.5, 2).
  - An ROI whose short side is under 64 px gets no estimate and stays
    translation-only, for example a 1000×30 beam ROI.
  - Small noisy ROIs get nothing from it. At σ = 2 grey-level noise the
    log-polar correlation finds no peak on a 110 px ROI from 5 to 45 deg,
    while on a 200 px ROI it is within 0.05 deg at every angle. Such frames
    stay translation-only, as before. Dropping the log compression rescues
    most of the noisy 110 px cases but breaks noise-free ones at 20–45 deg, so
    it is not a drop-in fix.
  - Anisotropic stretch is outside the model. Condition 3 returns such frames
    to the translation-only result, at the cost of solving the lattice twice.

## 5. Cost accounting

Wall clock alone is not enough. The anchor lattice is OpenMP-parallel while
AKAZE's `detectAndCompute` is largely serial, so a wall-clock win could hide a
CPU regression, which on a phone is battery.

Total CPU, real DICe pair, vs `akaze_pyramid`:

| configuration | akaze_pyramid | anchor_lattice | delta |
|---|---|---|---|
| 4 threads | 706.7 ms | 678.3 ms | −4.0% |
| pinned to one core (`taskset -c 0`) | 691.3 ms | 622.9 ms | −9.9% |

`OMP_NUM_THREADS=1` does not serialize this engine: Path A uses
`num_threads(safe_cores)` and Path B uses `std::thread`. CPU pinning is the
only way to measure the serial cost.

Seeding alone costs the lattice more: 13.5 ms vs 2.2 ms wall, 57.6 ms vs
5.8 ms CPU on the DICe pair. The solve repays it: mean ICGN iterations fall
from 2.56 to 2.21 there, and from 2.89 to 2.01 on Sample 5. Total CPU, not
isolated seeding cost, is the figure of merit. These are the original
measurements, kept because the AKAZE column cannot be re-measured; the lattice
side has since improved (§5.1).

As first measured, serial anchor seeding cost 39.2 ms of CPU and the 4-thread
run 57.6 ms, a 47% overhead, attributed to poor granularity of
`schedule(dynamic, 4)` over ~300 anchors. §5.1 shows that attribution was
wrong.

### 5.1 The anchor loop's OpenMP schedule, measured

DICe pair, after the 0.3.0 anchor-phase refactor, on Ubuntu 22.04 / GCC 11.4,
with `SEMPER_SEEDBENCH_REPEATS=21`. Values are medians, since one shot moves by
~10% here. `taskset` fixes the core count, because
`std::thread::hardware_concurrency()` reads the affinity mask and
`OMP_NUM_THREADS` does not reach `num_threads(safe_cores)`. Seeding phase only:

| schedule | 1 core CPU | 4 cores wall / CPU | 8 cores wall / CPU |
|---|---|---|---|
| `dynamic, 4` (shipping) | 27.2 ms | **11.0** / 29.7 ms | **8.8** / 33.6 ms |
| `dynamic, 16` | 26.4 ms | 11.7 / 29.8 ms | 9.1 / 33.8 ms |
| `static` | 26.7 ms | 12.6 / 30.5 ms | 10.0 / 32.6 ms |

The serial column spreads 26.4–27.2 ms across these three builds, so treat
anything under ~3% as noise.

- **`schedule(static)` is worse, not better.** It costs 14% more wall time at
  both 4 and 8 cores, outside the noise floor, for no CPU saving.
  `dynamic, 16` ties `dynamic, 4` on CPU and is slightly worse on wall. The
  shipping choice is the best of the three; no change was made.
- **The 47% overhead does not reproduce.** It is now 9% at 4 cores (29.7 vs
  27.2) and 23% at 8. The residual is ~0.9 ms of CPU per additional thread and
  is flat in the chunk size. That is the signature of fixed per-thread cost
  (each thread constructs its own `OptimizationEngine` and `SubsetData`, and
  first-touches ~100 KB of buffers) plus memory pressure. Scheduling
  granularity would move when the chunk moves.

The original figure most likely predates the 0.3.0 refactor. The anchor loop
then wrote `resultGrid` and incremented two atomics per accepted point inside
the parallel region; it now writes only its own `anchors[idx]` slot, and
publishing moved behind the median test. A different host cannot be ruled out,
but 47% becoming 9% is not a machine-speed factor: the serial column moved by
only 1.4x.

`OMP_WAIT_POLICY` is not a confounder. `passive` saves 2.7 ms of CPU at 4
cores against an explicit `active`, and the default is already within 0.5 ms
of `passive`.

## 6. Space

> Historical: v0.6.0 restored `features2d`, `calib3d` and `flann` for the AKAZE homography, so this saving no longer applies (CHANGELOG, 0.6.0).

At the time, `calib3d` was reached by exactly one line in the engine:
`cv::findHomography` in the descriptor seeding path. `features2d` and `flann`
served only that path and the `ReferenceCache` keypoint members. Removing the
descriptor front-end therefore reduced `SEMPER_OPENCV_BUILD_LIST` from

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

This is much more than the three modules' own footprint suggests (2.95 MB of
11.44 MB in the distro shared builds), because static linking also pulls in
their transitive dependencies.

### 6.1 aarch64

The engine ships as `arm64-v8a`, so the x86_64 figure alone is not enough.
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

The saving holds on ARM but is smaller than on x86_64: 29.1% and 2.755 MB
against 35.9% and 4.864 MB. Part of the gap is `WITH_CAROTENE`, which compiles
on aarch64 and was skipped on x86_64 ("NEON is not available, disabling
carotene"). It adds code to both aarch64 builds, which enlarges the shared
baseline and dilutes the percentage. Quote 29% and ~2.8 MB per ABI, not the
host number.

Both reduced libraries were verified working, not merely linked:
`semper_c_smoke` passes on x86_64 natively and on aarch64 under
`qemu-aarch64-static`, in both cases with the anchor lattice accepting 121/121
anchors and routing `FULL` through the C ABI.

No CI job runs the cross-build, by design. Cross-compiling the vendored OpenCV
submodule for aarch64 costs more CI minutes than a per-commit size readout is
worth. The toolchain file can therefore rot unnoticed; its header says so and
carries the recipe that produced the table above.

### 6.2 What the aarch64 measurement does not reproduce

The Android NDK could not be obtained here (`dl.google.com` does not respond
through the proxy), so this is a GNU cross-compile, not the shipping build:

- **GCC 13.3 aarch64, not NDK clang.** The before/after ratio depends mainly
  on which OpenCV modules link, not on the compiler, but absolute sizes will
  differ.
- **glibc, not Bionic.**
- **No `-flto`.** The Android build applies it to `semper_pipeline`
  (`SEMPER_ANDROID=ON`). Cross-TU dead-code elimination would plausibly
  increase the relative saving, so 29.1% is likely conservative. That is an
  expectation, not a measurement.
- **No 16 KB page alignment** (`-Wl,-z,max-page-size=16384`), which affects
  on-disk padding.

The one number that still needs an NDK is the shipping `libsemper_core.so`.

## 7. Throughput

`Perf.SubsetSolveThroughput` exercises only `SubsetPrecomputer` and
`OptimizationEngine`, so seeding changes cannot affect it. Measured on
identical hardware, three runs each:

| | solves/s |
|---|---|
| base `074602a` | 3609, 3635, 3673 |
| v0.2.2 branch | 3611, 3667 (3159 cold) |

No regression: −0.7% on medians, inside the run-to-run spread.

The absolute floor of 4557 solves/s in `docs/PERF_BASELINE_bd44af0.md` is not
reachable on this host, because the baseline was captured on faster hardware.
Compare against the base commit on the same host, not against the recorded
number.

## 8. Findings independent of any change

> Historical: observations from the v0.2.x measurement. The four struck-through items were fixed in 0.3.0.

- On real speckle the AKAZE pyramid, then shipping, routes `SPARSE` and leaves
  55% of the grid without a mesh guess (Sample 14: 24%). That was existing
  behaviour, not a consequence of anything measured here.
- At σ=5 additive noise, convergence collapses to ~13.8% for every candidate.
  That is `kCorrAccept = 0.15` being too strict for noisy input, not a seeding
  problem.
- ~~`run_full_field` returns `0`, not an error code, when the VSG strain window
  is too small for the step.~~ **Fixed in 0.3.0.** Every strain fit was
  rank-deficient, the sentinel fired, and the post-filter dropped the whole
  field without a diagnostic; at step 10 a 15 px window spans one grid node.
  `run_full_field` now rejects the pair with `SEMPER_ERR_STRAIN_WINDOW` (`-4`)
  and logs the smallest window that would work
  (`FullField.StrainWindowTooSmallForStep_IsRejected`). Every in-repo call
  site, including the C and Python smokes, had been on the broken side.
- ~~Anchor results were published inside the parallel solve loop, before the
  universal median test ran.~~ **Fixed in 0.3.0.** A rejected blunder could
  still be an output point, and Path B would flood fill from it: the
  periodic-speckle failure the median test exists to catch. Now
  `publish_anchor_results` applies the recorded outcomes afterwards. On the
  golden fixtures every output-gated anchor also survives the median test, so
  the recorded field did not move; the fix removes a failure mode.
- ~~A phase-correlation lock alone suppressed Path C.~~ **Fixed in 0.3.0.**
  With a lock but every anchor failing the result gate, Path C was skipped and
  Path B fell back to a blind, possibly textureless grid-centre seed. Now
  `have_seed` requires an actual point, the global shift becomes Path C's
  IC-GN guess instead of replacing Path C, and a Path C failure under a lock
  falls through to Path B instead of aborting. The change only adds fallbacks.
- ~~The anchor stride was clamped from the shorter grid axis.~~ **Fixed in
  0.3.0.** On a 1000×6 grid the short-axis 3-node floor forced stride 2 on
  both axes: a lattice several times `kAnchorTarget` on a plausible geometry
  (a beam, a weld seam). The floor is now per axis, with the pair coarsened
  back inside the isotropic budget; a no-op on square-ish ROIs.

## 9. Gate scorecard

> Historical: the acceptance criteria for the v0.2.2 swap. Criterion 6 was extended after §4.5.1.

`anchor_lattice` against `akaze_pyramid`, on the criteria agreed before the
measurement:

| # | criterion | verdict |
|---|---|---|
| 1 | median vertex error no worse | **pass except Sample 5** — 13× better on the DICe pair, 5.0–5.5× on Sample 14, better on all seven synthetics; 7% worse on Sample 5, where all seven candidates report an identical field RMS and the pair sits on a shared systematic floor |
| 2 | guess-gradient error improved ≥5× | **pass on every real-speckle set** — 17.2× (DICe), 8.9× (Sample 5), 5.2–7.6× (Sample 14). 2.0–3.5× on the clean synthetics, below the threshold |
| 3 | total wall time ≤ baseline | **pass** — every scenario. 192.0 vs 198.7 ms (DICe), 201.9 vs 283.9 (Sample 5), 372.7 vs 389.1 (Sample 14) |
| 4 | total CPU time ≤ baseline, single and multi-threaded | **pass** — 678.3 vs 706.7 ms at 4 threads, 622.9 vs 691.3 ms pinned to one core |
| 5 | peak RSS ≤ baseline | **pass** — 65.9 vs 67.9 MB, the lowest of all seven candidates |
| 6 | coverage and convergence no worse | **pass** — mesh coverage 1.000 vs 0.446 (DICe) and 0.762 (Sample 14), 1.000 at translations to 100 px, and, since the log-polar rotation estimate (§4.5.1), 1.000 at 2–30 deg rotation and 5% scale, against AKAZE's 0.961 at 15 deg. As first measured, translation-only, this failed above ~5 deg: 0.450 at 5 deg, 0.152 at 15 deg. Convergence and field RMS are identical in every case |
| 7 | binary size not larger | **pass** — aarch64 29.1% smaller (−2.755 MB per ABI); x86_64 35.9% (−4.864 MB). Quote the aarch64 figure |

Two criteria are not met literally:

- Criterion 1 fails on Sample 5 by 7%, at a level where the field result is
  identical across all candidates.
- Criterion 2 falls short of 5× on the synthetic scenarios while clearing it
  on all three real-speckle sets.

Both shortfalls are on the accuracy axis, where the lattice is already an
order of magnitude ahead everywhere else. Neither indicates a cost.

## 10. What of this is defended in CI (added 0.3.0)

> Historical in part: the list as of v0.5.x, with v0.6.0 renames inline. `docs/TESTING.md` has the current list.

Everything above is a measurement. Every sweep in
`tests/integration/test_seeding_bench.cpp` early-returns unless

    SEMPER_RUN_SEEDBENCH=1

is set, and three of them also need `SEMPER_DICE_REPO` pointed at an external
checkout. Until 0.3.0, CI therefore ran exactly one assertion-bearing case from
the file. Seven gates now run unconditionally (four since 0.3.0, three since
the rotation fix of §4.5.1):

| Test | What it pins |
|---|---|
| `SeedBench.AnchorSeedingQualityGate` | three of the seven synthetic scenarios (0.4 px translation, 0.5° rotation, 12 px translation): phase lock held, `MeshQuality::FULL`, coverage ≥ 0.90, and vertex- and gradient-error bounds at 3–5× the CandidateSweep medians, each annotated in-file. Loose by design: it fires when seeding stops working, not when a host is slower |
| `FullFieldGolden` | the whole orchestration layer (seeding, mesh, Path A, Path B, strain) against a committed binary baseline, including metrics slots 19–22, so a change that merely moves work between the three solving paths is caught |
| `SeedBench.PublishAnchorResultsRespectsMedianTest` | the publish gate fixed in 0.3.0 (§8): an anchor reaches the field only if it cleared `kCorrAccept` and survived the median test |
| `SeedBench.AnchorLatticeSizing` | the per-axis stride, including the 1000×6 ROI that motivated it. No fixture has that geometry, so the planner is tested directly |
| `SeedBench.LogPolarRotationScale` | the sign and size of the Fourier-Mellin rotation and scale (§4.5.1), the rotation-corrected translation, and the switch: below `kSimilarityMinCornerPx`, including FullFieldGolden's affine scenario, the result equals the translation-only one bit for bit |
| `SeedBench.AnchorSeedingRotationGate` | the 5 deg, 15 deg and combined 10 deg + 5% + 20 px rows of LargeMotionSweep: similarity applied, `FULL`, hull coverage ≥ 0.90, mesh fraction ≥ 0.95, vertex error ≤ 0.05 px |
| `SeedBench.SimilarityFallsBackUnderStretch` | the fallback of §4.5.1 on the IncBench frame that exposed it: the similarity is rejected and the field stays at the translation-only 0.014 px, not 0.215 px. Since v0.6.0 it is `StretchSeedsWithoutBlunders`: AKAZE finds no model on that frame, and the test bounds the p95 vertex error and the field instead |

Verified by perturbing the anchor stride by one in
`src/pipeline/full_field_anchors.cpp`: the golden reports slots 3, 8, 19 and 22
moved and fails. See `docs/TESTING.md` for both suites.

## 11. v0.6.0: the AKAZE homography as the global model

This is the current global model. v0.6.0 replaces the L0 of §4.5.1 (phase
correlation plus the log-polar similarity) with an AKAZE + RANSAC homography
(MATHEMATICS §8.1.2). The anchor lattice, its median test and the weak-lattice
fallback are unchanged. The acceptance run compared v0.5.0 with v0.6.0 on the
same inputs.

**Method.** Ubuntu 22.04 (WSL2), GCC 11.4, 20 logical CPUs,
`OMP_NUM_THREADS=8`, Python bindings, 3 interleaved rounds. Accuracy comes from
round 1; both builds were checked to be deterministic across rounds. Time is
the median over rounds; for incremental runs, the per-frame best of the rounds,
then the median. Synthetic frames are exact speckle at 2 GL noise; a node is
correct within 0.5 px of the truth. An Android emulator was running on the
host, so single timings vary by about ±15%: read ratios near 1 as equal.

**Fixed reference, 300 px ROI, subset 41, step 5** (3136 nodes):

| Motion | Coverage v0.5.0 / v0.6.0 | RMS px v0.5.0 / v0.6.0 | Median ms v0.5.0 / v0.6.0 |
|---|---|---|---|
| Sub-pixel shift (11 cases) | 100% / 100% | 0.0087 / 0.0087 | 69 / 71 |
| Pixel shift to 20 px (10) | 100% / 100% | 0.0078 / 0.0078 | 73 / 71 |
| Rotation 5–45° (9) | 100% / 100% | 0.0077 / 0.0077 | 76 / 71 |
| Rotation 50–90° (9) | 100% / 100% | 0.0077 / 0.0077 | 90 / 67 |
| Rotation 95–180° (11) | **0% / 100%** | — / 0.0076 | — |
| First solve on a new reference | | | 84 / 91 |

No wrong nodes in either build. v0.5.0 recovers no node past 90°, because its
log-polar estimate reads θ and θ ± 180° alike.

**ROI size sweep, fixed reference, subset 21, step 3, 6×6** (steady-state ms,
v0.5.0 / v0.6.0; coverage 100% and RMS equal to 4 decimals in every cell):

| ROI (px) | Shift (20, 10) px | Rotation 30° | Stretch 15% + rotation 15° |
|---|---|---|---|
| 60 | 36 / 40 | 80 / 42 | 123 / 43 |
| 80 | 45 / 52 | 138 / 46 | 100 / 46 |
| 120 | 56 / 54 | 53 / 53 | 66 / 57 |
| 160 | 58 / 53 | 59 / 55 | 332 / 64 |
| 240 | 74 / 69 | 69 / 70 | 558 / 63 |
| 320 | 88 / 93 | 87 / 81 | 1463 / 83 |

Under stretch, v0.5.0's similarity misfired, its lattice was rejected, and the
translation-only lattice left most nodes to the Simplex rescue. The homography
describes the stretch, and the lattice seeded from it is kept.

**Incremental, `IncrementalSession`, 40 frames, subset 21, step 3, 6×6.**
Accuracy at frame 40 is identical in every cell where both builds track, with
no wrong nodes. Two cells differ in coverage: rotation to 180° at 4.5° per
frame with N = 5 on 60 and 80 px ROIs. v0.5.0 loses every node there; v0.6.0
keeps all of them (256/441 and 484/729, RMS 0.029 and 0.026 px). Time per
frame, v0.6.0 / v0.5.0:

| ROI (px) | Stretch to 30% + 30°, N = 1 / 5 | Rotation to 180°, N = 1 / 5 |
|---|---|---|
| 60 | 0.51 / 0.45 | 0.77 / (v0.5.0 lost) |
| 80 | 1.11 / 0.94 | 0.78 / (v0.5.0 lost) |
| 120 | 1.05 / 0.87 | 0.98 / 0.79 |
| 160 | 0.90 / 0.89 | 0.74 / 0.98 |
| 240 | 0.88 / 0.82 | 0.82 / 0.77 |
| 320 | 0.86 / 0.73 | 0.82 / 0.74 |

**Tensile video** (real, about 730 frames, app ROI 80 × 72 and a taller
80 × 104):

| ROI, N | ms/frame v0.5.0 / v0.6.0 | Total s v0.5.0 / v0.6.0 | Nodes alive at the end v0.5.0 / v0.6.0 |
|---|---|---|---|
| app, 1 | 21.9 / 22.1 | 16.1 / 16.8 | 428 / 423 of 675 |
| app, 10 | 10.4 / 13.2 | 10.3 / 11.3 | 405 / 405 |
| tall, 1 | 26.0 / 27.8 | 18.9 / 20.0 | 657 / 657 of 945 |
| tall, 10 | 16.9 / 17.0 | 14.3 / 14.9 | 599 / 599 |

This is the one place v0.6.0 costs time: 1–10% on the run total. On an 80 px
ROI the coarse scales find too few inliers, so AKAZE runs at 1× on every
frame, about 1.5–3 ms. Phase correlation cost a fraction of a millisecond, and
between reference updates a frame here takes only about 10 ms.

**DIC Challenge Sample 14** (sinusoid of rising frequency, L1/L3/L5, subsets
21/31/41, 4×4 and 6×6): the two builds' u fields differ by at most 0.00008 px
on the line cut, and every Challenge metric matches.

**Thread pool.** Before AKAZE was kept off OpenCV's thread pool (ARCHITECTURE,
*OpenCV threads*), the first run of this comparison measured 7–19% slower
small-ROI incremental frames. OpenCV's idle workers were spinning into the
engine's OpenMP region: the early, near-static video frames took 31 ms instead
of v0.5.0's 10 ms. The tables above are from after the fix.
