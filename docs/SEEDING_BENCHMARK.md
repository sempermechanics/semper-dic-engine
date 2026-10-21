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

**Rotation — the translation-only lattice degraded from about 5 deg** (as first
measured; fixed in §4.5.1):

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

**Practical relevance.** The engine ships in an Android app, so the reference
and deformed frames may be captured handheld. Frame-to-frame camera rotation
above 5 deg is plausible there in a way it is not on a fixed laboratory rig.
This was the one scenario in which keeping a descriptor fallback — at the cost
of the 2.755 MB and the `features2d`/`flann`/`calib3d` modules — had a real
argument behind it. §4.5.1 removes that argument.

#### 4.5.1 Fixed: log-polar (Fourier-Mellin) rotation and scale

L0 now estimates rotation and scale as well as translation, still `imgproc`
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

Step 4 was not anticipated above, and it is half the fix. At 15 deg, lattice
neighbours 20 px apart differ by 5 px. On the lattice edge every neighbour
lies on one side, so the raw neighbourhood median is offset by that much and
each edge anchor fails the 1 px `kAnchorMedianTol` however accurate it is. With
better seeds alone the outer ring would still have been rejected, and the
hull would have shrunk to about 0.77 of the ROI.

The similarity is used only when all three of these hold:
- it moves the ROI corners more than `kSimilarityMinCornerPx` = 4 px beyond
  the bare translation;
- the rotation-corrected translation peak beats the uncorrected one;
- the lattice it seeds routes `FULL` and keeps at least
  `kSimilarityKeepFraction` = 90% of its anchors, or, failing that, it beats
  the lattice re-solved from the translation alone: it routes `FULL` and
  that one does not, or both route `FULL` and it keeps more (§4.5.3).

Otherwise every seed is the translation-only value, bit for bit. When the last
condition is decided by the comparison, the lattice has been solved twice.
`FullFieldGolden` passes unchanged: its affine scenario implies 2.8 px at the
corners and stays on the translation path, which
`SeedBench.LogPolarRotationScale` asserts directly.

The third condition was added after IncBench
([INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md)) showed what happens
without it. Under strong uniaxial stretch the log-polar correlation still
returns a peak, though a similarity cannot describe a stretch: −5.8 deg × 0.92
for a 45% stretch, and about 90 deg once noise is added. The stretch has
already weakened the translation peak, so the second condition passes. The
lattice that model seeds can hold a self-consistent cluster of wrong anchors
that neither the ZNSSD gate nor the residual median test rejects. On that
noise-free frame the field went from 0.014 to 0.215 px rms, and with noise
frames were lost outright: 27 against 14 dropped in the ring-mask scenario.
Counting kept anchors cannot pick the better lattice, since the bad one keeps
more (123 against 86). Only a lattice that keeps nearly everything shows that
the model fits. `SeedBench.SimilarityFallsBackUnderStretch` pinned that frame:
0.2150 px with the condition disabled, 0.0137 px with it. (Since §4.5.2 the
estimate reads that frame as −6.1 deg × 0.897, the corrected peak no longer
beats the uncorrected one, and the frame stops at the second condition with
the same 0.0137 px. The test then pinned a 40% stretch that still reached the
third, −42.4 deg, whose lattice kept 24 anchors at a 31.6 px median vertex
error against the translation lattice's 16 at 0.008 px. That frame's
log-polar response is 0.145, so §4.5.2's response floor now stops it before
the lattice; the test asserts that.) The "keeps more" half of the
comparison added in §4.5.3 is not the count rejected here: it only ranks two
`FULL` lattices. Every wrong cluster above sat in a lattice that routes
`SPARSE` (123, 86, 24 and 16 of 484), and between two `SPARSE` lattices the
translation's is used, as before.

**Measured** on Ubuntu 22.04 (WSL2), GCC 11.4, `taskset -c 0-3`,
`SEMPER_SEEDBENCH_REPEATS=11`, both columns on the same host. "Before" is
`045a500` with the same sweep rows added; the AKAZE column above cannot be
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

"mesh cov (frac)" is the fraction of grid points with a mesh guess, the
quantity the earlier tables in this section call mesh coverage. "hull cov" is
the anchors' convex hull as a fraction of the ROI area; 0.969 is the full
lattice, whose outermost nodes sit one step inside the ROI edge. Every rotated
scenario now locks (`lock` = `sim` in the sweep's table) and routes `FULL`.
The estimate reads 4.99, 14.94 and 9.96 deg × 1.049 on the 5 deg, 15 deg and
combined rows.

What this buys:

- **Rotation no longer costs.** Seeding CPU is flat at 32–38 ms from 2 to
  30 deg, against 84–805 ms before. Field CPU at 15 deg falls 2.8×, at 30 deg
  6.4×. Mean ICGN iterations fall too, because Path A now starts from a mesh
  that covers the whole ROI and carries the rotation's gradients.
- **Against AKAZE, by inference only.** On the old host the lattice cost
  1.12×, 1.56× and 2.19× AKAZE's CPU at 2, 5 and 15 deg. Applying this host's
  before/after ratios (0.87, 0.56, 0.35) puts it at about 0.97×, 0.88× and
  0.78×, i.e. at or below the descriptor seeder throughout. That combines two
  hosts, so treat it as an estimate, not a measurement.
- **Accuracy is unchanged.** Converged fraction (87.89%) and field RMS agree
  to the fifth decimal in every scenario. As before, Path B had been covering
  for the missing mesh; the fix removes the time cost.

What it costs:

- **The estimate is not free on pure translation.** One N² DFT plus a 256×128
  correlation adds 1.3–1.6 ms of serial time per solve: the L0 phase goes from
  3.0–3.3 to 4.3–4.4 ms on the 320 px ROI, and from 4.5 to 6.2 ms on the
  400 px DICe ROI. Measured field CPU on the translation rows and the DICe pair is up 1.7–3.4%.
  That is more than the estimate itself, but the solve that follows is
  identical work (same seeds, same iterations, same RMS), and the same rows
  spread 199.5–204.9 ms across repeats of the unchanged code. **Strictly, 0 deg
  regresses by the estimate's 1.3–1.6 ms; from 2 deg up, CPU is lower.**
- **Limits.**
  - The magnitude spectrum repeats every 180 deg, so rotation is recovered in
    (−90, 90) deg; within 10 deg of the wrap both readings are checked
    against the image (§4.5.2), which extends that to ±100 deg. Scale is
    accepted in (0.5, 2).
  - An ROI whose short side is under 64 px gets no estimate and stays
    translation-only — a 1000×30 beam ROI, for example.
  - ~~Small noisy ROIs get nothing from it.~~ Fixed in §4.5.2. As first
    measured: at σ = 2 grey-level noise the log-polar correlation found no
    peak on a 110 px ROI from 5 to 45 deg, while on a 200 px ROI it was
    within 0.05 deg at every angle. Dropping the log compression rescued most
    of the noisy 110 px cases but broke noise-free ones at 20–45 deg. The
    cause turned out to be where the peak was looked for, not the spectrum.
  - Anisotropic stretch is outside the model. Since §4.5.2 most such frames
    score under the log-polar response floor and stay translation-only at no
    extra cost. The few that pass it and keep under 90% are solved twice,
    and the third condition above keeps whichever lattice is better (§4.5.3).

#### 4.5.2 Fixed: the log-polar peak on small noisy ROIs

IncBench S4 ([INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md)) turns a
256 px image about its centre and correlates a centred 110 px ROI. At
σ = 2 GL, §4.5.1's estimate locked in only 0–7 of the 30 fixed-reference
solves per 10 frames between 5° and 85°, so the fixed reference lost coverage
from about 7° on, as it had without the estimate.

**Cause: the peak search, not the spectrum.** `cv::phaseCorrelate` puts the
peak at the highest single sample of the correlation surface and reports, as
the response, the 5×5 sum around it. Normalising the cross-power spectrum
whitens it, which turns noise into a field of one-sample spikes. The log-polar
peak, on the other hand, is spread over several samples: the spectrum varies
slowly in angle at small radii, and the window blurs it. On these frames the
true peak's response is 0.35–0.55, far above the 0.05 the estimate requires.
But the tallest single sample is almost always a spike elsewhere, whose own
response is −0.1 to 0.04. So the estimate reported no peak. A 200 px ROI has
a peak tall enough to win the single-sample search as well.

**Fix.** The estimate now runs its own phase correlation (same normalised
cross-power spectrum, response and sign convention) and puts the peak where
the response is largest: it box-sums the surface over 5×5 (periodically, as
the DFT is) and takes the maximum. The sub-pixel position is the same
weighted centroid over that window. The reference's windowed log-polar DFT is
computed once per reference instead of on every solve. The translation
correlation is untouched, so the translation-only path stays bit-identical:
`FullFieldGolden` passes without recapture, and `LogPolarRotationScale`'s
small-motion cases still match `phase_correlate_roi` bit for bit.

One related change: near ±90° the estimate wrapped to its 180° partner
(a 90.0° turn read as −89.4°), and on a noisy frame that wrong turn could
still beat the uncorrected translation peak. Within 10° of the wrap, both
readings are now undone and correlated, and the stronger peak wins. That
costs one more translation correlation on those frames only.

**Measured** with `SeedBench.LogPolarS4Sweep` on IncBench S4's frames (turn
to 90° in 0.45° steps, three noise seeds, the fixed reference: 600 solves, 300
of them up to 45°). A solve counts as locked when the similarity is applied
with the angle within 1° and the centre within 1 px:

| ROI, noise | locked to 45° (of 300) | locked to 90° (of 600) | applied but off by ≥ 1° or ≥ 1 px | worst error when applied |
|---|---|---|---|---|
| 110 px, σ = 2 | 22 → **283** | 63 → **583** | 13 → **0** | 2.25° → 0.76° |
| 110 px, σ = 0 | 204 → **282** | 456 → **582** | 3 → 0 | 1.13° → 0.31° |
| 110 px, σ = 4 | 5 → **282** | 19 → **581** | 18 → 6 | 94.5° → 1.34° |
| 200 px, σ = 2 | 291 → 291 | 591 → 591 | 0 → 0 | 0.27° → 0.25° |
| 80 px, σ = 2 | 1 → **259** | 10 → **533** | 31 → 45 | 151.5° → 2.23° |
| 80 px, σ = 0 | 18 → **273** | 33 → **573** | 30 → 0 | 110.4° → 0.72° |

Locks per 10 frames on the 110 px ROI at σ = 2, of 30:

| frames | 1–10 | 11–20 | 21–30 | 31–40 | 41–50 | 51–60 | 61–70 | 71–80 | 81–90 | 91–100 | 101–190 | 191–200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 5 | 2 | 2 | 2 | 4 | 2 | 2 | 1 | 2 | 0 | 0–7 | 14 |
| after | 13 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 each | 30 |

The first bucket cannot reach 30. Up to 2.7° (frame 6) the turn moves this
ROI's corners by less than `kSimilarityMinCornerPx`, so the translation is
kept by design. Locked angles are 0.27° rms off at σ = 2 on 110 px (0.12° at
σ = 0, 0.08° on 200 px). At σ = 4 and on 80 px the old estimate applied
similarities up to 94–152° off; the new one is never more than 2.3° off,
which moves those ROIs' corners by at most 2.2 px.

**Alternatives that were measured first**, on the same sweep (110 px,
σ = 2). They were counted with a stricter 0.5° tolerance, under which the
shipped fix locks 267 of 300 to 45°:

| change to §4.5.1's estimate | locked to 45° (0.5° tol.) |
|---|---|
| none | 19 |
| radially symmetric (raised-cosine disc) window instead of the separable Hann | 21 |
| radius band ending at 0.25 N, where σ = 2 noise overtakes this speckle | 14 (20 with the disc window) |
| no log compression, \|F\| | 114, with many of the rest applied 0.5–1° off |
| each radius whitened (log spectrum to zero mean, unit variance over angle) | 31 |
| … and weighted by its SNR/(1 + SNR), noise floor read beyond the Nyquist circle | 57 (71 with the disc window) |
| whitened \|F\| × SNR weight, disc window | 160 |
| four highest samples, ranked by response, each verified by the rotation-corrected translation peak | 192 (199 with the disc window) |
| **peak by response (shipped)** | **267** |

The partial gains fit the peak search being the problem: dropping the log
or weighting by SNR concentrates the peak on fewer samples, and ranking
candidates by response is the fix applied to four samples only. Once the
peak is placed by response, the disc window, a band ending at 0.3 N instead
of 0.45 N, SNR weighting and dropping the log each left the 1°-tolerance
count within 3 of 300 (283 → 282–286), and verifying three candidates added
nothing (283). None of them ships. A log-polar crop larger than the ROI was
not tried. It is not needed here, and it would read material the user left
out of the ROI, such as background or grips that do not turn with the
specimen, which would add a 0° peak.

**Stretch, and a response floor.** Placing the peak by response also finds
a peak on stretched frames, where the old search mostly found a spike and
gave up. A similarity cannot describe a stretch, so these peaks are weak but
not absent. With the estimate's floor left at the translation's 0.05, they
passed to the anchor lattice, failed the keep fraction (§4.5.1, third
condition) and were solved twice. The seeds and fields came out unchanged,
but CPU did not: IncBench's stretch scenarios (S2, S5, S6, S8, S9) solved the
lattice twice on 11–74 of every 800 quick-mode solves. Over full-mode S2 that
was 417 solves, against 6 before, and S2's fixed reference used 17–23% more
CPU.

The estimate now needs a response of 0.2 (`tuning::kLogPolarMinResponse`).
The two families separate, measured on frames rendered like IncBench's
(256 px image, 110 px ROI unless noted):

| frames | log-polar response |
|---|---|
| turns, 110 px ROI, σ = 0–4 | 0.27–0.86 |
| turns, 80 px ROI, σ = 2 | 0.24 and up |
| uniaxial stretch ≥ 10%, with or without a turn | 0.11–0.20 |

| floor | 110 px σ = 2 locked to 45° | 80 px σ = 4 locked to 45° | lattice solved twice, quick S2 / S5 / S8 / S6 / S9 (of 800 each) |
|---|---|---|---|
| 0.05 | 283 | 207 | 11 / 55 / 22 / 74 / 51 |
| **0.20** | **283** | 199 | **1 / 0 / 2 / 2 / 3** |
| 0.22 | 283 | 173 | 0 / 0 / 1 / 1 / 0 |
| 0.25 | 283 | 100 | 0 / 0 / 0 / 0 / 0 |

0.2 costs nothing on the 110 and 200 px rows of the table above, or on 80 px
at σ = 2, and gives up 8 of 207 locks at 80 px σ = 4, the noisiest and
smallest case measured. Higher floors cut into that case quickly.

A frame can still reach the keep fraction. `SimilarityFallsBackUnderStretch`
runs one: a 10% stretch turned by 15° reads +11.8° × 1.016 at response 0.214.
Its similarity lattice keeps 435 of 484 anchors (89.9%, routed `FULL`), just
under the 90% bar. The translation lattice that replaced it kept 77 (routed
`SPARSE`). §4.5.3 takes this up.

**Cost.** At 0–2°, where a 110 px ROI never applies the similarity and
every solve pays for the estimate for nothing, the per-solve cost went down.
Measured with `SeedBench.LogPolarCost` (`SEMPER_LP_COST=1`), one core
(`taskset`), 400 calls per cell, median of five passes alternating the
§4.5.1 and §4.5.2 builds. The host was loaded by other jobs on other cores.

| ROI | L0 CPU per solve, µs (translation + estimate) | estimate alone | reference prep, once per reference |
|---|---|---|---|
| 110 px, σ = 0 and 2, 0/1/2° | 1707–1850 → 1586–1739 (−12% to +2%) | 1235–1397 → 1094–1256 | 827–842 → 1052–1125 |
| 320 px, σ = 0 and 2, 0/1/2° | 6032–9975 → 5910–10078 (−4% to +1%) | 2519–6456 → 2241–6454 | 2546–2841 → 2767–2790 |

The saving is the reference's log-polar DFT, now computed once per reference
instead of on every solve. The box sum and the wider centroid cost less than
that. Prep grows by 0.2–0.3 ms on 110 px, once per reference. At 320 px the
1° and 2° cells apply the similarity in both builds and cost the same.
`LargeMotionSweep`'s field CPU, three alternated runs each: 0.975× at 2° and
0.985–0.998× on the translation rows, with identical field rms.

**IncBench.** These are re-runs of S4 and the stretch scenarios against
`5ea4fc0`; [INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md) has the tables.
S4 at σ = 2 on the 110 px ROI now keeps the fixed reference at ≥ 90%
coverage on all 600 solves at 0.45°/f (268 were below and 46 empty), and on
every frame at 2°/f and 5°/f. Incremental steps of 10–50° track at σ = 2: at
2°/f, N = 5 ends at 0.185 px / 95% instead of a wrong lock at 12.7 px, and
N = 10 at 0.134 px / 97% instead of losing the chain. The lowest response of
any similarity applied in S4 is 0.41, far above the floor. On S2, S5, S6, S8
and S9 the printed cells are unchanged, or within 0.012 px and mostly lower.
The exception is S6's fixed reference at frame 50, which improves from 0.233 to
0.016 px. Summed CPU per frame is 0.976–0.994× `5ea4fc0`'s. S4's accuracy
is identical, to every CSV column, at floors of 0.05 and 0.2.

#### 4.5.3 Fixed: which lattice the keep fraction falls back to

With the response floor in front of it, the question was whether
`kSimilarityKeepFraction` = 0.90 had become stricter than it needs to be.
Below 0.90 the lattice was re-solved from the translation alone and that
lattice was used, however it came out. The answer is that the threshold is
right but the fallback was not. The similarity's lattice is now kept when it
routes `FULL` and the translation's does not, or both route `FULL` and it keeps
more anchors. Otherwise the translation's is used, as before.

**Measured** against `16c924b` (§4.5.2), on Ubuntu 22.04 (WSL2), GCC 11.4, each
variant pinned to six cores:
- a SeedBench scan (a temporary test, not committed) of 168 frames rendered
  like `SimilarityFallsBackUnderStretch`: 384×256 px image, 110 px ROI, stretch
  5–50% (Poisson 0.3) × turn 0–45° × σ = 0 and 2, median of five runs;
- IncBench S2, S5, S6, S8 and S9 with one seed and every σ and variant: 16,750
  solves;
- the same in quick mode: 4000 solves.

**Where the fallback fires.** On 6 of the 168 scan frames, all noise-free; on
8 of 4000 quick-mode solves; on 115 of 16,750 one-seed solves (S2 31, S5 3,
S6 21, S8 43, S9 17). What separates the two outcomes is the scenario, not
the keep fraction:

| scenario | fallbacks | similarity lattice keeps | translation lattice |
|---|---|---|---|
| S2, pure slow stretch, read as a spurious −2.7 to −3.5° (once +2.8°) × 1.04–1.07 | 31 | 78–88%, `FULL` | `FULL`, keeps more in all 31 |
| S5, 60% stretch | 3 | 69–75%, `FULL` | `FULL`, keeps more in all 3 |
| S6, 40% stretch + 45° + 30 px | 21 | 67–90%, `FULL` | `SPARSE` in 15, keeps fewer in 20 |
| S8, 20% stretch + 10° | 43 | 59–90%, `FULL` in 42 | keeps fewer in all 43 |
| S9, ring mask, 72-node lattice | 17 | 13 `SPARSE`, 4 `FULL` | — |
| scan, 10–15% stretch + 10–45° | 6 | 72–90%, `FULL` | `SPARSE` in 5, with p95 vertex error 12–53 px |

The S2 events keep 78–88%, inside the range the S6 and S8 events span. So no
threshold sends S6 and S8 to the similarity without also sending S2 there,
where the translation is the right model.

**Thresholds against the comparison**, one-seed IncBench, rows of the CSV
(one per frame, mode and variant) against `16c924b`:

| fallback rule | lattice solved twice | rows changed | better / worse by > 0.002 px | by > 0.02 px | good points | CPU of the changed rows |
|---|---|---|---|---|---|---|
| 0.90, translation always (`16c924b`) | 115 | — | — | — | — | — |
| 0.85, translation always | 94 | 21 | 3 / 1 | 2 / 1 | +109 | 4.9 → 1.7 s |
| 0.80, translation always | 66 | 48 | 4 / 3 | 3 / 1 | +291 | 11.2 → 4.5 s |
| 0.90, the lattice that keeps more | 115 | 73 | 4 / 0, + 1 frame recovered | 3 / 0 | +935 | 22.8 → 16.2 s |
| **0.90, `FULL` and (the other not `FULL` or keeps more)** | **115** | **65** | **4 / 0** | **3 / 0** | **+773** | **21.4 → 14.3 s** |

The rows that moved by more than 0.02 px, or that a lower threshold made worse:

| row | `16c924b` | 0.85 | 0.80 | shipped |
|---|---|---|---|---|
| S6 σ = 0 fixed, frame 55 | 0.163 px, max 3.09, 361 good | **0.015**, max 0.12, 400 | **0.015** | **0.015** |
| S6 σ = 0 fixed, frame 57 | 0.167 px, max 3.31, 393 good | 0.167 | **0.014**, max 0.09, 400 | **0.014** |
| S8 σ = 2 fixed, frame 95 | 0.120 px, max 1.93 | **0.072**, max 0.35 | **0.072** | **0.072** |
| S8 σ = 2 fixed, frame 93 | 0.0998 px, 398 good | 0.0900, 390 | 0.0900, 390 | 0.0900, 390 |
| S2 σ = 1 fixed, frame 108 | 0.040 px, max 0.28 | 0.063, max 1.02 | 0.063, max 1.02 | 0.040 |
| S2 σ = 2 fixed, frame 108 (and its 6×6 variant) | 0.094 px, max 0.48 | 0.094 | 0.108, max 1.18 | 0.094 |

So under the old rule the fallback did the harm it was written to prevent.
On S6 frames 55 and 57 the translation lattice it fell back to was the wrong
cluster, and it carried 3 px errors into the field. Every lowered threshold
also takes S2's similarity at frame 108, which is worse. Comparing the two
lattices, as shipped, gets the four S6/S8 rows and leaves S2 alone. Quick mode
agrees: its eight events keep 28–83%, so 0.85 changes nothing there, while 0.80
changes four frames by up to 0.014 px in both directions.

**Why `FULL` decides, and not the count alone.** Over the 168 scan frames,
every lattice that routed `FULL` was clean, whichever model seeded it: vertex
p95 at most 0.028 px at σ = 0 and 0.32 px at σ = 2. Of the 122 frames whose
lattice routed `SPARSE` under `16c924b`, 94 had a p95 over 1 px, up to a 45 px
median. The wrong clusters of
§4.5.1 all sat in `SPARSE` lattices (123, 86, 24 and 16 of 484). There, the
wrong lattice kept more, which is why counting alone was rejected. Ranking
by count only between two `FULL` lattices keeps that case on the translation.
Counting between any two lattices, the fourth row of the table above, also
takes `SPARSE` over `NONE`. That recovered one empty S9 frame (0 → 136 points
at 0.083 px). It was not shipped, because one frame is not enough to trust a
`SPARSE` similarity lattice.

**The pinned frames** (`SimilarityFallsBackUnderStretch` now asserts three
of them):

| frame | similarity lattice | translation lattice | `16c924b`: vertex p95, field rms / points, field CPU | now |
|---|---|---|---|---|
| 10% + 15° | 435 `FULL` | 77 `SPARSE` | 12.1 px, 0.0148 / 400, 453 ms | similarity: 0.019 px, 0.0186 / 400, 224 ms |
| 10% + 45° | 428 `FULL` | 42 `SPARSE`, median 30 px | 53.4 px, 0.0136 / 315, 1509 ms | similarity: 0.019 px, 0.0135 / 400, 520 ms |
| 12% + 30° | 410 `FULL` | 24 `SPARSE` | 47.0 px, 0.0121 / 338, 584 ms | similarity: 0.019 px, 0.0141 / 400, 272 ms |
| 12% + 45° | 388 `FULL` | 37 `SPARSE` | 53.7 px, 0.0135 / 324, 1214 ms | similarity: 0.023 px, 0.0137 / 400, 572 ms |
| 15% + 10° | 351 `FULL` | 473 `FULL` | 0.022 px, 0.0160 / 400, 388 ms | translation, unchanged |
| 15% + 30° | 356 `FULL` | 25 `SPARSE` | 50.5 px, 0.0107 / 347, 741 ms | similarity: 0.022 px, 0.0116 / 389, 403 ms |

Where the similarity is now kept, the field covers 389–400 points instead
of 315–400. Its rms moves by −0.0001 to +0.0038 px, all well inside the
test's 0.05 px. Field CPU roughly halves on those frames. The worst rms move
is on the first row, the frame the question was asked about: it is 0.0148 px
from the translation lattice's mesh and 0.0186 px from the similarity's.

**Cost.** The lattice is still solved twice on every frame that keeps under
90%: 115 of 16,750 solves here, as before. A lower threshold would save some
of those second solves (94 at 0.85, 66 at 0.80), but only by giving up the
comparison, which is what fixes S6 and protects S2. On the rows that change,
CPU falls by a third, because the field solve after a `FULL` mesh is cheaper
than after a `SPARSE` one. Summed over a scenario, the difference is below
what this host resolves: identical code re-run on other cores differed by up
to 18% in summed CPU.

**The old misfire.** The 40% stretch that once seeded a 31.6 px cluster still
never reaches the lattice. All 70 scan frames at 30–50% stretch, σ = 0 and 2,
0–45°, score at most 0.186 against the 0.2 floor. If one did reach it, both its
lattices are `SPARSE` (24 against 16 of 484), so the translation's would be
used, as before.


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
CPU is the right one. (Every number in this paragraph is the original
measurement, kept because the AKAZE column cannot be re-measured — that code is
gone. The lattice half has since improved: 5.1.)

One inefficiency was visible in the data as first measured: serial anchor
seeding cost 39.2 ms of CPU but the 4-thread run cost 57.6 ms — a 47% overhead
— which was attributed to `schedule(dynamic, 4)` over ~300 anchors having poor
granularity. **That was measured, and the attribution was wrong.** See 5.1.

### 5.1 The anchor loop's OpenMP schedule, measured

Re-run on the DICe pair after the 0.3.0 anchor-phase refactor, on Ubuntu 22.04
/ GCC 11.4, with `SEMPER_SEEDBENCH_REPEATS=21` (medians, not single shots — one
shot moves by ~10% here) and `taskset` fixing the core count, since
`std::thread::hardware_concurrency()` reads the affinity mask and
`OMP_NUM_THREADS` does not reach `num_threads(safe_cores)`. Seeding phase only:

| schedule | 1 core CPU | 4 cores wall / CPU | 8 cores wall / CPU |
|---|---|---|---|
| `dynamic, 4` (shipping) | 27.2 ms | **11.0** / 29.7 ms | **8.8** / 33.6 ms |
| `dynamic, 16` | 26.4 ms | 11.7 / 29.8 ms | 9.1 / 33.8 ms |
| `static` | 26.7 ms | 12.6 / 30.5 ms | 10.0 / 32.6 ms |

Run-to-run spread on the serial column across these three builds is 26.4–27.2 ms,
so treat anything under ~3% as noise. Two things follow, and neither is what
the paragraph above predicted:

- **`schedule(static)` is worse, not better** — 14% more wall at both 4 and 8
  cores, which is outside the noise floor, for no CPU saving. `dynamic, 16` is
  indistinguishable from `dynamic, 4` on CPU and slightly worse on wall. The
  shipping choice is the best of the three; **no change was made.**
- **The 47% overhead does not reproduce.** It is now 9% at 4 cores
  (29.7 vs 27.2) and 23% at 8. The residual is ~0.9 ms of CPU per additional
  thread and is flat in the chunk size, which is the signature of fixed
  per-thread cost (each thread constructs its own `OptimizationEngine` and
  `SubsetData`, and first-touches ~100 KB of buffers) plus memory pressure —
  not of scheduling granularity, which would move when the chunk moves.

The likeliest reason the original figure was so much larger is that it predates
the 0.3.0 refactor: the anchor loop then wrote `resultGrid` and incremented two
atomics per accepted point *inside* the parallel region, and now writes only its
own `anchors[idx]` slot (publishing moved behind the median test). A different
host is the other candidate and cannot be ruled out from here — but a 47%
overhead becoming 9% is not a machine-speed scale factor, since the serial
column moved by only 1.4x.

`OMP_WAIT_POLICY` was checked as a possible confounder and is not one: `passive`
saves 2.7 ms of CPU at 4 cores against an explicit `active`, and the default is
already within 0.5 ms of `passive`.

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

**No CI job runs the cross-build**, and that is deliberate rather than an
oversight: cross-compiling the vendored OpenCV submodule for aarch64 costs
more CI minutes than a per-commit size readout is worth. The trade is that
the toolchain file can rot unnoticed, so the header of
`cmake/toolchains/aarch64-linux-gnu.cmake` now says so and carries the recipe
that produced the table above.

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
- ~~`run_full_field` returns `0`, not an error code, when the VSG strain window
  is too small for the step.~~ **Fixed in 0.3.0.** Every strain fit was
  rank-deficient, the sentinel fired, and the post-filter dropped the whole
  field, so a caller got an empty field with no diagnostic — at step 10 a
  15 px window spans one grid node. `run_full_field` now rejects the pair up
  front with `SEMPER_ERR_STRAIN_WINDOW` (`-4`) and logs the smallest window
  that would work. Pinned by `FullField.StrainWindowTooSmallForStep_IsRejected`.
  Every in-repo call site had been on the broken side of it, including the C
  and Python smokes, which is why nothing caught it.
- ~~Anchor results were published inside the parallel solve loop, before the
  universal median test ran.~~ **Fixed in 0.3.0.** A node could be a rejected
  blunder for the mesh and an accepted output point in the field at the same
  time, and Path B would then flood fill from it — which is exactly the
  periodic-speckle failure the median test exists to catch. The lattice now
  records outcomes and `publish_anchor_results` applies them afterwards. On the
  golden fixtures every output-gated anchor also survives the median test, so
  the recorded field did not move; the fix removes a failure mode rather than
  correcting a number.
- ~~A phase-correlation lock alone suppressed Path C.~~ **Fixed in 0.3.0.**
  Phase correlation can lock on a strong global shift while every anchor fails
  the result gate; `have_seed` was true, Path C was skipped, and Path B fell
  through to a blind grid-centre seed that may be textureless. `have_seed` now
  requires an actual point, the global shift is passed to Path C as its IC-GN
  guess instead of replacing it, and a Path C failure under a lock falls
  through to Path B rather than aborting the solve — so the change can only
  add fallbacks, never remove one.
- ~~The anchor stride was clamped from the shorter grid axis.~~ **Fixed in
  0.3.0.** On a 1000×6 grid the 3-node floor for the short axis forced stride 2
  on both, giving a lattice several times `kAnchorTarget` on a geometry — a
  beam, a weld seam — this engine plausibly sees. The floor is now applied per
  axis and the pair coarsened back inside the isotropic budget, which makes it
  a no-op on square-ish ROIs.

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
| 6 | coverage and convergence no worse | **pass** — mesh coverage 1.000 vs 0.446 (DICe) and 0.762 (Sample 14), 1.000 at translations to 100 px, and, since the log-polar rotation estimate (§4.5.1), 1.000 at 2–30 deg rotation and 5% scale, against AKAZE's 0.961 at 15 deg. As first measured, translation-only, this failed above ~5 deg: 0.450 at 5 deg, 0.152 at 15 deg. Convergence and field RMS are identical in every case |
| 7 | binary size not larger | **pass** — aarch64 29.1% smaller (−2.755 MB per ABI); x86_64 35.9% (−4.864 MB). Quote the aarch64 figure |

Two criteria are not met literally. Criterion 1 fails on Sample 5 by 7% at a
level where the field result is identical across all candidates, and criterion 2
falls short of 5× on the synthetic scenarios while clearing it on all three
real-speckle sets. Both shortfalls are on the accuracy axis where the lattice is
already an order of magnitude ahead everywhere else; neither indicates a cost.

## 10. What of this is defended in CI (added 0.3.0)

Everything above is a measurement. Until 0.3.0 none of it was a gate: every
sweep in `tests/integration/test_seeding_bench.cpp` early-returns unless

    SEMPER_RUN_SEEDBENCH=1

is set, and three of them additionally need `SEMPER_DICE_REPO` pointed at an
external checkout, so CI ran exactly one assertion-bearing case from the file.
Seven gates now run unconditionally (four since 0.3.0, three since the rotation fix of §4.5.1):

- `SeedBench.AnchorSeedingQualityGate` promotes three of the seven synthetic
  scenarios — 0.4 px translation, 0.5° rotation, 12 px translation — to
  always-on assertions: phase lock held, `MeshQuality::FULL`, coverage ≥ 0.90,
  and vertex- and gradient-error bounds set at 3–5× the medians
  CandidateSweep reports for those rows, each annotated in-file. Loose
  by design: it fires when seeding stops working, not when a host is slower.
- `FullFieldGolden` pins the whole orchestration layer — seeding, mesh,
  Path A, Path B, strain — against a committed binary baseline, including
  metrics slots 19–22, so a change that merely moves work between the three
  solving paths is caught rather than averaging out.
- `SeedBench.PublishAnchorResultsRespectsMedianTest` pins the publish gate
  fixed in 0.3.0 (§8): an anchor reaches the field only if it cleared
  `kCorrAccept` *and* survived the median test.
- `SeedBench.AnchorLatticeSizing` pins the per-axis stride, including the
  1000×6 ROI that motivated it. No fixture in this repo has that geometry, so
  the planner is tested directly rather than through a solve.
- `SeedBench.LogPolarRotationScale` pins the sign and size of the
  Fourier-Mellin rotation and scale (§4.5.1), the rotation-corrected
  translation, and the switch: below `kSimilarityMinCornerPx`, including
  FullFieldGolden's affine scenario, the result must equal the translation-only
  one bit for bit.
- `SeedBench.AnchorSeedingRotationGate` promotes the 5 deg, 15 deg and combined
  10 deg + 5% + 20 px rows of LargeMotionSweep: similarity applied, `FULL`, hull
  coverage ≥ 0.90, mesh fraction ≥ 0.95, vertex error ≤ 0.05 px.
- `SeedBench.SimilarityFallsBackUnderStretch` pins the stretch guards of
  §4.5.1–§4.5.3. A 40% stretch the estimate once misread must score under
  the log-polar response floor and seed from the translation alone. Three
  frames that reach the lattice comparison must keep the lattice that routes
  `FULL` with no blunders: the similarity's for a 10% stretch turned 15° or
  45°, and the translation's for a 15% stretch turned 10°. All four must give a
  field ≤ 0.05 px.
- `SeedBench.LogPolarNoisySmallRoi` pins §4.5.2: on IncBench S4's 110 px ROI
  at σ = 2, twelve turns from 9.9° to 70.2° must lock within 1° and 1 px.

Verified by perturbing the anchor stride by one in
`src/pipeline/full_field_anchors.cpp`: the golden reports slots 3, 8, 19 and 22
moved and fails. See `docs/TESTING.md` for both suites.
