# Fixed vs incremental reference benchmark

This compares the fixed reference (`update_interval = 0`, identical to `run_full_field`) against `IncrementalSession` moving the reference every N = 1, 2, 5 and 10 frames. It covers accuracy, precision, robustness and cost over long horizons (200–300 frames) and large motion.

## Outcome

**Where the fixed reference correlates, it is the more accurate mode, usually by a wide margin.** It correlates through more than intuition suggests. All of these solve against frame 0:
- 100% uniaxial stretch;
- 40% stretch with 45° rotation and 30 px of drift;
- smooth non-affine fields;
- noise-free, a full quarter turn, since the anchor lattice is seeded with rotation and scale.

Under heavy deformation with noise it degrades gently. It loses coverage and drops the odd frame (3–29 of 600), but stays accurate on the frames it solves.

**Incremental mode is the one that keeps going where the fixed reference stops:**
- rigid translation past the ~48 px seeding capture range;
- rotation past about 5–15° on noisy images when the ROI is small (here 110 px at σ = 2 GL);
- anything the speckle no longer resembles.

**Incremental error grows with the number of reference updates**, so the fewest updates that still correlate win, and N = 5–10 beats N = 1 wherever the larger step still tracks. Noise-free on slow loading the growth is roughly linear (interpolation bias of one sign); with noise it is a random walk.

**The 6×6 interpolator cuts the per-update bias 2–6×.**

**Incremental frames are cheaper under large deformation**, because each solve starts near its answer: 18–27 ms against 88 ms per frame on the combined scenario. Noise-free rotation now costs the fixed reference 32–43 ms per frame, against 108–117 ms with translation-only seeding.

## Rotation-aware seeding: what changed

This report was re-run when the anchor lattice gained rotation-aware seeding (SEEDING_BENCHMARK.md §4.5.1, MATHEMATICS.md §8.1.1). How it works:
- A log-polar (Fourier-Mellin) estimate gives rotation and scale.
- When those move the ROI corners by more than 4 px, they seed the lattice with a similarity instead of a bare translation.
- The lattice keeps the similarity only when it routes FULL and keeps ≥ 90% of its anchors. Otherwise it re-solves from the translation alone.

Both builds ran on the same host, interleaved scenario by scenario: `045a500` (translation-only seeding) and this one.

- **Only S4 (rigid rotation) moved, and only for the better.**
  - **Noise-free, the fixed reference holds the full quarter turn.** Coverage after the first quarter of the run averages 92–95% at every rate, against 70–71% before. One frame (f124 at 0.45°/f) is still empty, as before.
  - **Noise-free, incremental steps of 25–50° now track.** 5°/f at N = 5 and N = 10 ends at 0.010 / 0.007 px with every node, against 0.206 px / 86% and 3.4 px / 84%.
  - **With noise, small steps keep their coverage.** At 5°/f, N = 1 and N = 2 end at 94–95% instead of 62–63%. N = 5 no longer drops frames (13 → 0 of 54), and N = 10 drops 6 instead of 23.
  - **With noise, the fixed reference changes only near 90°.** On the 110 px ROI at σ = 2 the log-polar estimate locks in only a few solves between ~5° and ~85°, so these frames use the translation lattice exactly as before. See S4.
- **Every other scenario is unchanged** to the printed digit, except S6 N = 10: 0.069 → 0.065 px (σ = 0) and 0.337 → 0.336 px (σ = 2).
  - S2, S5, S8 and S9 are where the similarity misfired before the ≥ 90%-kept rule existed. Under anisotropic stretch the log-polar estimate reports a rotation and scale, and the similarity lattice can settle on a self-consistent wrong cluster. S5 at 60% then read 0.215 px at frame 150 instead of 0.014 px.
  - In this run the fallback (the second, translation-only lattice solve) fired in at most 15 of the 1000 solves of any variant (S8, T = 200). `SeedBench.SimilarityFallsBackUnderStretch` guards the case.
- **Cost:** see §3.

The tables below are the new build's. Tables other than S4 also differ in places from this report's first edition, because `045a500` landed in between. It added coasting across gaps, re-anchoring before reporting, and an early reference update when a frame reports 5% fewer points. Those numbers are identical in both builds of this run.

## Running it

```bash
SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_CSV=incbench.csv ./build/full/bin/dic_tests IncBench
python tools/plot_incremental_bench.py incbench.csv docs/images/incremental
```

| Variable | Effect |
|---|---|
| `SEMPER_INCBENCH_QUICK=1` | One seed at σ = 2 over T = 160 (the CI smoke) |
| `SEMPER_INCBENCH_ONLY=S4` | Runs one scenario |
| `SEMPER_INCBENCH_SEEDS=n` | Sets the seed count (default 3) |

The full run takes about 40 minutes on a 20-thread laptop (37 minutes for `045a500`).

![rms error vs frame, σ = 2](images/incremental/error_vs_frame_sigma2.png)

![coverage vs frame, σ = 2](images/incremental/coverage_vs_frame_sigma2.png)

---

## 1. Method

**Frames.** Every frame of a sequence is rendered from one analytic speckle (`SmoothSpeckleField`) under a known motion. The total displacement and the Green-Lagrange strain at every frame-0 node are therefore known in closed form, and no resampling error enters. Gaussian noise of σ = 0, 1, 2 or 4 GL is added per frame on 8-bit images. Noisy cases run over 3 seeds.

**Grid.** The app's defaults: subset 21, step 5, strain window 15, Keys 4×4 unless noted. Frames are 256² unless the motion needs more room:

| Scenario | Frame size |
|---|---|
| Translation (S3) | 448×192 |
| Large stretch (S5) | 384×256 |
| Combined (S6) | 384² |
| Ring mask (S9) | 320² |

**Modes.** Fixed, N = 1, 2, 5 and 10 run in lockstep on identical frames.

**Metrics.** Collected per frame and per mode. The CSV columns have the same names.

| Metric | Meaning |
|---|---|
| `rms` | RMS of the vector displacement error over reported points, px |
| `cov` | Points within 0.5 px of the truth, as % of frame 1's count. It can exceed 100% when frame 1 dropped a few edge points that later frames keep. |
| `bias_u/v` | Mean error, px: accuracy |
| `std_u/v` | Spread of the error across nodes, px: precision |
| `repeat` | Per-node standard deviation across noise seeds, px: repeatability |
| `rmsE` | RMS strain error over exx, eyy, exy |
| `fail` | First frame where any seed's solve returned no points. The run continues; three in a row end it. |
| `cov90` / `cov50` | First frame where coverage fell below 90% / 50% |
| `ms/f`, `cpu/f` | Wall and CPU time per frame |
| `upd ms` / `hold ms` | Mean wall time of frames that move the reference / that keep it |
| `drift` | ms/frame over the last 20 frames ÷ over the first 20 |

RSS growth from frame 20 to the end is reported for each scenario. It stayed under 3.5 MB in every scenario (3.4 MB on S6, the largest synthetic frames), and the process peak was 38 MB on the synthetic scenarios and 58 MB on S10: reference swaps do not leak.

**Host.** Intel i7-12700H laptop (20 threads), WSL2 Ubuntu 22.04, g++ 11.4, Release build (`-O3`, OpenMP), engine 0.4.0 with rotation-aware seeding. Use the timings to compare modes on one host, not as device figures. Frame times on this host move by ±20–40% between runs of identical code (S7 fixed: 13.6 and 9.3 ms/frame in the two builds, where the similarity is never applied), so read a single timing to about that precision.

---

## 2. Results

Tables give rms px / coverage % at the listed frames. "–" means no points.

### S1. Static, noise only, T = 300

This is the pure precision test: nothing moves, so any error is noise or drift.

| σ | mode | f10 | f100 | f200 | f300 | std_u at 300 | repeat |
|---|---|---|---|---|---|---|---|
| 0 | all | 0.000 | 0.000 | 0.000 | 0.000 | ≤ 0.0002 | – |
| 1 | fixed | 0.039 / 99 | 0.040 / 100 | 0.037 / 100 | 0.041 / 100 | 0.030 | 0.042 |
| 1 | N=1 | 0.053 / 99 | 0.103 / 94 | 0.116 / 91 | 0.124 / 83 | 0.077 | 0.119 |
| 1 | N=10 | 0.039 / 99 | 0.053 / 99 | 0.061 / 97 | 0.071 / 98 | 0.051 | 0.065 |
| 2 | fixed | 0.078 / 101 | 0.082 / 102 | 0.075 / 100 | 0.075 / 97 | 0.054 | 0.065 |
| 2 | N=1 | 0.110 / 92 | 0.225 / 73 | 0.319 / 58 | 0.370 / 52 | 0.209 | 0.335 |
| 2 | N=5 | 0.085 / 100 | 0.138 / 88 | 0.173 / 81 | 0.187 / 76 | 0.116 | 0.168 |
| 2 | N=10 | 0.080 / 100 | 0.121 / 93 | 0.137 / 88 | 0.148 / 82 | 0.095 | 0.140 |
| 4 | fixed | 0.127 / 113 | 0.121 / 116 | 0.111 / 114 | 0.117 / 106 | 0.078 | 0.108 |
| 4 | N=1 | 0.209 / 83 | 0.518 / 40 | 0.621 / 23 | 0.694 / 13 | 0.478 | 0.628 |
| 4 | N=10 | 0.135 / 110 | 0.222 / 84 | 0.255 / 62 | 0.320 / 56 | 0.202 | 0.273 |

Noise-free, every mode is exact for 300 frames: composition and the mask warp add nothing of their own.

With noise, the fixed reference's error is flat in time. Each reference update adds a fresh noise realisation, so the incremental error grows like √(updates). At σ = 2, N = 1 is 0.110 px at frame 10 and 0.370 px at frame 300: 30 times the updates, 3.4 times the error. Bias stays below 0.1 px in every mode, so what incremental mode loses is precision.

### S2. Slow loading, 0.1% stretch per frame, T = 200 (20% total)

Values are rms px / coverage % at frame 200.

| σ | interp | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|---|
| 0 | 4×4 | 0.010 / 97 | 0.225 / 98 | 0.184 / 99 | 0.121 / 100 | 0.074 / 100 |
| 0 | 6×6 | 0.009 / 97 | 0.126 / 99 | 0.071 / 100 | 0.035 / 100 | 0.022 / 100 |
| 1 | 4×4 | 0.037 / 98 | 0.342 / 92 | 0.248 / 96 | 0.176 / 98 | 0.125 / 100 |
| 2 | 4×4 | 0.094 / 101 | 0.611 / 64 | 0.465 / 78 | 0.368 / 91 | 0.299 / 96 |
| 2 | 6×6 | 0.095 / 100 | 0.567 / 68 | 0.443 / 79 | 0.330 / 87 | 0.274 / 95 |
| 4 | 4×4 | 0.136 / 117 | 0.828 / 26 | 0.678 / 39 | 0.496 / 70 | 0.403 / 85 |

This is the case incremental DIC is usually reached for, and the one it does worst on. Each increment is a ~0.1 px sub-pixel step, and the interpolation bias at a given sub-pixel offset has one sign. Noise-free, the error therefore grows linearly with the number of updates: at N = 1 it is 0.055 px at frame 50, 0.105 at 100 and 0.225 at 200.

The 6×6 interpolator halves that bias at N = 1 and cuts it 3× at N = 5 and N = 10. With noise, the random walk dominates and the interpolator no longer matters. A fixed reference handles 20% stretch without effort; at σ = 4 it drops 2 of 600 frames.

### S3. Rigid translation

Values are rms px / coverage % at the last frame.

| case | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|
| 1 px/f to 200 px, σ = 0 | fails at f49 | 0.050 / 100 | 0.047 / 96 | 0.024 / 100 | 0.013 / 100 |
| 1 px/f to 200 px, σ = 2 | fails at f49 | 0.281 / 67 | 0.219 / 79 | 0.145 / 86 | 0.154 / 96 |
| 5 px/f to 200 px, σ = 2 | fails at f10 | 0.168 / 92 | 0.146 / 96 | 0.107 / 105 | fails at f10 |
| 20 px/f to 200 px, σ = 2 | fails at f3 | 0.117 / 97 | 0.107 / 98 | fails at f3 | fails at f3 |

The fixed reference fails, and stays failed, once the total displacement passes the seeding capture range of about 48 px (half the 96 px phase-correlation window). Incremental mode carries the reference along and tracks the full 200 px.

The same limit applies to the step between references: N × (motion per frame) must stay under ~48 px. That is why N = 10 fails at 5 px/frame and N = 5 fails at 20 px/frame.

### S4. Rigid rotation to 90°

Values are rms px / coverage % at the listed frame. Where rotation-aware seeding changed a cell, the `045a500` value (translation-only seeding) comes first.

| case | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|
| 0.45°/f, σ = 0, at 45° (f100) | 0.009 / 81 → **0.010 / 100** | 0.423 / 78 | 0.176 / 99 | 0.046 / 100 | 0.023 / 100 |
| 0.45°/f, σ = 0, at 90° (f200) | 0.000 / 62 → **0.001 / 100** | 0.800 / 27 | 0.304 / 92 | 0.073 / 100 | 0.037 / 100 |
| 0.45°/f, σ = 2, at 45° (f100) | 0.067 / 82 | 0.688 / 58 | 0.493 / 86 | 0.265 / 97 → 0.264 / 97 | 0.176 / 100 → 0.168 / 100 |
| 0.45°/f, σ = 2, at 90° (f200) | – → **0.077 / 101** | 1.189 / 23 | 0.847 / 58 | 0.409 / 91 → 0.407 / 91 | 0.295 / 96 → 0.282 / 97 |
| 2°/f, σ = 0, at 90° (f45) | 0.000 / 62 → **0.001 / 100** | 0.083 / 100 | 0.045 / 100 | 0.020 / 99 → 0.020 / 100 | 0.059 / 100 → **0.013 / 100** |
| 2°/f, σ = 2, at 90° (f45) | 0.029 / 20 → **0.087 / 99** | 0.406 / 87 → 0.403 / 87 | 0.269 / 92 → 0.264 / 92 | 12.7 / 0 (wrong lock) | – |
| 5°/f, σ = 0, at 90° (f18) | 0.000 / 62 → **0.001 / 100** | 0.045 / 100 → 0.033 / 100 | 0.020 / 99 → 0.020 / 100 | 0.206 / 86 → **0.010 / 100** | 3.406 / 84 → **0.007 / 100** |
| 5°/f, σ = 2, at 90° (f18) | – → **0.087 / 99** | 0.193 / 62 → **0.279 / 94** | 0.129 / 63 → **0.184 / 95** | 0.194 / 60 → 0.220 / 89 | – → 0.070 / 60 |

Axis-aligned subsets stop matching a turning pattern, and translation-only seeding hands them a guess that is wrong by the rotation. The anchor lattice now measures rotation and scale from the log-polar spectrum and seeds the subsets with the turned shape.

**Noise-free**, that carries the fixed reference through the quarter turn at every rate:
- Coverage after the first quarter of the run averages 92–95%, against 70–71% before.
- The rms stays at 0.009–0.010 px throughout.
- Single frames still dip to 56–62% coverage, where the estimate does not lock. Frame 124 (55.8°) at 0.45°/f is still empty, as it was before.
- It is also 3–4× cheaper (§3), because IC-GN starts from the right shape.

Before, coverage fell from about 20° (88% at frame 50), and the 62% at exactly 90° was the pixel lattice mapping onto itself.

**With noise** (σ = 2) the fixed reference is unchanged up to about 85°:
- It is below 90% coverage by 7° (frame 15).
- Its first empty frame is at 13.5° (frame 30).
- It drops 46 of 600 frames, against 72.

On this 110 px ROI the log-polar estimate locks in 0–7 of the 30 fixed-reference solves per 10 frames between frame 11 and frame 190. Every other solve uses the translation lattice, exactly as before. SEEDING_BENCHMARK.md §4.5.1 traces this to the ROI size: at σ = 2 a 200 px ROI locks reliably.

Within 4° of the quarter turn the estimate locks in 27 of 30 solves, and the frames come back (0.077 px / 101% at frame 200). This is probably because the square ROI covers the same material there as at 0°, whereas at 45° only its inscribed disc is shared.

**Incremental mode** benefits wherever one step between references is a rotation the estimate resolves:
- Noise-free at 5°/frame, N = 5 and N = 10 (25° and 50° steps) now track every node to 0.010 / 0.007 px. They lost 14–16% of nodes before, and N = 10 ended 3.4 px off.
- With noise at 5°/frame, N = 1 and N = 2 keep 94–95% coverage at 90°, against 62–63%. Their rms rises because it now includes the roughly 30% of nodes they used to lose. N = 5 no longer drops frames (13 → 0 of 54), and N = 10 drops 6 instead of 23.
- With noise at 2°/frame, N = 5 (10° steps) locks onto a wrong solution at frame 45 in both builds: 12.7 px rms, no point within 0.5 px. It read 0.133 / 63 in this report's first edition, before `045a500`. N = 10 (20° steps) loses the chain from frame 8 in both builds.

N = 5–10 still follows the full quarter turn at 0.45°/frame, with or without noise: noise-free it ends at 0.04–0.07 px with every node tracked. N = 1 pays for its 200 updates and ends at 0.8 px.

Before engine 0.4.0 shipped, every incremental mode lost the chain between 50° and 85° here. Two causes:
- **Edge nodes fell out of the scan.** The ROI's material margin assumed axis-aligned subsets on an unrotated body.
- **The mask warp blanked whole bands of rows.** A per-pixel Newton inverse followed false branches.

MATHEMATICS.md §9.5 describes both fixes, and `Incremental.Rotation90_200Frames_TracksTheTurningRoi` guards them.

### S5. Large uniaxial stretch, T = 200

Values are rms px / coverage % at frame 200.

| case | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|
| to 60%, σ = 0 | 0.016 / 84 | 0.681 / 78 | 0.610 / 85 | 0.271 / 97 | 0.177 / 98 |
| to 60%, σ = 2 | 0.098 / 101 | 1.106 / 38 | 1.005 / 55 | 0.592 / 77 | 0.394 / 92 |
| to 100%, σ = 0 | 0.021 / 95 | 1.073 / 50 | 0.669 / 74 | 0.384 / 92 | 0.193 / 97 |
| to 100%, σ = 2 | 0.122 / 96 | 1.807 / 25 | 1.255 / 37 | 0.817 / 69 | 0.541 / 84 |

The first-order subset shape function absorbs uniform strain exactly, so the fixed reference follows even a 100% stretch. With noise it drops a few frames (3 and 5 of 600 seeded frames) but stays the most accurate mode on the rest. No incremental N comes close.

This is where an unguarded similarity seed did harm. Uniaxial stretch is not a similarity, and the log-polar estimate reports a rotation and scale for it. Without the ≥ 90%-kept rule, the 60% stretch read 0.215 px at frame 150 instead of 0.014 px. With the rule, every cell here matches translation-only seeding.

### S6. Combined: 40% stretch + 45° rotation + 30 px, T = 200

| σ | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|
| 0 | 0.017 / 100 | 1.014 / 14 | 0.435 / 78 | 0.141 / 100 | 0.065 / 100 |
| 2 | 0.085 / 97 | 1.480 / 17 | 0.973 / 46 | 0.437 / 83 | 0.336 / 92 |
| ms/frame (σ = 0) | 88 | 27 | 22 | 19 | 18 |

Rows σ = 0 and σ = 2 are rms px / coverage % at frame 200.

The fixed reference is again the most accurate, but it pays for it. Its time per frame rises 5–6× over the run, to about 90 ms, as seeding and IC-GN work harder against a pattern that has turned 45°. With noise it drops 29 of 600 frames. The incremental modes stay at 18–27 ms.

Rotation-aware seeding barely engages here. The similarity is applied in 31 of 1000 solves noise-free and 60 of 3000 with noise. On the rest, either the log-polar estimate does not lock under the anisotropic stretch, or the corrected peak does not beat the translation peak.

### S7. Non-affine fields, T = 200

Values are rms px / coverage % at frame 200.

| case | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|
| Gaussian bump 8 px, σ = 0 | 0.139 / 100 | 0.152 / 100 | 0.143 / 100 | 0.126 / 100 | 0.129 / 100 |
| Gaussian bump 8 px, σ = 2 | 0.155 / 104 | 0.460 / 73 | 0.354 / 85 | 0.241 / 94 | 0.254 / 99 |
| sinusoid 2 px, L 80, σ = 0 | 0.148 / 100 | 0.125 / 100 | 0.123 / 100 | 0.124 / 100 | 0.126 / 100 |
| bending 10% at edge, σ = 0 | 0.059 / 100 | 0.106 / 100 | 0.102 / 100 | 0.097 / 100 | 0.088 / 100 |
| bending 10% at edge, σ = 2 | 0.103 / 101 | 0.409 / 71 | 0.343 / 83 | 0.234 / 89 | 0.209 / 96 |

Noise-free, every mode carries about the same error. The first-order shape function cannot follow the curvature inside a 21 px subset, and that undermatching does not depend on the reference. Incremental mode is even marginally better on the sinusoid, where each increment has less curvature. With noise, the random walk appears again.

### S8. Same deformation, different frame counts (20% stretch + 10° rotation)

Values are rms px / coverage % at the last frame.

| T | σ | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|---|
| 5 | 0 | 0.009 / 90 | 0.019 / 100 | 0.013 / 100 | 0.012 / 92 | 0.012 / 92 |
| 20 | 0 | 0.009 / 90 | 0.090 / 100 | 0.046 / 100 | 0.018 / 100 | 0.013 / 100 |
| 50 | 0 | 0.009 / 90 | 0.212 / 99 | 0.115 / 100 | 0.046 / 100 | 0.019 / 100 |
| 200 | 0 | 0.009 / 90 | 0.350 / 87 | 0.305 / 90 | 0.181 / 100 | 0.090 / 100 |
| 20 | 2 | 0.082 / 100 | 0.303 / 94 | 0.235 / 98 | 0.148 / 100 | 0.144 / 103 |
| 200 | 2 | 0.100 / 99 | 0.771 / 59 | 0.571 / 69 | 0.396 / 91 | 0.287 / 95 |

With T ≤ N the reference moves only through the early update, which fires when a frame reports 5% fewer points than the first frame on the reference. Coverage at T = 5 falls to about 90% by the last frame, so N = 5 and N = 10 move early and differ slightly from the fixed reference.

The deciding variable is the **number of reference updates**, not the size of each step. Reaching the same end state through 4 updates costs about 0.02 px, and through 200 updates 0.35 px. A faster frame rate makes incremental DIC worse, not better, unless N grows with it.

### S9. Ring mask (hole in the ROI), 30% stretch, T = 200

Values are rms px / coverage % at frame 200.

| σ | fixed | N=1 | N=2 | N=5 | N=10 |
|---|---|---|---|---|---|
| 0 | 0.011 / 97 | 0.352 / 91 | 0.285 / 96 | 0.162 / 99 | 0.101 / 100 |
| 2 | 0.093 / 102 | 0.818 / 49 | 0.543 / 68 | 0.425 / 79 | 0.293 / 87 |

The mask is carried with the material: frame-0 material is warped forward onto each new reference. Subsets therefore never straddle the hole edge after an update, and the boundary does not erode. The fixed reference drops 14 of 600 frames with noise and is otherwise the more accurate mode.

### S10. DICe OHT CFRP ladder (real images)

Agreement with DICe's own fixed-reference solutions, as rms du / dv in px:

| rung | fixed | N=1 | N=2 |
|---|---|---|---|
| 01 | 0.0006 / 0.0006 | 0.0006 / 0.0006 | 0.0006 / 0.0006 |
| 03 | 0.0011 / 0.0010 | 0.0125 / 0.0185 | 0.0011 / 0.0010 |
| 06 | 0.0014 / 0.0017 | 0.0442 / 0.0227 | 0.0326 / 0.0172 |
| 11 | 0.0024 / 0.0029 | 0.0586 / 0.0257 | 0.0250 / 0.0168 |

Only five real frames exist, and the fixed reference correlates through all of them. This is therefore a pure cost-of-composition measurement: each update adds about 0.015–0.02 px against a reference that was never in trouble. Over four N = 1 updates the mean du drifts by −0.054 px, a bias of one sign, as in S2.

---

## 3. Cost

| | fixed | incremental |
|---|---|---|
| Frame time, small motion (S7, σ = 0) | 8–9 ms | 8–12 ms |
| Frame time, rotation (S4, σ = 0) | 32–43 ms (108–117 ms with translation-only seeding) | 18–28 ms |
| Frame time, combined (S6, σ = 0) | 88 ms | 18–27 ms |
| Reference update | – | no decode, no `prepare_data`: `adopt` hands over the deformed image already prepared for correlation |
| Memory | 2 images | 2 images. The old reference is freed when replaced. RSS growth ≤ 3.5 MB over 200–300 frames. |

`adopt` saves the `set_from_gray` work on every update: 0.16–1.5 ms per update at these frame sizes.

![accuracy against cost, every scenario with noise](images/incremental/accuracy_vs_cost.png)

On frames that keep the reference (N > 1), the per-reference cache (node scan and Hessian pool) is reused, so `hold ms` is at or below `upd ms`.

**Rotation-aware seeding.** On the 110 px ROI here, the log-polar estimate costs about 0.55 ms per solve on top of the 0.25 ms translation correlation (0.8 ms against 0.25 ms; at 200 px it is 0.9 against 0.7 ms, and at 320 px it is cheaper than the translation correlation). The reference half is prepared once per reference. It runs on every solve, in every mode, because only its result decides whether the similarity is used. Its effects:

- **Rotation, noise-free, fixed reference: 2.7–3.7× faster.** It drops from 108 / 116 / 117 ms to 37.5 / 43 / 32 ms per frame at 0.45 / 2 / 5°/frame. Over the last 10 frames of the 0.45°/frame run it is 14 ms against 146 ms, because IC-GN no longer fights a seed that is off by the rotation.
- **Everywhere else: about +1 ms per frame.** Two back-to-back runs per build of S2 and S5, alternated, measured this. The seeding stage grew from 7.4 to 8.4 ms (S2) and from 8.5 to 9.4 ms (S5) per solve, while the anchor IC-GN stage, which does identical work in both builds, stayed within 2%. Per frame that is +0–10% wall time, most on the cheapest frames (10–15 ms incremental frames at σ = 0). The fallback's second lattice solve is not a measurable part of it: it ran in at most 15 of 1000 solves of any variant.

Timings on this host drift by 10–40% between runs of identical code: in the second alternated pair the anchor IC-GN stage itself ran 6–10% slower. The first interleaved pass of the full benchmark showed the new build 1.1–1.9× slower on S2, S5 and S8. The same scenarios re-timed alternately show the +1 ms above, so treat single-run frame times in §2 to that precision.

**Fixed-reference cache.** The 0.4.0 refactor caches reference-only work: the node validity scan, the Hessian pool and the phase-correlation reference side. Normal (fixed-reference) runs benefit from it, not just incremental ones.

`tools/cache_bench.c` times one batch against `libsemper_c.so` from before the refactor (`434a77c`, 0.3.0) and from 0.4.0, on identical inputs. Wall time is per frame, two runs each:

| workload | 0.3.0 frames 2+ | 0.4.0 frames 2+ | pre-pass on frames 2+ |
|---|---|---|---|
| DICe OHT, 400×1040, 5411 pts | 128.5 / 125.3 ms | 118.4 / 115.5 ms (−8%) | 5–7 ms → 0 |
| 12 MP synthetic, 3000×2000 ROI, 59004 pts | 685.5 / 674.9 ms | 595.8 / 594.2 ms (−12%) | 54–59 ms → 0 |

The first frame against a reference costs the same as before or a little more, since it fills the cache. Outputs are bit-identical (`FullFieldGolden`, `FullField.RefCache_SequenceOnOneCache_BitIdenticalToFresh`).

---

## 4. Recommendations

1. **Default to the fixed reference.** It is more accurate whenever it correlates, and it correlates through more deformation than intuition says.
2. **Switch to incremental when the fixed reference stops correlating.** That happens with:
   - rigid motion past ~48 px;
   - rotation past ~5–15° on noisy images with a small ROI;
   - speckle that changes appearance (reflections, necking, lighting).

   Rotation-aware seeding carries the fixed reference through 90° where its log-polar estimate locks: on clean images, or with a ROI of about 200 px or more at σ = 2 GL (SEEDING_BENCHMARK.md §4.5.1). A 110 px ROI on noisy images does not get that far. In the app, losing correlation shows as coverage falling frame after frame.
3. **Move the reference as seldom as works.** Pick N so that N × (motion per frame) stays under ~40 px. Keep the rotation per step under ~5° on noisy images with a small ROI: 10° steps at σ = 2 locked onto a wrong solution here. On clean images, steps of 25–50° tracked. On video at full frame rate that is N = 5–10; on frames a second or more apart it is N = 1–2.
4. **Use the 6×6 interpolator** for incremental runs on clean images. It cuts the per-update bias 2–6×.
