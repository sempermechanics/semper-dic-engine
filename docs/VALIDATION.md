# External validation

Evidence that the engine is right, from sources other than its own golden
files: an independent DIC code on real experimental images, and the
large-deformation envelope measured against analytic ground truth.

---

## 1. Cross-check against pydic

[pydic](https://gitlab.com/damien.andre/pydic) is an independent open-source DIC
suite (Damien André, GPL-3.0). It ships three experimental image series with
analysis scripts. The engine was run on the same images, ROI and grid as pydic
and compared point by point.

Reproduce (pydic is not vendored; clone it separately):

```bash
git clone --depth 1 https://gitlab.com/damien.andre/pydic.git /path/to/pydic
pip install ./bindings/python numpy scipy matplotlib opencv-python-headless
python examples/python/pydic_crosscheck/crosscheck.py --pydic /path/to/pydic bending
python examples/python/pydic_crosscheck/crosscheck.py --pydic /path/to/pydic tension
python examples/python/pydic_crosscheck/crosscheck.py --pydic /path/to/pydic wedge
```

### Setup

| | pydic | engine |
|---|---|---|
| correlation | pyramidal Lucas-Kanade, translation-only window, **chained** frame to frame | 6-DOF ICGN, AKAZE-seeded, each frame **direct** to the first |
| window / subset | 80 px | 81 px (the engine needs an odd side) |
| grid step | 20 px | 20 px |
| strain | 5th-order bivariate spline + `np.gradient` (Green-Lagrange) | 41 px VSG plane fit (Green-Lagrange) |

The 41 px VSG diameter gives a plus stencil of ±1 grid step, the same gauge as
pydic's `np.gradient`. Upstream asks for the ROI with a mouse pick; the harness
fixes one per example, placed inside the speckle so both codes land on the
identical grid (pydic's `arange` grid and the engine's `rect_w / step` grid).

### Results

| example | frames | displacement rms | strain rms | derived constant |
|---|---|---|---|---|
| 4-point bending | 10 | 0.007–0.011 px | ~1×10⁻⁴ | E: engine **92.88** vs pydic **93.00** GPa (0.13%) |
| tension | 4 | 0.009–0.025 px | 2–4×10⁻⁴ | E: **6.22** vs **6.24** GPa (0.3%); ν: **0.523** vs **0.521** |
| wedge splitting | 3 | 0.023–0.027 px | 3–7×10⁻⁴ | — (crack located at x = 1220 vs 1240 px, one grid step apart) |

Displacement rms is against pydic's raw tracked points, before rigid-body
removal. After removing it with pydic's own routine, tension and wedge are
unchanged; bending v rises to 0.034 px rms at the last frame, most of it a
−0.03 px bias.

**Bending.** The specimen translates ~16 px bodily while the bending signal
after rigid-body removal is ~1.5 px; the engine recovers that signal to ~0.01 px
rms of pydic from an independent correlation path. The engine reports no value
on the outermost grid ring (the VSG stencil lacks neighbours there), so the
modulus is compared on the outermost row both codes populate (y = 1120 px). On
the example's literal row (y = 1140 px) pydic gives 86.12 GPa: moving one grid
row changes pydic's own answer by 8%, sixty times the engine–pydic difference.

**Tension.** ν > 0.5 is not admissible for an isotropic solid. It is pydic's own
result on this dataset (a polymer strained to ~1%), reproduced independently by
the engine — the codes agree on a number the experiment does not support.

**Wedge splitting.** Upstream analyses this series with `deep_flow` (dense
Farnebäck optical flow, window 2 px, grid 1 px), which is not a subset
correlation setting the engine can take. Both codes run at 80/20 instead, with
pydic's `raw` interpolation as upstream uses here: forcing `spline` onto a
cracked field smears the discontinuity across the ROI and raises the strain
disagreement twenty-fold. Across the open crack the engine **drops** points
(20 of 49 per column) where a subset straddles discontinuous material; pydic's LK
tracks each point independently and reports values straight through it.

### Where they differ

- **Point-to-point strain noise.** Engine ε_xx scatter about a smooth fit is
  ~3.5× pydic's on the bending field (1.5×10⁻⁴ vs 4.4×10⁻⁵). That is the strain
  estimator, not the correlation: pydic's global 5th-order spline is nearly the
  exact model for pure bending, and would equally smear a real localisation.
  The difference field is zero-mean, so row averages and moduli agree.
- **Discontinuities.** As above: the engine refuses to report across a crack.
- **Speed.** 0.3–0.8 s per frame for the engine on 2560×1920 images.

---

## 2. Large-deformation envelope

The shape function is 6-DOF first-order (affine), and the reported strain is
Green-Lagrange, E = ½(FᵀF − I). Both are exact for any homogeneous
deformation, and Green-Lagrange is zero for any rigid rotation. So the engine
has no kinematic limit on rotation or strain — the limit is **seeding**: AKAZE +
RANSAC must match enough features between reference and deformed images to
build the mesh. It escalates AKAZE scale 0.25× → 0.5× → 1.0× as deformation
grows; when full resolution still finds too few inliers it falls to Path C,
whose seed search (`estimate_initial_guess`) spans only ±15 px.

### Measured on real speckle (single-shot, pydic tension image, subset 81)

| mode | works to | error at that level | stops at |
|---|---|---|---|
| translation | ≥ 500 px (framing limit of the test) | 1×10⁻⁴ px | not reached |
| rigid rotation | ≥ 90° | 0.001 px; strain error 0 | not reached |
| uniaxial stretch | 60% (Exx = 0.78) | at 50%: 0.002 px; < 1×10⁻⁵ strain | 65% |
| simple shear | γ = 0.5 (Exy = 0.25) | 0.001 px; < 1×10⁻⁵ strain | by γ = 0.75 |

The stretch limit depends on the speckle: 60% (tension image, fine speckle),
65% (wedge), 80% (bending, coarse). On the host suite's synthetic speckle
(512²) it is 40% / γ = 0.4. Measure your own images with:

```bash
python examples/python/envelope_sweep.py --image my_ref.tif
```

**There is no accumulated-strain limit.** 100% stretch in five ~15% increments,
with the reference updated each step, converges fully at every stage where the
single-shot solve returns −1. The operative constraint is strain per
correlation increment.

The failure is abrupt — no points, not a degraded field — and near the edge it
is non-monotonic (on the synthetic speckle 70% stretch succeeds via Path C while
50–60% fail). The host suite therefore pins floors inside the envelope, not the
edge: see `Envelope` in [TESTING.md](TESTING.md).

### Limits of these numbers

- **Homogeneous deformation only.** A first-order shape function cannot
  represent curvature within a subset; in real tests the strain *gradient*
  across a subset usually binds before the strain magnitude does.
- **Sub-pixel accuracy is not measured by the real-image sweep.** It resamples
  the image (Lanczos), and the bias it reports changes with the resampling
  kernel (peak 0.029 px Lanczos vs 0.046 px cubic), so those figures are upper
  bounds dominated by the test itself. The host suite's analytic rendering and
  the pydic agreement above are the better evidence for the accuracy floor.

---

## 3. Known gap: return code −1

When seeding fails entirely, `run_full_field` returns **−1** from Path C
(`src/pipeline/full_field_path_c.cpp`). The Frozen return-code table in
[CONTRACT.md](CONTRACT.md) lists only `-2`, `-3` and `-99`. Large deformation
past the seeding limit is the most likely way to reach it, and a caller
switching on the documented codes will not handle it; the Python binding
surfaces it as `DicError` "engine error -1". Deciding whether to document −1 or
map it to an existing code is a contract change and is left open here.

A seeding failure does not always reach −1: it can also return **0** points
with no error (50% stretch on `examples/samples/translation/ref.tif`, subset 41).
Callers should treat `0` as "no result", not as success.
