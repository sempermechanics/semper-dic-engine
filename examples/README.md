# Examples

Beginner demos using **published DICe sample images** that this engine’s host
suite already verifies.

| Sample | Images | Verified result |
|---|---|---|
| [samples/translation](samples/translation/) | `ref.tif` / `def.tif` (512²) | DICe custom_app: \|u − 0.4\| ≤ 0.1 px at four subsets |
| [samples/oht_cfrp](samples/oht_cfrp/) | `ref.tiff` / `def.tiff` | Field agreement vs `DICe_solution_01.txt` (host test) |

Redistributed under [samples/LICENSE.DICe](samples/LICENSE.DICe) (Sandia / NTESS).

## 1. C++ — verified DICe contract (recommended first)

Uses the same **subset solver** path as
`tests/dice/test_translation_real_image.cpp`:

Needs **OpenCV 4.8+** (see the note in [../README.md](../README.md#host-tests));
this path builds the vendored copy, so the submodule checkout below covers it.

```bash
# from engine repo root
git submodule update --init --recursive
./scripts/sparse-opencv.sh          # Windows: scripts/sparse-opencv.ps1

cmake -S . -B build/sdk \
  -DCMAKE_BUILD_TYPE=Release \
  -DSEMPER_BUILD_EXAMPLES=ON
cmake --build build/sdk --target run_translation

./build/sdk/examples/cpp/run_translation \
  examples/samples/translation/ref.tif \
  examples/samples/translation/def.tif
```

Expected output ends with:

```text
OK — DICe custom_app contract: 4/4 subsets |u-0.4| <= 0.1 px
```

## 2. Python

```bash
pip install ./bindings/python

# A) Synthetic rigid shift — confirms the install (full-field API)
python examples/python/run_synthetic.py

# B) Same DICe TIFF pair via full-field Engine (reports metrics;
#    subset-level golden is the C++ demo / dic_tests)
python examples/python/run_translation.py
```

## Choosing `step`, `subset` and `strain_window`

`strain_window` is a diameter **in pixels**, not a number of grid points, and
it must be at least **`2 * step`**.

Strain is fitted by least squares over a circular window of radius
`strain_window / 2` around each grid point. The nearest neighbouring point is
`step` pixels away, so below `2 * step` no neighbour falls inside the window:
every window collapses to its own centre, fails the "at least 3 points" test,
and **the whole field is discarded** — the run returns zero points.

| `strain_window / step` | Points in window | Result |
|---|---|---|
| `< 2` | 1 (centre only) | Whole field dropped |
| `2` | 5 (plus shape) | Minimum workable |
| `>= 3` | 9 or more | Recommended |

The demos below use `3 * step`. `subset` is independent of this — it is the
correlation window for the displacement solve, not the strain fit.

## Output layout (Frozen)

Full-field points are 8 floats: `x, y, u, v, exx, eyy, exy, corr`.
See [docs/CONTRACT.md](../docs/CONTRACT.md) and
[samples/translation/expected.json](samples/translation/expected.json).

## Going further

- Inspect `samples/oht_cfrp/` for a real experiment + DICe solution file.
- Host suite: `dic_tests DiceTranslationReal` / `DiceFieldAgreement`.
- Catalog: [docs/TESTING.md](../docs/TESTING.md) · [docs/EXAMPLES.md](../docs/EXAMPLES.md).
