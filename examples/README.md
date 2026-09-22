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

## Output layout (Frozen)

Full-field points are 8 floats: `x, y, u, v, exx, eyy, exy, corr`.
See [docs/CONTRACT.md](../docs/CONTRACT.md) and
[samples/translation/expected.json](samples/translation/expected.json).

## Going further

- Inspect `samples/oht_cfrp/` for a real experiment + DICe solution file.
- `python examples/python/envelope_sweep.py [--image my_ref.tif]` — how large a
  translation, rotation, stretch or shear your own speckle can take in one solve.
- [python/pydic_crosscheck/](python/pydic_crosscheck/) — the engine against pydic
  on pydic's bending, tension and wedge-splitting series.
  Results: [docs/VALIDATION.md](../docs/VALIDATION.md).
- Host suite: `dic_tests DiceTranslationReal` / `DiceFieldAgreement`.
- Catalog: [docs/TESTING.md](../docs/TESTING.md) · [docs/EXAMPLES.md](../docs/EXAMPLES.md).
