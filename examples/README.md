# Examples

Three small programs that run the engine on sample images and check the
result. Run them from the repository root after the setup in the
[README](../README.md#quick-start). What each sample verifies:
[docs/EXAMPLES.md](../docs/EXAMPLES.md).

| Program | What it shows | Pass line |
|---|---|---|
| `cpp/run_translation` | Subset solver on the DICe 0.4 px pair: \|u − 0.4\| ≤ 0.1 px at four subsets | `OK — DICe custom_app contract: 4/4 subsets ...` |
| `python/run_synthetic.py` | Full-field solve of a synthetic +3 px shift; confirms the Python install | `OK — synthetic translation recovered` |
| `python/run_translation.py` | Full-field solve of the DICe pair from Python; prints the metrics | see below |

## C++

```bash
cmake -S . -B build/sdk -DCMAKE_BUILD_TYPE=Release -DSEMPER_BUILD_EXAMPLES=ON
cmake --build build/sdk --target run_translation
./build/sdk/examples/cpp/run_translation \
    examples/samples/translation/ref.tif examples/samples/translation/def.tif
```

## Python

```bash
pip install ./bindings/python
python examples/python/run_synthetic.py
python examples/python/run_translation.py
```

`run_translation.py` can report 0 points: on this sub-pixel pair the
full-field quality filter may keep none. The strict check for this pair is the
C++ program (or `dic_tests DiceTranslationReal`).

## Samples

| Directory | Images | Reference result |
|---|---|---|
| [samples/translation](samples/translation/) | `ref.tif`, `def.tif` (512 × 512) | 0.4 px diagonal shift; parameters in `expected.json` |
| [samples/oht_cfrp](samples/oht_cfrp/) | `ref.tiff`, `def.tiff` (400 × 1040) | DICe's solution `DICe_solution_01.txt`; parameters in `expected.json` |

The images are redistributed from DICe under
[samples/LICENSE.DICe](samples/LICENSE.DICe) (Sandia / NTESS).
