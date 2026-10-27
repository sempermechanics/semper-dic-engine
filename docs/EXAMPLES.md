# Beginner examples

Demos live under [`examples/`](../examples/README.md). They use
**DICe-published sample images** that this repository already treats as
goldens in the host suite.

## Samples

### Translation (`examples/samples/translation/`)

| File | Role |
|---|---|
| `ref.tif` / `def.tif` | 512×512 speckle; def ≈ ref shifted **0.4 px** (diagonal) |
| `expected.json` | Params + pass/fail tolerances |

**Verified subset contract** (C++ demo + host test
`DiceTranslationReal.CustomApp_0p4px_RealSpeckle`):

- Subset size **27** at `(100,100)`, `(200,200)`, `(300,300)`, `(400,400)`
- Assert `|u − 0.4| ≤ 0.1` (DICe’s published tolerance)
- `v` is typically ~0.4 as well (diagonal shift) but DICe only asserts `u`

### OHT CFRP (`examples/samples/oht_cfrp/`)

Real experimental first load step, 400×1040 (portrait). Host test
`DiceFieldAgreement.OhtCfrp_AgreesWithDiceSolution` compares the subset solver
to `DICe_solution_01.txt`. Suggested full-field params are in `expected.json`;
its `rect` is `[x, y, w, h]` as `Engine.run(rect=...)` takes it, and covers
DICe's grid (x 61–341, y 61–971, step 35).

## Code entry points

| Language | Path | What it proves |
|---|---|---|
| C++ | [`examples/cpp/run_translation.cpp`](../examples/cpp/run_translation.cpp) | Exact DICe 4-subset contract |
| Python | [`examples/python/run_synthetic.py`](../examples/python/run_synthetic.py) | Full-field install smoke (+3 px) |
| Python | [`examples/python/run_translation.py`](../examples/python/run_translation.py) | Load DICe TIFFs via `Engine` |

```bash
cmake -S . -B build/sdk -DSEMPER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/sdk --target run_translation
```

## License

Sample images: [`examples/samples/LICENSE.DICe`](../examples/samples/LICENSE.DICe)
(Sandia / NTESS). Engine code: root [`LICENSE`](../LICENSE).
