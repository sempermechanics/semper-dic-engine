# Sample images and what they verify

The examples in [`examples/`](../examples/README.md) use two image pairs
published with DICe. The host test suite checks the engine against both, so
the examples and the tests agree on what a correct result is.

## Translation — `examples/samples/translation/`

512 × 512 speckle. `def.tif` is `ref.tif` shifted about 0.4 px along the
diagonal.

| Check | Value |
|---|---|
| Subset size | 27 |
| Subset centres | (100, 100), (200, 200), (300, 300), (400, 400) |
| Pass condition | \|u − 0.4\| ≤ 0.1 px at all four (DICe's published tolerance) |
| Verified by | `examples/cpp/run_translation`, test `DiceTranslationReal.CustomApp_0p4px_RealSpeckle` |

v is also close to 0.4, but DICe asserts only u, and so do these checks.
`expected.json` holds the parameters and tolerances.

## OHT CFRP — `examples/samples/oht_cfrp/`

The first load step of a real open-hole tension test on a carbon-fibre
laminate, 400 × 1040 (portrait).

| Check | Value |
|---|---|
| Reference | DICe's solution `DICe_solution_01.txt` (grid x 61–341, y 61–971, step 35) |
| Verified by | `DiceFieldAgreement.OhtCfrp_AgreesWithDiceSolution` (subset solver against the DICe field) |
| Full-field parameters | `expected.json`; its `rect` is `[x, y, w, h]`, as `Engine.run(rect=...)` takes it, and covers DICe's grid |

## Code

| Program | Proves |
|---|---|
| [`examples/cpp/run_translation.cpp`](../examples/cpp/run_translation.cpp) | The four-subset translation check |
| [`examples/python/run_synthetic.py`](../examples/python/run_synthetic.py) | The Python install solves a +3 px synthetic shift |
| [`examples/python/run_translation.py`](../examples/python/run_translation.py) | The DICe TIFFs load and solve through `Engine` |

How to build and run them: [examples/README.md](../examples/README.md).

## License

Sample images: [`examples/samples/LICENSE.DICe`](../examples/samples/LICENSE.DICe)
(Sandia / NTESS). Engine code: [`LICENSE`](../LICENSE).
