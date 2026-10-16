# semper-dic

Python bindings for the **Semper** hybrid-Delaunay Digital Image Correlation (DIC)
engine — the same C++ core exposed as a NumPy-native API for desktop and research use.

## Install

```bash
pip install semper-dic          # from a published wheel
# or, from a source checkout of this engine repo:
pip install ./bindings/python
```

Building from source compiles the vendored OpenCV + engine, so a C++17 toolchain
and CMake are required (the published wheels are self-contained).

## Usage

```python
import numpy as np
import semper

eng = semper.Engine()
eng.set_reference(ref_u8)                 # 2-D uint8 array, or encoded PNG/JPEG bytes
res = eng.run(
    deformed_u8,
    rect=(0, 0, 512, 512),                # ROI: x, y, w, h
    step=10, subset=21, strain_window=21,
    progress=lambda pct: print(pct),      # optional
)

res.count            # number of solved points
res.points           # (N, 8) float32: x, y, u, v, exx, eyy, exy, corr
res.metrics          # (23,) float32 telemetry
```

Call `eng.cancel()` from another thread to stop a run in flight.

**Beginner demos** (DICe sample images + verified 0.4 px translation):

```bash
python examples/python/run_synthetic.py
python examples/python/run_translation.py
```

See [`examples/README.md`](../../examples/README.md) and [`docs/EXAMPLES.md`](../../docs/EXAMPLES.md).

The output packing, metrics layout, and error codes are **frozen** — see
[`docs/CONTRACT.md`](../../docs/CONTRACT.md) in this repository.
