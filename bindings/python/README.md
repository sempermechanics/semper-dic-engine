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
    step=10, subset=21, strain_window=30,  # window >= 2*step (see below)
    progress=lambda pct: print(pct),      # optional
)

res.count            # number of solved points
res.points           # (N, 8) float32: x, y, u, v, exx, eyy, exy, corr
res.metrics          # (17,) float32 telemetry
```

Call `eng.cancel()` from another thread to stop a run in flight.

### Choosing `strain_window`

`strain_window` is a diameter **in pixels**, not a count of grid points, and
it must be at least **`2 * step`**.

Strain is fitted by least squares over a circular window of radius
`strain_window / 2` centred on each grid point. The nearest neighbouring
point sits `step` pixels away, so if `strain_window < 2 * step` no neighbour
falls inside the window at all: every window collapses to its own centre,
fails the "at least 3 points" test, and **the entire field is discarded** —
`res.count` comes back `0`.

| `strain_window / step` | Points in the window | Result |
|---|---|---|
| `< 2` | 1 (centre only) | Whole field dropped |
| `2` | 5 (plus shape) | Minimum workable |
| `>= 3` | 9 or more | Recommended |

Prefer `3 * step` or above; a larger window smooths more but conditions the
fit better.

**Beginner demos** (DICe sample images + verified 0.4 px translation):

```bash
python examples/python/run_synthetic.py
python examples/python/run_translation.py
```

See [`examples/README.md`](../../examples/README.md) and [`docs/EXAMPLES.md`](../../docs/EXAMPLES.md).

The output packing, metrics layout, and error codes are **frozen** — see
[`docs/CONTRACT.md`](../../docs/CONTRACT.md) in this repository.
