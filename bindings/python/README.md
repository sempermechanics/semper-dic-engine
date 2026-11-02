# semper-dic

Python bindings for the Semper Digital Image Correlation (DIC) engine. Give it
a reference image and a deformed image; it returns the displacement and strain
at each point of a grid, as NumPy arrays.

## Install

```bash
pip install semper-dic                 # published wheel
pip install ./bindings/python          # from a clone of the engine repository
```

A source build compiles the engine and its vendored OpenCV, so it needs CMake
and a C++17 compiler. The wheels are self-contained.

## Use

```python
import semper

eng = semper.Engine()
eng.set_reference(ref)                     # 2-D uint8 array, or PNG/JPEG/TIFF bytes
res = eng.run(
    deformed,
    rect=(0, 0, 512, 512),                 # region: x, y, w, h
    step=10, subset=21, strain_window=21,  # grid spacing, subset size (odd), strain window
    progress=lambda pct: print(pct),       # optional
)

res.count      # number of solved points
res.points     # (N, 8) float32: x, y, u, v, exx, eyy, exy, corr
res.metrics    # (24,) float32 telemetry
```

| Topic | Behaviour |
|---|---|
| Errors | A negative engine code raises `semper.DicError` (`.code` holds it). Pass `raise_on_error=False` to get the code in `res.count` instead. |
| Masks | `mask=` selects the pixels to correlate. Bool arrays, non-uint8 arrays and uint8 arrays holding only 0 and 1: nonzero is included. Other uint8 masks: pixels below 128 are excluded. |
| Cancel | `eng.cancel()` stops a solve in progress; it is safe from any thread, including the progress callback. Ctrl-C also cancels. Any other method called from the progress callback raises `RuntimeError`. |
| Threads | Calls on one `Engine` run one at a time. Use one `Engine` per thread to solve in parallel. |

### Image sequences

`IncrementalSession` follows large deformation across frames by moving the
reference forward every `update_interval` frames (`0` keeps the first image).
It reports total displacement from the first image:

```python
s = semper.IncrementalSession(frames[0], rect=(0, 0, 512, 512), step=10,
                              subset=21, strain_window=21, update_interval=1)
for frame in frames[1:]:
    res = s.process(frame)
    res.points                     # (N, 8): totals from frames[0]
    res.info["reference_frame"]    # also: reference_updated, tracked_points, ...
```

## Examples and reference

- Examples with sample images:
  [examples/](https://github.com/sempermechanics/semper-dic-engine/tree/main/examples)
- Output layout, metrics slots and error codes (stable within a major version):
  [docs/CONTRACT.md](https://github.com/sempermechanics/semper-dic-engine/blob/main/docs/CONTRACT.md)
- Tests: `pytest bindings/python/tests` from the repository root.
