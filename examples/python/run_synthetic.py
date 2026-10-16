#!/usr/bin/env python3
"""Beginner example — synthetic rigid translation (always full-field friendly).

No external image files. Recovers an integer-pixel shift with the Python API.
Use this to confirm your install; use ``run_translation.py`` + the C++ demo
for the published DICe fixture.

    pip install ./bindings/python
    python examples/python/run_synthetic.py
"""
from __future__ import annotations

import sys

import numpy as np

try:
    import semper
except ImportError:
    sys.exit("semper not installed — run: pip install ./bindings/python")


def speckle(h=256, w=256, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w), dtype=np.float32)
    for _ in range(3):
        p = np.pad(base, 1, mode="edge")
        base = (
            p[:-2, :-2]
            + p[:-2, 1:-1]
            + p[:-2, 2:]
            + p[1:-1, :-2]
            + p[1:-1, 1:-1]
            + p[1:-1, 2:]
            + p[2:, :-2]
            + p[2:, 1:-1]
            + p[2:, 2:]
        ) / 9.0
    return (base * 255).astype(np.uint8)


def main() -> int:
    dx = 3
    ref = speckle()
    deformed = np.roll(ref, shift=dx, axis=1)

    print(f"semper {semper.__version__} — synthetic +{dx} px X shift")
    eng = semper.Engine()
    eng.set_reference(ref)
    res = eng.run(deformed, rect=(40, 40, 176, 176), step=20, subset=31, strain_window=41)

    if res.count <= 0:
        print("FAIL: no points", file=sys.stderr)
        return 1

    med_u = float(np.median(res.points[:, 2]))
    print(f"  points={res.count}  median u={med_u:.4f}  (expect ~{dx})")
    if abs(med_u - dx) > 0.5:
        print(f"FAIL: median u {med_u} not within 0.5 of {dx}", file=sys.stderr)
        return 1

    print("OK — synthetic translation recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
