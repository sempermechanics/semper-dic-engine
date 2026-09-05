#!/usr/bin/env python3
"""Beginner example — DICe translation pair via the full-field Python API.

Samples: examples/samples/translation/{ref,def}.tif

The **subset-level** DICe contract (|u-0.4|<=0.1 at four points) is proven by
the host test ``DiceTranslationReal`` and the C++ demo
``examples/cpp/run_translation``. This script shows the NumPy / full-field
path: load the same images, run ``Engine``, and report median displacement
when points survive the quality filter.

Install (from the engine repo root)::

    pip install ./bindings/python
    python examples/python/run_translation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import semper
except ImportError:
    sys.exit("semper not installed — run: pip install ./bindings/python")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = ROOT / "samples" / "translation" / "ref.tif"
DEFAULT_DEF = ROOT / "samples" / "translation" / "def.tif"

U_TRUE = 0.4
TOL = 0.15


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--def", dest="deformed", type=Path, default=DEFAULT_DEF)
    args = ap.parse_args()

    print(f"semper {semper.__version__} — translation demo (full-field)")
    print(f"  ref: {args.ref}")
    print(f"  def: {args.deformed}")
    print("  note: strict |u-0.4|<=0.1 subset contract → examples/cpp/run_translation")

    ref_bytes = args.ref.read_bytes()
    def_bytes = args.deformed.read_bytes()

    eng = semper.Engine()
    eng.set_reference(ref_bytes)

    res = eng.run(
        def_bytes,
        rect=(64, 64, 384, 384),
        step=32,
        subset=27,
        # In pixels, and at least 2*step or every VSG window degenerates to
        # its own centre and the whole field is dropped. 3*step here.
        strain_window=96,
        progress=lambda pct: print(f"\r  progress: {pct:3d}%", end="", flush=True),
    )
    print()

    print(
        f"  metrics: attempted={res.metrics[0]:.0f} "
        f"solved={res.metrics[1]:.0f} "
        f"convergence={res.metrics[15]:.1f}%"
    )

    if res.count <= 0:
        print(
            "  full-field quality filter kept 0 points on this pair "
            "(common for sub-pixel DICe fixtures).\n"
            "  Verified subset contract: build examples/cpp/run_translation "
            "or run `dic_tests DiceTranslationReal`."
        )
        return 0

    med_u = float(np.median(res.points[:, 2]))
    med_v = float(np.median(res.points[:, 3]))
    print(f"  points={res.count}  median u={med_u:.4f}  median v={med_v:.4f}")

    if abs(med_u - U_TRUE) > TOL or abs(med_v - U_TRUE) > TOL:
        print(
            f"WARN: median (u,v)=({med_u:.3f},{med_v:.3f}) outside "
            f"{U_TRUE}±{TOL} — inspect points; subset contract still holds in C++ demo",
            file=sys.stderr,
        )
        return 0

    print("OK — full-field median near DICe translation truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
