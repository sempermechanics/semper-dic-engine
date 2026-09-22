#!/usr/bin/env python3
"""Measure the engine's large-deformation envelope on YOUR speckle image.

Warps one reference image by known affine deformations — translation, rigid
rotation, uniaxial stretch, simple shear — of increasing size, solves each
against the reference, and reports where the engine stops returning points.

    pip install ./bindings/python opencv-python-headless
    python examples/python/envelope_sweep.py                       # in-repo sample
    python examples/python/envelope_sweep.py --image my_ref.tif

Why this is image-dependent: the shape function is 6-DOF affine and the strain
is Green-Lagrange, so no homogeneous warp is out of reach kinematically. The
limit is AKAZE + RANSAC seeding, which depends on how matchable the speckle
stays under the warp. On pydic's datasets single-shot stretch held to 60-80%;
see docs/VALIDATION.md. Past the limit, correlate in increments with the
reference updated each step — there is no accumulated-strain limit.

Past the first failure, larger levels can succeed again through the Path C
fallback: every warp here is applied about the ROI centre, where displacement
stays inside Path C's +-15 px seed search. A real test with bodily translation
does not get that rescue, so read the first failure as the limit.

What this measures and what it does not: the deformed images are produced by
resampling (cv2.warpAffine, Lanczos), which itself biases sub-pixel values by a
few hundredths of a pixel. Treat the reported errors as upper bounds. For exact
ground truth use the host suite (tests/integration/test_operating_envelope.cpp),
which renders an analytic speckle field instead.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    import semper
except ImportError:
    sys.exit("semper not installed — run: pip install ./bindings/python")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "samples" / "translation" / "ref.tif"

SWEEPS = {
    "translation": [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 400],     # px along (1, 0.3)
    "rotation": [1, 5, 10, 20, 30, 45, 60, 90],                    # degrees
    "stretch": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0],
    "shear": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0],
}


def deformation(mode: str, level: float):
    """Return (A, t): x' = A (x - c) + c + t."""
    A, t = np.eye(2), np.zeros(2)
    if mode == "translation":
        t = np.array([level, 0.3 * level])
    elif mode == "rotation":
        th = math.radians(level)
        A = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    elif mode == "stretch":
        A = np.diag([1.0 + level, 1.0])
    elif mode == "shear":
        A = np.array([[1.0, level], [0.0, 1.0]])
    return A, t


def label(mode: str, level: float) -> str:
    return {"translation": f"{level:g} px", "rotation": f"{level:g} deg",
            "stretch": f"{100 * level:g}%", "shear": f"g={level:g}"}[mode]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--subset", type=int, default=41)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--strain-window", type=int, default=41)
    ap.add_argument("--roi-fraction", type=float, default=0.4,
                    help="central ROI size as a fraction of the image (default 0.4)")
    ap.add_argument("--modes", nargs="+", choices=sorted(SWEEPS), default=list(SWEEPS))
    args = ap.parse_args()

    ref = cv2.imread(str(args.image), 0)
    if ref is None:
        sys.exit(f"cannot read {args.image}")
    H, W = ref.shape
    c = np.array([W / 2.0, H / 2.0])
    rw = int(args.roi_fraction * W) // args.step * args.step
    rh = int(args.roi_fraction * H) // args.step * args.step
    rect = (int(c[0]) - rw // 2, int(c[1]) - rh // 2, rw, rh)
    nx, ny = rw // args.step, rh // args.step
    gx = rect[0] + args.step * np.arange(nx)
    gy = rect[1] + args.step * np.arange(ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    half = args.subset / 2.0
    corners = np.array([[-half, -half], [half, -half], [-half, half], [half, half]])

    print(f"semper {semper.__version__}  |  {args.image} ({W}x{H})  rect {rect}  "
          f"subset {args.subset}  step {args.step}  VSG {args.strain_window}")

    eng = semper.Engine()
    eng.set_reference(ref)
    summary = {}
    for mode in args.modes:
        print(f"\n{mode.upper():12s} {'points':>9} {'disp max':>10} {'E max err':>10} {'seed':>6}")
        first_fail, rescued = None, False
        for level in SWEEPS[mode]:
            A, t = deformation(mode, level)
            # Skip levels where a deformed subset would leave the image: that is
            # a framing limit, not an engine one.
            pos = np.stack([GX - c[0], GY - c[1]], -1) @ A.T + c + t
            foot = pos[..., None, :] + corners @ A.T
            if (foot[..., 0].min() < 4 or foot[..., 0].max() > W - 5 or
                    foot[..., 1].min() < 4 or foot[..., 1].max() > H - 5):
                print(f"{label(mode, level):>12s}   (deformed ROI leaves the image — stopping)")
                break
            M = np.hstack([A, (c + t - A @ c)[:, None]])
            def_img = cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_REFLECT101)
            res = eng.run(def_img, rect=rect, step=args.step, subset=args.subset,
                          strain_window=args.strain_window, raise_on_error=False)
            if res.count <= 0:
                print(f"{label(mode, level):>12s} {'none':>9}   (return code {res.count})")
                first_fail = first_fail or label(mode, level)
                continue
            p = res.points
            dX = np.column_stack([p[:, 0] - c[0], p[:, 1] - c[1]])
            tu = dX @ (A - np.eye(2)).T + t
            E = 0.5 * (A.T @ A - np.eye(2))
            derr = np.hypot(p[:, 2] - tu[:, 0], p[:, 3] - tu[:, 1]).max()
            eerr = max(np.abs(p[:, 4] - E[0, 0]).max(), np.abs(p[:, 5] - E[1, 1]).max(),
                       np.abs(p[:, 6] - E[0, 1]).max())
            seed = {2: "full", 1: "sparse", 0: "pathC"}[int(res.metrics[16])]
            rescued = rescued or (first_fail is not None and seed == "pathC")
            print(f"{label(mode, level):>12s} {res.count:9d} {derr:10.4f} {eerr:10.5f} {seed:>6}")
        summary[mode] = (first_fail, rescued)

    print("\nfirst level with no points returned (seeding limit on this image):")
    for mode, (f, rescued) in summary.items():
        note = "   (larger levels recovered via Path C — centred-warp artefact)" if rescued else ""
        print(f"  {mode:12s} {f or 'none within the sweep'}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
