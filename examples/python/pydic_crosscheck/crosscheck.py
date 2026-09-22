#!/usr/bin/env python3
"""Cross-check the Semper engine against pydic on pydic's own example datasets.

pydic (https://gitlab.com/damien.andre/pydic, GPL-3.0) ships three image series:
4-point bending, tension, and wedge splitting. This script runs pydic on one of
them exactly as its example does, runs the Semper engine on the same images,
ROI and grid, and reports how far the two disagree — per point, per frame, and
on the elastic constants each example derives.

pydic is NOT vendored here. Clone it yourself and pass its path:

    git clone --depth 1 https://gitlab.com/damien.andre/pydic.git /path/to/pydic
    pip install ./bindings/python numpy scipy matplotlib opencv-python-headless
    python examples/python/pydic_crosscheck/crosscheck.py --pydic /path/to/pydic bending

Parameters. pydic's examples use an 80 px Lucas-Kanade window on a 20 px grid.
The engine gets subset 81 (it needs an odd side), step 20, and a 41 px VSG
strain window — the diameter whose plus stencil reproduces pydic's
np.gradient gauge of +-1 grid step. The ROI is fixed per example (upstream asks
for a mouse pick) and placed so both codes land on the identical grid.

The two algorithms differ: pydic chains pyramidal LK frame to frame with a
translation-only window; the engine correlates each frame directly against the
first with a 6-DOF ICGN. See docs/VALIDATION.md for results and caveats.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import warnings

import numpy as np
import scipy.interpolate
from scipy import stats

import cv2

try:
    import semper
except ImportError:
    sys.exit("semper not installed — run: pip install ./bindings/python")

WIN, GRID = (80, 80), (20, 20)            # pydic example parameters
SUBSET, STEP, STRAIN_WINDOW = 81, 20, 41  # engine equivalents

EXAMPLES = {
    "bending": dict(
        dir="4-pt-bending-test", pattern="img/*.bmp",
        area=((300, 640), (2300, 1160)),
        interpolation="spline", meta="img/meta-data.txt"),
    "tension": dict(
        dir="tension-test", pattern="img/*.bmp",
        area=((200, 400), (1640, 1480)),
        interpolation="spline", meta="img/meta-data.txt"),
    # Upstream runs this one with deep_flow (dense Farnebäck, window 2, grid 1),
    # which is not a subset-correlation setting. Both codes use LK / ICGN at
    # 80/20 here instead, and pydic's 'raw' interpolation as upstream does for
    # this test: a global spline cannot represent the crack discontinuity.
    "wedge": dict(
        dir="wedge-splitting-test", pattern="img/essai*.BMP",
        area=((460, 820), (1980, 1800)),
        interpolation="raw", meta=None),
}


# ----------------------------------------------------------------- pydic ----
def import_pydic(pydic_dir: str):
    if not os.path.isfile(os.path.join(pydic_dir, "pydic.py")):
        sys.exit(f"no pydic.py under {pydic_dir}")
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, pydic_dir)
    import pydic

    # Headless: upstream draw_opencv falls through to cv2.imshow without a
    # filename, and grid.write_result drops CSVs next to the images.
    draw = pydic.draw_opencv
    pydic.draw_opencv = lambda im, *a, **k: draw(im, *a, **k) if "filename" in k else None
    pydic.grid.write_result = lambda self: None
    return pydic


def run_pydic(pydic, ex_dir: str, cfg: dict, workdir: str):
    pydic.grid_list.clear()
    pattern = os.path.join(ex_dir, cfg["pattern"])
    result = os.path.join(workdir, "result.dic")
    pydic.init(pattern, WIN, GRID, result, area_of_intersest=[cfg["area"][0], cfg["area"][1]])
    kw = dict(interpolation=cfg["interpolation"], strain_type="cauchy", save_image=False)
    if cfg["meta"]:
        kw["meta_info_file"] = os.path.join(ex_dir, cfg["meta"])
    pydic.read_dic_file(result, **kw)
    return list(pydic.grid_list)


# ---------------------------------------------------------------- engine ----
def run_engine(files, rect):
    eng = semper.Engine()
    eng.set_reference(cv2.imread(files[0], 0))
    frames = []
    for f in files[1:]:
        res = eng.run(cv2.imread(f, 0), rect=rect, step=STEP, subset=SUBSET,
                      strain_window=STRAIN_WINDOW)
        frames.append(res)
    return frames


def to_grid(points, nx, ny, x0, y0):
    g = {k: np.full((nx, ny), np.nan) for k in ("u", "v", "exx", "eyy", "exy")}
    ix = np.rint((points[:, 0] - x0) / STEP).astype(int)
    iy = np.rint((points[:, 1] - y0) / STEP).astype(int)
    for k, col in (("u", 2), ("v", 3), ("exx", 4), ("eyy", 5), ("exy", 6)):
        g[k][ix, iy] = points[:, col]
    return g


# ------------------------------------------------------------ comparison ----
def stat(a, b, label, unit=""):
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    print(f"    {label:24s} n={m.sum():5d}  bias={np.mean(d):+.4f}{unit}  "
          f"rms={np.sqrt(np.mean(d ** 2)):.4f}{unit}  "
          f"p95={np.percentile(np.abs(d), 95):.4f}{unit}  max={np.max(np.abs(d)):.4f}{unit}")


def rigid_removed(pydic, ref_pts, cur_pts):
    """pydic's own rigid-body removal, applied to any matched point pair."""
    return np.asarray(pydic.compute_disp_and_remove_rigid_transform(cur_pts, ref_pts))


def engine_rigid_removed(pydic, S, GX, GY):
    m = np.isfinite(S["u"]) & np.isfinite(S["v"])
    cur = np.column_stack([(GX + S["u"])[m], (GY + S["v"])[m]]).astype(np.float32)
    ref = np.column_stack([GX[m], GY[m]]).astype(np.float32)
    rr = rigid_removed(pydic, ref, cur)
    u = np.full(GX.shape, np.nan); v = np.full(GX.shape, np.nan)
    u[m] = rr[:, 0]; v[m] = rr[:, 1]
    return u, v, m


def spline_gradient_exx(u, v, m, gx, gy):
    """pydic's 'spline' strain chain (5th-order bisplrep + np.gradient, same
    Green-Lagrange formula) evaluated on the engine's valid points only."""
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    tx = scipy.interpolate.bisplrep(GX[m], GY[m], u[m], kx=5, ky=5)
    ty = scipy.interpolate.bisplrep(GX[m], GY[m], v[m], kx=5, ky=5)
    su = scipy.interpolate.bisplev(gx, gy, tx)
    sv = scipy.interpolate.bisplev(gx, gy, ty)
    du_dx, _ = np.gradient(su, float(STEP), float(STEP), edge_order=2)
    dv_dx, _ = np.gradient(sv, float(STEP), float(STEP), edge_order=2)
    return du_dx + 0.5 * (du_dx ** 2 + dv_dx ** 2)


def regress(label, strain, stress):
    E, b, r, _, _ = stats.linregress(strain, stress)
    print(f"  {label:38s} E = {E * 1e-9:7.2f} GPa   r = {r:.5f}")
    return E


def post_bending(pydic, grids, S_list, gx, gy, GX, GY, nx, ny):
    # Geometry and stress formula from pydic's 4-pt-bending main.py.
    force = np.array([float(g.meta_info["force(N)"]) for g in grids])
    L, l, b, h = (41. - 5.) * 1e-3, (19. - 5.) * 1e-3, 7.66e-3, 4.06e-3
    stress = 1.5 * force * (L - l) / (b * h ** 2)
    xs = slice(1, nx - 1)
    # The engine reports no value on the outermost grid ring (VSG support),
    # so the matched tensile-face row is ny-2 for both codes.
    row = ny - 2
    pyd_lit = [np.nanmean(g.strain_xx[xs, ny - 1]) for g in grids]
    pyd = [np.nanmean(g.strain_xx[xs, row]) for g in grids]
    eng, chain = [0.0], [0.0]
    for S in S_list:
        eng.append(np.nanmean(S["exx"][xs, row]))
        u, v, m = engine_rigid_removed(pydic, S, GX, GY)
        chain.append(np.nanmean(spline_gradient_exx(u, v, m, gx, gy)[xs, row]))
    print(f"\n=== Young's modulus (tensile face, frames 1..{len(grids)}) ===")
    regress(f"pydic  (example's row, y={gy[ny - 1]})", np.array(pyd_lit), stress)
    regress(f"pydic  (matched row,  y={gy[row]})", np.array(pyd), stress)
    regress(f"engine (VSG strain,   y={gy[row]})", np.array(eng), stress)
    regress("engine (displacement + pydic chain)", np.array(chain), stress)


def post_tension(grids, S_list, nx, ny):
    # Section and averaging window from pydic's tension-test main.py.
    force = np.array([float(g.meta_info["force(N)"]) for g in grids])
    stress = force / (0.012 * 0.002)
    xr, yr = slice(nx // 4, 3 * nx // 4), slice(ny // 4, 3 * ny // 4)
    pxx = [np.nanmean(g.strain_xx[xr, yr]) for g in grids]
    pyy = [np.nanmean(g.strain_yy[xr, yr]) for g in grids]
    sxx = [0.0] + [np.nanmean(S["exx"][xr, yr]) for S in S_list]
    syy = [0.0] + [np.nanmean(S["eyy"][xr, yr]) for S in S_list]
    print("\n=== Elastic constants (central half of the ROI) ===")
    for label, xx, yy in (("pydic", pxx, pyy), ("engine", sxx, syy)):
        E, _, r, _, _ = stats.linregress(xx, stress)
        nu, _, r2, _, _ = stats.linregress(xx, -np.asarray(yy))
        print(f"  {label:7s}  E = {E * 1e-9:6.2f} GPa (r={r:.5f})   nu = {nu:.3f} (r={r2:.5f})")


def post_wedge(grids, S_list, gx, ny):
    P, S = grids[-1], S_list[-1]
    eng_col = np.nanmean(np.abs(S["exx"]), axis=1)
    pyd_col = np.nanmean(np.abs(P.strain_xx), axis=1)
    k = int(np.nanargmax(eng_col))
    print("\n=== Crack localisation (last frame, column-mean |exx|) ===")
    print(f"  engine peak at x = {gx[k]}  ({eng_col[k]:.4f})")
    print(f"  pydic  peak at x = {gx[int(np.nanargmax(pyd_col))]}  ({np.nanmax(pyd_col):.4f})")
    band = slice(max(0, k - 3), k + 4)
    dropped = [(int(x), ny - int(np.isfinite(S["exx"][i]).sum()))
               for i, x in zip(range(band.start, band.stop), gx[band])]
    print(f"  engine points dropped per column around the crack (of {ny}): {dropped}")


def plot_last(name, grids, S_list, gx, gy, path):
    import matplotlib.pyplot as plt
    P, S = grids[-1], S_list[-1]
    ext = [gx[0], gx[-1], gy[-1], gy[0]]
    v = np.nanpercentile(np.abs(P.strain_xx), 99)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for a, f, t, lim in ((ax[0], P.strain_xx, "pydic", v),
                         (ax[1], S["exx"], "Semper engine", v),
                         (ax[2], S["exx"] - P.strain_xx, "engine - pydic", v / 6)):
        im = a.imshow(f.T, extent=ext, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set_title(f"{t}  $\\epsilon_{{xx}}$", fontsize=10)
        fig.colorbar(im, ax=a, shrink=0.9)
    fig.suptitle(f"{name}: last frame", fontsize=11)
    fig.savefig(path, dpi=110)
    print(f"\nfigure written: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("example", choices=sorted(EXAMPLES))
    ap.add_argument("--pydic", required=True, help="path to a pydic clone (contains pydic.py)")
    ap.add_argument("--plot", metavar="PNG", help="write an exx comparison figure for the last frame")
    args = ap.parse_args()

    cfg = EXAMPLES[args.example]
    ex_dir = os.path.join(args.pydic, "examples", cfg["dir"])
    files = sorted(glob.glob(os.path.join(ex_dir, cfg["pattern"])))
    if len(files) < 2:
        sys.exit(f"no images for {args.example} under {ex_dir}")

    (x0, y0), (x1, y1) = cfg["area"]
    nx, ny = (x1 - x0) // GRID[0], (y1 - y0) // GRID[1]
    rect = (x0, y0, nx * STEP, ny * STEP)
    gx, gy = x0 + STEP * np.arange(nx), y0 + STEP * np.arange(ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")

    # Edge columns the engine drops entirely make nanmean warn; that is expected.
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    pydic = import_pydic(args.pydic)
    print(f"semper {semper.__version__}  |  {args.example}: {len(files)} frames, "
          f"grid {nx} x {ny}, rect {rect}")
    with tempfile.TemporaryDirectory() as work:
        grids = run_pydic(pydic, ex_dir, cfg, work)
    frames = run_engine(files, rect)
    S_list = [to_grid(r.points, nx, ny, x0, y0) for r in frames]

    print(f"\n=== engine (subset {SUBSET}, step {STEP}, VSG {STRAIN_WINDOW}) "
          f"vs pydic (window {WIN[0]}, grid {GRID[0]}, {cfg['interpolation']}) ===")
    for i, (P, S, res) in enumerate(zip(grids[1:], S_list, frames), start=2):
        rp, cp = P.reference_point, P.correlated_point
        raw = (cp - rp)
        print(f"--- frame {i}: engine {res.count}/{nx * ny} points, "
              f"convergence {res.metrics[15]:.1f}% ---")
        stat(S["u"], raw[:, 0].reshape(nx, ny), "u total [px]", " px")
        stat(S["v"], raw[:, 1].reshape(nx, ny), "v total [px]", " px")
        pr = rigid_removed(pydic, rp, cp)
        su, sv, _ = engine_rigid_removed(pydic, S, GX, GY)
        stat(su, pr[:, 0].reshape(nx, ny), "u rigid-removed [px]", " px")
        stat(sv, pr[:, 1].reshape(nx, ny), "v rigid-removed [px]", " px")
        stat(S["exx"], P.strain_xx, "exx")
        stat(S["eyy"], P.strain_yy, "eyy")
        stat(S["exy"], P.strain_xy, "exy")

    if args.example == "bending":
        post_bending(pydic, grids, S_list, gx, gy, GX, GY, nx, ny)
    elif args.example == "tension":
        post_tension(grids, S_list, nx, ny)
    else:
        post_wedge(grids, S_list, gx, ny)

    if args.plot:
        plot_last(args.example, grids, S_list, gx, gy, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
