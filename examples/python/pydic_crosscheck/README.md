# Cross-check against pydic

Runs [pydic](https://gitlab.com/damien.andre/pydic) and this engine on pydic's
own experimental image series — 4-point bending, tension, wedge splitting — with
the same images, ROI and grid, and reports how far they disagree: per point, per
frame, and on the elastic constants each pydic example derives.

```bash
git clone --depth 1 https://gitlab.com/damien.andre/pydic.git /path/to/pydic
pip install ./bindings/python numpy scipy matplotlib opencv-python-headless

python examples/python/pydic_crosscheck/crosscheck.py --pydic /path/to/pydic bending
python examples/python/pydic_crosscheck/crosscheck.py --pydic /path/to/pydic tension
python examples/python/pydic_crosscheck/crosscheck.py --pydic /path/to/pydic wedge --plot wedge.png
```

pydic is **GPL-3.0** and is deliberately not vendored: nothing from it (code or
images) is in this repository. The script imports it at run time from your clone
and does not write into it. Bending is the slowest of the three (pydic tracks
2600 points over 10 frames).

Results, parameter choices and caveats: [docs/VALIDATION.md](../../../docs/VALIDATION.md).
