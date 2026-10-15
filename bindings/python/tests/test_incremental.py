"""IncrementalSession at the Python boundary: marshalling, totals, contracts.

The DIC math is covered by the engine's own suite (Incremental.*); this proves
the binding passes images and settings through and returns totals from the
first image.
"""
import numpy as np
import pytest

import semper

from test_smoke import _speckle

RECT = (40, 40, 176, 176)
KW = dict(step=20, subset=31, strain_window=41)


def _frames(ref, n, dx=1):
    return [np.roll(ref, shift=dx * (i + 1), axis=1) for i in range(n)]


@pytest.mark.parametrize("interval", [1, 2])
def test_totals_accumulate_across_reference_updates(interval):
    ref = _speckle()
    s = semper.IncrementalSession(ref, RECT, update_interval=interval, **KW)
    for i, frame in enumerate(_frames(ref, 5)):
        res = s.process(frame)
        assert res.count > 0
        assert res.points.shape == (res.count, 8)
        assert res.metrics.shape == (23,)
        # Total from the first image, not the increment from the last reference.
        assert np.median(res.points[:, 2]) == pytest.approx(i + 1, abs=0.25)
        assert res.info["reference_frame"] == (i // interval) * interval
        assert res.info["reference_updated"] == ((i + 1) % interval == 0)
    assert s.frames_processed == 5


def test_interval_zero_matches_engine_run():
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    s = semper.IncrementalSession(ref, RECT, update_interval=0, **KW)
    for frame in _frames(ref, 3, dx=2):
        expected = eng.run(frame, rect=RECT, **KW)
        got = s.process(frame)
        assert got.count == expected.count
        assert np.array_equal(got.points, expected.points)
        assert got.info["reference_frame"] == 0


def test_init_errors_raise_with_codes():
    ref = _speckle()
    with pytest.raises(semper.DicError) as e:
        semper.IncrementalSession(ref, (0, 0, 10, 10), step=20, subset=31, strain_window=41)
    assert e.value.code == semper.ERR_ROI
    with pytest.raises(semper.DicError) as e:
        semper.IncrementalSession(ref, RECT, step=20, subset=31, strain_window=5)
    assert e.value.code == semper.ERR_STRAIN_WINDOW
    with pytest.raises(semper.DicError) as e:
        semper.IncrementalSession(ref, RECT, update_interval=-1, **KW)
    assert e.value.code == semper.ERR_INIT


def test_wrong_size_frame_is_rejected_without_advancing():
    ref = _speckle()
    s = semper.IncrementalSession(ref, RECT, **KW)
    res = s.process(np.zeros((64, 64), np.uint8), raise_on_error=False)
    assert res.count == semper.ERR_INIT
    assert s.frames_processed == 0
