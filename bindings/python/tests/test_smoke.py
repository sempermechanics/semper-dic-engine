"""Smoke test: a known rigid translation must be recovered.

Mirrors the C++ golden `test_translation_synthetic` at the Python boundary — it
proves the bindings marshal images/params/results correctly, not the DIC math
itself (that is covered by the engine's own suite).
"""
import numpy as np
import pytest

import semper


def _speckle(h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    # Blobby speckle: smooth noise thresholded, so ICGN has gradients to lock onto.
    base = rng.random((h, w)).astype(np.float32)
    k = np.ones((3, 3), np.float32) / 9.0
    for _ in range(3):  # cheap box blur
        base = np.pad(base, 1, mode="edge")
        base = (
            base[:-2, :-2] + base[:-2, 1:-1] + base[:-2, 2:]
            + base[1:-1, :-2] + base[1:-1, 1:-1] + base[1:-1, 2:]
            + base[2:, :-2] + base[2:, 1:-1] + base[2:, 2:]
        ) / 9.0
    return (base * 255).astype(np.uint8)


def test_version_exposed():
    assert isinstance(semper.__version__, str)
    assert semper.__version__.count(".") == 2


def test_recovers_rigid_translation():
    ref = _speckle()
    dx = 3
    deformed = np.roll(ref, shift=dx, axis=1)  # shift right by dx px

    eng = semper.Engine()
    eng.set_reference(ref)
    # strain_window is a diameter in PIXELS and must be >= 2*step, else every
    # VSG window holds only its own centre point, fails the valid_pts >= 3 test,
    # and the strain filter drops the whole field -- making res.count 0. This
    # test asserted count > 0 with strain_window=5 at step=20 (ratio 0.25), so
    # it could only ever have failed; nothing caught it because the binding
    # tests run solely under the manually-dispatched wheels workflow.
    res = eng.run(deformed, rect=(40, 40, 176, 176), step=20, subset=31,
                  strain_window=60)

    assert res.count > 0
    assert res.points.shape[1] == 8
    assert res.metrics.shape == (17,)
    # Column 2 is u (x-displacement). Recovered median should match the shift.
    u = res.points[:, 2]
    assert np.median(u) == pytest.approx(dx, abs=0.5)


def test_missing_reference_errors():
    eng = semper.Engine()
    with pytest.raises(semper.DicError) as excinfo:
        eng.run(_speckle(), rect=(0, 0, 128, 128), step=20, subset=31, strain_window=60)
    # Tighten: the code must be the documented INIT sentinel, not just "some error".
    assert excinfo.value.code == semper.ERR_INIT


def test_degenerate_roi_raises_with_roi_code():
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    # rect_w < step ⇒ empty grid ⇒ ROI error.
    with pytest.raises(semper.DicError) as excinfo:
        eng.run(ref, rect=(0, 0, 10, 10), step=20, subset=31, strain_window=60)
    assert excinfo.value.code == semper.ERR_ROI


def test_degenerate_roi_no_raise_returns_negative_result():
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    res = eng.run(
        ref, rect=(0, 0, 10, 10), step=20, subset=31, strain_window=60,
        raise_on_error=False,
    )
    # code < 0 branch in __init__.py: count carries the code, points is empty,
    # metrics is still the full (17,) array.
    assert res.count == semper.ERR_ROI
    assert res.points.shape == (0, 8)
    assert res.metrics.shape == (17,)


def test_set_reference_rejects_non_2d():
    eng = semper.Engine()
    rgb = np.zeros((16, 16, 3), np.uint8)  # 3-D → not a grayscale image
    with pytest.raises(ValueError):
        eng.set_reference(rgb)


def test_set_reference_float_array_is_cast_not_rejected():
    # to_gray uses pybind11 forcecast, so a float64 array is silently narrowed to
    # uint8 rather than rejected — the "must be uint8" message is NOT enforced.
    # Pin this surprising-but-real behavior so a future change is a conscious one.
    eng = semper.Engine()
    ref_f64 = _speckle().astype(np.float64)
    eng.set_reference(ref_f64)  # must NOT raise
    res = eng.run(
        ref_f64, rect=(0, 0, 128, 128), step=20, subset=31, strain_window=60,
        raise_on_error=False,
    )
    assert res.metrics.shape == (17,)


def test_error_codes_and_messages():
    assert semper.ERR_ROI == -2
    assert semper.ERR_INIT == -3
    assert semper.ERR_CANCELLED == -99
    # DicError carries the code and maps to the documented human message.
    err = semper.DicError(semper.ERR_ROI)
    assert err.code == semper.ERR_ROI
    assert "ROI" in str(err)
    assert semper.DicError(semper.ERR_INIT).code == semper.ERR_INIT
