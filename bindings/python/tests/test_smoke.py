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
    res = eng.run(deformed, rect=(40, 40, 176, 176), step=20, subset=31, strain_window=41)

    assert res.count > 0
    assert res.points.shape[1] == 8
    assert res.metrics.shape == (24,)
    # Column 2 is u (x-displacement). Recovered median should match the shift.
    u = res.points[:, 2]
    assert np.median(u) == pytest.approx(dx, abs=0.5)


def test_missing_reference_errors():
    eng = semper.Engine()
    with pytest.raises(semper.DicError) as excinfo:
        eng.run(_speckle(), rect=(0, 0, 128, 128), step=20, subset=31, strain_window=41)
    # Tighten: the code must be the documented INIT sentinel, not just "some error".
    assert excinfo.value.code == semper.ERR_INIT


def test_degenerate_roi_raises_with_roi_code():
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    # rect_w < step ⇒ empty grid ⇒ ROI error.
    with pytest.raises(semper.DicError) as excinfo:
        eng.run(ref, rect=(0, 0, 10, 10), step=20, subset=31, strain_window=41)
    assert excinfo.value.code == semper.ERR_ROI


def test_degenerate_roi_no_raise_returns_negative_result():
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    res = eng.run(
        ref, rect=(0, 0, 10, 10), step=20, subset=31, strain_window=41,
        raise_on_error=False,
    )
    # code < 0 branch in __init__.py: count carries the code, points is empty,
    # metrics is still the full (17,) array.
    assert res.count == semper.ERR_ROI
    assert res.points.shape == (0, 8)
    assert res.metrics.shape == (24,)


def test_set_reference_rejects_non_2d():
    eng = semper.Engine()
    rgb = np.zeros((16, 16, 3), np.uint8)  # 3-D → not a grayscale image
    with pytest.raises(ValueError):
        eng.set_reference(rgb)


def test_float_image_is_scaled_not_truncated():
    # A float image in [0, 1] used to go through pybind11's forcecast, which
    # truncated every pixel to 0 or 1: a black frame. Non-uint8 arrays are now
    # mapped from their own range onto 0..255 (io::to_gray8), the same rule as
    # a decoded 16-bit file, and ZNSSD does not see the affine map.
    ref = _speckle()
    dx = 3
    eng_u8 = semper.Engine()
    eng_u8.set_reference(ref)
    want = eng_u8.run(np.roll(ref, dx, axis=1), rect=(40, 40, 176, 176), step=20,
                      subset=31, strain_window=41)

    eng = semper.Engine()
    eng.set_reference(ref.astype(np.float64) / 255.0)
    got = eng.run(np.roll(ref, dx, axis=1).astype(np.float32) / 255.0,
                  rect=(40, 40, 176, 176), step=20, subset=31, strain_window=41)
    # Stretched to the full 0..255 range before quantizing, so not the same
    # bytes as the uint8 frame, but the same field.
    assert want.count > 0
    assert got.count >= 0.8 * want.count
    assert abs(float(np.median(got.points[:, 2])) - dx) < 0.05


def test_strain_window_too_small_raises():
    # The VSG plane fit needs 3 grid nodes in the strain window. Below that no
    # point can survive the strain post-filter, so the engine rejects the pair
    # up front instead of returning an empty field with a success code.
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    with pytest.raises(semper.DicError) as excinfo:
        eng.run(ref, rect=(40, 40, 176, 176), step=20, subset=31, strain_window=5)
    assert excinfo.value.code == semper.ERR_STRAIN_WINDOW


def test_error_codes_and_messages():
    assert semper.ERR_ROI == -2
    assert semper.ERR_INIT == -3
    assert semper.ERR_STRAIN_WINDOW == -4
    assert semper.ERR_SEED == -5
    assert "seed" in str(semper.DicError(semper.ERR_SEED))
    assert semper.ERR_CANCELLED == -99
    # DicError carries the code and maps to the documented human message.
    err = semper.DicError(semper.ERR_ROI)
    assert err.code == semper.ERR_ROI
    assert "ROI" in str(err)
    assert semper.DicError(semper.ERR_INIT).code == semper.ERR_INIT


def test_even_subset_rejected_with_init_code():
    # An even subset is off-centre by construction and used to write past the
    # engine's subset buffers; it is an argument error now.
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    for subset in (30, 0):
        with pytest.raises(semper.DicError) as excinfo:
            eng.run(ref, rect=(40, 40, 176, 176), step=20, subset=subset, strain_window=41)
        assert excinfo.value.code == semper.ERR_INIT


def test_rect_larger_than_image_is_clipped():
    # The rect is clipped to the image, so an oversized one solves the image
    # instead of sizing buffers from the rect.
    ref = _speckle()
    eng = semper.Engine()
    eng.set_reference(ref)
    big = eng.run(ref, rect=(0, 0, 2**31 - 1, 2**31 - 1), step=20, subset=31, strain_window=41)
    exact = eng.run(ref, rect=(0, 0, 256, 256), step=20, subset=31, strain_window=41)
    assert big.count == exact.count
    np.testing.assert_array_equal(big.points, exact.points)
    with pytest.raises(semper.DicError) as excinfo:
        eng.run(ref, rect=(300, 0, 64, 64), step=20, subset=31, strain_window=41)
    assert excinfo.value.code == semper.ERR_ROI
