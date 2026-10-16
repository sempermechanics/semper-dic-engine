"""Semper DIC engine — Python bindings.

Thin, NumPy-native wrapper over the same C++ core the Android app links. See
docs/CONTRACT.md for the frozen output/metrics/error contracts.

Example
-------
>>> import numpy as np, semper
>>> eng = semper.Engine()
>>> eng.set_reference(ref_u8)                     # 2-D uint8 array or encoded bytes
>>> res = eng.run(def_u8, rect=(0, 0, 512, 512),
...               step=10, subset=21, strain_window=21)
>>> res.points.shape          # (N, 8): x, y, u, v, exx, eyy, exy, corr
(2401, 8)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._semper import Engine as _Engine
from ._semper import IncrementalSession as _IncrementalSession
from ._semper import __version__

__all__ = [
    "Engine",
    "Result",
    "IncrementalSession",
    "FrameResult",
    "DicError",
    "ERR_ROI",
    "ERR_INIT",
    "ERR_STRAIN_WINDOW",
    "ERR_SEED",
    "ERR_CANCELLED",
    "__version__",
]

# Frozen error codes (see the contract doc). >= 0 is a valid point count.
ERR_ROI = -2
ERR_INIT = -3
ERR_STRAIN_WINDOW = -4
ERR_SEED = -5
ERR_CANCELLED = -99

_MESSAGES = {
    ERR_ROI: "invalid ROI",
    ERR_INIT: "init/argument failure (was set_reference called?)",
    ERR_STRAIN_WINDOW: "strain_window too small for step (need >= 2 * step)",
    ERR_SEED: "no seed found (no anchor converged and no global motion model locked)",
    ERR_CANCELLED: "cancelled",
}


class DicError(RuntimeError):
    """A negative engine return code surfaced as an exception."""

    def __init__(self, code: int):
        self.code = code
        super().__init__(_MESSAGES.get(code, f"engine error {code}"))


@dataclass
class Result:
    """One solve's output.

    ``points`` is an ``(N, 8)`` float32 array with columns
    ``[x, y, u, v, exx, eyy, exy, corr]``; ``metrics`` is the ``(24,)`` telemetry.
    """

    count: int
    points: np.ndarray
    metrics: np.ndarray


class Engine:
    """A single DIC engine instance. Not thread-safe except for :meth:`cancel`."""

    def __init__(self) -> None:
        self._engine = _Engine()

    def set_reference(self, image, mask=None, expected_w: int = 0, expected_h: int = 0) -> None:
        """Cache the reference image (a 2-D uint8 array or encoded bytes)."""
        self._engine.set_reference(image, mask, expected_w, expected_h)

    def run(
        self,
        deformed,
        rect,
        step: int,
        subset: int,
        strain_window: int,
        *,
        mask=None,
        use_6x6: bool = False,
        progress=None,
        raise_on_error: bool = True,
    ) -> Result:
        """Solve one deformed image against the cached reference.

        ``rect`` is ``(x, y, w, h)``. Raises :class:`DicError` on a negative code
        unless ``raise_on_error=False``, in which case ``Result.count`` is negative
        and ``points`` is empty.
        """
        code, points, metrics = self._engine.run(
            deformed, mask, tuple(rect), step, subset, strain_window, use_6x6, progress
        )
        if code < 0:
            if raise_on_error:
                raise DicError(code)
            return Result(code, np.empty((0, 8), np.float32), metrics)
        return Result(code, points, metrics)

    def cancel(self) -> None:
        """Request cancellation of the run in flight. Safe to call from any thread."""
        self._engine.cancel()


@dataclass
class FrameResult:
    """One frame of an incremental sequence.

    ``points`` is ``(N, 8)`` float32, ``[x0, y0, u, v, exx, eyy, exy, corr]``:
    total displacement from the first image at its grid points, with
    Green-Lagrange strain of that total. ``metrics`` describes the solve
    against the current reference; ``info`` holds ``reference_frame``,
    ``reference_updated``, ``tracked_points``, ``frame0_points``,
    ``taylor_fallbacks`` and ``lost_this_frame``.
    """

    count: int
    points: np.ndarray
    metrics: np.ndarray
    info: dict


class IncrementalSession:
    """Incremental (updated-reference) DIC over a frame sequence.

    The reference moves to the latest solved frame every ``update_interval``
    frames (``0`` keeps the first image, identical to :meth:`Engine.run` per
    frame). Each frame is correlated against the current reference and composed
    onto the running total, so large deformation can be followed frame by
    frame. Every update adds a little error, so on slow loading a larger
    interval (or the fixed reference) is more accurate. With an interval above
    1 the reference also moves as soon as a frame reports 5% fewer points. It
    never moves to a frame reporting less than half as many (a failed solve):
    the update falls to the next frame instead.

    A point the solve cannot reach on a frame (a band that does not correlate,
    such as glare or a background edge) coasts on its neighbours' motion and is
    not reported; it is reported again once it is measured, after a long coast
    only once its own earlier subset confirms where it is.

    Not thread-safe except for :meth:`cancel`.

    >>> s = semper.IncrementalSession(ref_u8, rect=(0, 0, 512, 512), step=10,
    ...                               subset=21, strain_window=21, update_interval=1)
    >>> for frame in frames:
    ...     res = s.process(frame)
    """

    def __init__(
        self,
        image,
        rect,
        step: int,
        subset: int,
        strain_window: int,
        *,
        update_interval: int = 1,
        mask=None,
        use_6x6: bool = False,
        expected_w: int = 0,
        expected_h: int = 0,
    ) -> None:
        self._session = _IncrementalSession()
        code = self._session.init(
            image, mask, tuple(rect), step, subset, strain_window, use_6x6,
            update_interval, expected_w, expected_h,
        )
        if code < 0:
            raise DicError(code)

    def process(self, deformed, *, progress=None, raise_on_error: bool = True) -> FrameResult:
        """Solve the next frame. Raises :class:`DicError` on a negative code
        unless ``raise_on_error=False``. A failed or cancelled frame leaves the
        session unchanged, so the same frame can be submitted again."""
        code, points, metrics, info = self._session.process(deformed, progress)
        if code < 0:
            if raise_on_error:
                raise DicError(code)
            return FrameResult(code, np.empty((0, 8), np.float32), metrics, info)
        return FrameResult(code, points, metrics, info)

    @property
    def frames_processed(self) -> int:
        """Frames solved so far (the first image not counted)."""
        return self._session.frames_processed

    def cancel(self) -> None:
        """Request cancellation of the frame in flight. Safe from any thread."""
        self._session.cancel()
