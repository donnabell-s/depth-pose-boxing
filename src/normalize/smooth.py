"""
normalize/smooth.py — Savitzky-Golay smoothing over the time axis.

Applied across all joints and dimensions simultaneously via scipy.
Chosen over One Euro Filter for batch video processing because:
  - No causality constraint — can use the full sequence window
  - The same polynomial fit used here is reused in features.py to extract
    velocity (deriv=1) and acceleration (deriv=2) without an extra smoothing pass
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def smooth(
    seq:           np.ndarray,  # (T, 9, 3)
    fps:           float,
    window_length: int = 7,
    polyorder:     int = 3,
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter along the time axis (axis=0).

    Parameters
    ----------
    seq           : (T, 9, 3) float32
    fps           : video frame rate — not used here but kept for API
                    consistency with features.py which needs it for derivatives
    window_length : frames in the SG window — must be odd and > polyorder.
                    Default 7 ≈ 0.23 s at 30 fps.
    polyorder     : polynomial order for the SG fit. Default 3.

    Returns
    -------
    smoothed seq — same shape and dtype as input.
    """
    T  = seq.shape[0]
    wl = _valid_window(window_length, polyorder, T)

    if wl != window_length:
        logger.warning(
            "SG window adjusted from %d → %d (T=%d, polyorder=%d)",
            window_length, wl, T, polyorder,
        )

    out = savgol_filter(seq, window_length=wl, polyorder=polyorder, axis=0)
    return out.astype(np.float32)


def _valid_window(window_length: int, polyorder: int, T: int) -> int:
    """Return the nearest valid odd window_length <= T and > polyorder."""
    wl = min(window_length, T)
    if wl % 2 == 0:
        wl -= 1
    min_odd = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
    wl = max(wl, min_odd)
    return wl