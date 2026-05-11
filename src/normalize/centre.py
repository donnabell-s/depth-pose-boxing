"""
normalize/centre.py — Waist centring.

Translates each frame so the midpoint of left_hip and right_hip
sits at the origin, making all joint positions waist-relative.
"""

from __future__ import annotations

import numpy as np

from src.constants import WAIST_LEFT_IDX, WAIST_RIGHT_IDX


def centre_on_waist(seq: np.ndarray) -> np.ndarray:
    """
    Translate each frame so the mid-hip (waist) point = origin.

    Parameters
    ----------
    seq : (T, 9, 3) float32

    Returns
    -------
    seq centred on mid-hip — same shape.
    """
    l_hip   = seq[:, WAIST_LEFT_IDX,  :]   # (T, 3)
    r_hip   = seq[:, WAIST_RIGHT_IDX, :]   # (T, 3)
    mid_hip = (l_hip + r_hip) / 2.0        # (T, 3)

    return seq - mid_hip[:, None, :]        # broadcast over 9 joints
