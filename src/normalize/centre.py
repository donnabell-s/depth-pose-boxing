"""
normalize/centre.py — Mid-shoulder centring.

Translates each frame so the midpoint of left_shoulder and right_shoulder
sits at the origin. This is the normalisation root for this project.

Why mid-shoulder instead of hips
---------------------------------
Videos are framed thigh-up. The shoulder girdle is always fully visible,
whereas hips may be partially cropped. A jittery root corrupts the entire
normalised sequence — shoulders are the reliable choice.
"""

from __future__ import annotations

import numpy as np

from src.constants import ROOT_LEFT_IDX, ROOT_RIGHT_IDX


def centre_on_shoulders(seq: np.ndarray) -> np.ndarray:
    """
    Translate each frame so the mid-shoulder point = origin.

    Parameters
    ----------
    seq : (T, 9, 3) float32

    Returns
    -------
    seq centred on mid-shoulder — same shape.
    """
    l_sho   = seq[:, ROOT_LEFT_IDX,  :]   # (T, 3)
    r_sho   = seq[:, ROOT_RIGHT_IDX, :]   # (T, 3)
    mid_sho = (l_sho + r_sho) / 2.0       # (T, 3)

    return seq - mid_sho[:, None, :]       # broadcast over 9 joints