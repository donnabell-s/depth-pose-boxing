"""
normalize/smooth.py — One Euro Filter smoothing over the time axis.

Applies one filter per active joint per spatial dimension (x, y, z).
The filter adapts to motion speed — minimal lag during fast punches,
heavy smoothing during slow guard movements.

This runs LAST in the normalisation pipeline so it operates on the
already-centred, scaled, and Y-flipped sequence.
"""

from __future__ import annotations

import numpy as np

from src.constants import NUM_ACTIVE_JOINTS
from src.filters import make_joint_filters


def smooth(
    seq:        np.ndarray,  # (T, 9, 3)
    fps:        float,
    min_cutoff: float = 1.0,
    beta:       float = 0.1,
) -> np.ndarray:
    """
    Apply One Euro Filter along the time axis per joint per dimension.

    Parameters
    ----------
    seq        : (T, 9, 3) float32
    fps        : video frame rate — sets the filter's time base
    min_cutoff : lower = more smoothing at rest
    beta       : higher = less lag during fast motion (punches)

    Returns
    -------
    smoothed seq — same shape and dtype as input.
    """
    out     = seq.copy()
    filters = make_joint_filters(freq=fps, min_cutoff=min_cutoff, beta=beta)
    T       = seq.shape[0]

    for t in range(T):
        for j in range(NUM_ACTIVE_JOINTS):
            for dim in range(3):
                out[t, j, dim] = filters[j][dim](seq[t, j, dim])

    return out