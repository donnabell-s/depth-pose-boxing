"""
normalize/impute.py — Missing joint interpolation.

Joints with score = 0 (occluded / low-confidence) are linearly
interpolated from their nearest valid neighbours in time.
Edge frames are forward/back-filled via np.interp's clamp behaviour.

This runs FIRST in the normalisation pipeline so that centring and
scaling never operate on zero-contaminated data.
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import NUM_ACTIVE_JOINTS

logger = logging.getLogger(__name__)


def impute_missing(
    seq:   np.ndarray,  # (T, 9, 3)
    valid: np.ndarray,  # (T, 9) bool — True where joint was detected
) -> np.ndarray:
    """
    Linearly interpolate missing joints over the time axis.

    Joints with no valid frames at all are left as zero — they are
    excluded from the ST-GCN graph in Phase 2.

    Parameters
    ----------
    seq   : (T, 9, 3) float32
    valid : (T, 9)    bool

    Returns
    -------
    out : (T, 9, 3) float32 — same shape, missing values filled.
    """
    T   = seq.shape[0]
    out = seq.copy()
    t_all = np.arange(T, dtype=float)

    for j in range(NUM_ACTIVE_JOINTS):
        col_valid = valid[:, j]

        if col_valid.all():
            continue

        if not col_valid.any():
            logger.debug("Joint %d has no valid frames — leaving as zero.", j)
            continue

        t_valid = t_all[col_valid]

        for dim in range(3):
            vals = out[col_valid, j, dim]
            out[:, j, dim] = np.interp(t_all, t_valid, vals)

    return out