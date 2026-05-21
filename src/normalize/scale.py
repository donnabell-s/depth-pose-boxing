"""
normalize/scale.py — Bone-length (scale) normalisation.

Divides the sequence by the median shoulder-to-shoulder distance
across all frames, making the output scale-invariant.

Effect
------
Different boxer sizes and camera distances produce comparable sequences.
A tall boxer and a short boxer performing the same punch will have
similar normalised trajectories after this step.

Reference bone
--------------
Shoulder-to-shoulder distance is used (not torso length) because both
shoulders are always visible in thigh-up framing. Torso length depends
on hip visibility which is unreliable in this setup.
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import ROOT_LEFT_IDX, ROOT_RIGHT_IDX

logger = logging.getLogger(__name__)


def normalise_scale(seq: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Scale by median shoulder-to-shoulder distance across all frames.

    Parameters
    ----------
    seq : (T, 9, 3) float32 — already centred on mid-hip (waist)

    Returns
    -------
    seq / median_shoulder_width : (T, 9, 3) — normalised sequence
    median_shoulder_width       : float      — scale factor used (same units as input)
    """
    l_sho = seq[:, ROOT_LEFT_IDX,  :]   # (T, 3)
    r_sho = seq[:, ROOT_RIGHT_IDX, :]   # (T, 3)

    shoulder_dist = np.linalg.norm(l_sho - r_sho, axis=1)  # (T,)
    valid_dist    = shoulder_dist[shoulder_dist > 1e-6]

    if len(valid_dist) == 0:
        logger.warning(
            "normalise_scale: all %d frames have near-zero shoulder distance — "
            "shoulders may be fully occluded. Skipping scale normalisation; "
            "kinematic features will NOT be scale-invariant.",
            len(shoulder_dist),
        )
        return seq, 1.0

    median_width = float(np.median(valid_dist))
    logger.debug("Median shoulder width: %.4f units", median_width)

    return seq / median_width, median_width