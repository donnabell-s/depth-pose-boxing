"""
normalize/__init__.py — Normalisation pipeline for depth-pose-boxing.

Public API
----------
    from src.normalize import normalize, NormConfig

Pipeline order (each step is a separate module):
    1. impute.py   — interpolate missing joints
    2. centre.py   — translate root to mid-shoulder
    3. scale.py    — divide by shoulder-to-shoulder distance
    4. flip Y      — +Y = up (image convention correction)
    5. smooth.py   — One Euro Filter along time axis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.constants import NUM_ACTIVE_JOINTS
from src.normalize.impute import impute_missing
from src.normalize.centre import centre_on_shoulders
from src.normalize.scale  import normalise_scale
from src.normalize.smooth import smooth

logger = logging.getLogger(__name__)


@dataclass
class NormConfig:
    """
    Tunable parameters for the normalisation pipeline.

    Attributes
    ----------
    fps        : video frame rate — used for One Euro Filter cutoff
    score_thr  : joints below this are treated as missing
    min_cutoff : One Euro Filter — lower = smoother at rest
    beta       : One Euro Filter — higher = less lag during fast motion
    flip_y     : flip Y axis so +Y = up (recommended: True)
    """
    fps:        float = 30.0
    score_thr:  float = 0.3
    min_cutoff: float = 1.0
    beta:       float = 0.1
    flip_y:     bool  = True


def normalize(
    points_3d: np.ndarray,          # (T, 9, 3) from backproject.py
    scores:    np.ndarray,          # (T, 9)    from depth_estimation.py
    config:    NormConfig | None = None,
) -> np.ndarray:
    """
    Run the full normalisation pipeline on a 3D skeleton sequence.

    Parameters
    ----------
    points_3d : (T, 9, 3) float32 — back-projected 3D keypoints
    scores    : (T, 9)    float32 — per-joint confidence scores
    config    : NormConfig — uses defaults if None

    Returns
    -------
    seq : (T, 9, 3) float32 — normalised skeleton, ready for ST-GCN.
    """
    cfg = config or NormConfig()
    T   = points_3d.shape[0]

    assert points_3d.shape == (T, NUM_ACTIVE_JOINTS, 3), (
        f"Expected (T, {NUM_ACTIVE_JOINTS}, 3), got {points_3d.shape}"
    )
    assert scores.shape == (T, NUM_ACTIVE_JOINTS), (
        f"Expected (T, {NUM_ACTIVE_JOINTS}), got {scores.shape}"
    )

    seq   = points_3d.copy().astype(np.float32)
    valid = scores >= cfg.score_thr   # (T, 9) bool

    logger.info("Normalising: T=%d joints=%d", T, NUM_ACTIVE_JOINTS)

    # 1. Impute missing joints
    seq = impute_missing(seq, valid)
    logger.debug("Step 1 — imputation done.")

    # 2. Mid-shoulder centring
    seq = centre_on_shoulders(seq)
    logger.debug("Step 2 — shoulder centring done.")

    # 3. Scale normalisation
    seq = normalise_scale(seq)
    logger.debug("Step 3 — scale normalisation done.")

    # 4. Y-axis flip (+Y = up)
    if cfg.flip_y:
        seq[:, :, 1] = -seq[:, :, 1]
        logger.debug("Step 4 — Y-axis flip done.")

    # 5. One Euro Filter smoothing
    seq = smooth(seq, cfg.fps, cfg.min_cutoff, cfg.beta)
    logger.debug("Step 5 — smoothing done.")

    logger.info(
        "Normalisation complete | "
        "X∈[%.3f, %.3f] Y∈[%.3f, %.3f] Z∈[%.3f, %.3f]",
        seq[:, :, 0].min(), seq[:, :, 0].max(),
        seq[:, :, 1].min(), seq[:, :, 1].max(),
        seq[:, :, 2].min(), seq[:, :, 2].max(),
    )

    return seq  # (T, 9, 3) float32