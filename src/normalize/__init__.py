"""
normalize/__init__.py — Normalisation pipeline for depth-pose-boxing.

Public API
----------
    from src.normalize import normalize, NormConfig

Pipeline order (each step is a separate module):
    1. impute.py   — interpolate missing joints
    2. centre.py   — translate root to mid-hip (waist)
    3. scale.py    — measure shoulder width (not applied to sequence)
    4. flip Y      — +Y = up (image convention correction)
    5. smooth.py   — Savitzky-Golay smoothing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.constants import NUM_ACTIVE_JOINTS
from src.normalize.impute import impute_missing
from src.normalize.centre import centre_on_waist
from src.normalize.scale  import normalise_scale
from src.normalize.smooth import smooth

logger = logging.getLogger(__name__)


@dataclass
class NormConfig:
    """
    Tunable parameters for the normalisation pipeline.

    Attributes
    ----------
    fps           : video frame rate — passed to features.py for derivative scaling
    score_thr     : joints below this are treated as missing
    sg_window     : Savitzky-Golay window length — must be odd and > sg_polyorder.
                    Default 7 ≈ 0.23 s at 30 fps. Increase for heavier smoothing.
    sg_polyorder  : Savitzky-Golay polynomial order. Default 3.
    flip_y        : flip Y axis so +Y = up (recommended: True)
    """
    fps:          float = 30.0
    score_thr:    float = 0.3
    sg_window:    int   = 7
    sg_polyorder: int   = 3
    flip_y:       bool  = True


def normalize(
    points_3d: np.ndarray,          # (T, 9, 3) from backproject.py
    scores:    np.ndarray,          # (T, 9)    from depth_estimation.py
    config:    NormConfig | None = None,
) -> tuple[np.ndarray, float]:
    """
    Run the full normalisation pipeline on a 3D skeleton sequence.

    Parameters
    ----------
    points_3d : (T, 9, 3) float32 — back-projected 3D keypoints
    scores    : (T, 9)    float32 — per-joint confidence scores
    config    : NormConfig — uses defaults if None

    Returns
    -------
    seq            : (T, 9, 3) float32 — centred + flip_y + smoothed (no shoulder-width scaling)
    shoulder_width : float — median shoulder-to-shoulder distance in input units
                     (metres when using metric depth). Divide raw kinematic
                     features by this value to make them scale-invariant.
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

    # 2. Waist (mid-hip) centring
    seq = centre_on_waist(seq)
    logger.debug("Step 2 — waist centring done.")

    # 3. Measure shoulder width — seq is NOT divided; width is returned for kinematic scaling
    _, shoulder_width = normalise_scale(seq)
    logger.debug("Step 3 — shoulder width measured (%.4f); scale not applied.", shoulder_width)

    # 4. Y-axis flip (+Y = up)
    if cfg.flip_y:
        seq[:, :, 1] = -seq[:, :, 1]
        logger.debug("Step 4 — Y-axis flip done.")

    # 5. Savitzky-Golay smoothing
    seq = smooth(seq, cfg.fps, cfg.sg_window, cfg.sg_polyorder)
    logger.debug("Step 5 — smoothing done.")

    logger.info(
        "Normalisation complete | shoulder_width=%.4f | "
        "X∈[%.3f, %.3f] Y∈[%.3f, %.3f] Z∈[%.3f, %.3f]",
        shoulder_width,
        seq[:, :, 0].min(), seq[:, :, 0].max(),
        seq[:, :, 1].min(), seq[:, :, 1].max(),
        seq[:, :, 2].min(), seq[:, :, 2].max(),
    )

    return seq, shoulder_width  # (T, 9, 3) float32, float