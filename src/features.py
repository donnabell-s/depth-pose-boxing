"""
features.py — Kinematic feature extraction for force estimation.

Two public functions:
  smooth_metric()              — SG filter on raw camera-space coordinates
  extract_kinematic_features() — 3D velocity/acceleration + elbow angle

All physics features are derived from camera-space (metric) data, keeping
metres, m/s, and m/s² clean and separate from the normalized geometry
used by ST-GCN (_pose_norm.npy).

Output conventions
------------------
- Wrist axis:  index 0 = left wrist, index 1 = right wrist
- Spatial axis: index 0 = X, index 1 = Y, index 2 = Z
- Elbow angle: degrees (not radians) for readable debug inspection
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import savgol_filter

from src.constants import ACTIVE_NAME_TO_IDX
from src.normalize.smooth import _valid_window

logger = logging.getLogger(__name__)

# Local joint indices
_L_SHOULDER = ACTIVE_NAME_TO_IDX["left_shoulder"]   # 1
_R_SHOULDER = ACTIVE_NAME_TO_IDX["right_shoulder"]  # 2
_L_ELBOW    = ACTIVE_NAME_TO_IDX["left_elbow"]      # 3
_R_ELBOW    = ACTIVE_NAME_TO_IDX["right_elbow"]     # 4
_L_WRIST    = ACTIVE_NAME_TO_IDX["left_wrist"]      # 5
_R_WRIST    = ACTIVE_NAME_TO_IDX["right_wrist"]     # 6


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def smooth_metric(
    points_3d:    np.ndarray,  # (T, 9, 3)
    sg_window:    int = 7,
    sg_polyorder: int = 3,
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to camera-space (metric) coordinates.

    Separate from normalize/smooth.py which runs on the normalized sequence.
    This operates on raw back-projected data so metric units (metres) are
    preserved throughout the kinematic extraction chain.

    Parameters
    ----------
    points_3d    : (T, 9, 3) float32 — output of backproject()
    sg_window    : SG window length — must be odd and > sg_polyorder
    sg_polyorder : SG polynomial order

    Returns
    -------
    (T, 9, 3) float32 — smoothed camera-space coordinates
    """
    T  = points_3d.shape[0]
    wl = _valid_window(sg_window, sg_polyorder, T)

    if wl != sg_window:
        logger.warning(
            "Metric SG window adjusted %d → %d (T=%d, polyorder=%d)",
            sg_window, wl, T, sg_polyorder,
        )

    out = savgol_filter(points_3d, window_length=wl, polyorder=sg_polyorder, axis=0)
    return out.astype(np.float32)


def extract_kinematic_features(
    points_3d_smooth: np.ndarray,  # (T, 9, 3) — output of smooth_metric()
    fps:              float,
    sg_window:        int = 7,
    sg_polyorder:     int = 3,
) -> dict[str, np.ndarray]:
    """
    Extract kinematic features from smoothed metric camera-space coordinates.

    Parameters
    ----------
    points_3d_smooth : (T, 9, 3) float32 — output of smooth_metric()
    fps              : video frame rate — scales derivatives to per-second units
    sg_window        : SG window (should match smooth_metric call)
    sg_polyorder     : SG polynomial order (should match smooth_metric call)

    Returns
    -------
    dict with:
      "velocity_3d"     : (T, 2, 3) float32 — left/right wrist 3D velocity (m/s)
      "acceleration_3d" : (T, 2, 3) float32 — left/right wrist 3D acceleration (m/s²)
      "elbow_angle_deg" : (T, 2)    float32 — left/right elbow angle (degrees)
    """
    T  = points_3d_smooth.shape[0]
    wl = _valid_window(sg_window, sg_polyorder, T)

    wrist_xyz = points_3d_smooth[:, [_L_WRIST, _R_WRIST], :]  # (T, 2, 3)

    vel = savgol_filter(
        wrist_xyz,
        window_length=wl,
        polyorder=sg_polyorder,
        deriv=1,
        delta=1.0 / fps,
        axis=0,
    ).astype(np.float32)   # (T, 2, 3) m/s

    acc = savgol_filter(
        wrist_xyz,
        window_length=wl,
        polyorder=sg_polyorder,
        deriv=2,
        delta=1.0 / fps,
        axis=0,
    ).astype(np.float32)   # (T, 2, 3) m/s²

    angles_deg = np.degrees(_elbow_angles(points_3d_smooth)).astype(np.float32)  # (T, 2)

    logger.info(
        "Kinematic features | T=%d | "
        "vel∈[%.3f, %.3f] m/s | acc∈[%.3f, %.3f] m/s² | angle∈[%.1f°, %.1f°]",
        T,
        vel.min(),        vel.max(),
        acc.min(),        acc.max(),
        angles_deg.min(), angles_deg.max(),
    )

    return {
        "velocity_3d":     vel,
        "acceleration_3d": acc,
        "elbow_angle_deg": angles_deg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _elbow_angles(sequence: np.ndarray) -> np.ndarray:
    """
    Compute 3D elbow angle (radians) per frame via dot product.

    Angle is measured at the elbow between:
      vec_upper = shoulder - elbow
      vec_lower = wrist    - elbow

    Frames where either vector has near-zero length (occluded joint) → 0.0 rad.

    Returns (T, 2) float32 — [left_elbow, right_elbow] in radians.
    Caller is responsible for converting to degrees if needed.
    """
    T   = sequence.shape[0]
    out = np.zeros((T, 2), dtype=np.float32)

    for side, (sh_idx, el_idx, wr_idx) in enumerate([
        (_L_SHOULDER, _L_ELBOW, _L_WRIST),
        (_R_SHOULDER, _R_ELBOW, _R_WRIST),
    ]):
        shoulder = sequence[:, sh_idx, :]  # (T, 3)
        elbow    = sequence[:, el_idx, :]  # (T, 3)
        wrist    = sequence[:, wr_idx, :]  # (T, 3)

        upper = shoulder - elbow           # (T, 3)
        lower = wrist    - elbow           # (T, 3)

        norm_upper = np.linalg.norm(upper, axis=1, keepdims=True)  # (T, 1)
        norm_lower = np.linalg.norm(lower, axis=1, keepdims=True)  # (T, 1)

        valid   = (norm_upper[:, 0] > 1e-6) & (norm_lower[:, 0] > 1e-6)
        upper_n = np.where(norm_upper > 1e-6, upper / norm_upper, 0.0)
        lower_n = np.where(norm_lower > 1e-6, lower / norm_lower, 0.0)

        cos_angle        = np.zeros(T, dtype=np.float32)
        cos_angle[valid] = np.einsum("td,td->t", upper_n, lower_n)[valid]
        cos_angle        = np.clip(cos_angle, -1.0, 1.0)

        out[:, side] = np.arccos(cos_angle)

    return out  # radians — np.degrees() applied by caller
