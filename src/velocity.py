"""
velocity.py — Per-joint velocity from a normalised 3D position sequence.

Input : (T, 9, 3) float32 from src/normalize output
Output: (T, 9, 3) float32 velocity array — (vx, vy, vz) per joint per frame

Finite difference scheme
------------------------
Interior frames — central difference (dt = 1 / fps):
    vel[t] = (pos[t+1] - pos[t-1]) / (2 * dt)
First frame — forward difference:
    vel[0] = (pos[1] - pos[0]) / dt
Last frame — backward difference:
    vel[-1] = (pos[-1] - pos[-2]) / dt

Joints whose position is exactly zero (imputed missing values from
normalize.impute) retain velocity = 0.0 to avoid step-function artefacts.

If shoulder_width_m is supplied, the normalised-unit velocity is converted
to metres per second:
    velocity_ms = velocity_normalised * shoulder_width_m * fps

where velocity_normalised is in normalised units per frame (before time
scaling by fps).  The scalar speed array (T, 9) is the L2 norm of the
velocity vector and is available via numpy.linalg.norm(velocity, axis=-1).
"""

from __future__ import annotations

import logging

import numpy as np

from src.constants import NUM_ACTIVE_JOINTS

logger = logging.getLogger(__name__)


def compute_velocity(
    sequence:         np.ndarray,         # (T, 9, 3) float32
    fps:              float,
    shoulder_width_m: float | None = None,
) -> np.ndarray:
    """
    Compute per-joint 3D velocity from a normalised skeleton sequence.

    Uses finite differences: central for interior frames, forward/backward
    for the first and last frames respectively.  Low-confidence joints whose
    position was imputed to zero keep velocity = 0.0.

    Parameters
    ----------
    sequence         : (T, 9, 3) float32 — normalised joint positions from
                       src.normalize.normalize()
    fps              : video frame rate (frames per second); dt = 1 / fps
    shoulder_width_m : shoulder-to-shoulder distance in metres.  When given,
                       velocity is converted from normalised units to m/s via:
                       velocity_ms = velocity_normalised * shoulder_width_m * fps
                       Omit (or pass None) to keep normalised units.

    Returns
    -------
    velocity : (T, 9, 3) float32 — velocity vector (vx, vy, vz) per joint.
               Units: normalised units per frame when shoulder_width_m is None;
                      metres per second when shoulder_width_m is provided.

    Notes
    -----
    Scalar speed per joint:
        speed = numpy.linalg.norm(velocity, axis=-1)   # (T, 9) float32
    """
    T = sequence.shape[0]
    assert sequence.shape == (T, NUM_ACTIVE_JOINTS, 3), (
        f"Expected (T, {NUM_ACTIVE_JOINTS}, 3), got {sequence.shape}"
    )

    velocity = _finite_differences(sequence)            # (T, 9, 3) float64, per frame
    velocity = _zero_imputed_joints(velocity, sequence) # (T, 9, 3)

    if shoulder_width_m is not None:
        velocity = velocity * (shoulder_width_m * fps)  # normalised/frame → m/s

    speed = _speed(velocity)                            # (T, 9) for logging

    logger.info(
        "Velocity | shape=%s | speed max=%.4f mean=%.4f",
        velocity.shape, float(speed.max()), float(speed.mean()),
    )

    return velocity.astype(np.float32)                  # (T, 9, 3) float32


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _finite_differences(seq: np.ndarray) -> np.ndarray:
    """
    Per-frame finite differences (normalised units per frame).

    Forward difference  : vel[0]    = pos[1]  - pos[0]
    Central difference  : vel[t]    = (pos[t+1] - pos[t-1]) / 2
    Backward difference : vel[-1]   = pos[-1] - pos[-2]
    """
    T   = seq.shape[0]
    vel = np.zeros_like(seq, dtype=np.float64)

    if T == 1:
        return vel                                          # degenerate — no motion

    vel[0]  = seq[1]  - seq[0]                             # forward
    vel[-1] = seq[-1] - seq[-2]                            # backward
    if T > 2:
        vel[1:-1] = (seq[2:] - seq[:-2]) / 2.0            # central

    return vel


def _zero_imputed_joints(velocity: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Zero velocity for any joint whose position is all-zero (imputed missing)."""
    imputed = np.all(positions == 0.0, axis=-1)            # (T, 9) bool
    velocity[imputed] = 0.0
    return velocity


def _speed(velocity: np.ndarray) -> np.ndarray:
    """Return (T, 9) scalar speed — L2 norm of velocity vector per joint."""
    return np.linalg.norm(velocity, axis=-1).astype(np.float32)
