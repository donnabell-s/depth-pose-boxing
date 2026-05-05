"""
backproject.py — Back-projection from 2D pixel + depth → metric 3D.

Input  : (T, 9, 3) array of (x, y, z_relative) from depth_estimation.py
         CameraIntrinsics for the recording camera
Output : (T, 9, 3) array of (X, Y, Z) in camera coordinate space

Pinhole camera model
--------------------
Given a pixel (u, v) and a depth value Z, the 3D camera-space point is:

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = Z

Where:
    fx, fy : focal lengths in pixels
    cx, cy : principal point (optical centre) in pixels

Z here is the relative depth from Depth Anything V2, normalised to [0, 1].
It is NOT metric depth in metres. The output 3D coordinates are therefore
in relative units — scale-invariant but geometrically correct in structure.
Absolute scale is recovered in normalize.py via bone-length normalisation.

Camera intrinsics
-----------------
Set your camera's intrinsics in CameraIntrinsics. If you don't have them
yet, use CameraIntrinsics.estimate_from_frame() as a placeholder — it
approximates fx/fy from image dimensions assuming a ~70° field of view,
which is typical for laptop webcams and phone front cameras.

To get accurate intrinsics, run:
    python calibrate.py --video data/raw/checkerboard.mp4

which will output the exact fx, fy, cx, cy values for your camera.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.utils import NUM_ACTIVE_JOINTS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Camera intrinsics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.

    Attributes
    ----------
    fx : focal length along x axis (pixels)
    fy : focal length along y axis (pixels)
    cx : principal point x — typically frame_width  / 2
    cy : principal point y — typically frame_height / 2

    How to fill these in
    --------------------
    Option A — OpenCV calibration (most accurate):
        Run calibrate.py with a checkerboard video.
        It prints the exact values for your camera.

    Option B — Manufacturer / spec sheet:
        Some phones publish focal length in mm. Convert to pixels:
            fx = (focal_length_mm / sensor_width_mm) * image_width_px

    Option C — Estimate from frame dimensions (least accurate):
        Use CameraIntrinsics.estimate_from_frame(width, height)
        Good enough for testing and punch classification (Phase 2 is
        scale-invariant after bone-length normalisation).

    Example values for common cameras
    ----------------------------------
    iPhone 14 front camera (1080p):  fx≈1450, fy≈1450, cx=540, cy=960
    MacBook Pro webcam (720p):       fx≈850,  fy≈850,  cx=640, cy=360
    Generic Android front (1080p):   fx≈1200, fy≈1200, cx=540, cy=960
    """
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def estimate_from_frame(
        cls,
        width:  int,
        height: int,
        fov_degrees: float = 70.0,
    ) -> "CameraIntrinsics":
        """
        Approximate intrinsics from frame dimensions.

        Assumes a horizontal field of view of fov_degrees (default 70°),
        which is typical for laptop webcams and phone front cameras.

        Use this only for testing — replace with calibrated values before
        collecting your final training data.
        """
        fx = (width / 2.0) / np.tan(np.radians(fov_degrees / 2.0))
        fy = fx  # assume square pixels
        cx = width  / 2.0
        cy = height / 2.0
        logger.warning(
            "Using estimated intrinsics (fov=%.0f°): "
            "fx=%.1f fy=%.1f cx=%.1f cy=%.1f — "
            "run calibrate.py for accurate values.",
            fov_degrees, fx, fy, cx, cy,
        )
        return cls(fx=fx, fy=fy, cx=cx, cy=cy)

    def as_matrix(self) -> np.ndarray:
        """Return the 3×3 camera intrinsic matrix K."""
        return np.array([
            [self.fx,  0.0,     self.cx],
            [0.0,      self.fy, self.cy],
            [0.0,      0.0,     1.0    ],
        ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Back-projection
# ──────────────────────────────────────────────────────────────────────────────

def backproject(
    keypoints_xyz: np.ndarray,      # (T, 9, 3) — (x_px, y_px, z_rel)
    intrinsics:    CameraIntrinsics,
    scores:        np.ndarray,      # (T, 9) — used to preserve zero mask
    score_thr:     float = 0.3,
) -> np.ndarray:
    """
    Back-project 2D pixel coordinates + relative depth into 3D camera space.

    Parameters
    ----------
    keypoints_xyz : (T, 9, 3) float32 — (x_pixel, y_pixel, z_relative)
                    from depth_estimation.py lift_to_3d()
    intrinsics    : CameraIntrinsics for the recording camera
    scores        : (T, 9) float32 — confidence scores from pose2d
    score_thr     : joints below this score stay at (0, 0, 0) — imputed later

    Returns
    -------
    points_3d : (T, 9, 3) float32 — (X, Y, Z) in camera coordinate space
                Units are relative (not metres) but geometrically correct.

    Coordinate convention
    ---------------------
        +X : right
        +Y : down  (image convention — flipped in normalize.py if needed)
        +Z : into the scene (away from camera)
    """
    T = keypoints_xyz.shape[0]
    assert keypoints_xyz.shape == (T, NUM_ACTIVE_JOINTS, 3), (
        f"Expected (T, {NUM_ACTIVE_JOINTS}, 3), got {keypoints_xyz.shape}"
    )

    u = keypoints_xyz[:, :, 0]   # (T, 9) pixel x
    v = keypoints_xyz[:, :, 1]   # (T, 9) pixel y
    z = keypoints_xyz[:, :, 2]   # (T, 9) relative depth

    # Pinhole back-projection
    X = (u - intrinsics.cx) * z / intrinsics.fx   # (T, 9)
    Y = (v - intrinsics.cy) * z / intrinsics.fy   # (T, 9)
    Z = z                                           # (T, 9)

    points_3d = np.stack([X, Y, Z], axis=-1).astype(np.float32)  # (T, 9, 3)

    # Re-apply zero mask for low-confidence joints
    # (depth_estimation.py already zeroes these but back-projection can reintroduce
    # small floating-point values near zero from the cx/cy subtraction)
    low_conf = scores < score_thr                  # (T, 9) bool
    points_3d[low_conf] = 0.0

    logger.info(
        "Back-projection complete | shape=%s | "
        "X∈[%.3f, %.3f] Y∈[%.3f, %.3f] Z∈[%.3f, %.3f]",
        points_3d.shape,
        X[scores >= score_thr].min() if (scores >= score_thr).any() else 0,
        X[scores >= score_thr].max() if (scores >= score_thr).any() else 0,
        Y[scores >= score_thr].min() if (scores >= score_thr).any() else 0,
        Y[scores >= score_thr].max() if (scores >= score_thr).any() else 0,
        Z[scores >= score_thr].min() if (scores >= score_thr).any() else 0,
        Z[scores >= score_thr].max() if (scores >= score_thr).any() else 0,
    )

    return points_3d  # (T, 9, 3)


def backproject_sequence(
    keypoints_xyz: np.ndarray,
    scores:        np.ndarray,
    frame_hw:      tuple[int, int],
    intrinsics:    CameraIntrinsics | None = None,
) -> np.ndarray:
    """
    Convenience wrapper — estimates intrinsics from frame dimensions if
    none are provided.

    Parameters
    ----------
    keypoints_xyz : (T, 9, 3) from depth_estimation.py
    scores        : (T, 9)    from depth_estimation.py
    frame_hw      : (height, width) of the source video frames
    intrinsics    : CameraIntrinsics — if None, estimated from frame_hw

    Returns
    -------
    points_3d : (T, 9, 3) float32
    """
    if intrinsics is None:
        h, w = frame_hw
        intrinsics = CameraIntrinsics.estimate_from_frame(w, h)

    return backproject(keypoints_xyz, intrinsics, scores)