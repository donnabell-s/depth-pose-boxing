"""
depth_estimation.py — Per-frame depth estimation using Depth Anything V2.

Input  : list of Pose2DResult from pose_detector.py + original video path
Output : (T, 9, 3) array — (x, y, z) per active joint per frame

Z values are sampled from the monocular depth map at each joint's (x, y)
pixel location. Depth is relative (not metric) — it will be scaled during
normalisation in normalize.py.

Depth Anything V2 returns disparity-like values where smaller = farther.
This module normalises to [0, 1] and flips so that:
    z = 0  →  closest to camera
    z = 1  →  farthest from camera

Model options (pick based on available VRAM)
--------------------------------------------
  "vit-s" : ~2 GB  — use on Colab free tier T4 or limited GPU
  "vit-b" : ~4 GB  — recommended for offline batch processing
  "vit-l" : ~8 GB  — highest quality, use if VRAM allows

Sampling strategy
-----------------
Z is sampled as the mean over a (2r+1) × (2r+1) patch centred on each
joint pixel, rather than a single point. This reduces depth-map noise at
joint boundaries (e.g. wrist against a contrasting background).
Default radius r=2 → 5×5 patch average.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.pose_detector import Pose2DResult
from src.utils import NUM_ACTIVE_JOINTS, load_video_frames

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

_MODELS: dict[str, str] = {
    "vit-s": "depth-anything/Depth-Anything-V2-Small-hf",
    "vit-b": "depth-anything/Depth-Anything-V2-Base-hf",
    "vit-l": "depth-anything/Depth-Anything-V2-Large-hf",
}


# ──────────────────────────────────────────────────────────────────────────────
# Depth estimator
# ──────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Wraps Depth Anything V2 via HuggingFace transformers.

    Parameters
    ----------
    model           : "vit-s" | "vit-b" | "vit-l"
    device          : "cuda:0" | "cpu"
    sampling_radius : patch radius for Z sampling at each joint pixel.
                      0 = single pixel, 2 = 5×5 patch (recommended).
    score_thr       : joints below this confidence get z = 0.0
    """

    def __init__(
        self,
        model:           str   = "vit-b",
        device:          str   = "cuda:0",
        sampling_radius: int   = 2,
        score_thr:       float = 0.3,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODELS)}")
        self.model           = model
        self.device          = device
        self.sampling_radius = sampling_radius
        self.score_thr       = score_thr
        self._pipe           = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._pipe is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed.\n"
                "Run:  pip install transformers"
            ) from exc

        repo  = _MODELS[self.model]
        gpu   = 0 if "cuda" in self.device else -1
        logger.info("Loading Depth Anything V2 — variant: %s", self.model)

        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=repo,
            device=gpu,
        )
        logger.info("Depth Anything V2 ready.")

    # ------------------------------------------------------------------
    # Depth map for one frame
    # ------------------------------------------------------------------

    def estimate_depth(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run depth estimation on one BGR frame.

        Returns
        -------
        depth : (H, W) float32, values in [0, 1]
                0 = closest to camera, 1 = farthest from camera
        """
        self._load()

        from PIL import Image as PILImage  # type: ignore
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = PILImage.fromarray(frame_rgb)

        raw = np.array(self._pipe(pil_img)["depth"], dtype=np.float32)  # (H, W)

        # Resize to match source frame if model downsampled
        h, w = frame_bgr.shape[:2]
        if raw.shape != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]
        d_min, d_max = raw.min(), raw.max()
        if d_max - d_min > 1e-6:
            depth = (raw - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(raw)

        # Flip: Depth Anything returns disparity (small = far)
        # After flip: 0 = near, 1 = far — standard depth convention
        depth = 1.0 - depth

        return depth  # (H, W) float32

    # ------------------------------------------------------------------
    # Sample Z at joint locations
    # ------------------------------------------------------------------

    def _sample_z(
        self,
        depth_map: np.ndarray,   # (H, W)
        keypoints: np.ndarray,   # (9, 2)  pixel (x, y)
        scores:    np.ndarray,   # (9,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample the depth map at each joint's pixel location.

        Low-confidence joints (score < score_thr) receive z = 0.0
        and will be imputed later in normalize.py.

        Returns
        -------
        z : (9,) float32
        """
        H, W = depth_map.shape
        r    = self.sampling_radius
        z    = np.zeros(NUM_ACTIVE_JOINTS, dtype=np.float32)

        for i in range(NUM_ACTIVE_JOINTS):
            if scores[i] < self.score_thr:
                z[i] = 0.0
                continue

            cx = int(np.clip(round(keypoints[i, 0]), 0, W - 1))
            cy = int(np.clip(round(keypoints[i, 1]), 0, H - 1))

            if r == 0:
                z[i] = depth_map[cy, cx]
            else:
                x0 = max(cx - r, 0);  x1 = min(cx + r + 1, W)
                y0 = max(cy - r, 0);  y1 = min(cy + r + 1, H)
                z[i] = depth_map[y0:y1, x0:x1].mean()

        return z

    # ------------------------------------------------------------------
    # Process a full video given 2D results
    # ------------------------------------------------------------------

    def lift_to_3d(
        self,
        pose2d_results: list[Pose2DResult],
        video_path:     str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Attach Z values to all 2D pose results, producing a (T, 9, 3) array.

        Reads the source video to get raw frames for depth estimation.
        Only frames that have a corresponding Pose2DResult are processed.

        Parameters
        ----------
        pose2d_results : output of PoseExtractor.process_video()
        video_path     : original video (needed to re-read frames for depth)

        Returns
        -------
        keypoints_3d : (T, 9, 3) float32 — (x, y, z) in pixel + depth space
        scores       : (T, 9)    float32 — unchanged from pose2d
        """
        self._load()

        video_path = Path(video_path)
        T          = len(pose2d_results)

        # Build a lookup: frame_idx → Pose2DResult
        frame_map  = {r.frame_idx: r for r in pose2d_results}
        frame_idxs = set(frame_map.keys())

        keypoints_3d = np.zeros((T, NUM_ACTIVE_JOINTS, 3), dtype=np.float32)
        scores_out   = np.zeros((T, NUM_ACTIVE_JOINTS),    dtype=np.float32)

        result_cursor = 0  # position in keypoints_3d / scores_out

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        raw_idx = 0
        try:
            while result_cursor < T:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if raw_idx not in frame_idxs:
                    raw_idx += 1
                    continue

                pose2d    = frame_map[raw_idx]
                depth_map = self.estimate_depth(frame_bgr)
                z         = self._sample_z(depth_map, pose2d.keypoints, pose2d.scores)

                keypoints_3d[result_cursor, :, :2] = pose2d.keypoints  # x, y
                keypoints_3d[result_cursor, :,  2] = z                  # z
                scores_out[result_cursor]           = pose2d.scores

                result_cursor += 1
                raw_idx       += 1

                if result_cursor % 50 == 0:
                    logger.info("Depth lifted %d / %d frames.", result_cursor, T)

        finally:
            cap.release()

        logger.info(
            "Depth lifting complete. Output shape: %s", keypoints_3d.shape
        )
        return keypoints_3d, scores_out  # (T, 9, 3),  (T, 9)
