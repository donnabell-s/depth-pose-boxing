"""
depth_estimation.py — Per-frame depth estimation using Depth Anything V2.

Input  : list of Pose2DResult from pose2d.py + original video path
Output : (T, 9, 3) array — (x, y, z) per active joint per frame

Z values are sampled from the monocular depth map at each joint's (x, y)
pixel location. Depth is relative (not metric) — it will be scaled during
normalisation in normalize.py.

Depth Anything V2 returns disparity-like values where smaller = farther.
This module normalises to [0, 1] and flips so that:
    z = 0  →  closest to camera
    z = 1  →  farthest from camera

Model variants
--------------
Relative models — "vit-s", "vit-b", "vit-l":
    Loaded from HuggingFace hub via transformers.pipeline().
    Output is normalised [0, 1], no physical units.

Metric models — "vit-s-metric", "vit-b-metric", "vit-l-metric":
    Loaded from local .pth checkpoints in models/ using DepthAnythingV2.
    Output is in metres (raw float32, not normalised).
    Intended for indoor scenes only (dataset=hypersim, max_depth=20).
    Requires the depth_anything_v2 package — clone the Depth Anything V2
    repo into the project root so that depth_anything_v2/ is importable.

Model options (pick based on available VRAM)
--------------------------------------------
  "vit-s" / "vit-s-metric" : ~2 GB  — use on Colab free tier T4 or limited GPU
  "vit-b" / "vit-b-metric" : ~4 GB  — recommended for offline batch processing
  "vit-l" / "vit-l-metric" : ~8 GB  — highest quality, use if VRAM allows

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

from src.pose_detector import Pose2DResult, stack_keypoints
from src.constants import NUM_ACTIVE_JOINTS
from src.video import load_video_frames

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

_MODELS: dict[str, str | dict] = {
    "vit-s": "depth-anything/Depth-Anything-V2-Small-hf",
    "vit-b": "depth-anything/Depth-Anything-V2-Base-hf",
    "vit-l": "depth-anything/Depth-Anything-V2-Large-hf",
    "vit-s-metric": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384],    "checkpoint": "models/depth_anything_v2_metric_hypersim_vits.pth"},
    "vit-b-metric": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768],    "checkpoint": "models/depth_anything_v2_metric_hypersim_vitb.pth"},
    "vit-l-metric": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "checkpoint": "models/depth_anything_v2_metric_hypersim_vitl.pth"},
}


# ──────────────────────────────────────────────────────────────────────────────
# Depth estimator
# ──────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Wraps Depth Anything V2 via HuggingFace transformers (relative) or local
    .pth checkpoint (metric).

    Parameters
    ----------
    model           : "vit-s" | "vit-b" | "vit-l" | "vit-s-metric" | "vit-b-metric" | "vit-l-metric"
    device          : "cuda:0" | "cpu"
    sampling_radius : patch radius for Z sampling at each joint pixel.
                      0 = single pixel, 2 = 5×5 patch (recommended).
    score_thr       : joints below this confidence get z = 0.0
    max_depth       : depth ceiling in metres for metric models (default 20).
                      Ignored by relative models. Raise for outdoor / large gyms.
    """

    def __init__(
        self,
        model:           str   = "vit-b",
        device:          str   = "cuda:0",
        sampling_radius: int   = 2,
        score_thr:       float = 0.3,
        max_depth:       float = 20.0,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODELS)}")
        self.model           = model
        self.device          = device
        self.sampling_radius = sampling_radius
        self.score_thr       = score_thr
        self.max_depth       = max_depth
        self._pipe           = None
        self._metric_model   = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_metric(self) -> bool:
        """True if this instance uses a metric depth model."""
        return self.model.endswith("-metric")

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.is_metric:
            if self._metric_model is not None:
                return

            try:
                import torch
                from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "depth_anything_v2 is not available.\n"
                    "Clone the Depth Anything V2 repo into the project root:\n"
                    "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
                    "so that depth_anything_v2/ is importable."
                ) from exc

            cfg = _MODELS[self.model]
            logger.info("Loading Depth Anything V2 metric model — variant: %s", self.model)
            model = DepthAnythingV2(
                encoder=cfg["encoder"],
                features=cfg["features"],
                out_channels=cfg["out_channels"],
                max_depth=self.max_depth,
            )
            model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
            self._metric_model = model.to(self.device).eval()
            logger.info("Depth Anything V2 metric model ready.")
        else:
            if self._pipe is not None:
                return

            try:
                from transformers import pipeline as hf_pipeline  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "transformers is not installed.\n"
                    "Run:  pip install transformers"
                ) from exc

            repo = _MODELS[self.model]
            gpu  = 0 if "cuda" in self.device else -1
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

    def estimate_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on one BGR frame.

        Returns
        -------
        depth : (H, W) float32
            Relative models: normalised [0, 1] — 0 = closest, 1 = farthest.
            Metric models  : raw depth in metres — larger = farther, no normalisation.
        """
        self._load()

        h, w = frame_bgr.shape[:2]

        if self.is_metric:
            # DepthAnythingV2.infer_image accepts BGR and returns (H, W) float32 in metres.
            depth = self._metric_model.infer_image(frame_bgr).astype(np.float32)
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            return depth

        from PIL import Image as PILImage  # type: ignore
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = PILImage.fromarray(frame_rgb)

        raw = np.array(self._pipe(pil_img)["depth"], dtype=np.float32)  # (H, W)

        # Resize to match source frame if model downsampled
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
    ) -> np.ndarray:
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
        pose2d_results:  list[Pose2DResult],
        video_path:      str | Path,
        keep_depth_maps: bool = False,
        front_camera:    bool = True,
    ) -> tuple:
        """
        Attach Z values to all 2D pose results, producing a (T, 9, 3) array.

        Reads the source video to get raw frames for depth estimation.
        Only frames that have a corresponding Pose2DResult are processed.

        Parameters
        ----------
        pose2d_results  : output of PoseExtractor.process_video()
        video_path      : original video (needed to re-read frames for depth)
        keep_depth_maps : if True, also return list of (H, W) depth maps
        front_camera    : passed to load_video_frames — flip horizontally for front-camera footage

        Returns
        -------
        keypoints_3d : (T, 9, 3) float32 — (x, y, z) where x and y are pixel coordinates and z is depth — relative normalised [0,1] when using vit-s/b/l models, or metric depth in metres when using vit-s/b/l-metric models
        scores       : (T, 9)    float32 — unchanged from pose2d
        depth_maps   : list of T (H, W) float32 arrays — only if keep_depth_maps=True
        """
        self._load()

        video_path = Path(video_path)
        T          = len(pose2d_results)

        # Build a lookup: frame_idx → Pose2DResult
        frame_map  = {r.frame_idx: r for r in pose2d_results}
        frame_idxs = set(frame_map.keys())

        keypoints_3d = np.zeros((T, NUM_ACTIVE_JOINTS, 3), dtype=np.float32)
        scores_out   = np.zeros((T, NUM_ACTIVE_JOINTS),    dtype=np.float32)
        depth_maps   = [] if keep_depth_maps else None

        result_cursor = 0

        for raw_idx, frame_bgr in load_video_frames(video_path, front_camera=front_camera):
            if result_cursor >= T:
                break

            if raw_idx not in frame_idxs:
                continue

            pose2d    = frame_map[raw_idx]
            depth_map = self.estimate_depth(frame_bgr)
            z         = self._sample_z(depth_map, pose2d.keypoints, pose2d.scores)

            keypoints_3d[result_cursor, :, :2] = pose2d.keypoints
            keypoints_3d[result_cursor, :,  2] = z
            scores_out[result_cursor]           = pose2d.scores

            if keep_depth_maps:
                depth_maps.append(depth_map)

            result_cursor += 1

            if result_cursor % 50 == 0:
                logger.info("Depth lifted %d / %d frames.", result_cursor, T)

        logger.info(
            "Depth lifting complete. Output shape: %s", keypoints_3d.shape
        )

        if keep_depth_maps:
            return keypoints_3d, scores_out, depth_maps
        return keypoints_3d, scores_out