"""
pose_detector.py — 2D keypoint extraction using YOLOv8-Pose via Ultralytics.

Input  : video file (MP4 / MOV / AVI)
Output : (T, 9, 2) keypoints in pixel space  — active joints only
         (T, 9)    confidence scores per joint

YOLOv8-Pose outputs COCO-17 keypoints natively so the ACTIVE_JOINTS
slice works without any remapping. See constants.py for the joint mapping.

Model options
-------------
  "nano"   : YOLOv8n-pose — fastest, lowest accuracy
  "small"  : YOLOv8s-pose — good balance for CPU
  "medium" : YOLOv8m-pose — recommended for GPU
  "large"  : YOLOv8l-pose — best accuracy
  "xlarge" : YOLOv8x-pose — highest accuracy, most VRAM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.constants import (
    ACTIVE_JOINTS,
    NUM_ACTIVE_JOINTS,
)
from src.video import load_video_frames

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# Models download automatically from Ultralytics on first run.
# ──────────────────────────────────────────────────────────────────────────────

_MODELS: dict[str, str] = {
    "nano":   "yolov8n-pose.pt",
    "small":  "yolov8s-pose.pt",
    "medium": "yolov8m-pose.pt",
    "large":  "yolov8l-pose.pt",
    "xlarge": "yolov8x-pose.pt",
}

DEFAULT_MODEL = "large"


# ──────────────────────────────────────────────────────────────────────────────
# Output container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Pose2DResult:
    """
    2D pose result for a single video frame.

    Attributes
    ----------
    frame_idx : absolute frame number in the source video
    keypoints : (9, 2) float32 — (x, y) in pixel coords, local joint indices
    scores    : (9,)   float32 — per-joint confidence in [0, 1]
    frame_hw  : (height, width) of the source frame
    """
    frame_idx: int
    keypoints: np.ndarray   # (9, 2)
    scores:    np.ndarray   # (9,)
    frame_hw:  tuple[int, int]

    def valid_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Boolean mask — True where joint confidence >= threshold."""
        return self.scores >= threshold

    def valid_keypoints(self, threshold: float = 0.3) -> np.ndarray:
        """Return keypoints with low-confidence joints zeroed out."""
        kps = self.keypoints.copy()
        kps[~self.valid_mask(threshold)] = 0.0
        return kps


# ──────────────────────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    Wraps YOLOv8-Pose inference via Ultralytics.
    Model is loaded once on first call (lazy loading).

    Parameters
    ----------
    model    : "nano" | "small" | "medium" | "large" | "xlarge"
    device   : "cuda:0" | "cpu"
    det_thr  : minimum person detection confidence
    pose_thr : joints below this confidence are zeroed out
    """

    def __init__(
        self,
        model:    str   = DEFAULT_MODEL,
        device:   str   = "cuda:0",
        det_thr:  float = 0.5,
        pose_thr: float = 0.3,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Choose from: {list(_MODELS)}"
            )
        self.model    = model
        self.device   = device
        self.det_thr  = det_thr
        self.pose_thr = pose_thr
        self._yolo    = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._yolo is not None:
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed.\n"
                "Run:  pip install ultralytics"
            ) from exc

        model_file = _MODELS[self.model]
        logger.info(
            "Loading YOLOv8-Pose — model: %s | device: %s",
            model_file, self.device,
        )
        self._yolo = YOLO(model_file)
        logger.info("YOLOv8-Pose ready.")

    # ------------------------------------------------------------------
    # Single frame
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
    ) -> Pose2DResult | None:
        """
        Run YOLOv8-Pose on one BGR frame.

        Internally produces (17, 2) COCO keypoints then slices ACTIVE_JOINTS
        before returning, so the caller always sees (9, 2).

        Returns None if no person is detected above det_thr.
        """
        self._load()

        h, w = frame_bgr.shape[:2]

        results = self._yolo(
            frame_bgr,
            device=self.device,
            conf=self.det_thr,
            verbose=False,
        )

        if not results or results[0].keypoints is None:
            logger.debug("Frame %d: no person detected.", frame_idx)
            return None

        result = results[0]

        if len(result.boxes) == 0:
            logger.debug("Frame %d: no person detected.", frame_idx)
            return None

        # Single boxer — keep highest confidence detection
        confs = result.boxes.conf.cpu().numpy()
        best  = int(np.argmax(confs))

        if confs[best] < self.det_thr:
            logger.debug("Frame %d: detection below threshold.", frame_idx)
            return None

        # Extract keypoints — shape (17, 3) where [:, 2] is confidence
        kps_all = result.keypoints.data[best].cpu().numpy()  # (17, 3)

        kps_full    = kps_all[:, :2].astype(np.float32)  # (17, 2)
        scores_full = kps_all[:, 2].astype(np.float32)   # (17,)

        # Slice to active joints only
        kps    = kps_full[ACTIVE_JOINTS]     # (9, 2)
        scores = scores_full[ACTIVE_JOINTS]  # (9,)

        # Zero out low-confidence joints
        low_conf = scores < self.pose_thr
        kps[low_conf]    = 0.0
        scores[low_conf] = 0.0

        return Pose2DResult(
            frame_idx=frame_idx,
            keypoints=kps,
            scores=scores,
            frame_hw=(h, w),
        )

    # ------------------------------------------------------------------
    # Full video
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        skip_frames: int = 0,
        max_frames:  int | None = None,
    ) -> list[Pose2DResult]:
        """
        Extract 2D poses for every processed frame of a video.

        Parameters
        ----------
        video_path  : path to the video file
        skip_frames : process every (skip_frames + 1)-th frame
        max_frames  : cap on number of frames processed

        Returns
        -------
        List of Pose2DResult, one per frame where a person was detected.
        Frames with no detection are omitted.
        """
        self._load()

        results   = []
        processed = 0

        for frame_idx, frame_bgr in load_video_frames(
            video_path, skip_frames, max_frames
        ):
            result = self.process_frame(frame_bgr, frame_idx)
            processed += 1
            if result is not None:
                results.append(result)

            if processed % 50 == 0:
                logger.info(
                    "Processed %d frames → %d detections so far.",
                    processed, len(results),
                )

        detection_rate = 100 * len(results) / max(processed, 1)
        logger.info(
            "Processed %d frames -> %d detections (%.1f%%)",
            processed, len(results), detection_rate,
        )

        if len(results) == 0:
            raise RuntimeError(
                "No persons detected in video. "
                "Try lowering --det-thr or check your video path."
            )

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for downstream stages
# ──────────────────────────────────────────────────────────────────────────────

def stack_keypoints(
    results: list[Pose2DResult],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack a list of Pose2DResult into dense arrays.

    Returns
    -------
    keypoints : (T, 9, 2) float32 — local joint indices
    scores    : (T, 9)    float32
    """
    keypoints = np.stack([r.keypoints for r in results], axis=0)
    scores    = np.stack([r.scores    for r in results], axis=0)
    return keypoints, scores


def frame_indices(results: list[Pose2DResult]) -> np.ndarray:
    """Return the original video frame indices as (T,) int32 array."""
    return np.array([r.frame_idx for r in results], dtype=np.int32)