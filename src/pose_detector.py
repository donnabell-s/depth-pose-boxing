"""
pose_detector.py — 2D keypoint extraction using RTMPose via MMPose.

Input  : video file (MP4 / AVI / MOV)
Output : (T, 9, 2) keypoints in pixel space  — active joints only
         (T, 9)    confidence scores per joint

RTMPose internally outputs all 17 COCO joints. This module slices
ACTIVE_JOINTS at the output boundary so every downstream file works
exclusively with local indices 0–8. See utils.py for the mapping.

Model options
-------------
  "body-s" : RTMPose-Small  — fastest, use on Colab free tier
  "body-m" : RTMPose-Medium — good balance
  "body-l" : RTMPose-Large  — best accuracy, recommended for final data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.utils import (
    ACTIVE_JOINTS,
    ACTIVE_JOINT_NAMES,
    NUM_ACTIVE_JOINTS,
    load_video_frames,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# Checkpoints download automatically on first run via MMPose.
# ──────────────────────────────────────────────────────────────────────────────

_MODELS: dict[str, dict] = {
    "body-s": {
        "config": "rtmpose-s_8xb256-420e_coco-256x192.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth"
        ),
    },
    "body-m": {
        "config": "rtmpose-m_8xb256-420e_coco-256x192.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
        ),
    },
    "body-l": {
        "config": "rtmpose-l_8xb256-420e_coco-256x192.py",
        "checkpoint": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth"
        ),
    },
}

_DET_CONFIG = "rtmdet-nano_320-8xb32_coco-person.py"
_DET_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmdet-nano_8xb32-100e_coco-obj365-person-05d8511e.pth"
)


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
    Wraps MMPose RTMPose inference. Model is loaded once on first call.

    Parameters
    ----------
    model    : "body-s" | "body-m" | "body-l"
    device   : "cuda:0" | "cpu"
    det_thr  : minimum person detection confidence
    pose_thr : joints below this confidence are zeroed out
    """

    def __init__(
        self,
        model:    str   = "body-l",
        device:   str   = "cuda:0",
        det_thr:  float = 0.5,
        pose_thr: float = 0.3,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(_MODELS)}")
        self.model    = model
        self.device   = device
        self.det_thr  = det_thr
        self.pose_thr = pose_thr
        self._inferencer = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._inferencer is not None:
            return

        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "MMPose is not installed.\n"
                "Run:  pip install openmim && "
                "mim install mmengine 'mmcv>=2.0.0' mmdet mmpose"
            ) from exc

        cfg = _MODELS[self.model]
        logger.info("Loading RTMPose — model: %s | device: %s", self.model, self.device)

        self._inferencer = MMPoseInferencer(
            pose2d=cfg["config"],
            pose2d_weights=cfg["checkpoint"],
            det_model=_DET_CONFIG,
            det_weights=_DET_CHECKPOINT,
            det_cat_ids=[0],    # 0 = person in COCO
            device=self.device,
        )
        logger.info("RTMPose ready.")

    # ------------------------------------------------------------------
    # Single frame
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
    ) -> Pose2DResult | None:
        """
        Run RTMPose on one BGR frame.

        Internally produces (17, 2) keypoints then slices ACTIVE_JOINTS
        before returning, so the caller always sees (9, 2).

        Returns None if no person is detected above det_thr.
        """
        self._load()

        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w      = frame_bgr.shape[:2]

        result_gen  = self._inferencer(frame_rgb, return_datasamples=False, progress_bar=False)
        predictions = next(result_gen).get("predictions", [[]])[0]

        if not predictions:
            logger.debug("Frame %d: no person detected.", frame_idx)
            return None

        # Single boxer — keep only the highest-confidence detection
        best = max(predictions, key=lambda p: p.get("bbox_score", 0.0))

        if best.get("bbox_score", 0.0) < self.det_thr:
            logger.debug("Frame %d: detection below threshold.", frame_idx)
            return None

        # Full COCO-17 arrays
        kps_full    = np.array(best["keypoints"],       dtype=np.float32)  # (17, 2)
        scores_full = np.array(best["keypoint_scores"], dtype=np.float32)  # (17,)

        # ── Slice to active joints only ───────────────────────────────
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

        for frame_idx, frame_bgr in load_video_frames(video_path, skip_frames, max_frames):
            result = self.process_frame(frame_bgr, frame_idx)
            processed += 1
            if result is not None:
                results.append(result)

        detection_rate = 100 * len(results) / max(processed, 1)
        logger.info(
            "Processed %d frames → %d detections (%.1f%%)",
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

def stack_keypoints(results: list[Pose2DResult]) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack a list of Pose2DResult into dense arrays.

    Returns
    -------
    keypoints : (T, 9, 2) float32 — local joint indices
    scores    : (T, 9)    float32
    """
    keypoints = np.stack([r.keypoints for r in results], axis=0)  # (T, 9, 2)
    scores    = np.stack([r.scores    for r in results], axis=0)  # (T, 9)
    return keypoints, scores


def frame_indices(results: list[Pose2DResult]) -> np.ndarray:
    """Return the original video frame indices as (T,) int32 array."""
    return np.array([r.frame_idx for r in results], dtype=np.int32)
