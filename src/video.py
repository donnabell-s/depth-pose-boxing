"""
video.py — Video I/O helpers for depth-pose-boxing.

Provides a single consistent interface for reading video frames
used by both pose_detector.py and depth_estimation.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import cv2

logger = logging.getLogger(__name__)


def load_video_frames(
    video_path:  str | Path,
    skip_frames: int = 0,
    max_frames:  int | None = None,
) -> Generator[tuple[int, any], None, None]:
    """
    Yield (frame_idx, bgr_frame) tuples from a video file.

    Parameters
    ----------
    video_path  : path to MP4 / AVI / MOV
    skip_frames : process every (skip_frames + 1)-th frame.
                  0 = every frame, 1 = every other frame.
    max_frames  : stop after this many yielded frames (None = full video)

    Yields
    ------
    (frame_idx, bgr_array) — frame_idx is the absolute frame number
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Video: %s | frames=%d | fps=%.1f | skip=%d",
        video_path.name, total, fps, skip_frames,
    )

    raw_idx = 0
    yielded = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if skip_frames == 0 or raw_idx % (skip_frames + 1) == 0:
                yield raw_idx, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break

            raw_idx += 1
    finally:
        cap.release()

    logger.info("Loaded %d frames from %s.", yielded, video_path.name)


def video_fps(video_path: str | Path) -> float:
    """Return the FPS of a video file without reading all frames."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def video_frame_size(video_path: str | Path) -> tuple[int, int]:
    """
    Return (height, width) of the video frames.

    Used by backproject.py to estimate camera intrinsics when
    no calibration values are available.
    """
    cap = cv2.VideoCapture(str(video_path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h, w