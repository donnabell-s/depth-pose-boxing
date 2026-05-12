"""
video.py — Video I/O helpers for depth-pose-boxing.

Provides a single consistent interface for reading video frames
used by both pose_detector.py and depth_estimation.py.

Uses the FFMPEG backend with CAP_PROP_ORIENTATION_AUTO=1 so that
rotation metadata (e.g. iPhone MOV files) is handled transparently
without any additional manual correction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import cv2

logger = logging.getLogger(__name__)


def load_video_frames(
    video_path:   str | Path,
    skip_frames:  int = 0,
    max_frames:   int | None = None,
    front_camera: bool = True,  # default True — most recordings are front-facing
) -> Generator[tuple[int, any], None, None]:
    """
    Yield (frame_idx, bgr_frame) tuples from a video file.

    Rotation metadata (e.g. iPhone MOV files) is handled automatically
    by the FFMPEG backend — frames are always yielded in the correct orientation.

    Front camera mirroring
    ----------------------
    Front-facing phone cameras produce a horizontally mirrored image. YOLOv8
    labels joints by their position in the frame, so left/right are swapped
    compared to the subject's actual body. When front_camera=True (the default),
    each frame is flipped horizontally after rotation correction so that joint
    labels match the subject's real left/right sides.

    Set front_camera=False for rear-camera footage or any recording that has
    already been de-mirrored.

    Parameters
    ----------
    video_path   : path to MP4 / AVI / MOV
    skip_frames  : process every (skip_frames + 1)-th frame.
                   0 = every frame, 1 = every other frame.
    max_frames   : stop after this many yielded frames (None = full video)
    front_camera : flip frames horizontally to correct front-camera mirroring

    Yields
    ------
    (frame_idx, bgr_array) — frame_idx is the absolute frame number
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # FFMPEG backend handles rotation metadata (MOV/iPhone) automatically.
    # MSMF (Windows default) ignores it, so we force FFMPEG here.
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

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
                if front_camera:
                    if yielded == 0:
                        logger.info("Front camera mode: applying horizontal flip to correct mirroring.")
                    frame = cv2.flip(frame, 1)  # 1 = horizontal flip
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
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def video_frame_size(video_path: str | Path) -> tuple[int, int]:
    """Return (height, width) of decoded video frames after auto-rotation."""
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame from {video_path}")
    return frame.shape[:2]  # (height, width)