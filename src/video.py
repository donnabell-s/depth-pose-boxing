"""
video.py — Video I/O helpers for depth-pose-boxing.

Provides a single consistent interface for reading video frames
used by both pose_detector.py and depth_estimation.py.

Handles iPhone MOV rotation metadata automatically — OpenCV ignores
the rotation flag embedded in MOV containers, so frames are corrected
here before being yielded to the rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Generator

import cv2

logger = logging.getLogger(__name__)


def _read_rotation_ffprobe(video_path: Path) -> int:
    """
    Read video rotation from container metadata via ffprobe.

    More reliable than CAP_PROP_ORIENTATION_META on Windows (MSMF backend
    ignores the rotation atom in MOV files). Returns 0 if ffprobe is not
    available or metadata is absent.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return 0
        data = json.loads(proc.stdout)
        for stream in data.get("streams", []):
            # Newer ffprobe: rotation stored in side_data_list
            for sd in stream.get("side_data_list", []):
                if "rotation" in sd:
                    return abs(int(float(sd["rotation"])))
            # Older ffprobe: rotation stored as a stream tag
            tags = stream.get("tags", {})
            for key in ("rotate", "rotation"):
                if key in tags:
                    return abs(int(float(tags[key])))
    except Exception:
        pass
    return 0


def load_video_frames(
    video_path:  str | Path,
    skip_frames: int = 0,
    max_frames:  int | None = None,
) -> Generator[tuple[int, any], None, None]:
    """
    Yield (frame_idx, bgr_frame) tuples from a video file.

    Automatically corrects rotation metadata from iPhone MOV files.

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

    # Force FFMPEG backend — Windows' default MSMF backend silently ignores
    # rotation metadata in MOV containers, so CAP_PROP_ORIENTATION_META
    # always returns 0 there. FFMPEG backend reads it correctly and is
    # bundled with all standard pip opencv-python builds.
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Video: %s | frames=%d | fps=%.1f | skip=%d",
        video_path.name, total, fps, skip_frames,
    )

    # Read rotation metadata — iPhones embed rotation in MOV container.
    # FFMPEG backend makes CAP_PROP_ORIENTATION_META reliable; ffprobe is
    # kept as a fallback for edge cases.
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    if rotation == 0:
        rotation = _read_rotation_ffprobe(video_path)
    rotation_map = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotate_code = rotation_map.get(rotation, None)
    if rotate_code is not None:
        logger.info("Applying rotation correction: %d degrees", rotation)

    raw_idx = 0
    yielded = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

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
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def video_frame_size(video_path: str | Path) -> tuple[int, int]:
    """
    Return (height, width) of the video frames after rotation correction.

    Used by backproject.py for camera intrinsics estimation.
    """
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    if rotation == 0:
        rotation = _read_rotation_ffprobe(video_path)
    cap.release()

    # If rotated 90 or 270, width and height are swapped
    if rotation in (90, 270):
        return w, h  # return as (height, width) after rotation

    return h, w