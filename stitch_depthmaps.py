#!/usr/bin/env python3
"""Stitch exported depth-map PNGs into a video.

This is a standalone utility for turning a folder of depth-map images into a
comparison video without changing the main pipeline.
"""

from __future__ import annotations

import argparse
import re
from math import gcd
from pathlib import Path

import cv2


FRAME_RE = re.compile(r"frame_(\d+)_depthmap\.png$", re.IGNORECASE)


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = FRAME_RE.search(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def collect_frames(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        frames = list(input_dir.rglob(pattern))
    else:
        frames = list(input_dir.glob(pattern))
    frames.sort(key=frame_sort_key)
    return frames


def infer_frame_step(frames: list[Path]) -> int:
    frame_numbers: list[int] = []
    for frame_path in frames:
        match = FRAME_RE.search(frame_path.name)
        if match:
            frame_numbers.append(int(match.group(1)))

    if len(frame_numbers) < 2:
        return 1

    deltas = [b - a for a, b in zip(frame_numbers, frame_numbers[1:]) if b > a]
    if not deltas:
        return 1

    step = deltas[0]
    for delta in deltas[1:]:
        step = gcd(step, delta)
    return max(step, 1)


def infer_output_path(input_dir: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_dir.with_suffix(".mp4")


def stitch_depthmaps(
    input_dir: Path,
    output_path: Path,
    playback_fps: float,
    pattern: str,
    recursive: bool,
) -> Path:
    frames = collect_frames(input_dir, pattern, recursive)
    if not frames:
        raise FileNotFoundError(
            f"No depth-map PNGs found in {input_dir} matching {pattern!r}"
        )

    first_frame = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        playback_fps,
        (width, height),
    )

    written = 0
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")

            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written == 0:
        raise RuntimeError(f"No frames were written to {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch exported depth-map PNGs into a video."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing depth-map PNGs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to <input_dir>.mp4.",
    )
    parser.add_argument(
        "--pattern",
        default="frame_*_depthmap.png",
        help="Glob pattern for frame PNGs inside input_dir.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Original video frame rate. When set, the stitcher compensates for skipped frames.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for matching frames recursively inside input_dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    frames = collect_frames(input_dir, args.pattern, args.recursive)
    if not frames:
        raise FileNotFoundError(
            f"No depth-map PNGs found in {input_dir} matching {args.pattern!r}"
        )

    output_path = infer_output_path(input_dir, args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    frame_step = infer_frame_step(frames)
    playback_fps = args.source_fps / frame_step if args.source_fps else 30.0 / frame_step

    stitched = stitch_depthmaps(
        input_dir=input_dir,
        output_path=output_path,
        playback_fps=playback_fps,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    print(f"Using playback FPS: {playback_fps:.3f} (frame step {frame_step})")
    print(stitched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())