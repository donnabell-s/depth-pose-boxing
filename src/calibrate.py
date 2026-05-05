"""
calibrate.py — Camera intrinsics calibration using a checkerboard video.

Detects checkerboard corners across video frames, runs OpenCV calibration,
and prints the fx, fy, cx, cy values to paste into CAMERA_PROFILES in main.py.

Usage
-----
    python calibrate.py --video data/raw/checkerboard.mp4

With a non-standard checkerboard size (default is 9x6 inner corners):
    python calibrate.py --video data/raw/checkerboard.mp4 --cols 8 --rows 5

Tips for filming the checkerboard
-----------------------------------
- Film for 10–15 seconds, moving the board to different angles and distances
- Cover corners of the frame, not just the centre
- Keep the board flat — avoid bending it
- Good lighting, no motion blur
- 20+ detected frames gives reliable results; 50+ is ideal

Output
------
Prints intrinsics ready to paste into main.py CAMERA_PROFILES:
    CameraIntrinsics(fx=849.3, fy=851.1, cx=638.7, cy=359.2)
Optionally saves to a .json file with --save.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("calibrate")


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate(
    video_path:    str | Path,
    board_cols:    int   = 9,     # inner corners along width
    board_rows:    int   = 6,     # inner corners along height
    square_size:   float = 1.0,   # physical size of each square (arbitrary units)
    skip_frames:   int   = 2,     # process every (skip+1)-th frame for speed
    min_frames:    int   = 20,    # minimum detected frames for reliable result
    save_path:     str | Path | None = None,
) -> dict:
    """
    Run OpenCV checkerboard calibration on a pre-recorded video.

    Parameters
    ----------
    video_path  : path to checkerboard video (MP4 / AVI / MOV)
    board_cols  : number of inner corners along the width  (default 9)
    board_rows  : number of inner corners along the height (default 6)
    square_size : physical square size — only affects scale, not fx/fy/cx/cy
    skip_frames : process every (skip+1)-th frame to speed up detection
    min_frames  : raise if fewer than this many frames detected
    save_path   : if provided, save results as JSON

    Returns
    -------
    dict with keys: fx, fy, cx, cy, rms, width, height, detected_frames
    """
    video_path  = Path(video_path)
    board_size  = (board_cols, board_rows)
    n_corners   = board_cols * board_rows

    # 3D object points for one checkerboard (z=0 plane)
    objp = np.zeros((n_corners, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []   # 3D points in real world space
    img_points = []   # 2D points in image plane

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)

    logger.info(
        "Video: %s | %dx%d | %.1f fps | %d frames",
        video_path.name, width, height, fps, total,
    )
    logger.info(
        "Checkerboard: %dx%d inner corners | skip=%d",
        board_cols, board_rows, skip_frames,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30, 0.001,
    )

    raw_idx   = 0
    detected  = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if raw_idx % (skip_frames + 1) != 0:
            raw_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            # Refine corner locations to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            obj_points.append(objp)
            img_points.append(corners_refined)
            detected += 1

            if detected % 10 == 0:
                logger.info("Detected corners in %d frames so far...", detected)

        raw_idx += 1

    cap.release()
    logger.info("Corner detection complete — %d / %d frames usable.", detected, raw_idx)

    if detected < min_frames:
        raise RuntimeError(
            f"Only {detected} frames with detected corners "
            f"(minimum required: {min_frames}).\n"
            "Tips:\n"
            "  - Film the checkerboard from more angles\n"
            "  - Ensure good lighting with no motion blur\n"
            "  - Check --cols and --rows match your actual board\n"
            "  - Try lowering --min-frames if your video is short"
        )

    # ── Run calibration ───────────────────────────────────────────────────────
    logger.info("Running calibration on %d frames...", detected)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (width, height), None, None
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    results = {
        "fx":              round(fx, 2),
        "fy":              round(fy, 2),
        "cx":              round(cx, 2),
        "cy":              round(cy, 2),
        "rms":             round(float(rms), 4),
        "width":           width,
        "height":          height,
        "detected_frames": detected,
        "board_cols":      board_cols,
        "board_rows":      board_rows,
    }

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Calibration Results")
    print("=" * 50)
    print(f"  Resolution       : {width} x {height}")
    print(f"  Frames used      : {detected}")
    print(f"  RMS error        : {rms:.4f}  (good: <1.0, great: <0.5)")
    print("-" * 50)
    print(f"  fx : {fx:.2f}")
    print(f"  fy : {fy:.2f}")
    print(f"  cx : {cx:.2f}")
    print(f"  cy : {cy:.2f}")
    print("-" * 50)
    print("  Paste into main.py CAMERA_PROFILES:")
    print(f'  "your_camera": CameraIntrinsics('
          f'fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}),')
    print("=" * 50 + "\n")

    if rms > 1.0:
        logger.warning(
            "RMS error %.4f is high (>1.0). Consider re-filming with:\n"
            "  - More angles and distances\n"
            "  - Better lighting\n"
            "  - A flatter checkerboard (no bending)",
            rms,
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to: %s", save_path)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Camera intrinsics calibration using a checkerboard video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--video",       required=True,
                    help="Path to checkerboard video (MP4 / AVI / MOV)")
    ap.add_argument("--cols",        type=int,   default=9,
                    help="Number of inner corners along the board width")
    ap.add_argument("--rows",        type=int,   default=6,
                    help="Number of inner corners along the board height")
    ap.add_argument("--square-size", type=float, default=1.0,
                    help="Physical square size in mm (optional — does not affect fx/fy/cx/cy)")
    ap.add_argument("--skip-frames", type=int,   default=2,
                    help="Process every (N+1)-th frame")
    ap.add_argument("--min-frames",  type=int,   default=20,
                    help="Minimum detected frames required for calibration")
    ap.add_argument("--save",        default=None,
                    help="Save results as JSON to this path")
    return ap


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    calibrate(
        video_path=args.video,
        board_cols=args.cols,
        board_rows=args.rows,
        square_size=args.square_size,
        skip_frames=args.skip_frames,
        min_frames=args.min_frames,
        save_path=args.save,
    )