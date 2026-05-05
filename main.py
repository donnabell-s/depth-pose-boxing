"""
main.py — depth-pose-boxing Phase 1 entry point.

Wires the full pipeline end-to-end:
    video → pose2d → depth → backproject → normalize → .npy

Usage
-----
    python main.py --video data/raw/boxer_01.mp4

With a specific camera profile:
    python main.py --video data/raw/boxer_01.mp4 --camera macbook

Validate against ground truth after processing:
    python main.py --video data/raw/boxer_01.mp4 --validate --gt data/athlete_pose_3d/boxer_01.npy

Camera profiles
---------------
Add your calibrated intrinsics to CAMERA_PROFILES below.
Run calibrate.py to get your fx, fy, cx, cy values.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.backproject import CameraIntrinsics, backproject
from src.depth_estimation import DepthEstimator
from src.normalize import NormConfig, normalize
from src.pose_detector import PoseExtractor
from src.data_validation import load_ground_truth, print_report
from src.video import video_fps, video_frame_size


CAMERA_PROFILES: dict[str, CameraIntrinsics] = {
    "iphone13": CameraIntrinsics(fx=1452.59, fy=1453.74, cx=996.58, cy=510.20),
    "oppo": CameraIntrinsics(fx=826.75, fy=827.42, cx=648.15, cy=345.17),
}
  

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("depth-pose-boxing")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:

    # ── Validate camera profile ───────────────────────────────────────────────
    if args.camera not in CAMERA_PROFILES:
        logger.error(
            "Unknown camera profile '%s'. Available: %s",
            args.camera, list(CAMERA_PROFILES)
        )
        sys.exit(1)

    intrinsics = CAMERA_PROFILES[args.camera]

    if intrinsics.fx == 0.0 or intrinsics.fy == 0.0:
        logger.error(
            "Camera profile '%s' has fx=0 or fy=0. "
            "Run calibrate.py and fill in your intrinsics in CAMERA_PROFILES.",
            args.camera,
        )
        sys.exit(1)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.npy"

    # ── Read video metadata ───────────────────────────────────────────────────
    fps    = video_fps(video_path)
    hw     = video_frame_size(video_path)

    logger.info("=" * 55)
    logger.info("depth-pose-boxing — Phase 1 Pipeline")
    logger.info("Video    : %s", video_path.name)
    logger.info("FPS      : %.1f | Frame size: %dx%d", fps, hw[1], hw[0])
    logger.info("Camera   : %s | fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                args.camera,
                intrinsics.fx, intrinsics.fy,
                intrinsics.cx, intrinsics.cy)
    logger.info("Device   : %s", args.device)
    logger.info("=" * 55)

    t_start = time.perf_counter()

    # ── Step 1 — 2D pose extraction ───────────────────────────────────────────
    logger.info("[1/4] Extracting 2D keypoints (RTMPose-%s)...", args.pose_model)
    extractor     = PoseExtractor(
        model=args.pose_model,
        device=args.device,
        det_thr=args.det_thr,
        pose_thr=args.pose_thr,
    )
    pose2d_results = extractor.process_video(
        video_path,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
    )

    # ── Step 2 — Depth estimation + Z sampling ────────────────────────────────
    logger.info("[2/4] Estimating depth (Depth Anything V2 — %s)...", args.depth_model)
    estimator = DepthEstimator(
        model=args.depth_model,
        device=args.device,
        sampling_radius=args.depth_radius,
        score_thr=args.pose_thr,
    )
    keypoints_3d, scores = estimator.lift_to_3d(pose2d_results, video_path)

    # ── Step 3 — Back-projection ──────────────────────────────────────────────
    logger.info("[3/4] Back-projecting to camera space...")
    points_3d = backproject(keypoints_3d, intrinsics, scores, score_thr=args.pose_thr)

    # ── Step 4 — Normalisation ────────────────────────────────────────────────
    logger.info("[4/4] Normalising skeleton sequence...")
    cfg      = NormConfig(
        fps=fps,
        score_thr=args.pose_thr,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        flip_y=True,
    )
    sequence = normalize(points_3d, scores, cfg)  # (T, 9, 3)

    # ── Save output ───────────────────────────────────────────────────────────
    np.save(output_path, sequence)
    t_total = time.perf_counter() - t_start

    logger.info("=" * 55)
    logger.info("Done in %.1f s", t_total)
    logger.info("Output shape : %s", sequence.shape)
    logger.info("Saved to     : %s", output_path)
    logger.info("=" * 55)

    # ── Optional validation ───────────────────────────────────────────────────
    if args.validate:
        if not args.gt:
            logger.error("--validate requires --gt <ground_truth.npy>")
            sys.exit(1)
        logger.info("Validating against ground truth...")
        gt = load_ground_truth(args.gt, num_frames=sequence.shape[0])
        print_report(sequence, gt)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="depth-pose-boxing — Phase 1 metric extraction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    ap.add_argument("--video", required=True,
                    help="Path to input video (MP4 / AVI / MOV)")

    # Camera
    ap.add_argument("--camera", default="default",
                    help="Camera profile name from CAMERA_PROFILES in main.py")

    # Output
    ap.add_argument("--output-dir", default="data/processed",
                    help="Directory for .npy output files")

    # Device
    ap.add_argument("--device", default="cuda:0",
                    help="PyTorch device — 'cuda:0' or 'cpu'")

    # Step 1 — RTMPose
    ap.add_argument("--pose-model", default="body-l",
                    choices=["body-s", "body-m", "body-l"],
                    help="RTMPose model size")
    ap.add_argument("--det-thr", type=float, default=0.5,
                    help="Person detection confidence threshold")
    ap.add_argument("--pose-thr", type=float, default=0.3,
                    help="Joint confidence threshold")
    ap.add_argument("--skip-frames", type=int, default=0,
                    help="Process every (N+1)-th frame. 0 = every frame")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N processed frames")

    # Step 2 — Depth Anything V2
    ap.add_argument("--depth-model", default="vit-b",
                    choices=["vit-s", "vit-b", "vit-l"],
                    help="Depth Anything V2 model variant")
    ap.add_argument("--depth-radius", type=int, default=2,
                    help="Depth sampling patch radius (0 = single pixel)")

    # Step 4 — One Euro Filter
    ap.add_argument("--min-cutoff", type=float, default=1.0,
                    help="One Euro Filter min cutoff — lower = smoother at rest")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="One Euro Filter beta — higher = less lag on fast motion")

    # Validation
    ap.add_argument("--validate", action="store_true",
                    help="Run MPJPE validation after processing")
    ap.add_argument("--gt", default=None,
                    help="Path to ground truth .npy for validation")

    return ap


if __name__ == "__main__":
    parser = _build_parser()
    run(parser.parse_args())