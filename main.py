"""
main.py — depth-pose-boxing Phase 1 entry point.

Wires the full pipeline end-to-end:
    video → pose2d → depth → backproject → normalize → .npy

Usage
-----
    python main.py --video data/raw/boxer_01.mp4 --camera iphone13

With debug video output:
    python main.py --video data/raw/boxer_01.mp4 --camera iphone13 --debug-video

Validate against ground truth after processing:
    python main.py --video data/raw/boxer_01.mp4 --camera iphone13 --validate --gt data/athlete_pose_3d/boxer_01.npy

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
from typing import NamedTuple

import numpy as np

from src.backproject import CameraIntrinsics, backproject
from src.depth_estimation import DepthEstimator
from src.normalize import NormConfig, normalize
from src.pose_detector import PoseExtractor, stack_keypoints, frame_indices
from src.data_validation import load_ground_truth, print_report
from src.video import video_fps, video_frame_size

# ──────────────────────────────────────────────────────────────────────────────
# Camera profiles — fill in calibrated values from calibrate.py.
# ──────────────────────────────────────────────────────────────────────────────

class CameraProfile(NamedTuple):
    intrinsics: CameraIntrinsics


CAMERA_PROFILES: dict[str, CameraProfile] = {
    "iphone13": CameraProfile(
        intrinsics=CameraIntrinsics(fx=1452.59, fy=1453.74, cx=996.58, cy=510.20),
    ),
    "oppo": CameraProfile(
        intrinsics=CameraIntrinsics(fx=826.75, fy=827.42, cx=648.15, cy=345.17),
    ),
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

    profile    = CAMERA_PROFILES[args.camera]
    intrinsics = profile.intrinsics

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
    stem        = video_path.stem
    output_path = output_dir / f"{stem}.npy"
    kp2d_path   = output_dir / f"{stem}_2d.npy"
    debug_path  = output_dir / f"{stem}_debug.mp4"

    # ── Read video metadata ───────────────────────────────────────────────────
    fps = video_fps(video_path)
    hw  = video_frame_size(video_path)

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
    logger.info("[1/4] Extracting 2D keypoints (YOLOv8-%s)...", args.pose_model)
    extractor = PoseExtractor(
        model=args.pose_model,
        device=args.device,
        det_thr=args.det_thr,
        pose_thr=args.pose_thr,
    )
    pose2d_results = extractor.process_video(
        video_path,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
        apply_rotation=not args.skip_rotation,
    )

    # Save 2D keypoints for debug video rendering
    kps2d, scores2d = stack_keypoints(pose2d_results)   # (T, 9, 2), (T, 9)
    fidxs           = frame_indices(pose2d_results)      # (T,)
    np.save(kp2d_path, kps2d)
    logger.info("2D keypoints saved → %s", kp2d_path)

    # ── Step 2 — Depth estimation + Z sampling ────────────────────────────────
    logger.info("[2/4] Estimating depth (Depth Anything V2 — %s)...", args.depth_model)
    estimator = DepthEstimator(
        model=args.depth_model,
        device=args.device,
        sampling_radius=args.depth_radius,
        score_thr=args.pose_thr,
    )

    keep_maps = args.depthmap_every is not None
    depth_result = estimator.lift_to_3d(
        pose2d_results,
        video_path,
        keep_depth_maps=keep_maps,
        apply_rotation=not args.skip_rotation,
    )

    if keep_maps:
        keypoints_3d, scores, depth_maps = depth_result
    else:
        keypoints_3d, scores = depth_result
        depth_maps = None

    # ── Step 3 — Back-projection ──────────────────────────────────────────────
    logger.info("[3/4] Back-projecting to camera space...")
    points_3d = backproject(keypoints_3d, intrinsics, scores, score_thr=args.pose_thr)

    # ── Step 4 — Normalisation ────────────────────────────────────────────────
    logger.info("[4/4] Normalising skeleton sequence...")
    cfg = NormConfig(
        fps=fps,
        score_thr=args.pose_thr,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        flip_y=True,
    )
    sequence = normalize(points_3d, scores, cfg)  # (T, 9, 3)

    # ── Save 3D output ────────────────────────────────────────────────────────
    np.save(output_path, sequence)

    t_total = time.perf_counter() - t_start

    logger.info("=" * 55)
    logger.info("Done in %.1f s", t_total)
    logger.info("Output shape : %s", sequence.shape)
    logger.info("Saved to     : %s", output_path)
    logger.info("=" * 55)

    # ── Optional debug video ──────────────────────────────────────────────────
    if args.debug_video:
        logger.info("Rendering debug video...")
        from src.visualize import render_skeleton_video
        render_skeleton_video(
            video_path=video_path,
            keypoints_2d=kps2d,         # (T, 9, 2)
            scores=scores2d,            # (T, 9)
            frame_indices=fidxs,        # (T,)
            output_path=debug_path,
            keypoints_3d=keypoints_3d,  # (T, 9, 3) — z in [0,1] drawn at each joint
            fps=fps,
            score_thr=args.pose_thr,
            apply_rotation=not args.skip_rotation,
        )
        logger.info("Debug video saved → %s", debug_path)

    # ── Optional depth map export ─────────────────────────────────────────────
    if args.depthmap_every is not None and depth_maps is not None:
        logger.info("Saving depth map frames (every %d)...", args.depthmap_every)
        from src.visualize import save_depthmaps
        depthmap_dir = output_dir / f"{stem}_depthmap"
        save_depthmaps(
            video_path=video_path,
            keypoints_2d=kps2d,
            scores=scores2d,
            frame_indices=fidxs,
            depth_maps=depth_maps,
            output_dir=depthmap_dir,
            every_n=args.depthmap_every,
            score_thr=args.pose_thr,
        )
        logger.info("Depth maps saved → %s", depthmap_dir)

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
                    help="Directory for output files")

    # Device
    ap.add_argument("--device", default="cuda:0",
                    help="PyTorch device — 'cuda:0' or 'cpu'")

    # Step 1 — YOLOv8-Pose
    ap.add_argument("--pose-model", default="large",
                    choices=["nano", "small", "medium", "large", "xlarge"],
                    help="YOLOv8-Pose model size")
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

    # Rotation
    ap.add_argument("--skip-rotation", action="store_true",
                    help="Disable automatic rotation correction from video metadata (for debugging)")

    # Debug video
    ap.add_argument("--debug-video", action="store_true",
                    help="Render annotated skeleton video to data/processed/")

    # Depth map export
    ap.add_argument("--depthmap-every", type=int, default=None,
                    help="Save depth map PNG every N frames (e.g. 10). "
                         "Saved to data/processed/<name>_depthmap/")

    # Validation
    ap.add_argument("--validate", action="store_true",
                    help="Run MPJPE validation after processing")
    ap.add_argument("--gt", default=None,
                    help="Path to ground truth .npy for validation")

    return ap


if __name__ == "__main__":
    parser = _build_parser()
    run(parser.parse_args())