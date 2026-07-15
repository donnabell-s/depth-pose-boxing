"""
inference_combined.py — End-to-end boxing punch analysis from a raw video.

Runs Phase 1 (pose extraction) then Phase 2 (classification + regression) to
produce an annotated MP4 showing detected punches with class, peak force, and
peak speed overlaid on each frame.

Usage
-----
    python inference_combined.py --video path/to/demo.mp4 \\
                                 --body-weight 58 \\
                                 --output annotated_demo.mp4

Skip Phase 1 if you already have a pose_norm.npy:
    python inference_combined.py --video path/to/demo.mp4 \\
                                 --body-weight 58 \\
                                 --skip-phase1 path/to/demo_pose_norm.npy \\
                                 --output annotated_demo.mp4
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────

T = 40
N_JOINTS = 9
N_CHANNELS = 3
N_FEATURES = N_JOINTS * N_CHANNELS  # 27

CLASS_TO_IDX = {"cross": 0, "hook": 1, "jab": 2, "uppercut": 3, "no_punch": 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

WINDOW_STEP = 2
CONFIDENCE_THRESHOLD = 0.82
MIN_PUNCH_LENGTH = 4   # minimum windows for a valid event
MAX_GAP_TO_MERGE = 4   # max window gap to merge same-class events

# Per-class annotation colors (BGR)
CLASS_COLORS_BGR = {
    "cross":    (180, 119,  31),
    "hook":     ( 14, 127, 255),
    "jab":      ( 44, 160,  44),
    "uppercut": ( 40,  39, 214),
    "no_punch": (200, 200, 200),
}

LABEL_DISPLAY_FRAMES = 30   # frames label stays visible after punch midpoint


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("inference_combined")


# ── Model Architectures ────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x + res)
        x = self.dropout(x)
        return x


class TCNClassifier(nn.Module):
    """Phase 2A — punch type classifier. channels=[32, 64, 128]."""

    def __init__(self, in_channels: int = N_FEATURES, n_classes: int = 5,
                 channels: list[int] | None = None,
                 kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        self.data_bn = nn.BatchNorm1d(in_channels)
        layers, prev = [], in_channels
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(prev, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.data_bn(x)
        x = self.tcn(x)
        x = x.mean(dim=2)
        return self.fc(x)


class TCNRegressor(nn.Module):
    """Phase 2B — peak force (N) + peak speed (m/s). channels=[32,64,128], ~184K params."""

    def __init__(self, in_channels: int = N_FEATURES,
                 channels: list[int] | None = None,
                 kernel_size: int = 5, dropout: float = 0.3, n_outputs: int = 2):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        self.data_bn = nn.BatchNorm1d(in_channels)
        layers, prev = [], in_channels
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(prev, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(prev + 1, 64),   # +1 for body weight scalar
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x: torch.Tensor, body_weight: torch.Tensor) -> torch.Tensor:
        x = self.data_bn(x)
        x = self.tcn(x)
        x = x.mean(dim=2)
        x = torch.cat([x, body_weight.unsqueeze(1)], dim=1)
        return self.fc(x)


# ── Camera Intrinsics (mirrors main.py profiles) ───────────────────────────────

_CAMERA_PROFILES: dict[str, tuple[float, float, float, float]] = {
    # (fx, fy, cx, cy)
    "iphone13": (1452.59, 1453.74, 996.58, 510.20),
    "oppo":     ( 826.75,  827.42, 648.15, 345.17),
}


def _resolve_intrinsics(args: argparse.Namespace, video_path: Path):
    from src.backproject import CameraIntrinsics
    from src.video import video_frame_size

    if args.intrinsics_json:
        import json
        with open(args.intrinsics_json) as f:
            cal = json.load(f)
        return CameraIntrinsics(
            fx=float(cal["fx"]), fy=float(cal["fy"]),
            cx=float(cal["cx"]), cy=float(cal["cy"]),
        )

    if args.camera in _CAMERA_PROFILES:
        fx, fy, cx, cy = _CAMERA_PROFILES[args.camera]
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    logger.warning(
        "Unknown camera profile '%s' — estimating intrinsics from frame size. "
        "Pass --intrinsics-json for accurate results.", args.camera
    )
    hw = video_frame_size(video_path)
    return CameraIntrinsics.estimate_from_frame(hw[1], hw[0])


# ── Phase 1 Pipeline ───────────────────────────────────────────────────────────

def run_phase1(video_path: Path, args: argparse.Namespace,
               device: str) -> np.ndarray:
    """Run the full Phase 1 pipeline and return the normalised pose array (T, 9, 3)."""
    from src.backproject import backproject
    from src.depth_estimation import DepthEstimator
    from src.normalize import NormConfig, normalize
    from src.pose_detector import PoseExtractor, stack_keypoints
    from src.video import video_fps

    logger.info("[Phase 1] Loading YOLOv8-%s + Depth Anything V2 (%s)...",
                args.pose_model, args.depth_model)

    extractor = PoseExtractor(
        model=args.pose_model, device=device, det_thr=0.5, pose_thr=0.3,
    )
    estimator = DepthEstimator(
        model=args.depth_model, device=device,
        sampling_radius=2, score_thr=0.3, max_depth=20.0,
    )

    logger.info("[Phase 1] Extracting 2D keypoints...")
    pose2d_results = extractor.process_video(
        video_path, front_camera=args.front_camera,
    )
    if not pose2d_results:
        raise RuntimeError(
            f"No poses detected in {video_path.name}. "
            "Check that the video contains a visible person."
        )
    kps2d, scores = stack_keypoints(pose2d_results)
    logger.info("[Phase 1] Detected %d frames with keypoints.", len(pose2d_results))

    logger.info("[Phase 1] Estimating depth and lifting to 3D...")
    keypoints_3d, scores3d = estimator.lift_to_3d(
        pose2d_results, video_path, keep_depth_maps=False,
        front_camera=args.front_camera,
    )

    logger.info("[Phase 1] Back-projecting to camera space...")
    intrinsics = _resolve_intrinsics(args, video_path)
    points_3d = backproject(keypoints_3d, intrinsics, scores3d, score_thr=0.3)

    logger.info("[Phase 1] Normalising skeleton sequence...")
    fps = video_fps(video_path)
    cfg = NormConfig(fps=fps, score_thr=0.3, sg_window=7, sg_polyorder=3, flip_y=True)
    pose_norm, shoulder_width = normalize(points_3d, scores3d, cfg)
    logger.info("[Phase 1] Done — pose shape: %s, shoulder_width: %.3f",
                pose_norm.shape, shoulder_width)
    return pose_norm


# ── Model Loading ──────────────────────────────────────────────────────────────

def _find_latest(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_classifier(
    checkpoint: Path, device: torch.device, channels: list[int] | None = None
) -> TCNClassifier:
    model = TCNClassifier(channels=channels)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    logger.info(
        "Classifier loaded:  %s  (%s params)",
        checkpoint.name, f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model


def load_regressor(checkpoint: Path, device: torch.device) -> TCNRegressor:
    model = TCNRegressor()
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    logger.info(
        "Regressor loaded:   %s  (%s params)",
        checkpoint.name, f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model


def load_target_stats(labels_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute force/speed normalisation stats from the training labels CSV."""
    df = pd.read_csv(labels_csv)
    mean = df[["peak_force_N", "peak_speed_mps"]].mean().values.astype(np.float32)
    std = df[["peak_force_N", "peak_speed_mps"]].std().values.astype(np.float32)
    logger.info(
        "Target stats — force: mean=%.1f std=%.1f | speed: mean=%.3f std=%.3f",
        mean[0], std[0], mean[1], std[1],
    )
    return mean, std


# ── Sliding Window Inference ───────────────────────────────────────────────────

def sliding_window_inference(
    pose_array: np.ndarray,
    cls_model: TCNClassifier,
    reg_model: TCNRegressor,
    body_weight_kg: float,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    window_size: int = T,
    step: int = WINDOW_STEP,
) -> dict:
    """Slide a T-frame window across pose_array and run both models per window.

    Returns dict:
        window_starts     (n,)     — start frame of each window
        window_centers    (n,)     — centre frame of each window
        predictions       (n,)     — argmax class index per window
        confidences       (n,)     — max softmax probability per window
        all_probs         (n, 5)   — full class probability distributions
        force_predictions (n,)     — denormalised peak force (N)
        speed_predictions (n,)     — denormalised peak speed (m/s)
    """
    total_frames = pose_array.shape[0]
    if total_frames < window_size:
        raise ValueError(
            f"Pose sequence too short: {total_frames} frames, need ≥{window_size}."
        )

    window_starts = list(range(0, total_frames - window_size + 1, step))
    n = len(window_starts)
    logger.info("Sliding window: %d windows over %d frames.", n, total_frames)

    # Build batch: (n, T, 9, 3) → (n, 27, T)
    batch = np.zeros((n, window_size, N_JOINTS, N_CHANNELS), dtype=np.float32)
    for i, start in enumerate(window_starts):
        batch[i] = pose_array[start : start + window_size]
    batch_flat = batch.reshape(n, window_size, -1).transpose(0, 2, 1)

    tensor = torch.from_numpy(batch_flat).to(device)
    bw = torch.full((n,), body_weight_kg, dtype=torch.float32, device=device)
    mean_t = torch.from_numpy(target_mean).to(device)
    std_t = torch.from_numpy(target_std).to(device)

    with torch.no_grad():
        cls_probs = F.softmax(cls_model(tensor), dim=1).cpu().numpy()
        reg_norm = reg_model(tensor, bw)
        reg_pred = (reg_norm * std_t + mean_t).cpu().numpy()

    return {
        "window_starts":      np.array(window_starts),
        "window_centers":     np.array(window_starts) + window_size // 2,
        "predictions":        cls_probs.argmax(axis=1),
        "confidences":        cls_probs.max(axis=1),
        "all_probs":          cls_probs,
        "force_predictions":  reg_pred[:, 0],
        "speed_predictions":  reg_pred[:, 1],
    }


# ── Event Merging ──────────────────────────────────────────────────────────────

def merge_predictions_to_events(
    results: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    min_punch_length: int = MIN_PUNCH_LENGTH,
    max_gap_to_merge: int = MAX_GAP_TO_MERGE,
) -> list[dict]:
    """Convert per-window predictions into discrete punch events.

    Each event dict: start_frame, end_frame, class, confidence,
                     peak_force_N, peak_speed_mps
    """
    predictions = results["predictions"].copy()
    confidences = results["confidences"]
    window_starts = results["window_starts"]
    force_preds = results["force_predictions"]
    speed_preds = results["speed_predictions"]
    no_punch_idx = CLASS_TO_IDX["no_punch"]

    # Step 1: low-confidence windows → no_punch
    predictions[confidences < confidence_threshold] = no_punch_idx

    if len(predictions) == 0:
        return []

    # Step 2: group consecutive same-class windows into raw events
    events: list[dict] = []
    cur_cls = int(predictions[0])
    cur_start = 0
    cur_forces = [float(force_preds[0])]
    cur_speeds = [float(speed_preds[0])]
    cur_confs = [float(confidences[0])]

    def _flush(end_idx: int) -> None:
        events.append({
            "start_frame":    int(window_starts[cur_start]),
            "end_frame":      int(window_starts[end_idx]) + T - 1,
            "class":          IDX_TO_CLASS[cur_cls],
            "confidence":     float(np.mean(cur_confs)),
            "peak_force_N":   float(np.max(cur_forces)),
            "peak_speed_mps": float(np.max(cur_speeds)),
        })

    for i in range(1, len(predictions)):
        if int(predictions[i]) == cur_cls:
            cur_forces.append(float(force_preds[i]))
            cur_speeds.append(float(speed_preds[i]))
            cur_confs.append(float(confidences[i]))
        else:
            _flush(i - 1)
            cur_cls = int(predictions[i])
            cur_start = i
            cur_forces = [float(force_preds[i])]
            cur_speeds = [float(speed_preds[i])]
            cur_confs = [float(confidences[i])]
    _flush(len(predictions) - 1)

    # Step 3: merge same-class events separated by a small gap
    merged: list[dict] = []
    for ev in events:
        if (
            merged
            and merged[-1]["class"] == ev["class"]
            and ev["start_frame"] - merged[-1]["end_frame"] <= max_gap_to_merge
        ):
            merged[-1]["end_frame"] = ev["end_frame"]
            merged[-1]["confidence"] = (merged[-1]["confidence"] + ev["confidence"]) / 2
            merged[-1]["peak_force_N"] = max(merged[-1]["peak_force_N"], ev["peak_force_N"])
            merged[-1]["peak_speed_mps"] = max(merged[-1]["peak_speed_mps"], ev["peak_speed_mps"])
        else:
            merged.append(ev)

    # Step 4: drop short punch events; remove no_punch entirely
    return [
        e for e in merged
        if e["class"] != "no_punch"
        and (e["end_frame"] - e["start_frame"] + 1) >= min_punch_length
    ]


# ── Annotated Video ────────────────────────────────────────────────────────────

def make_annotated_video(
    source_video: Path,
    output_path: Path,
    detected_punches: list[dict],
) -> None:
    """Render source_video with punch overlays written to output_path.

    Overlays:
    - Coloured border during active punch (class-specific colour)
    - Top-left box: punch class + force + speed (shown from midpoint for
      LABEL_DISPLAY_FRAMES frames)
    - History panel (bottom-right): last 3 completed punches
    - Frame counter (bottom-left)
    """
    # CAP_FFMPEG + ORIENTATION_AUTO handles rotation metadata (e.g. iPhone MOV)
    # without needing manual correction — same approach as src/video.py.
    cap = cv2.VideoCapture(str(source_video), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot create output video: {output_path}")

    # Pre-compute per-frame annotation state
    border_cls = ["no_punch"] * total_frames
    label_cls  = ["no_punch"] * total_frames
    label_conf  = [0.0] * total_frames
    label_force = [0.0] * total_frames
    label_speed = [0.0] * total_frames

    for ev in detected_punches:
        for f in range(ev["start_frame"], min(ev["end_frame"] + 1, total_frames)):
            border_cls[f] = ev["class"]

        mid = (ev["start_frame"] + ev["end_frame"]) // 2
        for f in range(mid, min(mid + LABEL_DISPLAY_FRAMES, total_frames)):
            label_cls[f]   = ev["class"]
            label_conf[f]  = ev["confidence"]
            label_force[f] = ev["peak_force_N"]
            label_speed[f] = ev["peak_speed_mps"]

    # Map each frame to its punch event (for history tracking)
    frame_to_event: dict[int, dict] = {}
    for ev in detected_punches:
        mid = (ev["start_frame"] + ev["end_frame"]) // 2
        frame_to_event[mid] = ev

    history: list[dict] = []   # last 3 completed punches
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fi = min(frame_idx, total_frames - 1)
        bc = border_cls[fi]
        lc = label_cls[fi]

        # Coloured border when inside a punch event
        if bc != "no_punch":
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1),
                          CLASS_COLORS_BGR[bc], 10)

        # Top-left label box
        if lc != "no_punch":
            # Update history at the first frame of a new label
            if fi in frame_to_event and (fi == 0 or label_cls[fi - 1] != lc):
                ev = frame_to_event[fi]
                if not history or history[-1] is not ev:
                    history.append(ev)
                    if len(history) > 3:
                        history.pop(0)

            cv2.rectangle(frame, (0, 0), (470, 130), (0, 0, 0), -1)
            cv2.putText(frame, lc.upper(), (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, CLASS_COLORS_BGR[lc], 2, cv2.LINE_AA)
            cv2.putText(frame, f"Force: {label_force[fi]:.0f} N", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Speed: {label_speed[fi]:.2f} m/s", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (0, 0), (310, 55), (0, 0, 0), -1)
            cv2.putText(frame, "no_punch", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, CLASS_COLORS_BGR["no_punch"],
                        2, cv2.LINE_AA)

        # History panel — bottom-right, last 3 punches
        if history:
            px = width - 290
            py_base = height - 10 - len(history) * 26
            for hi, h in enumerate(history):
                py = py_base + hi * 26
                txt = f"{h['class']}: {h['peak_force_N']:.0f}N  {h['peak_speed_mps']:.1f}m/s"
                cv2.putText(frame, txt, (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            CLASS_COLORS_BGR.get(h["class"], (200, 200, 200)),
                            1, cv2.LINE_AA)

        # Frame counter — bottom-left
        cv2.putText(frame, f"f={frame_idx}", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Annotated video → %s  (%d frames at %.1f fps)", output_path, frame_idx, fps)


# ── Console Summary ────────────────────────────────────────────────────────────

def print_summary(
    detected_punches: list[dict],
    cls_path: Path,
    reg_path: Path,
    video_path: Path,
    body_weight_kg: float,
    confidence_threshold: float,
) -> None:
    sep = "=" * 64
    print(sep)
    print("INFERENCE SUMMARY")
    print(sep)
    print(f"Video:                {video_path.name}")
    print(f"Classifier:           {cls_path.name}")
    print(f"Regressor:            {reg_path.name}")
    print(f"Body weight:          {body_weight_kg:.1f} kg")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Window size (T):      {T}")
    print(f"Window step:          {WINDOW_STEP}")
    print(f"\nDetected punches:     {len(detected_punches)}")

    if not detected_punches:
        print(sep)
        return

    forces = [p["peak_force_N"] for p in detected_punches]
    speeds = [p["peak_speed_mps"] for p in detected_punches]
    print(f"\nForce range:  {min(forces):.1f} – {max(forces):.1f} N  "
          f"(mean {np.mean(forces):.1f})")
    print(f"Speed range:  {min(speeds):.2f} – {max(speeds):.2f} m/s  "
          f"(mean {np.mean(speeds):.2f})")

    print(f"\nBy punch type:")
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for p in detected_punches:
        by_cls[p["class"]].append(p)
    for cls in sorted(by_cls):
        ps = by_cls[cls]
        fs = [p["peak_force_N"] for p in ps]
        ss = [p["peak_speed_mps"] for p in ps]
        print(f"  {cls:<10s}: {len(ps):>2} punches | "
              f"force {np.mean(fs):.0f} N | speed {np.mean(ss):.2f} m/s")

    print(f"\n{'#':>3}  {'Class':<10}  {'Frames':>13}  "
          f"{'Force (N)':>10}  {'Speed (m/s)':>11}  {'Conf':>6}")
    print("-" * 64)
    for i, p in enumerate(detected_punches, 1):
        print(f"{i:>3}  {p['class']:<10}  "
              f"{p['start_frame']:>5}–{p['end_frame']:<5}  "
              f"{p['peak_force_N']:>10.1f}  {p['peak_speed_mps']:>11.2f}  "
              f"{p['confidence']:>6.3f}")
    print(sep)


def save_csv(detected_punches: list[dict], csv_path: Path) -> None:
    fields = ["class", "start_frame", "end_frame",
              "peak_force_N", "peak_speed_mps", "confidence"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(detected_punches)
    logger.info("Punch events CSV → %s", csv_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Boxing inference: detect punches with class, force, and speed from raw video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    ap.add_argument("--video", required=True,
                    help="Input video file (MP4 / MOV / AVI)")
    ap.add_argument("--body-weight", type=float, required=True,
                    help="Subject body weight in kg (required by the regressor)")
    # Output
    ap.add_argument("--output", default="annotated_output.mp4",
                    help="Output annotated video path")
    ap.add_argument("--csv-output", default=None,
                    help="Optional path to save detected punches as CSV")
    ap.add_argument("--save-pose-norm", default=None, metavar="NPY_PATH",
                    help="Optional path to save the final normalised pose array from Phase 1")
    # Model overrides
    ap.add_argument("--classifier-channels", type=int, nargs="+", default=None,
                    help="Override TCN classifier channel sizes (default: 32 64 128)")
    ap.add_argument("--classifier-model", default=None,
                    help="Classifier checkpoint path "
                         "(default: newest *_best.pt in models/tcn/)")
    ap.add_argument("--regressor-model", default=None,
                    help="Regressor checkpoint path "
                         "(default: newest best.pt in models/tcn_regression/)")
    ap.add_argument("--labels-csv",
                    default="data/metadata/with_hardware/labels.csv",
                    help="Labels CSV for computing target normalisation statistics")
    # Inference params
    ap.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                    help="Minimum classifier confidence to register a punch window")
    # Phase 1 bypass
    ap.add_argument("--skip-phase1", default=None, metavar="POSE_NPY",
                    help="Path to a pre-computed *_pose_norm.npy to skip Phase 1")
    # Device
    ap.add_argument("--device", default="cuda:0",
                    help="PyTorch device ('cuda:0' or 'cpu')")
    # Phase 1 camera args
    ap.add_argument("--camera", default="iphone13",
                    help="Camera profile: iphone13 or oppo. "
                         "Ignored when --intrinsics-json is given.")
    ap.add_argument("--intrinsics-json", default=None,
                    help="JSON produced by src/calibrate.py (keys: fx, fy, cx, cy)")
    ap.add_argument("--pose-model", default="large",
                    choices=["nano", "small", "medium", "large", "xlarge"],
                    help="YOLOv8-Pose model size")
    ap.add_argument("--depth-model", default="vit-b-metric",
                    choices=["vit-s", "vit-b", "vit-l",
                             "vit-s-metric", "vit-b-metric", "vit-l-metric"],
                    help="Depth Anything V2 model variant")
    ap.add_argument("--front-camera", action="store_true", default=True,
                    help="Correct front-camera mirroring (default: on)")
    ap.add_argument("--no-front-camera", dest="front_camera", action="store_false",
                    help="Disable horizontal flip for rear camera footage")
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    # Device selection
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    project_root = Path(__file__).parent

    # ── Resolve model checkpoints ────────────────────────────────────────────
    if args.classifier_model:
        cls_path = Path(args.classifier_model)
    else:
        cls_path = _find_latest(project_root / "models" / "tcn", "*_best.pt")
    if cls_path is None or not cls_path.exists():
        logger.error(
            "No classifier checkpoint found. "
            "Train first (notebooks/pose_classification/04_train_tcn.ipynb) "
            "or pass --classifier-model."
        )
        sys.exit(1)

    if args.regressor_model:
        reg_path = Path(args.regressor_model)
    else:
        reg_path = _find_latest(
            project_root / "models" / "tcn_regression", "**/best.pt"
        )
    if reg_path is None or not reg_path.exists():
        logger.error(
            "No regressor checkpoint found. "
            "Train first (notebooks/regression/02_train_tcn.ipynb) "
            "or pass --regressor-model."
        )
        sys.exit(1)

    labels_csv = Path(args.labels_csv)
    if not labels_csv.is_absolute():
        labels_csv = project_root / labels_csv
    if not labels_csv.exists():
        logger.error("Labels CSV not found: %s", labels_csv)
        sys.exit(1)

    # ── Phase 1 — Pose Extraction ────────────────────────────────────────────
    if args.skip_phase1:
        npy_path = Path(args.skip_phase1)
        if not npy_path.exists():
            logger.error("Pose file not found: %s", npy_path)
            sys.exit(1)
        logger.info("Loading pre-computed pose array from %s", npy_path)
        pose_array = np.load(npy_path)
    else:
        logger.info("=== Phase 1: Pose Extraction ===")
        t0 = time.perf_counter()
        pose_array = run_phase1(video_path, args, device=str(device))
        logger.info("Phase 1 completed in %.1f s.", time.perf_counter() - t0)

    if args.save_pose_norm:
        npy_out_path = Path(args.save_pose_norm)
        npy_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_out_path, pose_array)
        logger.info("Saved normalised pose array to %s", npy_out_path)

    logger.info("Pose array shape: %s  dtype=%s", pose_array.shape, pose_array.dtype)

    # ── Load Models + Normalisation Stats ────────────────────────────────────
    logger.info("=== Phase 2: Loading Models ===")
    cls_model = load_classifier(cls_path, device, channels=args.classifier_channels)
    reg_model = load_regressor(reg_path, device)
    target_mean, target_std = load_target_stats(labels_csv)

    # ── Sliding Window Inference ─────────────────────────────────────────────
    logger.info("=== Phase 2: Sliding Window Inference ===")
    t0 = time.perf_counter()
    results = sliding_window_inference(
        pose_array, cls_model, reg_model,
        body_weight_kg=args.body_weight,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )
    logger.info("Inference: %d windows in %.2f s.",
                len(results["predictions"]), time.perf_counter() - t0)

    # ── Event Merging ────────────────────────────────────────────────────────
    detected_punches = merge_predictions_to_events(
        results, confidence_threshold=args.confidence,
    )

    # ── Console Summary ──────────────────────────────────────────────────────
    print_summary(detected_punches, cls_path, reg_path,
                  video_path, args.body_weight, args.confidence)

    # ── Annotated Video ──────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("=== Rendering Annotated Video ===")
    t0 = time.perf_counter()
    make_annotated_video(video_path, output_path, detected_punches)
    logger.info("Video rendering: %.1f s.", time.perf_counter() - t0)

    # ── Optional CSV ─────────────────────────────────────────────────────────
    if args.csv_output:
        save_csv(detected_punches, Path(args.csv_output))

    logger.info("Done.")


if __name__ == "__main__":
    main()
