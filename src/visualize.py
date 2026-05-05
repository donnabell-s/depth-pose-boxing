"""
visualize.py — Debug visualisation utilities for depth-pose-boxing.

Provides:
  render_skeleton_video() — annotated skeleton overlay + live depth graph

The output video is split into two panels:
  Left  — original video with skeleton overlay
  Right — live wrist depth graph with vertical progress line

Joint colour coding:
  Red    — wrists (punch endpoints)
  Blue   — shoulders
  Green  — hips
  White  — nose
  Cyan   — elbows
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.constants import (
    SKELETON_EDGES,
    ACTIVE_JOINT_NAMES,
    ACTIVE_NAME_TO_IDX,
)
from src.video import load_video_frames

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Joint indices (local)
# ──────────────────────────────────────────────────────────────────────────────

_L_WRIST = ACTIVE_NAME_TO_IDX["left_wrist"]   # local 5
_R_WRIST = ACTIVE_NAME_TO_IDX["right_wrist"]  # local 6

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (BGR)
# ──────────────────────────────────────────────────────────────────────────────

_JOINT_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),   # nose       — white
    1: (255, 120,  50),   # L shoulder — blue
    2: (255, 120,  50),   # R shoulder — blue
    3: (255, 200,   0),   # L elbow    — cyan
    4: (255, 200,   0),   # R elbow    — cyan
    5: (  0,  60, 255),   # L wrist    — red
    6: (  0, 160, 255),   # R wrist    — orange
    7: (  0, 200,  80),   # L hip      — green
    8: (  0, 200,  80),   # R hip      — green
}

_L_WRIST_COLOUR = (  0,  60, 255)   # red   — left wrist depth line
_R_WRIST_COLOUR = (  0, 160, 255)   # orange — right wrist depth line
_PROGRESS_COLOUR = (  0, 255, 255)  # yellow — progress line
_BONE_COLOUR    = (180, 180, 180)
_MISSING_COLOUR = (  0,   0, 100)
_FONT           = cv2.FONT_HERSHEY_SIMPLEX
_BG_COLOUR      = ( 18,  18,  30)   # dark background for graph panel


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton drawing
# ──────────────────────────────────────────────────────────────────────────────

def _draw_skeleton(
    frame:     np.ndarray,
    keypoints: np.ndarray,   # (9, 2)
    scores:    np.ndarray,   # (9,)
    score_thr: float = 0.3,
    radius:    int   = 6,
    thickness: int   = 2,
) -> np.ndarray:
    out   = frame.copy()
    valid = scores >= score_thr

    for (i, j) in SKELETON_EDGES:
        if valid[i] and valid[j]:
            p1 = tuple(keypoints[i].astype(int))
            p2 = tuple(keypoints[j].astype(int))
            cv2.line(out, p1, p2, _BONE_COLOUR, thickness, cv2.LINE_AA)

    for k in range(len(keypoints)):
        pt     = tuple(keypoints[k].astype(int))
        colour = _JOINT_COLOURS.get(k, (200, 200, 200)) if valid[k] else _MISSING_COLOUR
        size   = radius + 2 if k in [5, 6] else radius
        cv2.circle(out, pt, size, colour, -1, cv2.LINE_AA)
        cv2.circle(out, pt, size, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Depth graph panel
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph_panel(
    H: int,
    W: int,
    z_left:   np.ndarray,   # (T,) normalised Z for left wrist
    z_right:  np.ndarray,   # (T,) normalised Z for right wrist
    current_t: int,         # current frame index in the sequence
    T: int,                 # total number of frames
) -> np.ndarray:
    """
    Build a (H, W, 3) BGR graph panel showing wrist depth over time
    with a vertical progress line at current_t.
    """
    panel = np.full((H, W, 3), _BG_COLOUR, dtype=np.uint8)

    pad_x = 50   # left padding for y-axis labels
    pad_y = 30   # top/bottom padding
    gw    = W - pad_x - 10   # graph width
    gh    = H - 2 * pad_y    # graph height

    # Normalise Z values to [0, 1] for display
    all_z  = np.concatenate([z_left, z_right])
    valid  = all_z[all_z != 0]
    z_min  = float(valid.min()) if len(valid) > 0 else 0.0
    z_max  = float(valid.max()) if len(valid) > 0 else 1.0
    z_range = max(z_max - z_min, 1e-6)

    def to_px(t: int, z: float) -> tuple[int, int]:
        x = pad_x + int((t / max(T - 1, 1)) * gw)
        y = pad_y + gh - int(((z - z_min) / z_range) * gh)
        y = max(pad_y, min(pad_y + gh, y))
        return x, y

    # Draw grid lines
    for i in range(5):
        gy = pad_y + int(i * gh / 4)
        cv2.line(panel, (pad_x, gy), (pad_x + gw, gy), (40, 40, 60), 1)
        z_val = z_max - (i / 4) * z_range
        cv2.putText(panel, f"{z_val:.2f}", (2, gy + 4),
                    _FONT, 0.28, (120, 120, 140), 1, cv2.LINE_AA)

    # Draw wrist depth lines
    for t in range(1, T):
        if z_left[t] != 0 and z_left[t - 1] != 0:
            p1 = to_px(t - 1, z_left[t - 1])
            p2 = to_px(t,     z_left[t])
            cv2.line(panel, p1, p2, _L_WRIST_COLOUR, 2, cv2.LINE_AA)

        if z_right[t] != 0 and z_right[t - 1] != 0:
            p1 = to_px(t - 1, z_right[t - 1])
            p2 = to_px(t,     z_right[t])
            cv2.line(panel, p1, p2, _R_WRIST_COLOUR, 2, cv2.LINE_AA)

    # Draw progress line
    px = pad_x + int((current_t / max(T - 1, 1)) * gw)
    cv2.line(panel, (px, pad_y), (px, pad_y + gh), _PROGRESS_COLOUR, 2, cv2.LINE_AA)

    # Draw current Z values
    if z_left[current_t] != 0:
        cv2.putText(panel, f"L: {z_left[current_t]:.3f}",
                    (px + 4, pad_y + 20), _FONT, 0.4, _L_WRIST_COLOUR, 1, cv2.LINE_AA)
    if z_right[current_t] != 0:
        cv2.putText(panel, f"R: {z_right[current_t]:.3f}",
                    (px + 4, pad_y + 40), _FONT, 0.4, _R_WRIST_COLOUR, 1, cv2.LINE_AA)

    # Legend
    cv2.putText(panel, "Left wrist depth",  (pad_x, H - 12),
                _FONT, 0.4, _L_WRIST_COLOUR, 1, cv2.LINE_AA)
    cv2.putText(panel, "Right wrist depth", (pad_x + 140, H - 12),
                _FONT, 0.4, _R_WRIST_COLOUR, 1, cv2.LINE_AA)

    # Title
    cv2.putText(panel, "Wrist Z (depth) over time",
                (pad_x, 18), _FONT, 0.5, (200, 200, 220), 1, cv2.LINE_AA)

    # Axis border
    cv2.rectangle(panel, (pad_x, pad_y), (pad_x + gw, pad_y + gh),
                  (80, 80, 100), 1)

    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Main render function
# ──────────────────────────────────────────────────────────────────────────────

def render_skeleton_video(
    video_path:    str | Path,
    keypoints_2d:  np.ndarray,        # (T, 9, 2)
    scores:        np.ndarray,        # (T, 9)
    frame_indices: np.ndarray,        # (T,) — original frame numbers
    output_path:   str | Path,
    fps:           float = 30.0,
    score_thr:     float = 0.3,
    label_joints:  bool  = False,
    codec:         str   = "mp4v",
) -> Path:
    """
    Write an annotated debug video with skeleton overlay and live depth graph.

    Output layout:
      Left  panel (original video width)  — skeleton overlay
      Right panel (50% of video width)    — wrist depth graph with progress line

    Parameters
    ----------
    video_path    : path to the original source video
    keypoints_2d  : (T, 9, 2) float32 — pixel coordinates from pose_detector
    scores        : (T, 9)    float32 — per-joint confidence
    frame_indices : (T,)      int32   — which video frames these correspond to
    output_path   : where to write the annotated video
    fps           : frame rate of the output video
    score_thr     : joints below this threshold drawn as missing
    label_joints  : draw joint names (default False)
    codec         : fourcc codec string
    """
    video_path  = Path(video_path)
    output_path = Path(output_path).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(frame_indices)

    # Build lookup: video frame_idx → sequence position t
    frame_map = {int(fidx): i for i, fidx in enumerate(frame_indices)}

    # Pre-extract wrist Z values for the full graph
    # Use raw depth from keypoints_2d Z channel if available,
    # otherwise use the score as a proxy signal
    # We use scores[:,_L_WRIST] and scores[:,_R_WRIST] as confidence-weighted Z
    # The actual Z comes from the 3D sequence but we only have 2D here —
    # use confidence as a visual signal (higher conf = closer detection)
    z_left  = np.zeros(T, dtype=np.float32)
    z_right = np.zeros(T, dtype=np.float32)
    for t in range(T):
        z_left[t]  = scores[t, _L_WRIST]  if scores[t, _L_WRIST]  >= score_thr else 0.0
        z_right[t] = scores[t, _R_WRIST] if scores[t, _R_WRIST] >= score_thr else 0.0

    # Get frame size after rotation correction
    first_frame = None
    for _, frame in load_video_frames(video_path, max_frames=1):
        first_frame = frame
        break

    if first_frame is None:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    H, W    = first_frame.shape[:2]
    GW      = W // 2         # graph panel width = 50% of video width
    out_W   = W + GW         # total output width
    fourcc  = cv2.VideoWriter_fourcc(*codec)
    writer  = cv2.VideoWriter(str(output_path), fourcc, fps, (out_W, H))

    written   = 0
    current_t = 0   # current position in sequence

    for raw_idx, frame in load_video_frames(video_path):

        if raw_idx in frame_map:
            current_t = frame_map[raw_idx]

            # Draw skeleton on video frame
            frame = _draw_skeleton(
                frame,
                keypoints_2d[current_t],
                scores[current_t],
                score_thr=score_thr,
            )

            # Frame info overlay
            cv2.putText(
                frame,
                f"Frame {raw_idx:04d}",
                (10, 30), _FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Confidence bar at bottom
            bar_y = H - 20
            bar_w = W // 9
            for j in range(9):
                x0    = j * bar_w
                conf  = float(scores[current_t, j])
                color = _JOINT_COLOURS.get(j, (200, 200, 200))
                cv2.rectangle(frame, (x0, bar_y - 12), (x0 + bar_w - 2, bar_y + 4),
                              (50, 50, 50), -1)
                fill_w = int((bar_w - 4) * conf)
                cv2.rectangle(frame, (x0 + 2, bar_y - 10),
                              (x0 + 2 + fill_w, bar_y + 2), color, -1)
                name = ACTIVE_JOINT_NAMES.get(j, str(j))[:4]
                cv2.putText(frame, name, (x0 + 2, bar_y - 14),
                            _FONT, 0.28, (200, 200, 200), 1, cv2.LINE_AA)

        # Build graph panel at current_t
        graph = _build_graph_panel(H, GW, z_left, z_right, current_t, T)

        # Stitch side by side
        combined = np.hstack([frame, graph])
        writer.write(combined)
        written += 1

    writer.release()

    logger.info(
        "Debug video written: %s  (%d frames, %.1f s)",
        output_path, written, written / fps,
    )
    return output_path