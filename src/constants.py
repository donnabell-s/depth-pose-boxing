"""
constants.py — Joint definitions, active joint subset, and skeleton structure.

All joint indices in this project reference LOCAL indices (0–8) into
ACTIVE_JOINTS, not the original COCO-17 indices.

Local index → joint name:
  0  nose
  1  left_shoulder   ← normalisation root (mid-shoulder = mean of 1 and 2)
  2  right_shoulder  ← normalisation root
  3  left_elbow
  4  right_elbow
  5  left_wrist      ← primary punch endpoint
  6  right_wrist     ← primary punch endpoint
  7  left_hip
  8  right_hip

Framing assumption
------------------
Videos are framed thigh-up. Knees and ankles are excluded.
Hips (local 7, 8) may be partially visible — mid-shoulder is used
as the normalisation root for reliability.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# COCO-17 full joint map (RTMPose output order — do not reorder)
# ──────────────────────────────────────────────────────────────────────────────

COCO17_JOINTS: dict[int, str] = {
    0:  "nose",
    1:  "left_eye",
    2:  "right_eye",
    3:  "left_ear",
    4:  "right_ear",
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# ──────────────────────────────────────────────────────────────────────────────
# Active joint subset — 9 upper-body joints
# These are COCO-17 indices used to slice RTMPose output:
#   keypoints[:, ACTIVE_JOINTS, :]  → (T, 9, 2 or 3)
# ──────────────────────────────────────────────────────────────────────────────

ACTIVE_JOINTS: list[int] = [0, 5, 6, 7, 8, 9, 10, 11, 12]

NUM_ACTIVE_JOINTS: int = len(ACTIVE_JOINTS)  # 9

# Local index → joint name
ACTIVE_JOINT_NAMES: dict[int, str] = {
    local: COCO17_JOINTS[coco]
    for local, coco in enumerate(ACTIVE_JOINTS)
}

# Joint name → local index
ACTIVE_NAME_TO_IDX: dict[str, int] = {
    v: k for k, v in ACTIVE_JOINT_NAMES.items()
}

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation root (local indices)
# ──────────────────────────────────────────────────────────────────────────────

ROOT_LEFT_IDX:  int = ACTIVE_NAME_TO_IDX["left_shoulder"]   # local 1
ROOT_RIGHT_IDX: int = ACTIVE_NAME_TO_IDX["right_shoulder"]  # local 2

# ──────────────────────────────────────────────────────────────────────────────
# Skeleton edges — local indices (0–8)
# Used for visualisation and ST-GCN adjacency matrix in Phase 2.
# ──────────────────────────────────────────────────────────────────────────────

SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),   # nose          → left_shoulder
    (0, 2),   # nose          → right_shoulder
    (1, 2),   # left_shoulder → right_shoulder
    (1, 3),   # left_shoulder → left_elbow
    (3, 5),   # left_elbow    → left_wrist
    (2, 4),   # right_shoulder → right_elbow
    (4, 6),   # right_elbow    → right_wrist
    (1, 7),   # left_shoulder  → left_hip
    (2, 8),   # right_shoulder → right_hip
    (7, 8),   # left_hip       → right_hip
]

# ──────────────────────────────────────────────────────────────────────────────
# Boxing-critical joints (local indices)
# ──────────────────────────────────────────────────────────────────────────────

BOXING_JOINTS: list[int] = [
    1, 2,   # shoulders — rotation signal
    3, 4,   # elbows    — arm extension
    5, 6,   # wrists    ← most important for punch endpoint
]