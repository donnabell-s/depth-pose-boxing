"""
data_validation.py — Validation of 3D pose estimates against AthletePose3D ground truth.

Metric: MPJPE (Mean Per Joint Position Error)
---------------------------------------------
The standard metric for 3D pose estimation quality.

    MPJPE = mean over all joints and frames of ||predicted - ground_truth||₂

Lower is better. Reported in the same units as the normalised output
(dimensionless after bone-length normalisation). To get millimetres,
run validation before normalisation using metric depth.

AthletePose3D format assumptions
---------------------------------
Ground truth .npy files are expected to have shape (T, J, 3) where J >= 9.
If J > 9 (e.g. full 17-joint COCO), the file is sliced to ACTIVE_JOINTS
before comparison. If J == 9, it is assumed to already match ACTIVE_JOINTS.

Usage
-----
    python -m src.data_validation \\
        --pred  data/processed/boxer_01.npy \\
        --gt    data/athlete_pose_3d/boxer_01.npy \\
        --report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.utils import ACTIVE_JOINTS, ACTIVE_JOINT_NAMES, NUM_ACTIVE_JOINTS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core metric
# ──────────────────────────────────────────────────────────────────────────────

def mpjpe(
    predicted:    np.ndarray,   # (T, 9, 3)
    ground_truth: np.ndarray,   # (T, 9, 3)
    valid_mask:   np.ndarray | None = None,  # (T, 9) bool — exclude missing
) -> float:
    """
    Mean Per Joint Position Error between predicted and ground truth.

    Parameters
    ----------
    predicted    : (T, 9, 3) float32 — normalised output from normalize.py
    ground_truth : (T, 9, 3) float32 — AthletePose3D ground truth
    valid_mask   : (T, 9) bool — if provided, only valid joints are included
                   in the mean. Use this to exclude occluded joints that were
                   imputed rather than directly estimated.

    Returns
    -------
    mpjpe_score : float — mean Euclidean distance across all valid joints
    """
    assert predicted.shape == ground_truth.shape, (
        f"Shape mismatch: predicted={predicted.shape} gt={ground_truth.shape}"
    )

    # Per-joint Euclidean distance: (T, 9)
    error = np.linalg.norm(predicted - ground_truth, axis=-1)

    if valid_mask is not None:
        assert valid_mask.shape == error.shape, (
            f"valid_mask shape {valid_mask.shape} must match error shape {error.shape}"
        )
        if valid_mask.sum() == 0:
            logger.warning("valid_mask has no True entries — returning 0.0")
            return 0.0
        return float(error[valid_mask].mean())

    return float(error.mean())


def mpjpe_per_joint(
    predicted:    np.ndarray,   # (T, 9, 3)
    ground_truth: np.ndarray,   # (T, 9, 3)
    valid_mask:   np.ndarray | None = None,  # (T, 9) bool
) -> np.ndarray:
    """
    Per-joint MPJPE — useful for identifying which joints are least accurate.

    Returns
    -------
    errors : (9,) float32 — one MPJPE value per active joint
    """
    error = np.linalg.norm(predicted - ground_truth, axis=-1)  # (T, 9)

    per_joint = np.zeros(NUM_ACTIVE_JOINTS, dtype=np.float32)
    for j in range(NUM_ACTIVE_JOINTS):
        if valid_mask is not None:
            mask_j = valid_mask[:, j]
            per_joint[j] = error[mask_j, j].mean() if mask_j.any() else 0.0
        else:
            per_joint[j] = error[:, j].mean()

    return per_joint


def pck(
    predicted:    np.ndarray,   # (T, 9, 3)
    ground_truth: np.ndarray,   # (T, 9, 3)
    threshold:    float = 0.15,
    valid_mask:   np.ndarray | None = None,
) -> float:
    """
    Percentage of Correct Keypoints (PCK).

    A joint is considered correct if its error is within `threshold`
    (in normalised units — after bone-length normalisation, threshold=0.15
    corresponds to roughly 15% of shoulder width).

    Returns
    -------
    pck_score : float in [0, 1] — higher is better
    """
    error = np.linalg.norm(predicted - ground_truth, axis=-1)  # (T, 9)
    correct = error < threshold

    if valid_mask is not None:
        if valid_mask.sum() == 0:
            return 0.0
        return float(correct[valid_mask].mean())

    return float(correct.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ground_truth(
    path:        str | Path,
    num_frames:  int | None = None,
) -> np.ndarray:
    """
    Load an AthletePose3D .npy ground truth file.

    Handles two cases:
      - Shape (T, 17, 3): slices ACTIVE_JOINTS → (T, 9, 3)
      - Shape (T, 9, 3):  used as-is (assumes ACTIVE_JOINTS ordering)

    Parameters
    ----------
    path       : path to the .npy ground truth file
    num_frames : if provided, truncates or raises if lengths don't match

    Returns
    -------
    gt : (T, 9, 3) float32
    """
    gt = np.load(path).astype(np.float32)

    if gt.ndim != 3 or gt.shape[2] != 3:
        raise ValueError(
            f"Expected ground truth shape (T, J, 3), got {gt.shape}"
        )

    _, J, _ = gt.shape

    if J == 17:
        logger.info("Ground truth has 17 joints — slicing to ACTIVE_JOINTS.")
        gt = gt[:, ACTIVE_JOINTS, :]   # (T, 9, 3)
    elif J == NUM_ACTIVE_JOINTS:
        pass  # already 9 joints
    else:
        raise ValueError(
            f"Ground truth has {J} joints — expected 17 (COCO) or "
            f"{NUM_ACTIVE_JOINTS} (active joints)."
        )

    if num_frames is not None and gt.shape[0] != num_frames:
        raise ValueError(
            f"Ground truth has {gt.shape[0]} frames but prediction has "
            f"{num_frames} frames. Ensure they correspond to the same clip."
        )

    return gt   # (T, 9, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def print_report(
    predicted:    np.ndarray,
    ground_truth: np.ndarray,
    valid_mask:   np.ndarray | None = None,
    pck_threshold: float = 0.15,
) -> dict:
    """
    Print a human-readable validation report and return results as a dict.

    Parameters
    ----------
    predicted    : (T, 9, 3) normalised prediction
    ground_truth : (T, 9, 3) ground truth
    valid_mask   : (T, 9) bool — joints to include (None = all)
    pck_threshold: PCK threshold in normalised units

    Returns
    -------
    results : dict with keys "mpjpe", "pck", "per_joint_mpjpe"
    """
    overall_mpjpe = mpjpe(predicted, ground_truth, valid_mask)
    overall_pck   = pck(predicted, ground_truth, pck_threshold, valid_mask)
    per_joint     = mpjpe_per_joint(predicted, ground_truth, valid_mask)

    print("\n" + "=" * 50)
    print("  Validation Report — depth-pose-boxing")
    print("=" * 50)
    print(f"  Frames evaluated : {predicted.shape[0]}")
    print(f"  Active joints    : {NUM_ACTIVE_JOINTS}")
    print(f"  MPJPE            : {overall_mpjpe:.4f}  (lower is better)")
    print(f"  PCK @ {pck_threshold:.2f}        : {overall_pck * 100:.1f}%  (higher is better)")
    print("-" * 50)
    print("  Per-joint MPJPE:")
    for j, err in enumerate(per_joint):
        name  = ACTIVE_JOINT_NAMES[j]
        bar   = "█" * int(err * 40)
        print(f"    [{j}] {name:<16} {err:.4f}  {bar}")
    print("=" * 50 + "\n")

    return {
        "mpjpe":           overall_mpjpe,
        "pck":             overall_pck,
        "per_joint_mpjpe": per_joint.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate 3D pose predictions against AthletePose3D ground truth."
    )
    ap.add_argument("--pred",      required=True, help="Path to predicted .npy (T, 9, 3)")
    ap.add_argument("--gt",        required=True, help="Path to ground truth .npy")
    ap.add_argument("--pck-thr",   type=float, default=0.15, help="PCK threshold (default 0.15)")
    ap.add_argument("--report",    action="store_true", help="Print full per-joint report")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args   = _parse_args()

    pred   = np.load(args.pred).astype(np.float32)
    gt     = load_ground_truth(args.gt, num_frames=pred.shape[0])

    if args.report:
        print_report(pred, gt, pck_threshold=args.pck_thr)
    else:
        score = mpjpe(pred, gt)
        print(f"MPJPE: {score:.4f}")


if __name__ == "__main__":
    main()
