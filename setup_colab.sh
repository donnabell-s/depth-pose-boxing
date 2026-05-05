#!/bin/bash
# setup_colab.sh — Install all dependencies for depth-pose-boxing on Google Colab.
#
# Run this in the first cell of your Colab notebook:
#   !bash setup_colab.sh
#
# Colab already provides: torch, torchvision, numpy, opencv, matplotlib
# This script installs: openmim, mmpose stack, transformers, scipy

set -e  # exit on any error

echo "========================================"
echo " depth-pose-boxing — Colab Setup"
echo "========================================"

# ── Step 1 — Base deps not pre-installed on Colab ────────────────────
echo "[1/4] Installing scipy, transformers, Pillow..."
pip install -q scipy transformers>=4.40 Pillow>=10.0

# ── Step 2 — OpenMIM (MMPose installer) ──────────────────────────────
echo "[2/4] Installing OpenMIM..."
pip install -q openmim

# ── Step 3 — MMPose stack via mim ────────────────────────────────────
# Install order matters — mmengine first, then mmcv, then mmdet, then mmpose.
# mim handles CUDA-matched mmcv builds automatically.
echo "[3/4] Installing mmengine, mmcv, mmdet, mmpose via mim..."
mim install -q mmengine
mim install -q "mmcv>=2.0.0"
mim install -q mmdet
mim install -q mmpose

# ── Step 4 — Verify ──────────────────────────────────────────────────
echo "[4/4] Verifying install..."
python - <<'EOF'
import mmpose
import transformers
import cv2
import scipy
print("mmpose     :", mmpose.__version__)
print("transformers:", transformers.__version__)
print("opencv     :", cv2.__version__)
print("scipy      :", scipy.__version__)
print("")
print("Setup complete. You're ready to run main.py.")
EOF

echo "========================================"
echo " Setup complete!"
echo "========================================"