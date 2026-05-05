# depth-pose-boxing

Phase 1 of a boxing AI pipeline. Takes a raw boxing video and outputs a clean, normalised 3D skeleton sequence as a `.npy` file for punch classification in Phase 2.

---

## What it does

```
video.mp4  →  2D keypoints  →  depth map  →  3D coordinates  →  normalised sequence  →  .npy
              (RTMPose)         (Depth          (backproject)      (normalize)
                                Anything V2)
```

Output shape: `(T, 9, 3)` — T frames, 9 upper-body joints, XYZ coordinates.

---

## Project structure

```
depth-pose-boxing/
├── src/
│   ├── constants.py          # joint definitions, skeleton edges
│   ├── filters.py            # One Euro Filter
│   ├── video.py              # video I/O helpers
│   ├── utils.py              # re-exports
│   ├── pose_detector.py      # RTMPose 2D extraction
│   ├── depth_estimation.py   # Depth Anything V2
│   ├── backproject.py        # pinhole back-projection
│   ├── normalize/
│   │   ├── __init__.py       # pipeline entry point
│   │   ├── impute.py         # missing joint interpolation
│   │   ├── centre.py         # mid-shoulder centring
│   │   ├── scale.py          # bone-length normalisation
│   │   └── smooth.py         # One Euro Filter smoothing
│   └── data_validation.py    # MPJPE + PCK metrics
├── data/
│   ├── raw/                  # input boxing videos
│   ├── processed/            # output .npy files
│   └── athlete_pose_3d/      # ground truth for validation
├── models/                   # RTMPose + Depth Anything checkpoints
├── notebooks/
│   └── visualize.ipynb       # 3D skeleton inspection
├── main.py                   # pipeline entry point
├── calibrate.py              # camera intrinsics calibration
├── requirements.txt          # GPU dependencies
├── requirements.local.txt    # CPU-only (gitignored)
└── setup_colab.sh            # Colab environment setup
```

---

## Active joints

9 upper-body joints — thigh-up framing, knees and ankles excluded.

| Local index | Joint          | Role                        |
|-------------|----------------|-----------------------------|
| 0           | nose           | head position               |
| 1           | left_shoulder  | normalisation root          |
| 2           | right_shoulder | normalisation root          |
| 3           | left_elbow     | arm extension               |
| 4           | right_elbow    | arm extension               |
| **5**       | **left_wrist** | **punch endpoint ← key**    |
| **6**       | **right_wrist**| **punch endpoint ← key**    |
| 7           | left_hip       | trunk rotation              |
| 8           | right_hip      | trunk rotation              |

---

## Setup

### Teammate (RTX 4060, CUDA 13.0)

```bash
pip install -r requirements.txt
pip install openmim
mim install mmengine "mmcv>=2.0.0" mmdet mmpose
```

### Local development (CPU only)

```bash
pip install -r requirements.local.txt
```

### Colab

```bash
!bash setup_colab.sh
```

---

## Camera calibration

Before running the pipeline you need camera intrinsics (`fx, fy, cx, cy`).

**Option A — Film a checkerboard and calibrate:**
```bash
python calibrate.py --video data/raw/checkerboard.mp4
```
Copy the printed values into `CAMERA_PROFILES` in `main.py`.

**Option B — Look up values for your device:**

Search `"<device name> camera intrinsics fx fy cx cy"`. Make sure the values match your recording resolution and which camera (front/rear).

**Add a profile in `main.py`:**
```python
CAMERA_PROFILES = {
    "default":  CameraIntrinsics(fx=0.0,    fy=0.0,    cx=0.0,  cy=0.0),
    "macbook":  CameraIntrinsics(fx=849.3,  fy=851.1,  cx=638.7, cy=359.2),
    "iphone14": CameraIntrinsics(fx=1450.0, fy=1450.0, cx=540.0, cy=960.0),
}
```

---

## Running the pipeline

```bash
python main.py --video data/raw/boxer_01.mp4 --camera macbook
```

Output saved to `data/processed/boxer_01.npy`.

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `default` | Camera profile from `CAMERA_PROFILES` |
| `--pose-model` | `body-l` | `body-s`, `body-m`, `body-l` |
| `--depth-model` | `vit-b` | `vit-s`, `vit-b`, `vit-l` |
| `--skip-frames` | `0` | Process every N+1 frames |
| `--device` | `cuda:0` | `cuda:0` or `cpu` |

**With validation against ground truth:**
```bash
python main.py \
  --video data/raw/boxer_01.mp4 \
  --camera macbook \
  --validate \
  --gt data/athlete_pose_3d/boxer_01.npy
```

---

## Inspecting output

```bash
cd notebooks
jupyter notebook visualize.ipynb
```

Set `NPY_PATH` in the first cell to your `.npy` file. The notebook shows:
- 3D skeleton for a single frame (interactive, rotatable)
- Wrist and shoulder trajectories over time
- Raw XYZ values for the first 5 frames

---

## Normalisation

Applied in this order to every sequence:

1. **Missing joint imputation** — linear interpolation over occluded frames
2. **Mid-shoulder centring** — origin at midpoint of shoulders each frame
3. **Scale normalisation** — divide by median shoulder-to-shoulder distance
4. **Y-axis flip** — `+Y = up` (corrects image coordinate convention)
5. **One Euro Filter** — adaptive smoothing, minimal lag during punches

---

## Validation

```bash
python -m src.data_validation \
  --pred data/processed/boxer_01.npy \
  --gt   data/athlete_pose_3d/boxer_01.npy \
  --report
```

Prints MPJPE, PCK @ 0.15, and a per-joint breakdown.