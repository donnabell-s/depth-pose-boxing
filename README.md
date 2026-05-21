# depth-pose-boxing

Phase 1 of a View-Invariant Force Estimation pipeline for boxing. Takes a raw monocular front-facing boxing video and outputs two data streams:

- **`_pose_norm.npy`** — normalised 3D skeleton geometry for ST-GCN punch classification
- **`_kinematics.npz`** — metric kinematic features (velocity, acceleration, elbow angle) for IMU force label alignment

---

## Pipeline overview

```
                         ┌─────────────────────────────────────────────┐
                         │           METRIC STREAM (Physics)           │
                         │  points_3d → SG Smooth → Vel/Acc/Angle     │
                         │                              ↓              │
video.mp4                │                    _kinematics.npz          │
   │                     └─────────────────────────────────────────────┘
   ▼
YOLOv8-Pose → Depth Anything V2 → backproject() → points_3d
                                                        │
                         ┌──────────────────────────────┘
                         │       NORMALISED STREAM (Geometry)
                         │  Impute → Hip-Centre → Scale →
                         │  Y-Flip → SG Smooth
                         │              ↓
                         │       _pose_norm.npy
                         └─────────────────────────────────────────────
```

---

## Project structure

```
depth-pose-boxing/
├── src/
│   ├── constants.py          # joint definitions, skeleton edges
│   ├── video.py              # video I/O — rotation + front camera flip
│   ├── utils.py              # re-exports
│   ├── pose_detector.py      # YOLOv8-Pose 2D extraction
│   ├── depth_estimation.py   # Depth Anything V2 (relative + metric)
│   ├── backproject.py        # pinhole back-projection
│   ├── features.py           # metric smoothing + kinematic feature extraction
│   ├── normalize/
│   │   ├── __init__.py       # normalisation pipeline entry point
│   │   ├── impute.py         # missing joint interpolation
│   │   ├── centre.py         # mid-hip (waist) centring
│   │   ├── scale.py          # shoulder-width scale normalisation
│   │   └── smooth.py         # Savitzky-Golay smoothing
│   └── visualize.py          # skeleton overlay video + depth map export
├── depth_anything_v2/        # metric depth model source (cloned from DA V2 repo)
├── data/
│   ├── raw/                  # input boxing videos
│   └── processed/            # output files (see Output files section)
├── models/                   # Depth Anything V2 metric checkpoints (.pth)
├── notebooks/
│   └── visualize.ipynb       # 3D skeleton + kinematic feature inspection
├── main.py                   # pipeline entry point
├── calibrate.py              # camera intrinsics calibration
├── requirements.txt          # GPU dependencies (WSL2 / Linux / native GPU)
└── requirements.local.txt    # CPU-only for local development (gitignored)
```

---

## Active joints

9 upper-body joints — thigh-up framing, knees and ankles excluded.

| Local index | Joint           | Role                     |
|-------------|-----------------|--------------------------|
| 0           | nose            | head position            |
| 1           | left_shoulder   | normalisation root       |
| 2           | right_shoulder  | normalisation root       |
| 3           | left_elbow      | arm extension            |
| 4           | right_elbow     | arm extension            |
| **5**       | **left_wrist**  | **punch endpoint ← key** |
| **6**       | **right_wrist** | **punch endpoint ← key** |
| 7           | left_hip        | normalisation root       |
| 8           | right_hip       | normalisation root       |

---

## Output files

Every pipeline run produces four files in `data/processed/`:

| File | Shape | Description |
|------|-------|-------------|
| `{stem}_2d.npy` | `(T, 9, 2)` | 2D pixel keypoints from YOLOv8-Pose |
| `{stem}_camera.npy` | `(T, 9, 3)` | Back-projected camera-space coordinates (metres if metric model) |
| `{stem}_pose_norm.npy` | `(T, 9, 3)` | Hip-relative, scale-normalised, smoothed skeleton — input for ST-GCN |
| `{stem}_kinematics.npz` | named arrays | Metric kinematic features — input for IMU force label alignment |

### `_kinematics.npz` arrays

| Key | Shape | Unit | Description |
|-----|-------|------|-------------|
| `velocity_3d` | `(T, 2, 3)` | m/s | Left/right wrist 3D velocity — axis 0=left, 1=right, axis 2=XYZ |
| `acceleration_3d` | `(T, 2, 3)` | m/s² | Left/right wrist 3D acceleration |
| `elbow_angle_deg` | `(T, 2)` | degrees | Left/right elbow angle via Law of Cosines (Shoulder→Elbow→Wrist) |

Loading the kinematics file:
```python
data = np.load('data/processed/boxer_01_kinematics.npz')
vel  = data['velocity_3d']      # (T, 2, 3)
acc  = data['acceleration_3d']  # (T, 2, 3)
ang  = data['elbow_angle_deg']  # (T, 2)

# Example: right wrist Z-velocity (punch snap toward camera)
right_wrist_z_vel = vel[:, 1, 2]
```

---

## Setup

### GPU environment (RTX 4060, CUDA 13.0)

```bash
# 1. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fix numpy/xtcocotools compatibility
pip install "numpy<2.0"
pip install cython
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi && python setup.py build_ext --inplace && pip install -e . && cd ..

# 4. Install YOLOv8
pip install ultralytics

# 5. (Optional) Metric depth setup
git clone https://github.com/DepthAnything/Depth-Anything-V2.git /tmp/depth-anything-v2
cp -r /tmp/depth-anything-v2/metric_depth/depth_anything_v2 .
mkdir -p models
# Download checkpoint (vitb recommended):
wget -O models/depth_anything_v2_metric_hypersim_vitb.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth"
```

### Local development (CPU only)

```bash
pip install -r requirements.local.txt
```

> **Windows note:** rotation correction for MOV files requires `ffprobe` (ships with [FFmpeg](https://ffmpeg.org/download.html)). Add it to your PATH, or install via `winget install ffmpeg`.

---

## Camera calibration

Before running the pipeline you need camera intrinsics (`fx, fy, cx, cy`).

**Option A — Film a checkerboard and calibrate:**
```bash
python calibrate.py --video data/raw/checkerboard.mp4
python calibrate.py --video data/raw/checkerboard.mp4 --cols 10 --rows 7 --square-size 50 --skip-frames 10
```
Copy the printed values into `CAMERA_PROFILES` in `main.py`.

**Option B — Look up values for your device:**

Search `"<device name> camera intrinsics fx fy cx cy"`. Make sure the values match your recording resolution and which camera (front/rear).

**Add a profile in `main.py`:**
```python
CAMERA_PROFILES = {
    "iphone13": CameraProfile(
        intrinsics=CameraIntrinsics(fx=1452.59, fy=1453.74, cx=996.58, cy=510.20),
    ),
    "oppo": CameraProfile(
        intrinsics=CameraIntrinsics(fx=826.75, fy=827.42, cx=648.15, cy=345.17),
    ),
}
```

---

## Running the pipeline

```bash
python main.py --video data/raw/boxer_01.MOV --camera iphone13
```

**Recommended for force estimation (metric depth):**
```bash
python main.py --video data/raw/boxer_01.MOV --camera iphone13 --depth-model vit-b-metric
```

**With debug video and depth maps:**
```bash
python main.py --video data/raw/boxer_01.MOV --camera iphone13 --debug-video --depthmap-every 10
```

**Stitch depth map PNGs into a video:**
```bash
python stitch_depthmaps.py data/processed/boxer_01_depthmap -o data/processed/boxer_01_depthmap.mp4 --source-fps 30
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `iphone13` | Camera profile from `CAMERA_PROFILES` in `main.py` |
| `--output-dir` | `data/processed` | Directory for all output files |
| `--pose-model` | `large` | YOLOv8-Pose size: `nano`, `small`, `medium`, `large`, `xlarge` |
| `--depth-model` | `vit-b` | `vit-s/b/l` (relative) or `vit-s/b/l-metric` (metric, requires `.pth`) |
| `--sg-window` | `7` | Savitzky-Golay window length — must be odd and > `--sg-poly` |
| `--sg-poly` | `3` | Savitzky-Golay polynomial order |
| `--skip-frames` | `0` | Process every N+1 frames |
| `--device` | `cuda:0` | `cuda:0` or `cpu` |
| `--front-camera` | `True` | Flip frames horizontally for front-facing camera recordings |
| `--debug-video` | off | Render annotated skeleton overlay video with normalised Z labels |
| `--depthmap-every` | off | Save raw depth map PNGs every N frames |

---

## Normalisation pipeline

Applied inside `normalize()` to produce `_pose_norm.npy`:

| Step | Module | Description |
|------|--------|-------------|
| 1 | `impute.py` | Linear interpolation over occluded/low-confidence frames |
| 2 | `centre.py` | Translate all joints so mid-hip = `(0, 0, 0)` each frame |
| 3 | `scale.py` | Divide by median shoulder-to-shoulder distance (scale-invariant) |
| 4 | `__init__.py` | Flip Y axis — `+Y = up` (corrects image coordinate convention) |
| 5 | `smooth.py` | Savitzky-Golay filter (window=7, polyorder=3) along time axis |

---

## Depth model options

| Model | Type | Output | Notes |
|-------|------|--------|-------|
| `vit-s` | Relative | `[0, 1]` normalised | Fastest |
| `vit-b` | Relative | `[0, 1]` normalised | Recommended default |
| `vit-l` | Relative | `[0, 1]` normalised | Highest quality |
| `vit-s-metric` | Metric | metres | Requires `.pth` checkpoint |
| `vit-b-metric` | Metric | metres | Recommended for force estimation |
| `vit-l-metric` | Metric | metres | Highest quality metric |


> **Note:** `_kinematics.npz` velocity and acceleration values are only physically meaningful in metres/s and metres/s² when using a metric depth model. With relative models the units are arbitrary but the signal shape remains valid for comparative analysis.

---

## Inspecting output

```bash
cd notebooks
jupyter notebook visualize.ipynb
```

Set `NPY_PATH` and `NPZ_PATH` in the respective load cells. The notebook shows:
- 3D skeleton for a single frame (interactive, rotatable)
- Hip-relative wrist and shoulder depth trajectories
- Camera-space depth trajectories
- Wrist Z-velocity and Z-acceleration (punch snap and impact)
- Elbow angle over time (jab ≈ 180°, hook < 120°)
- Raw XYZ joint position table for the first 5 frames
