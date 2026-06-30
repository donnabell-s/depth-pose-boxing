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

IMU wrist sensor (Phase 2B)
   │
ESP/Feather ──USB──▶ receive_serial.py ──▶ raw _imu.csv
                                                │
                                        clean_imu_csv.py
                                                │
                                    _imu_cleaned.csv (sync_offset_ms)
                                                │
                              align with _kinematics.npz via sync clap
                              → force regression labels
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
├── firmware/
│   ├── imu_wired/
│   │   └── imu_wired.ino     # USB serial IMU firmware (production)
│   ├── imu_wifi/
│   │   └── imu_wifi.ino      # UDP WiFi IMU firmware (backup, latency issues)
│   └── README.md             # hardware setup notes
├── scripts/
│   ├── imu/
│   │   ├── receive_serial.py   # wired IMU data collection (recommended)
│   │   ├── receive_udp.py      # wireless IMU data collection
│   │   ├── clean_imu_csv.py    # single-file cleanup + sync clap detection
│   │   ├── clean_imu_batch.py  # batch cleanup across a subject directory
│   │   └── plot_imu.py         # IMU signal visualisation
│   ├── calibrate.py            # camera intrinsics calibration
│   └── stitch_depthmaps.py     # stitch depth map PNGs into video
├── depth_anything_v2/        # metric depth model source (cloned from DA V2 repo)
├── data/
│   ├── raw/
│   │   ├── no_hardware/      # Phase 1: video-only recordings
│   │   └── with_hardware/    # Phase 2B: paired video + IMU recordings
│   └── processed/            # output files (see Output files section)
├── models/                   # Depth Anything V2 metric checkpoints (.pth)
├── notebooks/
│   └── visualize.ipynb       # 3D skeleton + kinematic feature inspection
├── main.py                   # pipeline entry point
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
python scripts/calibrate.py --video data/checkerboard/checkerboard.mp4
python scripts/calibrate.py --video data/checkerboard/checkerboard.mp4 --cols 10 --rows 7 --square-size 50 --skip-frames 10
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
python main.py --video data/raw/no_hardware/test/test_video.MOV --camera iphone13 --depth-model vit-b-metric
```

```bash
python main.py --subject-dir data/raw/no_hardware/subject03 --camera iphone13 --depth-model vit-b-metric
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
python scripts/stitch_depthmaps.py data/processed/boxer_01_depthmap -o data/processed/boxer_01_depthmap.mp4 --source-fps 30
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

## Phase 2B — IMU data collection

Phase 2B pairs each video recording with a wrist-mounted IMU (ESP/Feather with ICM-42688-P).
The IMU stream is aligned to the video via sync claps bookending the recording.

### Data collection workflow

1. Flash `firmware/imu_wired/imu_wired.ino` to the ESP/Feather.

2. Edit session metadata at the top of `scripts/imu/receive_serial.py`:
   ```python
   SUBJECT_ID = "subject01"
   PUNCH_TYPE = "jab"
   DISTANCE_M = 1
   HAND = "right"
   ```

3. Start IMU recording:
   ```bash
   python scripts/imu/receive_serial.py
   ```
   Saves to `data/raw/with_hardware/<subject_id>/<session>_imu.csv`. Press **Q** to stop.

4. Record the boxing video simultaneously. **Clap once at start and once at end** with the
   gloved hand clearly visible to the camera — these bookend claps are the sync signal.

5. Clean the IMU CSV and detect sync claps:
   ```bash
   python scripts/imu/clean_imu_csv.py data/raw/with_hardware/subject01/test_imu.csv
   ```
   Outputs `test_imu_cleaned.csv` with a `sync_offset_ms` column (first clap = 0) and
   prints a quality verdict (target: EXCELLENT ≥ 180 Hz, mean interval ≤ 6 ms).

6. Visualise the signal to verify clap detection and data quality:
   ```bash
   python scripts/imu/plot_imu.py data/raw/with_hardware/subject01/test_imu_cleaned.csv
   ```

7. Run the Phase 1 video pipeline on the paired video to get `_kinematics.npz`, then align
   the two streams using `sync_offset_ms` for force regression label extraction.

**Batch cleanup** across a full subject directory:
```bash
python scripts/imu/clean_imu_batch.py data/raw/with_hardware/subject01
```

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
