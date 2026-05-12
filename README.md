# depth-pose-boxing

Phase 1 of a boxing AI pipeline. Takes a raw boxing video and outputs a clean, normalised 3D skeleton sequence as a `.npy` file for punch classification in Phase 2.

---

## What it does

```
video.mp4  →  2D keypoints  →  depth map  →  3D coordinates  →  normalised sequence  →  .npy
              (YOLOv8-Pose)    (Depth          (backproject)      (normalize)
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
│   ├── video.py              # video I/O — rotation + front camera flip
│   ├── utils.py              # re-exports
│   ├── pose_detector.py      # YOLOv8-Pose 2D extraction
│   ├── depth_estimation.py   # Depth Anything V2 (relative + metric)
│   ├── backproject.py        # pinhole back-projection
│   ├── velocity.py           # per-joint velocity + approximate m/s
│   ├── normalize/
│   │   ├── __init__.py       # pipeline entry point
│   │   ├── impute.py         # missing joint interpolation
│   │   ├── centre.py         # mid-shoulder centring
│   │   ├── scale.py          # bone-length normalisation
│   │   └── smooth.py         # One Euro Filter smoothing
│   ├── visualize.py          # skeleton overlay video + depth map export
├── depth_anything_v2/        # metric depth model source (cloned from DA V2 repo)
├── data/
│   ├── raw/                  # input boxing videos
│   ├── processed/            # output .npy files
├── models/                   # Depth Anything V2 metric checkpoints (.pth)
├── notebooks/
│   └── visualize.ipynb       # 3D skeleton + velocity inspection
├── main.py                   # pipeline entry point
├── calibrate.py              # camera intrinsics calibration
├── requirements.txt          # GPU dependencies (WSL2 / Linux / native GPU)
└── requirements.local.txt    # CPU-only for local development (gitignored)
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
# Download checkpoint (vitb recommended for balance of speed/accuracy):
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
 
Output saved to `data/processed/boxer_01.npy`.
 
**Front camera recordings (default):**
Front-facing phone cameras produce a mirrored image. The pipeline automatically flips frames horizontally to correct left/right joint labelling:
```bash
python main.py --video data/raw/boxer_01.MOV --camera iphone13 --front-camera   # default
python main.py --video data/raw/boxer_01.MOV --camera iphone13 --no-front-camera # rear camera
```
 
**Common options:**
 
| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `default` | Camera profile from `CAMERA_PROFILES` |
| `--pose-model` | `large` | `nano`, `small`, `medium`, `large`, `xlarge` |
| `--depth-model` | `vit-b` | `vit-s`, `vit-b`, `vit-l`, `vit-s-metric`, `vit-b-metric`, `vit-l-metric` |
| `--metric-scale` | `0.699` | Scale correction for metric depth (calibrated against OAK-D) |
| `--skip-frames` | `0` | Process every N+1 frames |
| `--device` | `cuda:0` | `cuda:0` or `cpu` |
| `--front-camera` | `True` | Flip frames horizontally for front camera recordings |
| `--debug-video` | `False` | Render annotated skeleton overlay video |
| `--depthmap-every` | `None` | Save depth map PNGs every N frames |
 
**With metric depth:**
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


---

## Inspecting output

```bash
cd notebooks
jupyter notebook visualize.ipynb
```

Set `NPY_PATH` and `VELOCITY_PATH` in the first cell. The notebook shows:
- 3D skeleton for a single frame (interactive, rotatable)
- Wrist and shoulder trajectories over time
- Raw XYZ values for the first 5 frames

---

## Normalisation

Applied in this order to every sequence:

1. **Missing joint imputation** — linear interpolation over occluded frames
2. **Mid-hip (waist) centring** — origin at midpoint of left and right hip each frame
3. **Scale normalisation** — divide by median shoulder-to-shoulder distance
4. **Y-axis flip** — `+Y = up` (corrects image coordinate convention)
5. **One Euro Filter** — adaptive smoothing, minimal lag during punches

---

## Depth model options
 
| Model | Type | Output | Notes |
|-------|------|--------|-------|
| `vit-s` | Relative | [0, 1] normalised | Fastest |
| `vit-b` | Relative | [0, 1] normalised | Recommended default |
| `vit-l` | Relative | [0, 1] normalised | Highest quality |
| `vit-s-metric` | Metric | metres | Requires `.pth` checkpoint |
| `vit-b-metric` | Metric | metres | Recommended for metric use |
| `vit-l-metric` | Metric | metres | Highest quality metric |
 
Metric depth is calibrated against OAK-D stereo ground truth (--metric-scale 0.699).