# Phase 2B Recording Session Protocol

Quick-start checklist for collecting paired video + IMU data for boxing punch regression labels.

---

## Prerequisites (one-time setup)

- [ ] ESP/Feather flashed with `firmware/imu_wired/imu_wired.ino`
- [ ] All 4 IMU pins connected: 3V3, GND, IO0→SDA, IO2→SCL
- [ ] Phone camera mounted on tripod, frontal view
- [ ] Floor markings for 1m and 2m subject distances
- [ ] Python dependencies installed: `pip install -r requirements.txt`

---

## Per-subject setup (once per subject)

- [ ] Add subject to metadata:
```bash
  python scripts/imu/add_subject_metadata.py
```
  Enter: subject ID (e.g. `subject01`), body weight in kg, dominant hand

- [ ] Create a Label Studio project for this subject named `{subject_id}_with_hardware`
- [ ] Configure label classes in Label Studio: `clap`, `jab`, `cross`, `hook`, `uppercut`

---

## Per-session recording (repeat for each punch type × distance × hand)

Recording protocol matches Phase 2A: 4 punch types × 2 distances × 2 hands = 16 sessions per subject.

### Pre-recording

1. **Attach IMU to subject's hand** (velcro/strap, firmly, on the back of the hand)
2. **Plug ESP into laptop via USB**, wait 5 seconds
3. **Edit `scripts/imu/receive_serial.py`** session metadata at top:
```python
   SUBJECT_ID = "subject01"
   PUNCH_TYPE = "jab"          # jab | cross | hook | uppercut
   DISTANCE_M = 1              # 1 or 2
   HAND = "right"              # left or right
```
4. **Position subject** at the correct distance marking, facing camera
5. **Confirm IMU hand is visible** to the camera throughout the expected punch range

### Start IMU recording

6. Run from project root:
```bash
   python scripts/imu/receive_serial.py
```
7. **Pre-flight check:** wave the IMU for 5 seconds when prompted
   - ✓ "IMU is healthy" → continue
   - ❌ "FAILED: IMU appears frozen" → unplug, replug ESP, try again
8. **Press ENTER** to start recording

### Start video recording

9. **Start phone video recording**
10. **Subject stands at marker** facing camera, IMU hand raised

### Recording sequence

11. **CLAP 1** — subject claps both hands together hard in front of camera
12. **Wait 1-2 seconds** in guard position
13. **Throw 8-10 punches** of the specified type with the specified hand
    - Return to guard between each punch
    - Vary intensity if natural (some hard, some soft)
14. **Wait 1-2 seconds** in guard position
15. **CLAP 2** — subject claps both hands together hard

### Stop recording

16. **Stop phone video recording** (on phone)
17. **Press Q on laptop** to stop IMU recording
18. **Note the output file path** printed in terminal

### Transfer video file

19. Move/copy the phone video to:
data/raw/with_hardware/{subject_id}/{punch_type}_{distance}m_{hand}.mp4
    Example: `data/raw/with_hardware/subject01/jab_1m_right.mp4`

---

## Post-recording verification

### Quality check

20. **Clean the IMU CSV:**
```bash
    python scripts/imu/clean_imu_csv.py data/raw/with_hardware/subject01/jab_1m_right_imu.csv
```
    - ✓ Verdict: ACCEPTABLE or better → continue
    - ❌ Verdict: UNACCEPTABLE → re-record
    - ✓ "2 clap events detected" → sync ready
    - ❌ "1 clap detected" or "no clap" → re-record with louder/sharper claps

21. **Plot to verify visually:**
```bash
    python scripts/imu/plot_imu.py data/raw/with_hardware/subject01/jab_1m_right_imu_cleaned.csv
```
    Check the plot:
    - ✓ Baseline at ~1g (gravity)
    - ✓ Two large clap spikes (~25-30g) at the recording bookends
    - ✓ Medium punch spikes (~5-15g) between the claps
    - ✓ Punch count matches what was thrown

22. **Verify sync (one-time per recording protocol):**
    - Open video, find clap 1 frame and clap 2 frame
    - Compute video clap-to-clap duration: `(frame2 - frame1) / 30 * 1000` ms
    - Compare to IMU's "Duration between claps" reported by cleanup
    - Should match within ±50 ms

### If anything fails verification

- Re-record the session before moving on
- A bad recording included in the dataset will create wrong labels

---

## Run Phase 1 pose extraction

23. Process the video through main.py:
```bash
    python main.py --video data/raw/with_hardware/subject01/jab_1m_right.mp4 --camera iphone13 --depth-model vit-b-metric
```
    Outputs:
    - `data/processed/with_hardware/subject01/jab_1m_right_pose_norm.npy`
    - `data/processed/with_hardware/subject01/jab_1m_right_kinematics.npz`

---

## Annotation in Label Studio

24. Upload video to the subject's Label Studio project
25. **Annotate clap 1**: drag a small range (2-3 frames) at the exact moment hands meet
26. **Annotate each punch**: drag a range covering from punch start (wind-up) to full extension
27. **Annotate clap 2**: drag a small range at the end clap (optional but recommended for verification)
28. **Export as JSON-MIN** to:
data/annotations/with_hardware/subject01/labelstudio/subject01_with_hardware_annotations.json

---

## Generate training labels

29. After all sessions for a subject are recorded, annotated, and processed:
Open notebooks/regression/preprocessing.ipynb
Run all cells
    
30. Inspect outputs:
    - `data/clips/with_hardware/{class}/{clip_id}.npy` — pose clips
    - `data/metadata/with_hardware/labels.csv` — peak force, peak speed per clip
    - Sanity check plots in the notebook

### Expected label ranges (sanity check)

For a typical adult subject:
- **Peak force:** 50-300 N (jabs lighter, hooks/uppercuts heavier)
- **Peak speed:** 2-12 m/s
- **Peak acceleration:** 4-20 g

Anything wildly outside these ranges suggests a label calculation problem or hardware issue with that session.

---

## Troubleshooting

### IMU pre-flight fails
- Unplug ESP, wait 10 seconds, replug
- Check all 4 pins are firmly seated
- Verify ESP is flashed with `imu_wired.ino` (not the WiFi version)

### IMU "frozen" warning during recording
- Stop immediately (press Q)
- Unplug + replug ESP
- Restart from step 6

### Cleanup script reports UNACCEPTABLE
- Check if Python had high CPU load during recording
- Close other applications and re-record

### Wrong number of claps detected
- Re-record with sharper, louder claps
- Make sure no other large impacts (e.g., dropping device) happened during recording

### IMU and video duration don't match (>50ms off)
- Re-record the session
- Ensure both recordings start before clap 1 and end after clap 2

---

## File structure reference

After successful processing, each session produces:
data/
├── raw/with_hardware/subject01/
│   ├── jab_1m_right.mp4              ← original video
│   ├── jab_1m_right_imu.csv          ← raw IMU
│   └── jab_1m_right_imu_cleaned.csv  ← cleaned + sync_offset_ms
├── processed/with_hardware/subject01/
│   ├── jab_1m_right_pose_norm.npy    ← skeleton for model input
│   └── jab_1m_right_kinematics.npz   ← computed kinematics
├── annotations/with_hardware/subject01/labelstudio/
│   └── subject01_with_hardware_annotations.json
└── metadata/with_hardware/
├── subjects.csv                   ← body weights
└── labels.csv                     ← regression labels (after preprocessing)

---

## Full subject recording plan

For each subject, complete all 16 sessions:

| # | Punch Type | Distance | Hand |
|---|---|---|---|
| 1 | jab | 1m | right |
| 2 | jab | 1m | left |
| 3 | jab | 2m | right |
| 4 | jab | 2m | left |
| 5 | cross | 1m | right |
| 6 | cross | 1m | left |
| 7 | cross | 2m | right |
| 8 | cross | 2m | left |
| 9 | hook | 1m | right |
| 10 | hook | 1m | left |
| 11 | hook | 2m | right |
| 12 | hook | 2m | left |
| 13 | uppercut | 1m | right |
| 14 | uppercut | 1m | left |
| 15 | uppercut | 2m | right |
| 16 | uppercut | 2m | left |

Estimated time per session: 2-3 minutes (recording) + 3-5 minutes (verification + processing)
Estimated time per subject: 90-120 minutes total