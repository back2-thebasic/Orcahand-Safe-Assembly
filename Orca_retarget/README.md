# ORCA Hand Retargeting Test Comparison
[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

ORCA v1 right hand · MediaPipe · Adaptive Analytical · MuJoCo simulation

This repository shares YAML parameters and test videos. It includes configuration changes and comparison videos for three scenarios. The README uses H.264 previews for better compatibility; the original MOV files can be downloaded from the link below each video.

## Configuration and Change Notes

### A: Original Official Configuration

**File: [baseline.yaml](configs/baseline.yaml)**

- Purpose: Serves as the baseline and retains the additional fingertip offsets
- `w_pos: 1.0`, `w_dir: 10.0`, `w_full_hand: 1.0`
- `norm_delta: 0.04`, `lp_alpha: 1.0` (no low-pass smoothing of the output)

### B: Zero-Offset Configuration

**File: [baseline_zero_offsets.yaml](configs/baseline_zero_offsets.yaml)**

- Relative to A: Sets `fingertip_offsets_m` to `[0, 0, 0]` for all five fingers, directly using each URDF fingertip frame origin and eliminating fingertip offsets
- All other YAML parameters are identical to A
- Purpose of the change: Avoid adding an extra fingertip length to an already defined fingertip position
- Note: This change also affects automatic scale calibration, so it is not a pure geometry comparison at a fixed scale

### C: Configuration with Adjusted Parameters

**File: [last.yaml](configs/last.yaml)**

- Current changes relative to B: `w_pos: 1.0 → 2.0`, `w_full_hand: 1.0 → 0.5`
- Purpose of the change: Increase the position-constraint weight and reduce the full-hand shape-constraint weight to observe whether pinching improves while checking for pose degradation

> B and C require retargeting code that supports `fingertip_offsets_m`; unmodified upstream code may not read this field.

## Fingertip Offsets: Rationale and Effects

### Why Does the Default Configuration Use Offsets?

Orca's retargeting implementation uses empirical offsets to compensate for the difference between the URDF coordinate-frame origins and the actual finger-pad positions. This is a geometric position correction, not a motor zero-position or joint-angle offset.

The upstream code labels these values as manually tuned empirical values and plans to replace them with definitions based on collision geometry, so they may not suit every model version. By default, the thumb, index, middle, ring, and little fingers are extended along their local z-axes by **30.5, 43.3, 45.3, 45.3, and 38.3 mm**, respectively.

Sources: [Orca upstream offset definitions](https://github.com/orcahand/orca_teleop/blob/main/src/orca_teleop/retargeting/constants.py), [related geometry utility notes](https://github.com/orcahand/orca_teleop/blob/main/src/orca_teleop/retargeting/utils.py).

### Why Can the Fingers Pinch After Setting the Offsets to Zero?

The fingertip positions used by the optimizer are:

- **Default offsets:** The positions of the URDF fingertip frames plus the local offsets rotated by those frames.
- **Zero offsets:** The positions of the URDF fingertip frames directly.

The current v1 URDF already defines fingertip positions through fixed joints. For example, the index-finger fingertip frame is located at `[-9, 0, 35] mm` relative to the distal link. Because this fixed joint has no rotation, adding the default offset changes the computed point to `[-9, 0, 78.3] mm`. These coordinates are relative to the distal link; they do not represent the full finger length.

If an additional offset moves the computed point away from the actual contact position, the optimizer may bring these "virtual fingertips" close to the target while a gap remains between the real fingertips in the simulation. Setting the offsets to zero returns the computed points to the URDF fingertip frames and may therefore improve actual fingertip alignment and pinching.

**Observation in this test:** With the original configuration, the thumb and index finger could not complete a pinch. They could complete it after the offsets were set to zero.

This result supports the interpretation that the empirical offsets may not match the current model, but it does not prove that the default configuration is wrong for every model or that the offsets are the only cause. The URDF fingertip frames may not lie exactly on the actual contact surfaces, so smaller, geometrically validated corrections may still be needed.

**Zero offsets only change the points used for retargeting calculations. They do not change the model geometry, collision bodies, or joint limits, nor do they directly provide smoothing or grip-force control.** Visual closure does not mean that physical contact or stable grasping has been verified.

## How to Download and Run the YAML Files

### 1. Where Should the YAML Files Go?

In this repository, click **Code → Download ZIP**. After extracting it, place `configs/` inside `orca_adaptive_test/` in your local Orca project, preserving the structure below. You can also open an individual YAML page and click Raw to download that file.

```text
Orcahand/
├── orca_teleop/
│   ├── .venv/
│   └── scripts/teleop_sim.py
├── orca_sim/
├── orcahand_description/
│   └── v1/models/urdf/orcahand_right.urdf
└── orca_adaptive_test/
    └── configs/
        ├── baseline.yaml
        ├── baseline_zero_offsets.yaml
        └── last.yaml
```

Do not place them in `.venv` or Python's site-packages. If experiment files with the same names already exist, preserve the old versions first so that older videos do not lose their corresponding parameters.

### 2. Prerequisites

The zero-offset field used by B and C requires code support. Open `orca_teleop/src/orca_teleop/retargeting/adaptive_analytical.py` and check whether `_build_frame_indices()` already reads `fingertip_offsets_m`. The code used for this test includes that support.

If it does not, replace the small block at the end of the function that sets `_frame_offsets` with the following code. Keep it inside the function with the same indentation, and leave all other code unchanged:

```python
        self._frame_offsets = [np.zeros(3, dtype=np.float64) for _ in self._computed_frame_names]
        tip_offsets = self._retarget_cfg.get("fingertip_offsets_m", {})
        for finger, frame_name in zip(FINGERS, self._frame_map.tip, strict=True):
            frame_idx = self._computed_frame_names.index(frame_name)
            offset = np.asarray(
                tip_offsets.get(finger, FINGERTIP_OFFSETS_M[finger]), dtype=np.float64
            )
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise ValueError(f"fingertip_offsets_m.{finger} must contain three finite values")
            self._frame_offsets[frame_idx] = offset
```

Without this support, simply downloading the B or C YAML file does not guarantee that zero offsets will take effect. A does not set this field and therefore continues to use the old offsets. Ensure that the runtime imports the source code you modified.

### 3. Select A, B, or C, Then Launch

First, enter `orca_teleop`, replacing the path with your actual path:

```bash
cd /xxx/Orcahand/orca_teleop
```

Choose one of the following and set the configuration for this run in the same terminal:

```bash
# A: Original configuration
RETARGET_CONFIG=../orca_adaptive_test/configs/baseline.yaml
```

```bash
# B: Zero offsets for all five fingers
RETARGET_CONFIG=../orca_adaptive_test/configs/baseline_zero_offsets.yaml
```

```bash
# C: Current latest configuration
RETARGET_CONFIG=../orca_adaptive_test/configs/last.yaml
```

Then run the common launch command. Use `mjpython` from the virtual environment directly; there is no need to activate the environment first:

```bash
.venv/bin/mjpython scripts/teleop_sim.py \
  --env right \
  --version v1 \
  --hand right \
  --local \
  --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config "$RETARGET_CONFIG"
```

Do not add spaces after the backslash at the end of each line. The v2 YAML files are not compatible with this v1 command.

One issue encountered during testing: Do not run two simulations that use the same port at the same time. If the green skeleton moves normally but the simulation does not, check the `Publisher connected` log and run `lsof -nP -iTCP:50051 -sTCP:LISTEN` to see whether an old instance is still running.

## S01: Open Hand, Half-Close, and Fist

Motion: Open hand → half-close → make a fist → open, repeated three times

### A: Original Configuration

https://github.com/user-attachments/assets/c61373b0-8059-412d-8560-4a3ba6c9c63a

[View or download the original MOV](videos/S01/S01-baseline.mov)

### B: Zero Offsets

https://github.com/user-attachments/assets/29e975da-077b-48ba-9e53-38181359c819

[View or download the original MOV](videos/S01/S01-zero%20offsets.mov)

### C: Current Latest Configuration

https://github.com/user-attachments/assets/25fd48fd-ee29-44b5-9cb1-a7689365d1fd

[View or download the original MOV](videos/S01/S01-last.mov)

**Conclusion for this scenario:** All YAML configurations performed well. Thumb retargeting in B was less accurate.

## S02: Slow Thumb–Index Pinch and Release

Motion: Slowly approach → pinch → release, repeated three times

### A: Original Configuration

https://github.com/user-attachments/assets/4c55b450-9961-4b50-bdd5-de74a9e2b264

[View or download the original MOV](videos/S02/S02-baseline.mov)

### B: Zero Offsets

https://github.com/user-attachments/assets/8d29cfac-a2db-4344-89fa-7863eeb2a6a5

[View or download the original MOV](videos/S02/S02-zero%20offsets.mov)

### C: Current Latest Configuration

https://github.com/user-attachments/assets/c37a71f9-74a2-4a70-8b3a-697967042f29

[View or download the original MOV](videos/S02/S02-last.mov)

**Conclusion for this scenario:** The official default configuration could not complete the pinch; a gap remained between the thumb and index finger. B and C performed this task well, but the thumb's direction during the pinch looked less natural in B.

## S03: Three-Finger Grasp and Release

Motion: Thumb–index–middle-finger pinch → release, repeated three times.

### A: Original Configuration

https://github.com/user-attachments/assets/52aaa9e6-61f5-478b-94e8-735014e2cfac

[View or download the original MOV](videos/S03/S03-baseline.mov)

### B: Zero Offsets

https://github.com/user-attachments/assets/57ebade0-5c62-416b-878a-cc3ec69e8192

[View or download the original MOV](videos/S03/S03-zero%20offsets.mov)

### C: Current Latest Configuration

https://github.com/user-attachments/assets/505e5a70-ac18-4eff-b557-5c8502d0ffa5

[View or download the original MOV](videos/S03/S03-last.mov)

**Conclusion for this scenario:** Testing without an object only verifies the grasping pose; it does not demonstrate that the hand can hold an object. The official configuration still could not complete the pinch. B and C performed well, but when B released, the thumb retargeting was less accurate, and the retargeting of the thumb IP joint—the joint closest to the fingertip—looked less natural.

## How to Add Videos

1. Store the original videos in the corresponding `videos/S01/`, `S02/`, and `S03/` directories.
2. Regular repository video links do not reliably generate embedded players in a README. This README uses `user-attachments` URLs generated by uploading videos in GitHub's Markdown editor.
3. Use H.264 MP4 for preview videos to provide better browser compatibility.
4. Show both the human hand and the simulation in every recording, and use a consistent calibration pose. When updating a YAML file, update the description on this page as well. If existing videos still correspond to older parameters, add a new configuration identifier instead of overwriting the old parameters.

See the [official instructions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files) for working with GitHub video attachments.

## File Structure

```text
configs/                 All shared YAML files
videos/S01/              S01-baseline.mov, S01-zero offsets.mov, S01-last.mov
videos/S02/              S02-baseline.mov, S02-zero offsets.mov, S02-last.mov
videos/S03/              S03-baseline.mov, S03-zero offsets.mov, S03-last.mov
README.md                English version (repository default)
README.zh-CN.md          Chinese version
README.ko.md             Korean version
```
