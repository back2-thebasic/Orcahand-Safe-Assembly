# ORCA Hand Retargeting Improvements and Simulation Comparison

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

This document records incremental improvements to fingertip reference points, optimization weights, joint limits, and pinch targets, starting from the official reference configuration. The current focus is improving coordination between the thumb and other fingers and observing how a collision safety layer changes the resulting motion.

## Latest Comparison Videos

Both videos use the same MediaPipe landmark recording, `mediapipe-02.jsonl`. Each variant produces 807 valid simulation frames, replayed in sync.

Left to right in the recorded view: Baseline → Retarget 03 → Retarget 04.

### Safety Layer OFF

The additional safety layer is disabled for all three variants. MuJoCo's original collision handling and physical joint limits remain enabled.

[View or download the comparison video](videos/retarget-comparison/comparison_Safety_OFF.mov)

https://github.com/user-attachments/assets/20ff53f4-c8f9-4d7f-b451-67a239d0fad6

### Safety Layer ON

All three variants use the same v2 CBF-QP safety layer, which filters retargeting outputs before simulation execution.

[View or download the comparison video](videos/retarget-comparison/comparison_Safety_ON.mov)

https://github.com/user-attachments/assets/95f271ae-0a19-46e6-8f49-8c624a4a2b04

## Configuration Changes

The official reference configuration is named `baseline.yaml`. Subsequent configurations in this directory are named `retarget_01.yaml` through `retarget_04.yaml`. Numbers indicate the order of changes, not that each successive variant performs better on every motion.

### Baseline: Official Reference Configuration

File: [baseline.yaml](configs/baseline.yaml)

Retains the original additional fingertip offsets; `w_pos: 1.0`, `w_dir: 10.0`, and `w_full_hand: 1.0`.

This runs the official reference configuration in the current codebase.

### Retarget 01: Remove Additional Fingertip Offsets

File: [retarget_01.yaml](configs/retarget_01.yaml)

Starting from Baseline, sets `fingertip_offsets_m` to `[0, 0, 0]` for all five fingers, directly using the URDF fingertip frame origins. Other configuration parameters remain unchanged.

This changes the reference points used for retargeting, not motor or joint zero positions or collision geometry. Changing the fingertip reference points may also affect automatic scale calibration.

### Retarget 02: Adjust Position and Full-Hand Shape Weights

File: [retarget_02.yaml](configs/retarget_02.yaml)

Starting from Retarget 01, changes `w_pos` from 1.0 to 2.0 and `w_full_hand` from 1.0 to 0.5.

### Retarget 03: Add a Relative Fingertip Position Objective

File: [retarget_03.yaml](configs/retarget_03.yaml)

Adds the following to Retarget 02:

```yaml
retarget:
  w_relative_tip: 1.0
```

The accompanying Python code adds a relative fingertip position loss and its analytical gradient. It tracks target vectors from the thumb tip to the other four fingertips and reuses the existing pinch activation weights. This emphasizes coordination between fingertip positions but does not directly require the finger pads to face each other.

This comparison also enables the **simulation joint-limit coordinate correction**:

The original retargeting joint ranges and MuJoCo control angles use inconsistent reference coordinates. This can exclude poses that the simulation could execute or produce target angles that the simulation subsequently clips. The correction takes the intersection of MuJoCo joint ranges and actuator control ranges as the optimizer's effective bounds, so optimization and execution use consistent angular ranges.

### Retarget 04: Add Pinch Target Distance Remapping

File: [retarget_04.yaml](configs/retarget_04.yaml)

Retains Retarget 03's relative objective and adds:

```yaml
retarget:
  relative_tip_remap:
    enabled: true
    close_cm: 1.0
    start_cm: 5.0
    min_target_cm: 0.5
```

The accompanying function preserves the target vector's direction while adjusting its length. Distances are expressed in calibrated, aligned robot-scale centimeters: distances of at least 5 cm remain unchanged; 1–5 cm maps linearly to 0.5–5 cm; distances of at most 1 cm generally use 0.5 cm. Extremely small distances taper continuously to zero to handle undefined direction at zero.

This may encourage closure near a pinch, but it may also compress the representation of small gaps or increase collision-related blocking. This comparison also enables the simulation joint-limit correction.

## Code Changes for Retargeting and Teleoperation

New YAML parameters require corresponding code support. Replacing configuration files alone does not enable all improvements. Paths below are relative to the local `orca_teleop/` project.

### `src/orca_teleop/retargeting/adaptive_analytical.py`

- Reads `fingertip_offsets_m`, allowing zero fingertip offsets.
- Adds `set_sim_joint_limits()` so the optimizer uses effective joint ranges consistent with MuJoCo.
- Adds the relative fingertip position loss and gradient, controlled by `w_relative_tip`, for Retarget 03.
- Adds `remap_relative_vectors()` to adjust target distances according to `relative_tip_remap`, for Retarget 04.

Retarget 02 only adjusts the existing `w_pos` and `w_full_hand` parameters; it does not require a new loss term. The relative fingertip term and distance remapping are disabled by default.

### Real-Time Teleoperation Integration

- `src/orca_teleop/sim_diagnostics.py`: `simulation_joint_limits()` extracts the intersection of MuJoCo joint ranges and actuator control ranges.
- `scripts/teleop_sim.py`, `src/orca_teleop/sim.py`, and `src/orca_teleop/pipeline.py`: `--sim-joint-limits` passes these bounds to the retargeter and enables the correction before human-hand input is processed.
- With `--collision-safety`, the simulation invokes the independent `orca_safety_v2` layer to filter retargeting outputs before MuJoCo executes them.

Selecting a YAML configuration, enabling the joint-limit correction, and enabling the safety layer are three independent settings. Changing the YAML does not automatically enable the latter two.

## Earlier Experiments

Earlier A/B/C configurations and S01–S03 videos retain their original settings. They must not be retroactively described as using the current joint-limit correction or new objectives.

See the [earlier experiment record](README.history.md) for configuration details, launch instructions, and S01–S03 videos.

## Installation and Usage

Five complete source files are provided for replacement at their original paths; no manual code editing is needed. See the [installation guide](INSTALL.md). Recording modules and the safety layer implementation are not included. The configs directory contains only baseline.yaml and retarget_01.yaml through retarget_04.yaml; historical document paths have been updated.
