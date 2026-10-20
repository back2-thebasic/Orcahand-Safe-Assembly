# Installation and Usage

Prerequisite: a working ORCA v1 right-hand MediaPipe / MuJoCo environment.

## 1. Replace the Files

Copy the **5 Python files** from [modified_code/orca_teleop/](modified_code/orca_teleop/) into your local `orca_teleop/`, preserving their relative paths and replacing the existing files. Back up your files first.

Copy the YAML files from [configs/](configs/) into your local `orca_adaptive_test/configs/`.

## 2. Use the Local Code

Run from your local `orca_teleop/` directory (skip if already installed in editable mode):

```bash
uv pip install --python .venv/bin/python --no-deps -e .
```

## 3. Start Teleoperation

Run from `orca_teleop/` (macOS uses `mjpython`):

```bash
.venv/bin/mjpython scripts/teleop_sim.py \
  --env right --version v1 --hand right \
  --local --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config ../orca_adaptive_test/configs/retarget_04.yaml \
  --sim-joint-limits
```

- **Retarget 03:** use `retarget_03.yaml` and keep `--sim-joint-limits`.
- **Official Baseline:** use `baseline.yaml` and remove `--sim-joint-limits`.
- **Retarget 01 / 02:** select the corresponding YAML; `--sim-joint-limits` controls the joint-limit correction separately.

The command above leaves the additional safety layer disabled. These five files do not include the safety layer implementation or recording modules; do not add `--record-video` or `--landmark-log`.
