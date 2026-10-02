# Experiment and Replay Scripts

[中文](README.zh.md) | [English](README.md) | [한국어](README.ko.md)

Install dependencies using the [main README](../README.md). Except for the webcam recording example, run commands from the workspace root containing `Orca_safety` and `orca_teleop`.

## Scripts

- `verify_geometry.py`: static poses, collision distances, and gradient verification.
- `compare_simulation.py`: OFF/ON data experiments with identical target trajectories.
- `generate_retarget_replay.py`: pass synthetic landmarks through the original retargeter to create NPZ replay input.
- `generate_crossing_motion.py`: generate JSONL targets for index/middle-finger crossing.
- `replay_video.py`: read live or synthetic JSONL input and export OFF, ON, and side-by-side MP4s.
- `replay_viewer.py`: interactively inspect OFF or ON in MuJoCo.

## Interactive MuJoCo Replay

On macOS, launch the viewer with `mjpython`:

```bash
orca_teleop/.venv/bin/mjpython Orca_safety/scripts/replay_viewer.py \
  Orca_safety/output/teleop/retarget-session-02.jsonl \
  --mode OFF --view front --speed 0.5 --paused
```

Change OFF to ON to compare. Space plays/pauses, N/B moves forward/backward one step, and R returns to the beginning. Left-drag rotates, right-drag pans, and the scroll wheel zooms. The script computes the trajectory before opening the window; ON preparation may take some time.

For the [synthetic crossing motion](../output/synthetic/index-middle-crossing/README.md), replace the input with `Orca_safety/output/synthetic/index-middle-crossing/motion.jsonl`. No webcam timing file is needed.

## Video Export

```bash
orca_teleop/.venv/bin/python Orca_safety/scripts/replay_video.py \
  Orca_safety/output/teleop/retarget-session-02.jsonl \
  --video-timing Orca_safety/output/video/camera/session-02.timing.json \
  --clip 7:14 --view front --speed 0.5 \
  --output Orca_safety/output/video/my-key-actions
```

Outputs: `OFF.mp4`, `ON.mp4`, and `OFF-ON.mp4`. The clip is an example; select the actual range from your webcam recording. Remove `--clip` to export the complete sequence.

Parameters shared by interactive and video replay:

- `--clip START:END`: repeatable. With `--video-timing`, use webcam video seconds; otherwise use elapsed wall time from the first logged frame. Synthetic input uses trajectory time. Simulation history before selected clips is still computed.
- `--speed 0.5`: half-speed playback in simulation time, without changing the control timestep. ON recomputes safe targets rather than playing recorded `q_safe` values.
- `--view front`: palm-facing view. Use `--azimuth` and `--elevation` for other angles. A larger `--camera-distance` moves farther away; the current v1 default is approximately 0.45 m.

## Optional Webcam Recording

When launching teleoperation from `orca_teleop`, add these arguments to your existing `teleop_sim.py --local` command:

```bash
--collision-safety-log ../Orca_safety/output/teleop/retarget-session-04.jsonl \
--record-video ../Orca_safety/output/video/camera/session-04.mp4
```

Without `--record-video`, no video is recorded. Recording is independent of `--show-video`. Stop normally with Ctrl+C. A `.timing.json` file is also generated for clip alignment via `--video-timing`. Video and post-dispatch logs are approximately aligned by wall time, including detection and queue delays.

## Data Experiments and Input Generation

```bash
orca_teleop/.venv/bin/python Orca_safety/scripts/verify_geometry.py \
  --output Orca_safety/output/my-passive --render
orca_teleop/.venv/bin/python Orca_safety/scripts/compare_simulation.py \
  --steps 120 --poses Orca_safety/output/my-passive/poses.json \
  --output Orca_safety/output/my-comparison
orca_teleop/.venv/bin/python Orca_safety/scripts/generate_retarget_replay.py \
  --steps 120 --output Orca_safety/output/replay/my-replay.npz
orca_teleop/.venv/bin/python Orca_safety/scripts/compare_simulation.py \
  --replay Orca_safety/output/replay/my-replay.npz \
  --output Orca_safety/output/my-retarget-comparison
orca_teleop/.venv/bin/python Orca_safety/scripts/generate_crossing_motion.py \
  --output Orca_safety/output/synthetic/my-crossing
```

120 control steps correspond to 1.2 seconds of simulation time. Use new output directories for comparisons, video export, and crossing-motion generation; static verification overwrites files with the same names. macOS rendering requires graphics access. Offline experiments need neither a webcam nor physical hardware.

See [output/README.md](../output/README.md) for output files, [tests/README.md](../tests/README.md) for code tests, and [implementation.md](../docs/implementation.md) (Chinese) for validation scope.
