# Index–Middle Crossing: Synthetic Target Motion

[中文](README.zh.md) | [English](README.md) | [한국어](README.ko.md)

`motion.jsonl` contains synthetic nominal input for the v1 right hand. It is not a live retargeting recording and contains no executed-motion results. It works with `replay_viewer.py` and `replay_video.py`; use the same file for OFF and ON.

The sequence has 600 control steps of 0.01 seconds each, totaling 6 seconds of simulation time:

- 0–1 s: hold the model's open-hand reset pose.
- 1–2.5 s: gradually move the index and middle fingers toward the crossing target.
- 2.5–3.5 s: hold the crossing target.
- 3.5–5 s: gradually return to the open pose.
- 5–6 s: hold open, allowing the actual motion to settle.

Only the index and middle abduction joints change; other targets remain fixed. The crossing target is index abd=0.23 rad (about 13.18°) and middle abd=−0.40 rad (about −22.92°), with smooth cubic interpolation. These are absolute model angles using its coordinate conventions, not increments from zero.

The target is within joint limits, but configured index–middle collision geometry intersects. OFF retains MuJoCo's control limits and contact dynamics; contact forces may stop the fingers in a compressed pose rather than allowing the full crossing target. ON should stop further approach earlier; failing to complete the crossing is expected. Zoom and rotate to inspect the fingers. Actual behavior depends on the replay.

Run from the workspace root:

```bash
orca_teleop/.venv/bin/mjpython Orca_safety/scripts/replay_viewer.py \
  Orca_safety/output/synthetic/index-middle-crossing/motion.jsonl \
  --mode OFF --view front --speed 0.5 --paused
```

Change OFF to ON to inspect filtering. On macOS, use `mjpython` for the interactive window. Space plays/pauses, N/B steps forward/backward, and R returns to the start. Use the mouse to adjust the view. No `--video-timing` is needed because there is no webcam video; optional `--clip` ranges use synthetic trajectory time.

ON reruns the current Safety Layer on the synthetic targets; it is not a manually authored avoidance trajectory. Checks cover configured pairs and sampled states, not arbitrary continuous trajectories or physical hardware. This input is for simulation demonstrations only.
