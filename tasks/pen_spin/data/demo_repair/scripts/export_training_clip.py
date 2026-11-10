"""Package repaired OrcaHand trajectories as a wuji-mjlab pen-spin clip group.

Writes ``<clips_root>/<group>/{index.json, clip_*.npz}`` in Wuji's clip format,
with joints and links renamed to the task's ``right_finger{N}_joint{K}`` /
``right_finger{N}_link{K}`` convention, and a recipe selecting the group.

  python export_training_clip.py --clips 01_single_axis/clip_003
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PEN_SPIN = Path(__file__).resolve().parents[3]
RESULTS = PEN_SPIN / "data/demo_repair/results"
WUJI_PEN_SPIN = PEN_SPIN.parent / "reorient/wuji-mjlab/src/wuji_mjlab/tasks/pen_spin"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS = {
  "thumb_mcp": "right_finger1_joint1",
  "thumb_abd": "right_finger1_joint2",
  "thumb_pip": "right_finger1_joint3",
  "thumb_dip": "right_finger1_joint4",
  **{
    f"{finger}_{joint}": f"right_finger{i}_joint{k}"
    for i, finger in enumerate(FINGERS[1:], start=2)
    for joint, k in (("mcp", 1), ("abd", 2), ("pip", 3))
  },
}
LINKS = {
  "right_palm": "right_palm_link",
  **{f"right_thumb_{link}": f"right_finger1_link{k}" for k, link in enumerate(("mp", "pp", "ip", "dp"), start=1)},
  **{
    f"right_{finger}_{link}": f"right_finger{i}_link{k}"
    for i, finger in enumerate(FINGERS[1:], start=2)
    for k, link in enumerate(("mp", "pp", "ip"), start=1)
  },
}
# Order of the task's link_names: palm, then each finger's links base to tip.
TASK_LINKS = ("right_palm_link",) + tuple(
  f"right_finger{i}_link{k}" for i in range(1, 6) for k in range(1, 5 if i == 1 else 4)
)


def sha256(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clips", nargs="+", default=["01_single_axis/clip_003"])
  ap.add_argument("--group", default="orcahand_v1_single_axis")
  ap.add_argument("--clips-root", type=Path, default=WUJI_PEN_SPIN / "data/rawdata/clips")
  ap.add_argument("--recipe-dir", type=Path, default=WUJI_PEN_SPIN / "data/recipe")
  a = ap.parse_args()

  out = a.clips_root / a.group
  out.mkdir(parents=True, exist_ok=True)
  entries, frame_dt, wrist = [], None, None
  for clip in a.clips:
    run = np.load(RESULTS / "trajectory" / clip.replace("/", "__") / "reference.npz", allow_pickle=True)
    joint_names = [JOINTS[str(n).removeprefix("right_")] for n in run["joint_names"]]
    link_index = {LINKS[n]: i for i, n in enumerate(run["link_names"]) if n in LINKS}
    dt = float(run["frame_dt"])
    if frame_dt is not None and abs(dt - frame_dt) > 1e-9:
      raise ValueError(f"{clip}: frame_dt {dt} differs from {frame_dt}")
    frame_dt = dt
    if wrist is not None and not np.allclose(run["wrist_pose_w"][0], wrist):
      raise ValueError(f"{clip}: wrist pose differs from the group's")
    wrist = run["wrist_pose_w"][0]

    clip_id = clip.split("/")[1]
    path = out / f"{clip_id}.npz"
    np.savez(
      path,
      reference_qpos=run["reference_qpos"].astype(np.float32),
      wrist_pose_w=run["wrist_pose_w"].astype(np.float32),
      object_pose_w=run["object_pose_w"].astype(np.float32),
      link_pos_w=run["link_pos_w"][:, [link_index[n] for n in TASK_LINKS]].astype(np.float32),
    )
    entries.append({"clip_id": clip_id, "file": path.name, "sha256": sha256(path)})
    print(f"{path}: {len(run['reference_qpos'])} frames, joints {joint_names}")

  (out / "index.json").write_text(
    json.dumps(
      {
        "schema": "wuji_pen_spin_clips:1",
        "group_id": a.group,
        "group_name": "OrcaHand v1, repaired from Wuji clips",
        "frame_dt": frame_dt,
        "clip_count": len(entries),
        "clips": entries,
      },
      indent=2,
    )
    + "\n"
  )
  a.recipe_dir.mkdir(parents=True, exist_ok=True)
  recipe = a.recipe_dir / f"{a.group}.json"
  recipe.write_text(json.dumps({"schema": "wuji_pen_spin_recipe:1", "groups": [a.group]}, indent=2) + "\n")
  print(f"recipe {recipe}")
  print("fixed wrist pose (palm, world):", np.array2string(wrist, separator=", ", precision=10))


if __name__ == "__main__":
  main()
