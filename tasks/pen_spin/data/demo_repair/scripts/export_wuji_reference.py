"""Export what the Wuji Hand 2 reference looks like in Wuji's own simulator.

For every frame of one official clip this records, in the Wuji palm frame
(``right_palm_link``), the pen pose and which hand links touch the pen when the
hand is posed at ``reference_qpos``. It also records the knuckle positions and
the gravity direction of Wuji's fixed wrist mount.

Needs the mjlab environment (wuji-mjlab + mjlab), not the plain-mujoco one:

  python export_wuji_reference.py --clip 01_single_axis/clip_003
"""

import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PEN_SPIN = Path(__file__).resolve().parents[3]
CLIPS = PEN_SPIN / "data/official/rawdata/clips"
OUT = PEN_SPIN / "data/demo_repair/results/wuji"

import wuji_mjlab.tasks.pen_spin.tooling.clip_store as clip_store

clip_store._DEFAULT_CLIPS_ROOT = CLIPS

from mjlab.envs import ManagerBasedRlEnv

from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.env_cfgs import (
  wuji_hand2_pen_spin_env_cfg,
)
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.hand2_constants import (
  RIGHT_FINGER_JOINT_NAMES,
  RIGHT_FIXED_WRIST_POSE_W,
)

LINKS = ("right_palm_link",) + tuple(
  f"right_finger{f}_{part}"
  for f in range(1, 6)
  for part in ("link1", "link2", "link3", "link4", "tip_sensor")
)
CONTACT_DIST_M = 0.0005


def to_rot(q_wxyz):
  return Rotation.from_quat(np.roll(q_wxyz, -1, axis=-1))


def to_wxyz(rot):
  return np.roll(rot.as_quat(), 1, axis=-1)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  a = ap.parse_args()

  recipe = Path(tempfile.mkdtemp()) / "recipe.json"
  recipe.write_text(json.dumps({"schema": "wuji_pen_spin_recipe:1", "clips": [a.clip]}))
  cfg = wuji_hand2_pen_spin_env_cfg(play=True)
  cfg.commands["motion"].recipe_path = str(recipe)
  cfg.commands["motion"].debug_vis = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  m = env.sim.mj_model
  env.close()
  d = mujoco.MjData(m)

  raw = np.load(CLIPS / f"{a.clip}.npz")
  wrist, obj = raw["wrist_pose_w"].astype(float), raw["object_pose_w"].astype(float)
  qref = raw["reference_qpos"].astype(float)
  frames = len(qref)

  wrist_rot = to_rot(wrist[:, 3:])
  pen_pos_palm = wrist_rot.inv().apply(obj[:, :3] - wrist[:, :3])
  pen_rot_palm = wrist_rot.inv() * to_rot(obj[:, 3:])
  pen_pose_palm = np.concatenate([pen_pos_palm, to_wxyz(pen_rot_palm)], axis=1)
  link_pos_palm = np.stack(
    [wrist_rot[k].inv().apply(raw["link_pos_w"][k] - wrist[k, :3]) for k in range(frames)]
  )

  fixed = np.asarray(RIGHT_FIXED_WRIST_POSE_W, dtype=float)
  fixed_rot = to_rot(fixed[3:])
  palm_id = m.body("robot/right_palm_link").id
  pen_id = m.body("object/cube").id
  pen_q = m.jnt_qposadr[m.joint("object/cube_freejoint").id]
  qadr = [m.jnt_qposadr[m.joint(f"robot/{n}").id] for n in RIGHT_FINGER_JOINT_NAMES]
  link_ids = [m.body(f"robot/{n}").id for n in LINKS]

  mujoco.mj_forward(m, d)
  assert np.allclose(d.xpos[palm_id], fixed[:3], atol=1e-6)
  knuckles_palm = np.stack(
    [
      fixed_rot.inv().apply(d.xpos[m.body(f"robot/right_finger{f}_link1").id] - fixed[:3])
      for f in range(1, 6)
    ]
  )
  zero_pose_link_pos_palm = fixed_rot.inv().apply(d.xpos[link_ids] - fixed[:3])

  contacts = np.zeros((frames, len(LINKS)), dtype=bool)
  min_dist = np.full((frames, len(LINKS)), np.inf)
  # Direction in which each touching link pushes the pen, in the palm frame.
  push_palm = np.zeros((frames, len(LINKS), 3))
  for k in range(frames):
    d.qpos[qadr] = qref[k]
    d.qpos[pen_q : pen_q + 3] = fixed[:3] + fixed_rot.apply(pen_pos_palm[k])
    d.qpos[pen_q + 3 : pen_q + 7] = to_wxyz(fixed_rot * pen_rot_palm[k])
    mujoco.mj_forward(m, d)
    for c in d.contact[: d.ncon]:
      b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
      if pen_id not in (b1, b2):
        continue
      other = b2 if b1 == pen_id else b1
      if other in link_ids:
        i = link_ids.index(other)
        min_dist[k, i] = min(min_dist[k, i], c.dist)
        if c.dist < CONTACT_DIST_M:
          contacts[k, i] = True
          normal = c.frame[:3] * (1.0 if b2 == pen_id else -1.0)
          push_palm[k, i] += fixed_rot.inv().apply(normal)
  norm = np.linalg.norm(push_palm, axis=-1, keepdims=True)
  push_palm = np.divide(push_palm, norm, out=np.zeros_like(push_palm), where=norm > 0)

  OUT.mkdir(parents=True, exist_ok=True)
  out = OUT / f"{a.clip.replace('/', '__')}.npz"
  np.savez_compressed(
    out,
    clip=a.clip,
    frame_dt=0.02,
    links=np.array(LINKS),
    pen_pose_palm=pen_pose_palm,
    link_pos_palm=link_pos_palm,
    contacts=contacts,
    contact_min_dist=min_dist,
    contact_push_palm=push_palm,
    knuckles_palm=knuckles_palm,
    zero_pose_link_pos_palm=zero_pose_link_pos_palm,
    gravity_palm=fixed_rot.inv().apply([0.0, 0.0, -1.0]),
    fixed_wrist_pose_w=fixed,
  )
  touched = [LINKS[i] for i in np.flatnonzero(contacts[0])]
  print(f"saved {out}")
  print(f"frames={frames} first-frame contacts ({len(touched)}): {touched}")
  print(f"frames with >=3 fingers touching: {np.mean(_fingers(contacts) >= 3):.2f}")


def _fingers(contacts):
  per_finger = contacts[:, 1:].reshape(len(contacts), 5, 5).any(-1)
  return per_finger.sum(-1)


if __name__ == "__main__":
  main()
