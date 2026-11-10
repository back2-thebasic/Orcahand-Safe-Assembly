"""Open-loop survival of a Wuji clip on Wuji Hand 2, the bar for the OrcaHand.

Uses the compiled pen-spin model of wuji-mjlab (play configuration) in plain
MuJoCo and the rollouts of rollout.py. Needs the mjlab environment.

  python wuji_baseline.py --clip 01_single_axis/clip_003
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rollout import shaft_swing_deg, summarize, survival_profile

PEN_SPIN = Path(__file__).resolve().parents[3]
CLIPS = PEN_SPIN / "data/official/rawdata/clips"
RESULTS = PEN_SPIN / "data/demo_repair/results"

import wuji_mjlab.tasks.pen_spin.tooling.clip_store as clip_store

clip_store._DEFAULT_CLIPS_ROOT = CLIPS

from mjlab.envs import ManagerBasedRlEnv

from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.env_cfgs import (
  wuji_hand2_pen_spin_env_cfg,
)
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.hand2_constants import (
  RIGHT_FINGER_JOINT_NAMES,
)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  ap.add_argument("--stride", type=int, default=5)
  ap.add_argument("--horizon-s", type=float, default=0.5)
  a = ap.parse_args()
  tag = a.clip.replace("/", "__")

  recipe = Path(tempfile.mkdtemp()) / "recipe.json"
  recipe.write_text(json.dumps({"schema": "wuji_pen_spin_recipe:1", "clips": [a.clip]}))
  cfg = wuji_hand2_pen_spin_env_cfg(play=True)
  cfg.commands["motion"].recipe_path = str(recipe)
  cfg.commands["motion"].debug_vis = False
  env = ManagerBasedRlEnv(cfg, device="cpu")
  m = env.sim.mj_model
  env.close()

  w = np.load(RESULTS / "wuji" / f"{tag}.npz")
  fixed = w["fixed_wrist_pose_w"]
  fixed_rot = Rotation.from_quat(np.roll(fixed[3:], -1))
  pen_palm = w["pen_pose_palm"]
  pen_rot = fixed_rot * Rotation.from_quat(np.roll(pen_palm[:, 3:], -1, axis=-1))
  pen_ref = np.concatenate(
    [fixed[:3] + fixed_rot.apply(pen_palm[:, :3]), np.roll(pen_rot.as_quat(), 1, axis=-1)], axis=1
  )
  qref = np.load(CLIPS / f"{a.clip}.npz")["reference_qpos"].astype(float)

  joints = [m.joint(f"robot/{n}").id for n in RIGHT_FINGER_JOINT_NAMES]
  act_joint = list(m.actuator_trnid[:, 0])
  assert m.nu == 20 and act_joint == joints, "actuators must follow RIGHT_FINGER_JOINT_NAMES"
  qadr, vadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
  pen_j = m.joint("object/cube_freejoint").id
  pen_body = m.body("object/cube").id

  end = len(pen_palm)
  while end > 1 and np.allclose(pen_palm[end - 2], pen_palm[-1]):
    end -= 1
  starts = np.arange(0, end, a.stride)
  dt = float(w["frame_dt"])
  args = (m, qadr, vadr, m.jnt_qposadr[pen_j], m.jnt_dofadr[pen_j], pen_body, qref, pen_ref)
  times, pos_err, shaft_err = survival_profile(*args, starts, a.horizon_s, dt)
  from_start = survival_profile(*args, [0], 3.0, dt)[0][0]
  swing = [shaft_swing_deg(pen_ref, s, int(round(a.horizon_s / dt))) for s in starts]
  summary = summarize(times, pos_err, shaft_err, a.horizon_s)
  summary["reference_mean_shaft_swing_deg"] = round(float(np.mean(swing)), 1)
  report = {
    "clip": a.clip,
    "hand": "Wuji Hand 2",
    "open_loop_from_frame0_s": round(float(from_start), 2),
    "start_frames": starts.tolist(),
    "survival_s": times.round(3).tolist(),
    "summary": summary,
  }
  (RESULTS / "wuji" / f"{tag}_baseline.json").write_text(json.dumps(report, indent=1))
  print(json.dumps({k: v for k, v in report.items() if k not in ("start_frames", "survival_s")}, indent=2))


if __name__ == "__main__":
  main()
