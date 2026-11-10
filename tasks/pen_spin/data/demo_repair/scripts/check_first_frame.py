"""Free-dynamics check of a first frame: hand targets fixed, pen free.

Compares the repaired first frame with the two earlier versions:
  codex     original retarget scene and pen mapping (results/converted, from convert.py)
  aligned   Wuji-aligned mount and pen mapping, retargeted joints
  repaired  aligned + contact-aware first-frame repair

Wuji Hand 2 under the same test (its own physics) loses the pen beyond 5 cm
after 0.38 s on clip_003 and 0.6 s on clip_006.

  python check_first_frame.py --clip 01_single_axis/clip_003
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from contact_map import pen_contacts

PEN_SPIN = Path(__file__).resolve().parents[3]
RESULTS = PEN_SPIN / "data/demo_repair/results"
CONVERTED = RESULTS / "converted"
DT = 0.02


def hold(scene, qpos, pen_pose, gravity=True, seconds=3.0, friction_scale=1.0):
  m = mujoco.MjModel.from_xml_path(str(scene))
  if not gravity:
    m.opt.gravity[:] = 0
  m.geom_friction[:, 0] *= friction_scale
  d = mujoco.MjData(m)
  qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
  pen_q = m.jnt_qposadr[m.joint("pen_free").id]
  pen = m.body("pen").id
  d.qpos[qadr] = qpos
  d.qpos[pen_q : pen_q + 7] = pen_pose
  d.ctrl[:] = qpos
  mujoco.mj_forward(m, d)
  start = pen_contacts(m, d)
  steps = int(round(seconds / DT))
  checkpoints = {int(round(t / DT)) for t in (0.5, 1.0)} | {steps}
  t_fail, disp = None, {}
  for i in range(1, steps + 1):
    mujoco.mj_step(m, d, nstep=round(DT / m.opt.timestep))
    err = float(np.linalg.norm(d.xpos[pen] - pen_pose[:3]))
    if t_fail is None and err > 0.05:
      t_fail = round(i * DT, 2)
    if i in checkpoints:
      disp[f"{i * DT:.1f}s"] = round(100 * err, 2)
  return {
    "contacts_t0": start,
    "contacts_end": pen_contacts(m, d),
    "first_over_5cm_s": t_fail,
    "pen_displacement_cm": disp,
  }


def robustness(scene, qpos, pen_pose, trials=30, seed=0):
  """Hold time under small pen pose errors and different friction."""
  rng = np.random.default_rng(seed)
  out = {}
  for friction in (0.5, 0.7, 1.0):
    times = []
    for _ in range(trials):
      pose = pen_pose.copy()
      pose[:3] += rng.uniform(-0.001, 0.001, 3)
      tilt = Rotation.from_rotvec(np.radians(rng.uniform(-2.0, 2.0, 3)))
      pose[3:] = np.roll((tilt * Rotation.from_quat(np.roll(pose[3:], -1))).as_quat(), 1)
      t = hold(scene, qpos, pose, seconds=3.0, friction_scale=friction)["first_over_5cm_s"]
      times.append(3.0 if t is None else t)
    times = np.array(times)
    out[f"friction_x{friction}"] = {
      "median_s": float(np.median(times)),
      "p10_s": float(np.percentile(times, 10)),
      "share_over_1s": float(np.mean(times > 1.0)),
    }
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  ap.add_argument("--robust", action="store_true")
  a = ap.parse_args()
  tag = a.clip.replace("/", "__")
  aligned = RESULTS / "aligned" / tag
  codex = np.load(CONVERTED / a.clip / "reference.npz")
  mapped = np.load(aligned / "mapped.npz")
  repaired = np.load(RESULTS / "first_frame" / tag / "first_frame.npz")
  cases = {
    "codex": (CONVERTED / a.clip / "scene.xml", codex["reference_qpos"][0], codex["object_pose_w"][0]),
    "aligned": (aligned / "scene.xml", codex["reference_qpos"][0], mapped["object_pose_w"][0]),
    "repaired": (aligned / "scene.xml", repaired["qpos"], repaired["object_pose_w"]),
  }
  report = {}
  for name, (scene, q, pen) in cases.items():
    report[name] = {
      "gravity": hold(scene, q, pen),
      "zero_gravity": hold(scene, q, pen, gravity=False),
    }
    g = report[name]["gravity"]
    print(
      f"{name:9s} contacts@t0={len(g['contacts_t0'])} first>5cm={g['first_over_5cm_s']} "
      f"disp={g['pen_displacement_cm']} zero-g disp={report[name]['zero_gravity']['pen_displacement_cm']}"
    )
    if a.robust:
      report[name]["robustness"] = robustness(scene, q, pen)
      print(f"{'':9s} robustness (first >5 cm, capped at 3 s): {report[name]['robustness']}")
  (RESULTS / "first_frame" / tag / "check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
