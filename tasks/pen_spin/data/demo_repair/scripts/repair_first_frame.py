"""Contact-aware repair of the first reference frame on the aligned OrcaHand.

Jointly solves the 16 finger joints and a small pen pose correction so that

- every OrcaHand body corresponding to a Wuji first-frame pen contact touches
  the pen, pressed in about as deep as Wuji's link is (the PD preload), and
  pushes the pen in the same world direction as the Wuji link does;
- every other hand body stays clear of the pen;
- the pen keeps its mapped pose and the hand stays close to the retarget.

  python repair_first_frame.py --clip 01_single_axis/clip_003
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from check_first_frame import hold
from contact_map import PenDistance, pen_contacts, wuji_to_orca_body
from portable_scene import copy_scene

PEN_SPIN = Path(__file__).resolve().parents[3]

RESULTS = PEN_SPIN / "data/demo_repair/results"
CONVERTED = RESULTS / "converted"

PEN_RADIUS_M = 0.00745
DEPTH_RANGE_M = (0.0003, 0.002)
CLEARANCE_M = 0.002
MIN_CLEARANCE_M = 0.001
SELECT_HOLD_S = 1.0
MAX_PEN_SHIFT_M = 0.03
MAX_PEN_TILT_RAD = np.radians(15.0)
W_CONTACT = 1000.0  # 1 per mm
W_PUSH = 1.0  # ~0.5 per 30 deg
W_PEN_POS = 20.0  # 0.2 per cm
W_PEN_ROT = 5.0  # ~0.9 per 10 deg
W_JOINT = 0.3


def to_rot(q_wxyz):
  return Rotation.from_quat(np.roll(q_wxyz, -1, axis=-1))


def to_wxyz(rot):
  return np.roll(rot.as_quat(), 1, axis=-1)


def orca_targets(mapped, frame, palm_rot_w):
  """Target depth and world push direction per OrcaHand body, from Wuji's links."""
  links = list(mapped["wuji_links"])
  touching = np.flatnonzero(mapped["wuji_contacts"][frame])
  push_w = palm_rot_w.apply(mapped["wuji_contact_push_palm"][frame])
  targets = {}
  for i in touching:
    body = wuji_to_orca_body(links[i])
    t = targets.setdefault(body, {"push": np.zeros(3), "depth": 0.0, "wuji": []})
    t["push"] += push_w[i]
    t["depth"] = max(t["depth"], -float(mapped["wuji_contact_min_dist"][frame, i]))
    t["wuji"].append(links[i])
  for t in targets.values():
    t["push"] /= np.linalg.norm(t["push"])
    t["depth"] = float(np.clip(t["depth"], *DEPTH_RANGE_M))
  return targets


class FirstFrameProblem:
  def __init__(self, m, pen_pose_w, q_init, targets):
    self.m, self.d = m, mujoco.MjData(m)
    self.dist = PenDistance(m)
    names = self.dist.body_names()
    self.target = np.array([n in targets for n in names])
    self.depth = np.array([targets[n]["depth"] if n in targets else 0.0 for n in names])
    self.push = np.array([targets[n]["push"] if n in targets else np.zeros(3) for n in names])
    self.qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    self.pen_q = m.jnt_qposadr[m.joint("pen_free").id]
    self.pen_pos0, self.pen_rot0 = pen_pose_w[:3], to_rot(pen_pose_w[3:])
    self.q_init = q_init
    lo, hi = m.actuator_ctrlrange.T
    self.lower = np.r_[lo, -MAX_PEN_SHIFT_M * np.ones(3), -MAX_PEN_TILT_RAD * np.ones(3)]
    self.upper = np.r_[hi, MAX_PEN_SHIFT_M * np.ones(3), MAX_PEN_TILT_RAD * np.ones(3)]

  def pen_pose(self, x):
    rot = Rotation.from_rotvec(x[19:22]) * self.pen_rot0
    return np.r_[self.pen_pos0 + x[16:19], to_wxyz(rot)]

  def pose(self, x):
    self.d.qpos[self.qadr] = x[:16]
    self.d.qpos[self.pen_q : self.pen_q + 7] = self.pen_pose(x)
    mujoco.mj_kinematics(self.m, self.d)
    return self.dist(self.d)

  def residuals(self, x, side_only=False):
    dist, push, point = self.pose(x)
    clearance = np.maximum(0.0, CLEARANCE_M - dist)[~self.target]
    if side_only:
      side = self.dist.side_error(
        self.d, point[self.target], self.push[self.target], PEN_RADIUS_M
      )
      contact = np.r_[side.ravel(), clearance]
      push_err = np.zeros(0)
    else:
      contact = np.r_[(dist + self.depth)[self.target], clearance]
      push_err = (push - self.push)[self.target].ravel()
    return np.r_[
      W_CONTACT * contact,
      W_PUSH * push_err,
      W_PEN_POS * x[16:19],
      W_PEN_ROT * x[19:22],
      W_JOINT * (x[:16] - self.q_init),
    ]

  def max_push_error_deg(self, x):
    _, push, _ = self.pose(x)
    cos = np.clip(np.sum(push * self.push, axis=1)[self.target], -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).max())

  def solve(self, x0):
    """Put each contact on the Wuji side of the shaft first, then refine."""
    x0 = np.clip(x0, self.lower + 1e-9, self.upper - 1e-9)
    kw = dict(bounds=(self.lower, self.upper), diff_step=1e-3, max_nfev=400)
    side = least_squares(self.residuals, x0, kwargs={"side_only": True}, **kw)
    return least_squares(self.residuals, side.x, **kw)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  ap.add_argument("--frame", type=int, default=0)
  ap.add_argument("--restarts", type=int, default=8)
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()
  tag = a.clip.replace("/", "__")
  aligned = RESULTS / "aligned" / tag
  mapped = np.load(aligned / "mapped.npz")
  m = mujoco.MjModel.from_xml_path(str(aligned / "scene.xml"))
  k = a.frame

  d0 = mujoco.MjData(m)
  mujoco.mj_forward(m, d0)
  palm_rot_w = Rotation.from_matrix(d0.xmat[m.body("right_palm").id].reshape(3, 3))
  targets = orca_targets(mapped, k, palm_rot_w)
  q_retarget = np.load(CONVERTED / a.clip / "reference.npz")["reference_qpos"][k]

  prob = FirstFrameProblem(m, mapped["object_pose_w"][k], q_retarget, targets)
  rng = np.random.default_rng(a.seed)
  # The OrcaHand cannot always press from Wuji's side with every finger, so
  # candidates are ranked by how well the pen actually stays put.
  candidates = []
  for r in range(a.restarts):
    x0 = np.r_[q_retarget, np.zeros(6)]
    if r:
      x0[:16] += rng.normal(0.0, 0.3, 16)
    sol = prob.solve(x0)
    dist, _, _ = prob.pose(sol.x)
    valid = bool(
      (dist[prob.target] < 0).all() and (dist[~prob.target] >= MIN_CLEARANCE_M).all()
    )
    sim = hold(aligned / "scene.xml", sol.x[:16], prob.pen_pose(sol.x), seconds=SELECT_HOLD_S)
    candidates.append(
      {
        "restart": r,
        "x": sol.x,
        "cost": float(sol.cost),
        "valid": valid,
        "max_push_error_deg": prob.max_push_error_deg(sol.x),
        "hold_disp_cm": sim["pen_displacement_cm"][f"{SELECT_HOLD_S:.1f}s"],
      }
    )
    c = candidates[-1]
    print(
      f"restart {r}: valid={valid} cost={c['cost']:.3f} "
      f"max_push_error={c['max_push_error_deg']:.0f}deg disp@{SELECT_HOLD_S}s={c['hold_disp_cm']}cm",
      flush=True,
    )
  pool = [c for c in candidates if c["valid"]] or candidates
  chosen = min(pool, key=lambda c: (c["hold_disp_cm"], c["cost"]))

  x = chosen["x"]
  dist, push, _ = prob.pose(x)
  names = prob.dist.body_names()
  d = prob.d
  d.qvel[:] = 0
  mujoco.mj_forward(m, d)
  out = RESULTS / "first_frame" / tag
  out.mkdir(parents=True, exist_ok=True)
  per_target = {}
  for i, n in enumerate(names):
    if n in targets:
      cos = float(np.clip(push[i] @ targets[n]["push"], -1, 1))
      per_target[n] = {
        "wuji_links": targets[n]["wuji"],
        "dist_mm": round(1000 * dist[i], 2),
        "target_dist_mm": round(-1000 * targets[n]["depth"], 2),
        "push_dir_w": push[i].round(3).tolist(),
        "push_error_deg": round(float(np.degrees(np.arccos(cos))), 1),
      }
  report = {
    "clip": a.clip,
    "frame": k,
    "chosen_restart": chosen["restart"],
    "cost": chosen["cost"],
    "hold_disp_cm_at_selection": chosen["hold_disp_cm"],
    "candidates": [
      {k: v for k, v in c.items() if k != "x"} for c in candidates
    ],
    "targets": per_target,
    "worst_non_target_clearance_mm": round(
      1000 * float(min(v for n, v in zip(names, dist) if n not in targets)), 2
    ),
    "mujoco_contacts": pen_contacts(m, d),
    "pen_shift_mm": (1000 * x[16:19]).round(2).tolist(),
    "pen_tilt_deg": round(float(np.degrees(np.linalg.norm(x[19:22]))), 2),
    "joint_change_from_retarget_deg": dict(
      zip(
        [m.actuator(i).name.removesuffix("_actuator") for i in range(m.nu)],
        np.degrees(x[:16] - q_retarget).round(1).tolist(),
      )
    ),
  }
  np.savez_compressed(
    out / "first_frame.npz",
    clip=a.clip,
    frame=k,
    qpos=x[:16],
    object_pose_w=prob.pen_pose(x),
    mapped_object_pose_w=mapped["object_pose_w"][k],
    q_retarget=q_retarget,
  )
  # One-frame clip so tasks/pen_spin/view.py --run <out> can show it.
  np.savez_compressed(
    out / "reference.npz",
    reference_qpos=x[None, :16],
    object_pose_w=prob.pen_pose(x)[None],
    frame_dt=float(mapped["frame_dt"]),
  )
  copy_scene(aligned / "scene.xml", out / "scene.xml")
  (out / "repair.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
