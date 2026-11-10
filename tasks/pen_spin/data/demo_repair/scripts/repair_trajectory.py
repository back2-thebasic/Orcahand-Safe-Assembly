"""Contact-aware repair of a whole clip on the aligned OrcaHand.

Starts from the repaired first frame and solves every following frame warm
started from the previous one. Per frame, the OrcaHand bodies corresponding to
Wuji's pen contacts touch the pen, every other hand body stays out of it, the
pen correction stays small and smooth, and each joint moves at most
``MAX_JOINT_STEP_RAD`` per frame. Because a finger cannot pass through the pen
between frames, the side it presses from is carried over from the first frame;
the push direction is only a weak preference.

Wuji often cradles the pen in one curled finger (link3, link4 and tip pressing
from different sides), which all map to the single OrcaHand distal body. So
the other phalanges of a target finger may touch the pen too, and the contacts
must be able to produce the wrench the reference pen motion needs (gravity
plus its acceleration) within a friction cone of ``MU_PLAN``.

Frames after the clip's motion (Wuji's static tail) repeat the last solved frame.

  python repair_trajectory.py --clip 01_single_axis/clip_003
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.spatial.transform import Rotation

from contact_map import PenDistance
from portable_scene import copy_scene
from repair_first_frame import orca_targets, to_rot, to_wxyz

PEN_SPIN = Path(__file__).resolve().parents[3]

RESULTS = PEN_SPIN / "data/demo_repair/results"
CONVERTED = RESULTS / "converted"

CLEARANCE_M = 0.001
MAX_JOINT_STEP_RAD = 0.08
MAX_PEN_SHIFT_M = 0.03
MAX_PEN_TILT_RAD = np.radians(15.0)
SOFT_PEN_SHIFT_M = 0.04
SOFT_PEN_TILT_RAD = np.radians(20.0)
MAX_PEN_SHIFT_STEP_M = 0.002
MAX_PEN_TILT_STEP_RAD = np.radians(1.0)
LOOKAHEAD_FRAMES = 5
MAX_ALLOWED_DEPTH_M = 0.002
MU_PLAN = 0.5  # the scene uses 1.0
SUPPORT_RANGE_M = 0.001  # contact strength ramps from this gap to touching
SUPPORT_ARM_M = 0.05  # torque scale
SUPPORT_REG = 0.05  # cost of contact force, per pen weight
W_CONTACT = 1000.0  # 1 per mm
W_SWEPT = 500.0
W_APPROACH = 200.0  # bodies about to become targets, divided by frames ahead
W_SUPPORT = 3.0  # per pen weight of missing wrench
W_PEN_LIMIT = 2000.0
W_PUSH = 0.5
W_JOINT_SMOOTH = 1.0
W_PEN_POS = 20.0
W_PEN_ROT = 5.0
W_PEN_POS_SMOOTH = 200.0  # 0.2 per mm of change between frames
W_PEN_ROT_SMOOTH = 20.0
W_JOINT_RETARGET = 0.1


class TrackingProblem:
  def __init__(self, m):
    self.m, self.d = m, mujoco.MjData(m)
    self.dist = PenDistance(m)
    self.names = self.dist.body_names()
    self.qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    self.pen_q = m.jnt_qposadr[m.joint("pen_free").id]
    self.pen_body = m.body("pen").id
    self.weight = float(m.body_mass[self.pen_body] * np.linalg.norm(m.opt.gravity))
    self.q_lo, self.q_hi = m.actuator_ctrlrange.T
    self.wrench = np.zeros(6)

  def free(self, targets):
    """Bodies that may touch the pen: targets and the phalanges of their fingers."""
    fingers = {n.split("_")[1] for n in targets}
    return np.array(
      [n in targets or (n.split("_")[1] in fingers and not n.endswith("_mp")) for n in self.names]
    )

  def set_frame(self, targets, pen_pose_w, q_retarget, x_prev, upcoming=(), wrench=None):
    """``upcoming``: (mapped pen pose, target body names) of the next frames.
    ``wrench``: force and torque (about the centre of mass) the hand must
    apply to the pen, or None to skip the support term."""
    self.target = np.array([n in targets for n in self.names])
    self.allowed = self.free(targets) & ~self.target
    self.wrench = wrench
    self.upcoming = [
      (
        to_rot(pose[3:]),
        pose[:3],
        ~self.free(targets) & ~self.free(t),
        np.array([n in t for n in self.names]) & ~self.target,
      )
      for pose, t in upcoming
    ]
    self.depth = np.array([targets[n]["depth"] if n in targets else 0.0 for n in self.names])
    self.push = np.array(
      [targets[n]["push"] if n in targets else np.zeros(3) for n in self.names]
    )
    self.pen_pos0, self.pen_rot0 = pen_pose_w[:3], to_rot(pen_pose_w[3:])
    self.q_retarget, self.x_prev = q_retarget, x_prev
    step = np.r_[
      MAX_JOINT_STEP_RAD * np.ones(16),
      MAX_PEN_SHIFT_STEP_M * np.ones(3),
      MAX_PEN_TILT_STEP_RAD * np.ones(3),
    ]
    lo = np.r_[self.q_lo, -MAX_PEN_SHIFT_M * np.ones(3), -MAX_PEN_TILT_RAD * np.ones(3)]
    hi = np.r_[self.q_hi, MAX_PEN_SHIFT_M * np.ones(3), MAX_PEN_TILT_RAD * np.ones(3)]
    self.lower = np.maximum(lo, x_prev - step)
    self.upper = np.minimum(hi, x_prev + step)

  def pen_pose(self, x):
    rot = Rotation.from_rotvec(x[19:22]) * self.pen_rot0
    return np.r_[self.pen_pos0 + x[16:19], to_wxyz(rot)]

  def pose(self, x):
    self.d.qpos[self.qadr] = x[:16]
    self.d.qpos[self.pen_q : self.pen_q + 7] = self.pen_pose(x)
    mujoco.mj_kinematics(self.m, self.d)
    return self.dist(self.d)

  def lookahead(self, x):
    """With the pen already at its next poses (same correction): penetration
    of bodies that stay non-contact there, so fingers clear its path early,
    and the gap of bodies about to touch it, so they arrive on time."""
    swept, approach = [], []
    correction = Rotation.from_rotvec(x[19:22])
    self.d.qpos[self.qadr] = x[:16]
    for ahead, (rot, pos, clear, coming) in enumerate(self.upcoming, start=1):
      self.d.qpos[self.pen_q : self.pen_q + 3] = pos + x[16:19]
      self.d.qpos[self.pen_q + 3 : self.pen_q + 7] = to_wxyz(correction * rot)
      mujoco.mj_kinematics(self.m, self.d)
      dist = self.dist(self.d)[0]
      swept.append(np.maximum(0.0, -dist)[clear])
      approach.append(np.maximum(0.0, dist)[coming] / ahead)
    if not swept:
      return np.zeros(0), np.zeros(0)
    return np.concatenate(swept), np.concatenate(approach)

  def support_miss(self, dist, push, point):
    """Wrench the touching bodies cannot supply, in pen weights.

    Each contact is a friction pyramid of strength ramping from 0 at
    ``SUPPORT_RANGE_M`` gap to 1 when touching; weak contacts cost more force.
    """
    if self.wrench is None:
      return np.zeros(0)
    strength = np.clip(1.0 - dist / SUPPORT_RANGE_M, 0.0, 1.0)
    strength[~(self.target | self.allowed)] = 0.0
    centre = self.d.xipos[self.pen_body]
    cols, reg = [], []
    for i in np.flatnonzero(strength > 1e-3):
      n = push[i]
      t1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
      t1 /= np.linalg.norm(t1)
      t2 = np.cross(n, t1)
      for e in (n + MU_PLAN * t1, n - MU_PLAN * t1, n + MU_PLAN * t2, n - MU_PLAN * t2):
        cols.append(np.r_[e, np.cross(point[i] - centre, e) / SUPPORT_ARM_M])
        reg.append(SUPPORT_REG / strength[i])
    b = self.wrench / self.weight
    b = np.r_[b[:3], b[3:] / SUPPORT_ARM_M]
    if not cols:
      return b
    a = np.array(cols).T
    lam, _ = nnls(np.r_[a, np.diag(reg)], np.r_[b, np.zeros(len(reg))])
    return a @ lam - b

  def residuals(self, x):
    swept, approach = self.lookahead(x)
    dist, push, point = self.pose(x)
    other = ~self.target & ~self.allowed
    contact = np.r_[
      (dist + self.depth)[self.target],
      np.maximum(0.0, CLEARANCE_M - dist)[other],
      np.maximum(0.0, -dist - MAX_ALLOWED_DEPTH_M)[self.allowed],
    ]
    return np.r_[
      W_CONTACT * contact,
      W_SWEPT * swept,
      W_APPROACH * approach,
      W_SUPPORT * self.support_miss(dist, push, point),
      W_PEN_LIMIT * max(0.0, np.linalg.norm(x[16:19]) - SOFT_PEN_SHIFT_M),
      W_PEN_LIMIT * SUPPORT_ARM_M * max(0.0, np.linalg.norm(x[19:22]) - SOFT_PEN_TILT_RAD),
      W_PUSH * (push - self.push)[self.target].ravel(),
      W_JOINT_SMOOTH * (x[:16] - self.x_prev[:16]),
      W_PEN_POS * x[16:19],
      W_PEN_ROT * x[19:22],
      W_PEN_POS_SMOOTH * (x[16:19] - self.x_prev[16:19]),
      W_PEN_ROT_SMOOTH * (x[19:22] - self.x_prev[19:22]),
      W_JOINT_RETARGET * (x[:16] - self.q_retarget),
    ]

  def solve(self):
    x0 = np.clip(self.x_prev, self.lower, self.upper)
    return least_squares(
      self.residuals,
      x0,
      bounds=(self.lower - 1e-12, self.upper + 1e-12),
      diff_step=1e-3,
      max_nfev=200,
    )


def motion_end(pen_pose):
  """First frame of the static tail (all following frames identical)."""
  end = len(pen_pose)
  while end > 1 and np.allclose(pen_pose[end - 2], pen_pose[-1]):
    end -= 1
  return end


def required_wrench(m, pen_pose_w, dt, smooth_frames=5):
  """Force and torque about the centre of mass that move the pen along
  ``pen_pose_w`` against gravity (accelerations smoothed over a few frames)."""
  body = m.body("pen").id
  mass, inertia = m.body_mass[body], m.body_inertia[body]
  kernel = np.ones(smooth_frames) / smooth_frames

  def smooth(v):
    return np.stack([np.convolve(v[:, i], kernel, mode="same") for i in range(3)], axis=1)

  acc = smooth(np.gradient(np.gradient(pen_pose_w[:, :3], dt, axis=0), dt, axis=0))
  rot = to_rot(pen_pose_w[:, 3:])
  omega = np.zeros((len(rot), 3))
  omega[1:-1] = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
  alpha = smooth(np.gradient(omega, dt, axis=0))
  r = rot.as_matrix()
  inertia_w = r @ np.diag(inertia) @ r.transpose(0, 2, 1)
  i_alpha = np.einsum("kij,kj->ki", inertia_w, alpha)
  i_omega = np.einsum("kij,kj->ki", inertia_w, omega)
  return np.concatenate([mass * (acc - m.opt.gravity), i_alpha + np.cross(omega, i_omega)], axis=1)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  a = ap.parse_args()
  tag = a.clip.replace("/", "__")
  aligned = RESULTS / "aligned" / tag
  mapped = np.load(aligned / "mapped.npz")
  first = np.load(RESULTS / "first_frame" / tag / "first_frame.npz")
  q_retarget = np.load(CONVERTED / a.clip / "reference.npz")["reference_qpos"]
  m = mujoco.MjModel.from_xml_path(str(aligned / "scene.xml"))
  d0 = mujoco.MjData(m)
  mujoco.mj_forward(m, d0)
  palm_id = m.body("right_palm").id
  palm_rot_w = Rotation.from_matrix(d0.xmat[palm_id].reshape(3, 3))
  palm_pose_w = np.r_[d0.xpos[palm_id], d0.xquat[palm_id]]

  pen_mapped = mapped["object_pose_w"]
  frames = len(pen_mapped)
  end = motion_end(mapped["object_pose_palm"])
  wrench = required_wrench(m, pen_mapped, float(mapped["frame_dt"]))
  wrench[end:] = wrench[end]
  prob = TrackingProblem(m)

  # Express the repaired first frame as a correction of the mapped pen pose.
  rot0 = to_rot(first["object_pose_w"][3:]) * to_rot(pen_mapped[0, 3:]).inv()
  x = np.r_[first["qpos"], first["object_pose_w"][:3] - pen_mapped[0, :3], rot0.as_rotvec()]
  xs = [x]
  per_frame = []
  for k in range(frames):
    targets = orca_targets(mapped, min(k, end - 1), palm_rot_w)
    if 0 < k < end:
      upcoming = [
        (pen_mapped[j], orca_targets(mapped, j, palm_rot_w))
        for j in range(k + 1, min(k + 1 + LOOKAHEAD_FRAMES, end))
      ]
      prob.set_frame(targets, pen_mapped[k], q_retarget[k], xs[-1], upcoming, wrench[k])
      x = prob.solve().x
      xs.append(x)
    elif k >= end:
      xs.append(xs[-1])
    k_ret = min(k, len(q_retarget) - 1)
    prob.set_frame(targets, pen_mapped[k], q_retarget[k_ret], xs[-1], wrench=wrench[k])
    dist, push, point = prob.pose(xs[-1])
    target_idx = np.flatnonzero(prob.target)
    other = ~prob.target & ~prob.allowed
    cos = np.clip(np.sum(push * prob.push, axis=1)[target_idx], -1, 1)
    miss = prob.support_miss(dist, push, point)
    per_frame.append(
      {
        "frame": k,
        "targets": [prob.names[i] for i in target_idx],
        "target_dist_mm": (1000 * dist[target_idx]).round(2).tolist(),
        "push_error_deg": np.degrees(np.arccos(cos)).round(1).tolist(),
        "touching_others": [prob.names[i] for i in np.flatnonzero(prob.allowed & (dist < 0.0005))],
        "worst_allowed_mm": round(1000 * float(dist[prob.allowed].min(initial=0.05)), 2),
        "worst_non_target_mm": round(1000 * float(dist[other].min()), 2),
        "support_miss_force": round(float(np.linalg.norm(miss[:3])), 3),
        "support_miss_torque": round(float(np.linalg.norm(miss[3:])), 3),
      }
    )
    if k % 25 == 0 or k == end - 1:
      f = per_frame[-1]
      worst_target = max(f["target_dist_mm"], default=0.0)
      print(
        f"frame {k:3d}: targets={[n.removeprefix('right_') for n in f['targets']]} "
        f"+{[n.removeprefix('right_') for n in f['touching_others']]} "
        f"worst target dist={worst_target:+.2f}mm non-target min={f['worst_non_target_mm']:+.2f}mm "
        f"support miss={f['support_miss_force']:.2f}/{f['support_miss_torque']:.2f}",
        flush=True,
      )

  xs = np.array(xs[:frames])
  qpos = xs[:, :16]
  pen_rot_w = Rotation.from_rotvec(xs[:, 19:22]) * to_rot(pen_mapped[:, 3:])
  pen_pose_w = np.concatenate([pen_mapped[:, :3] + xs[:, 16:19], to_wxyz(pen_rot_w)], axis=1)

  link_names = [m.body(i).name for i in range(1, m.nbody) if m.body(i).name.startswith("right_")]
  link_ids = [m.body(n).id for n in link_names]
  link_pos_w = []
  d = mujoco.MjData(m)
  for k in range(frames):
    d.qpos[prob.qadr] = qpos[k]
    mujoco.mj_kinematics(m, d)
    link_pos_w.append(d.xpos[link_ids].copy())

  out = RESULTS / "trajectory" / tag
  out.mkdir(parents=True, exist_ok=True)
  copy_scene(aligned / "scene.xml", out / "scene.xml")
  np.savez_compressed(
    out / "reference.npz",
    reference_qpos=qpos,
    object_pose_w=pen_pose_w,
    wrist_pose_w=np.tile(palm_pose_w, (frames, 1)),
    link_pos_w=np.array(link_pos_w),
    joint_names=np.array([m.actuator(i).name.removesuffix("_actuator") for i in range(m.nu)]),
    link_names=np.array(link_names),
    frame_dt=float(mapped["frame_dt"]),
    motion_end_frame=end,
    source_clip=a.clip,
  )
  steps = np.abs(np.diff(qpos, axis=0))
  summary = {
    "clip": a.clip,
    "frames": frames,
    "motion_end_frame": end,
    "max_joint_step_deg": round(float(np.degrees(steps.max())), 2),
    "p95_joint_step_deg": round(float(np.degrees(np.percentile(steps, 95))), 2),
    "max_pen_correction_mm": round(float(1000 * np.linalg.norm(xs[:, 16:19], axis=1).max()), 2),
    "max_pen_correction_deg": round(float(np.degrees(np.linalg.norm(xs[:, 19:22], axis=1).max())), 2),
    "frames_all_targets_touching": float(
      np.mean([all(v < 0.5 for v in f["target_dist_mm"]) for f in per_frame])
    ),
    "worst_target_dist_mm": float(max(max(f["target_dist_mm"], default=0.0) for f in per_frame)),
    "worst_non_target_penetration_mm": float(
      -min(0.0, min(f["worst_non_target_mm"] for f in per_frame))
    ),
    "worst_allowed_penetration_mm": float(
      -min(0.0, min(f["worst_allowed_mm"] for f in per_frame))
    ),
    "mean_support_miss_force": round(
      float(np.mean([f["support_miss_force"] for f in per_frame[:end]])), 3
    ),
    "mean_support_miss_torque": round(
      float(np.mean([f["support_miss_torque"] for f in per_frame[:end]])), 3
    ),
  }
  (out / "repair.json").write_text(json.dumps({"summary": summary, "frames": per_frame}, indent=1))
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
