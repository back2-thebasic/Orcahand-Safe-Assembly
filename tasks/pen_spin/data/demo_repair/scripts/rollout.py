"""Open-loop reference rollouts shared by the OrcaHand and Wuji checks.

A rollout starts from the reference state of one frame (joints, pen pose and
their finite-difference velocities, as Wuji's reset does), then sends the
reference joint targets frame by frame. It ends on Wuji's training
terminations: the pen more than ``FAIL_M`` from the reference position, or its
shaft (local x) more than ``FAIL_SHAFT_DEG`` from the reference shaft.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

FAIL_M = 0.05
FAIL_SHAFT_DEG = 60.0


def _rot(q_wxyz):
  return Rotation.from_quat(np.roll(q_wxyz, -1, axis=-1))


def _shaft(q_wxyz):
  return _rot(q_wxyz).apply([1.0, 0.0, 0.0])


def survival_s(m, d, qadr, vadr, pen_q, pen_v, pen_body, qref, pen_ref, start, horizon_s, dt):
  frames = len(qref)
  nxt = min(start + 1, frames - 1)
  mujoco.mj_resetData(m, d)
  d.qpos[qadr] = qref[start]
  d.qpos[pen_q : pen_q + 7] = pen_ref[start]
  if nxt > start:
    d.qvel[vadr] = (qref[nxt] - qref[start]) / dt
    d.qvel[pen_v : pen_v + 3] = (pen_ref[nxt, :3] - pen_ref[start, :3]) / dt
    rel = _rot(pen_ref[start, 3:]).inv() * _rot(pen_ref[nxt, 3:])
    d.qvel[pen_v + 3 : pen_v + 6] = rel.as_rotvec() / dt
  mujoco.mj_forward(m, d)
  nsub = round(dt / m.opt.timestep)
  steps = int(round(horizon_s / dt))
  pos_err, shaft_err = [], []
  for i in range(1, steps + 1):
    k = min(start + i, frames - 1)
    d.ctrl[:] = qref[k]
    mujoco.mj_step(m, d, nstep=nsub)
    pos = np.linalg.norm(d.xpos[pen_body] - pen_ref[k, :3])
    cos = np.dot(_shaft(d.xquat[pen_body]), _shaft(pen_ref[k, 3:]))
    shaft = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    if pos > FAIL_M or shaft > FAIL_SHAFT_DEG:
      return (i - 1) * dt, pos_err, shaft_err
    pos_err.append(pos)
    shaft_err.append(shaft)
  return horizon_s, pos_err, shaft_err


def survival_profile(m, qadr, vadr, pen_q, pen_v, pen_body, qref, pen_ref, starts, horizon_s, dt):
  """Survival time per start frame, plus the tracking errors while alive."""
  d = mujoco.MjData(m)
  times, pos_err, shaft_err = [], [], []
  for s in starts:
    t, p, a = survival_s(m, d, qadr, vadr, pen_q, pen_v, pen_body, qref, pen_ref, s, horizon_s, dt)
    times.append(t)
    pos_err += p
    shaft_err += a
  return np.array(times), np.array(pos_err), np.array(shaft_err)


def summarize(times, pos_err, shaft_err, horizon_s):
  return {
    "mean_s": round(float(times.mean()), 3),
    "median_s": round(float(np.median(times)), 3),
    "share_over_0.2s": round(float(np.mean(times >= 0.2)), 3),
    f"share_full_{horizon_s}s": round(float(np.mean(times >= horizon_s)), 3),
    "alive_mean_pos_err_cm": round(float(pos_err.mean() * 100), 2) if len(pos_err) else None,
    "alive_mean_shaft_err_deg": round(float(shaft_err.mean()), 1) if len(shaft_err) else None,
  }


def shaft_swing_deg(pen_ref, start, horizon_frames):
  """Largest shaft angle the reference reaches from its start within the horizon."""
  ref = _shaft(pen_ref[start : start + horizon_frames + 1, 3:])
  return float(np.degrees(np.arccos(np.clip(ref @ ref[0], -1.0, 1.0))).max())
