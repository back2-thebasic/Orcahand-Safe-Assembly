"""Open-loop survival of a repaired OrcaHand clip against the earlier versions.

Every ``--stride`` frames of the motion, the hand and pen start from the
reference state and follow the reference joint targets open loop (see
rollout.py). Versions compared:

  codex     the Codex retarget in its own scene (results/converted, from convert.py)
  aligned   the Codex joints with the pen mapped into the aligned palm frame
  repaired  repair_trajectory.py output

Compare with wuji_baseline.py, which runs the same test on Wuji Hand 2.

  python check_trajectory.py --clip 01_single_axis/clip_003
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rollout import shaft_swing_deg, summarize, survival_profile

PEN_SPIN = Path(__file__).resolve().parents[3]
RESULTS = PEN_SPIN / "data/demo_repair/results"
CONVERTED = RESULTS / "converted"


def motion_end(pen_pose):
  end = len(pen_pose)
  while end > 1 and np.allclose(pen_pose[end - 2], pen_pose[-1]):
    end -= 1
  return end


def profile(scene, qref, pen_ref, end, stride, horizon_s, dt):
  m = mujoco.MjModel.from_xml_path(str(scene))
  joints = [m.actuator_trnid[i, 0] for i in range(m.nu)]
  pen_j = m.joint("pen_free").id
  args = (
    m, m.jnt_qposadr[joints], m.jnt_dofadr[joints],
    m.jnt_qposadr[pen_j], m.jnt_dofadr[pen_j], m.joint("pen_free").bodyid[0],
    qref, pen_ref,
  )
  starts = np.arange(0, end, stride)
  times, pos_err, shaft_err = survival_profile(*args, starts, horizon_s, dt)
  from_start = survival_profile(*args, [0], 3.0, dt)[0][0]
  swing = [shaft_swing_deg(pen_ref, s, int(round(horizon_s / dt))) for s in starts]
  summary = summarize(times, pos_err, shaft_err, horizon_s)
  summary["reference_mean_shaft_swing_deg"] = round(float(np.mean(swing)), 1)
  return {
    "open_loop_from_frame0_s": round(float(from_start), 2),
    "summary": summary,
    "survival_s": times.round(3).tolist(),
  }


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  ap.add_argument("--stride", type=int, default=5)
  ap.add_argument("--horizon-s", type=float, default=0.5)
  a = ap.parse_args()
  tag = a.clip.replace("/", "__")

  codex = np.load(CONVERTED / a.clip / "reference.npz", allow_pickle=True)
  mapped = np.load(RESULTS / "aligned" / tag / "mapped.npz")
  run = RESULTS / "trajectory" / tag
  repaired = np.load(run / "reference.npz", allow_pickle=True)
  dt = float(repaired["frame_dt"])
  end = int(repaired["motion_end_frame"])
  q_codex = codex["reference_qpos"].astype(float)

  versions = {
    "codex": (CONVERTED / a.clip / "scene.xml", q_codex, codex["object_pose_w"]),
    "aligned": (RESULTS / "aligned" / tag / "scene.xml", q_codex, mapped["object_pose_w"]),
    "repaired": (run / "scene.xml", repaired["reference_qpos"], repaired["object_pose_w"]),
  }
  report = {"clip": a.clip, "start_frames": list(range(0, end, a.stride)), "horizon_s": a.horizon_s}
  for name, (scene, q, pen) in versions.items():
    report[name] = profile(scene, q, pen, end, a.stride, a.horizon_s, dt)
    print(name, json.dumps({k: v for k, v in report[name].items() if k != "survival_s"}))

  wuji = RESULTS / "wuji" / f"{tag}_baseline.json"
  if wuji.exists():
    w = json.loads(wuji.read_text())
    report["wuji"] = {k: w[k] for k in ("open_loop_from_frame0_s", "summary", "survival_s")}
    print("wuji", json.dumps({k: v for k, v in report["wuji"].items() if k != "survival_s"}))
  (run / "check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
  main()
