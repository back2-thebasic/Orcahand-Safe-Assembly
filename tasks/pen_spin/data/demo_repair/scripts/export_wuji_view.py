"""Export a Wuji clip on Wuji Hand 2 as view.py runs, to compare with OrcaHand.

Writes two run directories under results/wuji_view/<group>__<clip>/:

  reference/  Wuji's demonstration: the clip's joint and pen reference on
              Wuji Hand 2. View with --mode reference, hold or dynamic like
              the OrcaHand runs.
  policy/     The released closed-loop policy of the clip's group, simulated
              in Wuji's play environment from frame 0 and recorded. Its
              reference.npz holds the simulated states, so only
              --mode reference is meaningful (it plays the recording).

Needs the mjlab environment.

  python export_wuji_view.py --clip 01_single_axis/clip_003
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

PEN_SPIN = Path(__file__).resolve().parents[3]
CLIPS = PEN_SPIN / "data/official/rawdata/clips"
CHECKPOINTS = PEN_SPIN.parents[1] / "Wuji/wuji-mjlab-v2026.10.22-assets/pen_spin/checkpoints"
RESULTS = PEN_SPIN / "data/demo_repair/results"
PEN_JOINT = "object/cube_freejoint"

import wuji_mjlab.tasks.pen_spin.tooling.clip_store as clip_store

clip_store._DEFAULT_CLIPS_ROOT = CLIPS

import wuji_mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.spec import export_spec

from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.env_cfgs import (
  wuji_hand2_pen_spin_env_cfg,
)
from wuji_mjlab.tasks.pen_spin.config.wuji_hand2.hand2_constants import (
  RIGHT_FINGER_JOINT_NAMES,
)

PEN_TEXTURES = Path(wuji_mjlab.__file__).parent / "assets/objects/aruco_pen250/textures"
OPTION_FIELDS = ("timestep", "impratio", "cone", "solver", "iterations", "ls_iterations", "tolerance", "integrator", "gravity")


def make_env(clip):
  recipe = Path(tempfile.mkdtemp()) / "recipe.json"
  recipe.write_text(json.dumps({"schema": "wuji_pen_spin_recipe:1", "clips": [clip]}))
  cfg = wuji_hand2_pen_spin_env_cfg(play=True)
  cfg.commands["motion"].recipe_path = str(recipe)
  cfg.commands["motion"].debug_vis = False
  cfg.commands["motion"].first_frame_hold_prob = 0.0
  return ManagerBasedRlEnv(cfg, device="cpu")


def write_scene(env, out):
  """The compiled model as scene.xml, with the pen joint named as view.py expects."""
  spec = env.scene.spec.copy()
  spec.joint(PEN_JOINT).name = "pen_free"
  compiled = env.sim.mj_model
  for field in OPTION_FIELDS:
    setattr(spec.option, field, getattr(compiled.opt, field))
  tmp = Path(tempfile.mkdtemp())
  export_spec(spec, tmp)
  out.mkdir(parents=True, exist_ok=True)
  xml = (tmp / "scene.xml").read_text()
  xml = xml.replace('meshdir="assets"', 'meshdir="assets" texturedir="assets/textures"', 1)
  (out / "scene.xml").write_text(xml)
  shutil.rmtree(out / "assets", ignore_errors=True)
  shutil.copytree(tmp / "assets", out / "assets")
  shutil.copytree(PEN_TEXTURES, out / "assets/textures")


def reference_pen_w(tag):
  w = np.load(RESULTS / "wuji" / f"{tag}.npz")
  fixed = w["fixed_wrist_pose_w"]
  fixed_rot = Rotation.from_quat(np.roll(fixed[3:], -1))
  palm = w["pen_pose_palm"]
  rot = fixed_rot * Rotation.from_quat(np.roll(palm[:, 3:], -1, axis=-1))
  return np.concatenate([fixed[:3] + fixed_rot.apply(palm[:, :3]), np.roll(rot.as_quat(), 1, axis=-1)], axis=1)


def record_policy(env, policy_path, qadr, pen_q, pen_ref, frames):
  import onnxruntime as ort

  sess = ort.InferenceSession(str(policy_path))
  name = sess.get_inputs()[0].name
  obs, _ = env.reset()
  motion = env.command_manager.get_term("motion")
  start = int(motion.time_steps[0])
  qpos = env.sim.data.qpos
  hand, pen, err = [], [], []

  def grab():
    q = qpos[0].cpu().numpy()
    hand.append(q[qadr].copy())
    pen.append(q[pen_q : pen_q + 7].copy())
    err.append(float(np.linalg.norm(pen[-1][:3] - pen_ref[min(start + len(err), len(pen_ref) - 1), :3])))

  grab()
  for _ in range(frames - start - 1):
    action = torch.from_numpy(sess.run(None, {name: obs["policy"].numpy().astype(np.float32)})[0])
    obs, _, term, trunc, _ = env.step(action)
    if bool(term[0]) or bool(trunc[0]):
      break
    grab()
  return start, np.array(hand), np.array(pen), np.array(err)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  a = ap.parse_args()
  tag = a.clip.replace("/", "__")
  out = RESULTS / "wuji_view" / tag

  env = make_env(a.clip)
  m = env.sim.mj_model
  joints = [m.joint(f"robot/{n}").id for n in RIGHT_FINGER_JOINT_NAMES]
  assert list(m.actuator_trnid[:, 0]) == joints, "actuators must follow RIGHT_FINGER_JOINT_NAMES"
  qadr = m.jnt_qposadr[joints]
  pen_q = m.jnt_qposadr[m.joint(PEN_JOINT).id]
  dt = float(env.step_dt)
  qref = np.load(CLIPS / f"{a.clip}.npz")["reference_qpos"].astype(float)
  pen_ref = reference_pen_w(tag)
  joint_names = np.array(RIGHT_FINGER_JOINT_NAMES)

  write_scene(env, out / "reference")
  np.savez_compressed(
    out / "reference" / "reference.npz",
    reference_qpos=qref, object_pose_w=pen_ref, joint_names=joint_names, frame_dt=dt, source_clip=a.clip,
  )

  policy = CHECKPOINTS / a.clip.split("/")[0] / "policy.onnx"
  start, hand, pen, err = record_policy(env, policy, qadr, pen_q, pen_ref, len(qref))
  write_scene(env, out / "policy")
  np.savez_compressed(
    out / "policy" / "reference.npz",
    reference_qpos=hand, object_pose_w=pen, reference_object_pose_w=pen_ref[start : start + len(pen)],
    joint_names=joint_names, frame_dt=dt, start_frame=start, source_clip=a.clip, policy=str(policy),
  )
  summary = {
    "clip": a.clip,
    "policy": str(policy),
    "start_frame": start,
    "recorded_frames": len(hand),
    "pen_error_mean_cm": round(100 * float(err.mean()), 2),
    "pen_error_max_cm": round(100 * float(err.max()), 2),
  }
  (out / "policy" / "summary.json").write_text(json.dumps(summary, indent=1))
  env.close()
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
