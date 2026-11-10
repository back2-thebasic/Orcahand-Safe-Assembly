"""Initial conversion of Wuji pen clips to the OrcaHand: geometric retarget, no repair.

The MediaPipe landmarks are retargeted to OrcaHand joints with the orca_teleop
retargeter, calibrated on the clip's first (grasping) frame. The pen pose goes
through exactly the spatial normalization the retargeter applies to the hand,
so hand and pen stay consistent. Wuji's official 250 mm / 30 g pen replaces the
legacy pen, and two feasibility checks run on the result:

  metrics.json       kinematic pass (pen forced to the reference) and a rollout
                     with hand targets only and the pen free
  hold_metrics.json  3 s static hold of the first frame

Writes results/converted/<group>/<clip>/ (scene.xml, reference.npz,
rollout.npz, metrics.json, hold_metrics.json), which align.py and the repair
scripts read, and lists every converted clip in results/converted/catalog.json
for view.py --catalog. Run with the orca_teleop environment:

  MPLCONFIGDIR=/tmp/mpl orca_teleop/.venv/bin/python convert.py \\
    --clips 02_continuous_single_axis/clip_002 02_continuous_single_axis/clip_004
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import orca_core
import orca_sim
from scipy.spatial.transform import Rotation

from official_pen import attach_official_pen, verify_official_pen
from orca_teleop.retargeting import utils
from orca_teleop.retargeting.constants import MANO_TO_URDF_TRANSLATION
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose
from portable_scene import copy_scene

DEMO_REPAIR = Path(__file__).resolve().parents[1]
PEN_SPIN = DEMO_REPAIR.parents[1]
CLIPS = PEN_SPIN / "data/official/rawdata/clips"
CONVERTED = DEMO_REPAIR / "results/converted"
RETARGET_CONFIG = DEMO_REPAIR / "assets/retarget_04.yaml"
HAND_MJCF = Path(orca_sim.__file__).resolve().parent / "models/v1/right_cube_orientation.mjcf"
HAND_CONFIG = Path(orca_core.__file__).parent / "models/v1/orcahand_right/config.yaml"
URDF = PEN_SPIN.parents[1] / "orcahand_description/v1/models/urdf/orcahand_right.urdf"
TIMESTEP = 0.002
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def rot(q):
  return Rotation.from_quat(np.roll(q, -1)).as_matrix()


def quat(r):
  return np.roll(Rotation.from_matrix(r).as_quat(), 1)


def hand_scene(dt):
  """OrcaHand with the wrist removed, floor, light and a legacy capsule pen.

  The legacy pen only settles in the palm here; attach_official_pen replaces it.
  """
  xml = ET.parse(HAND_MJCF).getroot()
  for parent in xml.iter():
    for child in list(parent):
      if child.get("name") == "right_wrist" or child.get("joint") == "right_wrist":
        parent.remove(child)
  for mesh in xml.findall(".//asset/mesh"):
    mesh.set("file", str((HAND_MJCF.parent / mesh.get("file")).resolve()))
  world = xml.find("worldbody")
  ET.SubElement(world, "light", pos=".1 -.1 1", dir="0 0 -1")
  ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .01", rgba=".2 .25 .3 1")
  length, radius, mass = 0.14, 0.007, 0.018
  pen = ET.SubElement(world, "body", name="pen", pos=".18 .04 .20")
  ET.SubElement(pen, "freejoint", name="pen_free")
  half = length / 2 - radius
  ET.SubElement(pen, "geom", name="pen_barrel", type="capsule", fromto=f"0 {-half} 0 0 {half} 0",
                size=str(radius), mass=str(mass), rgba=".9 .85 .65 1", friction="1 .005 .0001", condim="4")
  for name, sign, color in [("blue", 1, ".1 .35 1 1"), ("red", -1, "1 .2 .15 1")]:
    ET.SubElement(pen, "geom", name=f"pen_{name}_end", type="sphere", pos=f"0 {sign * half} 0",
                  size=str(radius * 1.01), mass="0", contype="0", conaffinity="0", rgba=color)
  ET.SubElement(xml, "option", timestep=str(TIMESTEP), gravity="0 0 -9.81", integrator="implicitfast")
  m = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
  m.opt.timestep = dt / max(1, round(dt / m.opt.timestep))
  d = mujoco.MjData(m)
  qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
  for joint, index in zip(m.actuator_trnid[:, 0], qadr):
    name = m.joint(int(joint)).name
    if "thumb" not in name and name.endswith(("mcp", "pip")):
      d.qpos[index] = 0.2
  d.ctrl[:] = np.clip(d.qpos[qadr], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
  mujoco.mj_forward(m, d)
  palm = m.body("right_palm").id
  geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == palm and m.geom_contype[g]]
  pen_q = m.jnt_qposadr[m.joint("pen_free").id]
  d.qpos[pen_q : pen_q + 3] = d.geom_xpos[geoms].mean(axis=0) + [0.025, 0.0, 0.05]
  mujoco.mj_forward(m, d)
  mujoco.mj_step(m, d, nstep=round(3.0 / m.opt.timestep))
  mujoco.mj_forward(m, d)
  return m, d


def load_clip(clip):
  group = CLIPS / clip.parent
  index = json.loads((group / "index.json").read_text())
  entry = next((e for e in index["clips"] if Path(e["file"]).stem == clip.name), None)
  if entry is None:
    raise SystemExit(f"{clip} is not in {group / 'index.json'}")
  source = group / entry["file"]
  digest = hashlib.sha256(source.read_bytes()).hexdigest()
  assert digest == entry["sha256"], f"{source} does not match its index sha256"
  with np.load(source) as d:
    raw = {k: d[k].astype(float) for k in d.files}
  assert all(np.isfinite(v).all() for v in raw.values())
  return source, digest, raw, float(index["frame_dt"])


def retarget(raw, m, d):
  # OrcaHand rewrites the calibration.yaml next to its config on construction,
  # so it gets a scratch copy instead of the installed orca_core package.
  with tempfile.TemporaryDirectory() as tmp:
    for name in ("config.yaml", "calibration.yaml"):
      shutil.copy(HAND_CONFIG.parent / name, tmp)
    r = Retargeter.from_paths(model_path=str(Path(tmp) / "config.yaml"), urdf_path=str(URDF),
                              config_path=str(RETARGET_CONFIG))
    return retarget_frames(r, raw, m, d)


def retarget_frames(r, raw, m, d):
  """Joint reference plus pen, fingertip and link poses in the OrcaHand world."""
  joints = m.actuator_trnid[:, 0]
  qadr = m.jnt_qposadr[joints]
  names = [m.joint(int(j)).name.removeprefix("right_") for j in joints]
  limits = {n: tuple(m.actuator_ctrlrange[i]) for i, n in enumerate(names)}
  limits["wrist"] = (-1e-6, 1e-6)
  r.set_sim_joint_limits(limits)
  # Calibrate on the first recorded grasp, not a neutral open hand.
  for _ in range(100):
    if r._calibration_done:
      break
    r.retarget(TargetPose(raw["mediapipe_landmarks_w"][0]))
  assert r._calibration_done
  urdf = ET.parse(r.urdf_path)
  tip_info = []
  for finger in FINGERS:
    child = f"right_{finger}_fingertip"
    joint = next(j for j in urdf.findall("joint") if j.find("child").get("link") == child)
    tip_info.append((m.body(joint.find("parent").get("link")).id,
                     np.fromstring(joint.find("origin").get("xyz"), sep=" ")))
  palm = m.body("right_palm").id
  out = {k: [] for k in ("reference_qpos", "object_pose_w", "target_tips_w", "actual_tips_w", "link_pos_w")}
  fk_errors = []
  for k, lm in enumerate(raw["mediapipe_landmarks_w"]):
    q = np.deg2rad(r.retarget(TargetPose(lm)).as_array(names))
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    pin_q = r._full_model_qpos(q)
    pin_tips, _ = r._robot.compute_positions_and_jacobians(pin_q, r._computed_frame_indices, r._frame_offsets)
    pin_palm = r._robot.data.oMf[r._computed_frame_indices[r._palm_index]]
    A = d.xmat[palm].reshape(3, 3) @ pin_palm.rotation.T
    b = d.xpos[palm] - A @ pin_palm.translation
    center, H = utils.get_hand_center_and_rotation(lm[1], lm[5], lm[9], lm[13], lm[17], lm[0])
    extra = np.eye(3) if r._rotation_matrix is None else r._rotation_matrix
    B = r._urdfhand_rot_matrix @ extra @ H.T
    C = r._urdfhand_center + np.array(MANO_TO_URDF_TRANSLATION)

    def transform(x):
      return ((x - center) @ B.T * r._mano_scale + C) @ A.T + b

    pose = raw["object_pose_w"][k]
    tips = np.array([d.xpos[body] + d.xmat[body].reshape(3, 3) @ offset for body, offset in tip_info])
    out["reference_qpos"].append(q)
    out["object_pose_w"].append(np.r_[transform(pose[:3]), quat(A @ B @ rot(pose[3:]))])
    out["target_tips_w"].append(transform(lm[[4, 8, 12, 16, 20]]))
    out["actual_tips_w"].append(tips)
    out["link_pos_w"].append(d.xpos[1:].copy())
    fk_errors.append(np.linalg.norm(tips - (pin_tips[r._tip_indices] @ A.T + b), axis=1))
  out = {k: np.array(v) for k, v in out.items()}
  fk_errors = np.array(fk_errors)
  assert fk_errors.max() < 0.002, f"URDF/MJCF mapping mismatch: {fk_errors.max()}"
  assert np.all(out["reference_qpos"] >= m.actuator_ctrlrange[:, 0] - 1e-7)
  assert np.all(out["reference_qpos"] <= m.actuator_ctrlrange[:, 1] + 1e-7)
  return out, names, float(fk_errors.max()), r._mano_scale


def pen_contacts(m, d, pen):
  contacts, self_depth = [], []
  for c in d.contact[: d.ncon]:
    a, b = [int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
    if pen in (a, b):
      other = b if a == pen else a
      if m.body(other).name.startswith("right_"):
        contacts.append((m.body(other).name, float(c.dist)))
    elif m.body(a).name.startswith("right_") and m.body(b).name.startswith("right_"):
      self_depth.append(max(0.0, -float(c.dist)))
  return contacts, max(self_depth, default=0.0)


def feasibility(scene, ref, dt):
  """Kinematic pass with the pen forced, then free-pen rollout on hand targets."""
  m = mujoco.MjModel.from_xml_path(str(scene))
  d = mujoco.MjData(m)
  qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
  pen_q = m.joint("pen_free").qposadr[0]
  pen_v = m.joint("pen_free").dofadr[0]
  pen = m.body("pen").id
  refq, obj = ref["reference_qpos"], ref["object_pose_w"]
  kinematic = []
  for k, q in enumerate(refq):
    d.qpos[qadr] = q
    d.qpos[pen_q : pen_q + 7] = obj[k]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    c, s = pen_contacts(m, d, pen)
    kinematic.append(dict(contact_bodies=sorted(set(a for a, _ in c)),
                          max_penetration_m=max([max(0, -v) for _, v in c], default=0.0),
                          self_penetration_m=s))
  mujoco.mj_resetData(m, d)
  d.qpos[qadr] = refq[0]
  d.qpos[pen_q : pen_q + 7] = obj[0]
  d.ctrl[:] = refq[0]
  # Differentiated initial velocities, as reference-state initialization does.
  d.qvel[m.jnt_dofadr[m.actuator_trnid[:, 0]]] = (refq[1] - refq[0]) / dt
  d.qvel[pen_v : pen_v + 3] = (obj[1, :3] - obj[0, :3]) / dt
  d.qvel[pen_v + 3 : pen_v + 6] = Rotation.from_matrix(rot(obj[0, 3:]).T @ rot(obj[1, 3:])).as_rotvec() / dt
  mujoco.mj_forward(m, d)
  palm = d.xpos[m.body("right_palm").id].copy()
  dynamic, actual = [], []
  for k, q in enumerate(refq):
    c, _ = pen_contacts(m, d, pen)
    p = d.xpos[pen].copy()
    actual.append(np.r_[p, d.xquat[pen]])
    angle = np.rad2deg(np.arccos(np.clip(d.xmat[pen].reshape(3, 3)[:, 0] @ rot(obj[k, 3:])[:, 0], -1, 1)))
    dynamic.append(dict(t=k * dt, position_error_m=float(np.linalg.norm(p - obj[k, :3])),
                        axis_error_deg=float(angle), contact=bool(c),
                        fallen=bool(p[2] < 0.10 or np.linalg.norm(p - palm) > 0.22),
                        joint_rmse_rad=float(np.sqrt(np.mean((d.qpos[qadr] - q) ** 2)))))
    d.ctrl[:] = q
    d.xfrc_applied[:] = 0
    d.qfrc_applied[:] = 0
    mujoco.mj_step(m, d, nstep=round(dt / m.opt.timestep))
    mujoco.mj_forward(m, d)
    assert np.isfinite(d.qpos).all()
  return kinematic, dynamic, np.array(actual)


def hold_screen(scene, ref, dt, seconds=3.0):
  """Static hold of the first frame: hand targets fixed, pen free."""
  m = mujoco.MjModel.from_xml_path(str(scene))
  d = mujoco.MjData(m)
  q, pose = ref["reference_qpos"][0], ref["object_pose_w"][0]
  qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
  pen_q = m.joint("pen_free").qposadr[0]
  pen = m.body("pen").id
  d.qpos[qadr] = q
  d.ctrl[:] = q
  d.qpos[pen_q : pen_q + 7] = pose
  d.qvel[:] = 0
  mujoco.mj_forward(m, d)
  initial_axis = d.xmat[pen].reshape(3, 3)[:, 0].copy()
  rows = []
  for k in range(round(seconds / dt) + 1):
    fingers, depth, self_depth, floor = set(), 0.0, 0.0, False
    for c in d.contact[: d.ncon]:
      a, b = [int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
      if pen in (a, b):
        other = b if a == pen else a
        floor |= other == 0
        name = m.body(other).name
        if name.startswith("right_"):
          depth = max(depth, -float(c.dist))
          fingers |= {f for f in FINGERS if name.startswith(f"right_{f}_") and c.dist <= 0}
      elif m.body(a).name.startswith("right_") and m.body(b).name.startswith("right_"):
        self_depth = max(self_depth, -float(c.dist))
    angle = np.rad2deg(np.arccos(np.clip(initial_axis @ d.xmat[pen].reshape(3, 3)[:, 0], -1, 1)))
    rows.append(dict(time_s=k * dt, displacement_m=float(np.linalg.norm(d.xpos[pen] - pose[:3])),
                     axis_drift_deg=float(angle), finger_contacts=sorted(fingers), pen_depth_m=depth,
                     self_depth_m=self_depth, floor_contact=floor))
    if k < round(seconds / dt):
      d.xfrc_applied[:] = 0
      d.qfrc_applied[:] = 0
      mujoco.mj_step(m, d, nstep=round(dt / m.opt.timestep))
      mujoco.mj_forward(m, d)
      if not np.isfinite(d.qpos).all():
        raise RuntimeError("Nonfinite hold rollout")
  pair_share = float(np.mean([len(x["finger_contacts"]) >= 2 for x in rows]))
  displacement = max(x["displacement_m"] for x in rows)
  axis = max(x["axis_drift_deg"] for x in rows)
  max_depth = max(x["pen_depth_m"] for x in rows)
  max_self = max(x["self_depth_m"] for x in rows)
  pen_v = m.joint("pen_free").dofadr[0]
  v = float(np.linalg.norm(d.qvel[pen_v : pen_v + 3]))
  w = float(np.linalg.norm(d.qvel[pen_v + 3 : pen_v + 6]))
  summary = dict(
    hold_seconds=seconds, initial_finger_contacts=rows[0]["finger_contacts"],
    initial_penetration_mm=rows[0]["pen_depth_m"] * 1000, max_penetration_mm=max_depth * 1000,
    max_self_penetration_mm=max_self * 1000, max_displacement_mm=displacement * 1000,
    max_axis_drift_deg=axis, two_finger_contact_fraction=pair_share,
    final_speed_m_s=v, final_angular_speed_rad_s=w,
    passes_static_screen=bool(displacement <= 0.01 and axis <= 15 and pair_share >= 0.9
                              and max_depth <= 0.002 and max_self <= 0.002
                              and not any(x["floor_contact"] for x in rows) and v < 0.01 and w < 0.5),
    caveat="Heuristic static screen only; does not certify force closure or dynamic pen spin.")
  return summary, rows


def convert(clip):
  source, digest, raw, dt = load_clip(clip)
  m, d = hand_scene(dt)
  ref, names, fk_error, mano_scale = retarget(raw, m, d)
  link_names = [m.body(i).name for i in range(1, m.nbody)]
  out = CONVERTED / clip
  if out.exists():
    shutil.rmtree(out)
  out.mkdir(parents=True)
  base = out / "scene_base.xml"
  mujoco.mj_saveLastXML(str(base), m)
  tree = ET.parse(base)
  base.unlink()
  pen_source, pen_inertia = attach_official_pen(tree)
  scene = out / "scene.xml"
  tree.write(scene)
  copy_scene(scene, scene)
  pen_properties = verify_official_pen(mujoco.MjModel.from_xml_path(str(scene)), pen_source, pen_inertia)
  np.savez_compressed(
    out / "reference.npz", **ref, joint_names=names, link_names=link_names, frame_dt=dt,
    metadata=json.dumps(dict(training_eligible=False, artifact_type="kinematic_retarget_probe",
                             quaternion_order="wxyz", joint_units="radians", source=str(source),
                             note="Includes diagnostic world body positions; NOT a Wuji training clip.")))
  kinematic, dynamic, actual = feasibility(scene, ref, dt)
  np.savez_compressed(out / "rollout.npz", object_pose_w=actual)
  errs = np.linalg.norm(ref["actual_tips_w"] - ref["target_tips_w"], axis=2)
  summary = dict(
    source=str(source), source_sha256=digest, frames=len(ref["reference_qpos"]), frame_dt=dt,
    duration_s=(len(ref["reference_qpos"]) - 1) * dt, config=RETARGET_CONFIG.stem.removeprefix("retarget_"),
    mano_scale=mano_scale, pen_length_m=0.250, pen_radius_m=0.00745, pen_mass_kg=0.030,
    pen_model="official", pen_properties=pen_properties, urdf_mjcf_max_error_m=fk_error,
    tip_target_mean_error_mm=(errs.mean(axis=0) * 1000).tolist(),
    tip_target_p95_error_mm=float(np.percentile(errs, 95) * 1000),
    kinematic_contact_frame_fraction=float(np.mean([bool(x["contact_bodies"]) for x in kinematic])),
    kinematic_max_penetration_mm=max(x["max_penetration_m"] for x in kinematic) * 1000,
    kinematic_max_self_penetration_mm=max(x["self_penetration_m"] for x in kinematic) * 1000,
    first_fallen_s=next((x["t"] for x in dynamic if x["fallen"]), None),
    first_over_5cm_error_s=next((x["t"] for x in dynamic if x["position_error_m"] > 0.05), None),
    dynamic_contact_frame_fraction=float(np.mean([x["contact"] for x in dynamic])),
    limitations=["Geometric retarget baseline, no object-aware IK, no PPO.",
                 "Calibration uses a bent first-frame grasp rather than neutral human hand.",
                 "Official 250mm/30g pen; reference demonstrates a different nominal length; contacts not optimized.",
                 "Kinematic preview overwrites pen state and is NOT evidence of manipulation success.",
                 "reference.npz is a probe artifact, not a drop-in Wuji training clip."])
  (out / "metrics.json").write_text(json.dumps(dict(summary=summary, kinematic=kinematic, dynamic=dynamic), indent=2))
  hold, rows = hold_screen(scene, ref, dt)
  (out / "hold_metrics.json").write_text(json.dumps(dict(summary=hold, frames=rows), indent=2))
  return summary, hold


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--clips", nargs="+", type=Path, help="e.g. 02_continuous_single_axis/clip_002")
  p.add_argument("--groups", nargs="+", default=[], help="convert every clip of these groups")
  a = p.parse_args()
  clips = list(a.clips or [])
  for group in a.groups:
    index = json.loads((CLIPS / group / "index.json").read_text())
    clips += [Path(group) / Path(e["file"]).stem for e in index["clips"]]
  if not clips:
    p.error("give --clips or --groups")
  catalog = CONVERTED / "catalog.json"
  entries = {e["clip"]: e for e in json.loads(catalog.read_text())["entries"]} if catalog.exists() else {}
  for i, clip in enumerate(clips):
    entry = dict(clip=str(clip), run=str(clip))
    try:
      summary, hold = convert(clip)
    except Exception as e:
      entry.update(error=f"{type(e).__name__}: {e}")
      print(f"[{i + 1}/{len(clips)}] {clip}: FAILED {entry['error']}", flush=True)
    else:
      entry.update(hold)
      print(f"[{i + 1}/{len(clips)}] {clip}: tip p95 {summary['tip_target_p95_error_mm']:.1f} mm, "
            f"free pen over 5 cm at {summary['first_over_5cm_error_s']} s, "
            f"static hold pass={hold['passes_static_screen']} "
            f"(drift {hold['max_displacement_mm']:.1f} mm, contacts {hold['initial_finger_contacts']})", flush=True)
    entries[entry["clip"]] = entry
    CONVERTED.mkdir(parents=True, exist_ok=True)
    catalog.write_text(json.dumps(dict(entries=[entries[k] for k in sorted(entries)],
                                       scope="first frame only, no contact optimization"), indent=2))
  print("Catalog:", catalog, flush=True)


if __name__ == "__main__":
  main()
