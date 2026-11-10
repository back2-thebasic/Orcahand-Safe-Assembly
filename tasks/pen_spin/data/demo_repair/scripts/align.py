"""Map a Wuji pen trajectory into an OrcaHand scene mounted like the Wuji hand.

Both palms get an anatomical frame built the same way (finger direction, index
-> pinky knuckle line, palm normal), which gives one constant rotation between
the Wuji ``right_palm_link`` frame and the OrcaHand ``right_palm`` frame. The
OrcaHand is then mounted so that gravity has the same direction in its palm
frame as on Wuji's fixed wrist, and every pen pose is mapped through the same
transform. Pen offsets from the knuckle centre are scaled per anatomical axis
because the OrcaHand is wider but has shorter fingers than Wuji Hand 2.

Run with the plain-mujoco environment:

  python align.py --clip 01_single_axis/clip_003
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from portable_scene import rebase_scene

PEN_SPIN = Path(__file__).resolve().parents[3]

BASE_SCENE = PEN_SPIN / "data/demo_repair/base_scene.xml"
RESULTS = PEN_SPIN / "data/demo_repair/results"
ORCA_FINGERS = ("index", "middle", "ring", "pinky")
# Palm origin height of the Wuji fixed wrist, so the pen falls as far as there.
PALM_POS_W = np.array([0.0, 0.0, 0.4953])


def to_rot(q_wxyz):
  return Rotation.from_quat(np.roll(q_wxyz, -1, axis=-1))


def to_wxyz(rot):
  return np.roll(rot.as_quat(), 1, axis=-1)


def anatomical_frame(knuckle_index, knuckle_pinky, mcp, pip):
  finger = (pip - mcp) / np.linalg.norm(pip - mcp)
  lateral = knuckle_pinky - knuckle_index
  lateral -= finger * (lateral @ finger)
  lateral /= np.linalg.norm(lateral)
  return np.column_stack([finger, lateral, np.cross(finger, lateral)])


def wuji_features(w):
  links = list(w["links"])
  zero = w["zero_pose_link_pos_palm"]
  at = lambda name: zero[links.index(name)]
  knuckles = np.stack([at(f"right_finger{f}_link1") for f in range(2, 6)])
  frame = anatomical_frame(
    knuckles[0], knuckles[3], at("right_finger3_link1"), at("right_finger3_link3")
  )
  return frame, knuckles.mean(0), np.linalg.norm(knuckles[3] - knuckles[0]), np.linalg.norm(
    at("right_finger3_link3") - at("right_finger3_link1")
  )


def orca_features(m):
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  palm = m.body("right_palm").id
  rot, pos = d.xmat[palm].reshape(3, 3), d.xpos[palm]
  at = lambda name: rot.T @ (d.xpos[m.body(name).id] - pos)
  knuckles = np.stack([at(f"right_{f}_mp") for f in ORCA_FINGERS])
  frame = anatomical_frame(
    knuckles[0], knuckles[3], at("right_middle_mp"), at("right_middle_ip")
  )
  return frame, knuckles.mean(0), np.linalg.norm(knuckles[3] - knuckles[0]), np.linalg.norm(
    at("right_middle_ip") - at("right_middle_mp")
  )


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", default="01_single_axis/clip_003")
  a = ap.parse_args()
  tag = a.clip.replace("/", "__")
  w = np.load(RESULTS / "wuji" / f"{tag}.npz")

  base = mujoco.MjModel.from_xml_path(str(BASE_SCENE))
  frame_w, centre_w, width_w, finger_w = wuji_features(w)
  frame_o, centre_o, width_o, finger_o = orca_features(base)
  q_map = frame_o @ frame_w.T
  # finger direction, lateral, palm normal
  scale = np.array([finger_o / finger_w, width_o / width_w, finger_o / finger_w])

  palm_rot_w = to_rot(w["fixed_wrist_pose_w"][3:]) * Rotation.from_matrix(q_map).inv()
  gravity_palm_o = palm_rot_w.inv().apply([0.0, 0.0, -1.0])
  assert np.allclose(gravity_palm_o, q_map @ w["gravity_palm"], atol=1e-9)

  pen = w["pen_pose_palm"]
  offset = (pen[:, :3] - centre_w) @ frame_w
  pen_pos_palm = centre_o + (offset * scale) @ frame_o.T
  pen_rot_palm = Rotation.from_matrix(q_map) * to_rot(pen[:, 3:])
  pen_pos_w = PALM_POS_W + palm_rot_w.apply(pen_pos_palm)
  pen_quat_w = to_wxyz(palm_rot_w * pen_rot_palm)

  out = RESULTS / "aligned" / tag
  out.mkdir(parents=True, exist_ok=True)
  tree = rebase_scene(BASE_SCENE, out / "scene.xml")
  tower = tree.find(".//body[@name='right_tower']")
  palm = tree.find(".//body[@name='right_palm']")
  assert palm.get("quat") is None
  palm_offset = np.fromstring(palm.get("pos"), sep=" ")
  tower.set("pos", " ".join(f"{v:.6f}" for v in PALM_POS_W - palm_rot_w.apply(palm_offset)))
  tower.set("quat", " ".join(f"{v:.8f}" for v in to_wxyz(palm_rot_w)))
  tree.write(out / "scene.xml", encoding="unicode")

  m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  palm_id = m.body("right_palm").id
  assert np.allclose(d.xpos[palm_id], PALM_POS_W, atol=1e-6)
  assert np.allclose(d.xmat[palm_id].reshape(3, 3), palm_rot_w.as_matrix(), atol=1e-6)

  np.savez_compressed(
    out / "mapped.npz",
    clip=a.clip,
    frame_dt=float(w["frame_dt"]),
    object_pose_w=np.concatenate([pen_pos_w, pen_quat_w], axis=1),
    object_pose_palm=np.concatenate([pen_pos_palm, to_wxyz(pen_rot_palm)], axis=1),
    wuji_contacts=w["contacts"],
    wuji_contact_push_palm=w["contact_push_palm"] @ q_map.T,
    wuji_contact_min_dist=w["contact_min_dist"],
    wuji_links=w["links"],
  )
  info = {
    "clip": a.clip,
    "wuji_palm_to_orca_palm_rotation": q_map.round(6).tolist(),
    "rotation_angle_deg": float(np.degrees(Rotation.from_matrix(q_map).magnitude())),
    "anatomical_scale_finger_lateral_normal": scale.round(4).tolist(),
    "knuckle_width_m": {"wuji": float(width_w), "orca": float(width_o)},
    "mcp_to_pip_m": {"wuji": float(finger_w), "orca": float(finger_o)},
    "gravity_in_palm": {
      "wuji_frame": w["gravity_palm"].round(4).tolist(),
      "orca_frame": gravity_palm_o.round(4).tolist(),
    },
    "orca_palm_pose_w": PALM_POS_W.tolist() + to_wxyz(palm_rot_w).round(8).tolist(),
  }
  (out / "alignment.json").write_text(json.dumps(info, indent=2))
  print(json.dumps(info, indent=2))


if __name__ == "__main__":
  main()
