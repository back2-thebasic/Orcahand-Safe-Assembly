"""Wuji Hand 2 link -> OrcaHand v1 body correspondence and geometry helpers."""

import mujoco
import numpy as np

ORCA_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PEN_GEOMS = ("wuji_pen_cube", "wuji_pen_A_bottom", "wuji_pen_A_top")


def wuji_to_orca_body(wuji_link):
  """OrcaHand fingers have one distal body (``ip``) where Wuji has link3 + link4."""
  if wuji_link == "right_palm_link":
    return "right_palm"
  finger, part = wuji_link.removeprefix("right_finger").split("_", 1)
  name = ORCA_FINGERS[int(finger) - 1]
  if name == "thumb":
    orca = {"link1": "mp", "link2": "pp", "link3": "ip", "link4": "dp", "tip_sensor": "dp"}
  else:
    orca = {"link1": "mp", "link2": "pp", "link3": "ip", "link4": "ip", "tip_sensor": "ip"}
  return f"right_{name}_{orca[part]}"


class PenDistance:
  """Signed distance between each OrcaHand collision body and the pen."""

  def __init__(self, m, distmax=0.05):
    self.m = m
    self.distmax = distmax
    self.pen_geoms = [m.geom(n).id for n in PEN_GEOMS]
    pen_body = m.body("pen").id
    self.bodies = [
      i
      for i in range(1, m.nbody)
      if i != pen_body and m.body(i).name.startswith("right_")
    ]
    self.body_geoms = {
      b: [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])
      ]
      for b in self.bodies
    }
    self._fromto = np.zeros(6)

  def body_names(self):
    return [self.m.body(b).name for b in self.bodies]

  def __call__(self, d):
    """Per body: signed distance, the closest hand point, and the push
    direction on the pen there (from the hand point toward the shaft axis)."""
    out = np.full(len(self.bodies), self.distmax)
    push = np.zeros((len(self.bodies), 3))
    point = np.zeros((len(self.bodies), 3))
    pen = self.m.body("pen").id
    centre, axis = d.xpos[pen], d.xmat[pen].reshape(3, 3)[:, 0]
    for i, b in enumerate(self.bodies):
      for g in self.body_geoms[b]:
        for p in self.pen_geoms:
          dist = mujoco.mj_geomDistance(self.m, d, g, p, self.distmax, self._fromto)
          if dist < out[i]:
            out[i] = dist
            point[i] = self._fromto[:3]
            on_axis = centre + axis * ((point[i] - centre) @ axis)
            radial = on_axis - point[i]
            push[i] = radial / max(np.linalg.norm(radial), 1e-9)
    return out, push, point

  def side_error(self, d, point, push_target, radius):
    """Offset of each hand point from the shaft surface point on the side
    given by ``push_target`` (well-conditioned even from the wrong side)."""
    pen = self.m.body("pen").id
    centre, axis = d.xpos[pen], d.xmat[pen].reshape(3, 3)[:, 0]
    on_axis = centre + np.outer((point - centre) @ axis, axis)
    return point - (on_axis - push_target * radius)


def pen_contacts(m, d, threshold=0.0005):
  """Bodies in MuJoCo contact with the pen after mj_forward/mj_step."""
  pen = m.body("pen").id
  touched = set()
  for c in d.contact[: d.ncon]:
    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
    if pen in (b1, b2) and c.dist < threshold:
      touched.add(m.body(b2 if b1 == pen else b1).name)
  return sorted(touched)
