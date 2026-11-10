"""Policy joints (training MuJoCo qpos, radians) <-> orca_core joints (degrees).

    orca_deg = sign * degrees(qpos - offset)

Default: sign +1 and offset = the MJCF joint ``ref`` (qpos0), so orca_core's zero
is the pose the MJCF bodies are drawn in. Evidence: with that mapping the
orca_core v1 joint_roms reproduce the MJCF ranges exactly, including the
asymmetric ranges of the joints with a nonzero ref (thumb_abd [-20, 42] deg +
ref -0.733 rad = [-1.082, 0] rad, pinky_abd [-37, 37] deg + ref 0.523 =
[-0.122, 1.169]), and every orca_core neutral_position lands inside the MJCF range.
orca_teleop's sim sink drives the same MJCF without the offset; that cannot
hold the neutral pose (thumb_abd 42 deg = 0.733 rad is above the MJCF's 0 limit).
The convention still has to be confirmed on the hand: see ``check_against_roms``
and the per-joint override file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from penspin_deploy.bundle import Bundle


class JointMap:
    def __init__(self, bundle: Bundle, overrides: dict | None = None):
        self.orca_ids = list(bundle.orca_ids)
        self.sign = np.ones(bundle.num_joints)
        self.offset = bundle.qpos0.copy()
        for orca_id, entry in (overrides or {}).items():
            i = self.orca_ids.index(orca_id)
            self.sign[i] = float(entry.get("sign", self.sign[i]))
            if "offset_deg" in entry:
                self.offset[i] = np.radians(float(entry["offset_deg"]))
        if not np.all(np.isin(self.sign, (-1.0, 1.0))):
            raise ValueError("joint map signs must be +1 or -1")

    @classmethod
    def from_file(cls, bundle: Bundle, path: str | Path | None) -> "JointMap":
        """Overrides as JSON: {"thumb_abd": {"sign": -1, "offset_deg": -42.0}, ...}."""
        return cls(bundle, json.loads(Path(path).read_text()) if path else None)

    def to_orca(self, qpos: np.ndarray) -> dict[str, float]:
        deg = self.sign * np.degrees(np.asarray(qpos, dtype=np.float64) - self.offset)
        return dict(zip(self.orca_ids, deg.tolist()))

    def from_orca(self, joints: dict[str, float | None]) -> np.ndarray:
        missing = [j for j in self.orca_ids if joints.get(j) is None]
        if missing:
            raise RuntimeError(f"orca_core reports no position for {missing} (uncalibrated?)")
        deg = np.array([float(joints[j]) for j in self.orca_ids])
        return self.offset + np.radians(self.sign * deg)

    def check_against_roms(self, bundle: Bundle, roms: dict[str, tuple[float, float]]) -> list[str]:
        """Where the training range and the orca_core ROM disagree by more than 1 degree."""
        problems = []
        for i, orca_id in enumerate(self.orca_ids):
            lo, hi = sorted(self.sign[i] * np.degrees(bundle.ctrl_range[i] - self.offset[i]))
            rom_lo, rom_hi = roms[orca_id]
            if abs(lo - rom_lo) > 1.0 or abs(hi - rom_hi) > 1.0:
                problems.append(
                    f"{orca_id:11s} training range [{lo:7.1f}, {hi:7.1f}] deg"
                    f"  orca_core ROM [{rom_lo:6.1f}, {rom_hi:6.1f}] deg"
                )
        return problems
