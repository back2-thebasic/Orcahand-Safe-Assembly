"""The hand as the control loop sees it: policy-order joint angles in radians.

``OrcaCoreHand`` wraps any orca_core ``BaseHand``: the real ``OrcaHand``, its
``MockOrcaHand``, or the MuJoCo twin's ``SimOrcaHand``. Every command goes
through orca_core's ``set_joint_positions`` and with it orca_core's ROM clamp,
so the twin exercises the same path as the hardware.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from penspin_deploy.joint_map import JointMap

V1_RIGHT = ("v1", "orcahand-right")


class OrcaCoreHand:
    def __init__(self, hand, joint_map: JointMap, *, wrist_deg: float | None = 0.0):
        self.hand = hand
        self.map = joint_map
        # The policy has no wrist. Training fixes the palm, so the wrist motor is
        # held at one angle; None leaves it uncommanded.
        self.wrist_deg = wrist_deg
        self.has_wrist = "wrist" in hand.config.joint_ids

    def read(self) -> np.ndarray:
        return self.map.from_orca(self.hand.get_joint_position().as_dict())

    def command(self, qpos: np.ndarray) -> None:
        target = self.map.to_orca(qpos)
        if self.has_wrist and self.wrist_deg is not None:
            target["wrist"] = float(self.wrist_deg)
        self.hand.set_joint_positions(target)

    def rom_problems(self, bundle) -> list[str]:
        return self.map.check_against_roms(bundle, self.hand.config.joint_roms_dict)

    def close(self) -> None:
        disconnect = getattr(self.hand, "disconnect", None)
        if disconnect is not None:
            disconnect()


def connect_orca(*, mock: bool, config_path: str | Path | None = None):
    """Connect an orca_core v1 right hand and make it ready, without calibrating.

    A calibration run sweeps every joint to its hard stops; it is never started
    from here. An uncalibrated hand is refused: calibrate it with orca_core first.
    """
    from orca_core.hand_factory import load_hand

    if config_path is None:
        version, name = V1_RIGHT
        hand = load_hand(model_version=version, model_name=name, mock=mock, engage_feedback=False)
    else:
        hand = load_hand(config_path=str(config_path), mock=mock, engage_feedback=False)
    ok, message = hand.connect(interactive=False)
    if not ok:
        raise RuntimeError(f"orca_core connect failed: {message}")
    if not hand.calibrated:
        hand.disconnect()
        raise RuntimeError("the hand is not calibrated; run orca_core calibration first")
    # Torque on, control mode and current limit from config.yaml; no move.
    hand.init_joints(force_calibrate=False, move_to_neutral=False)
    return hand
