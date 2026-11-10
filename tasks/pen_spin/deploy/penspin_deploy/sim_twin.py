"""MuJoCo twin of the training scene, standing in for the hand and the camera.

``Twin``          the compiled training scene (scene.mjb): OrcaHand v1 position
                  actuators, the 250 mm pen, 0.005 s steps, 4 per control tick.
``SimOrcaHand``   an orca_core ``BaseHand`` over the twin: degrees in and out,
                  orca_core's ROM clamp, so the runtime drives it exactly as it
                  drives ``OrcaHand``.
``SimPenCamera``  the pen pose as the camera pipeline would deliver it, through
                  the training camera model (latency, calibration bias, jitter,
                  dropout bursts), or with the latency only.
"""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np
from orca_core.base_hand import BaseHand
from orca_core.joint_position import OrcaJointPositions

from penspin_deploy.bundle import Bundle
from penspin_deploy.geometry import (
    pose_from_frame,
    pose_in_frame,
    quat_from_axis_angle,
    quat_mul,
    quat_unique,
)
from penspin_deploy.hand import V1_RIGHT
from penspin_deploy.joint_map import JointMap


class Twin:
    def __init__(self, bundle: Bundle, *, actuation_delay_s: float = 0.0):
        m = mujoco.MjModel.from_binary_path(str(bundle.path / "scene.mjb"))
        self.m, self.d = m, mujoco.MjData(m)
        self.decimation = int(round(bundle.ctrl_dt / m.opt.timestep))
        if abs(self.decimation * m.opt.timestep - bundle.ctrl_dt) > 1e-9:
            raise ValueError("scene timestep does not divide the control period")
        joints = [m.joint(f"robot/{name}").id for name in bundle.joint_names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.vadr = np.array([m.jnt_dofadr[j] for j in joints])
        self.act = np.array([m.actuator(f"robot/{name}_actuator").id for name in bundle.joint_names])
        self.palm = m.body(bundle.contract["palm"]["body"]).id
        pen = m.body(bundle.contract["pen"]["body"]).id
        pen_joint = m.body_jntadr[pen]
        if m.jnt_type[pen_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("the pen body has no free joint")
        self.pen = pen
        self.pen_qadr = m.jnt_qposadr[pen_joint]
        self.pen_vadr = m.jnt_dofadr[pen_joint]
        # Commands reach the actuators this many physics steps after they are sent.
        self.delay_steps = int(round(actuation_delay_s / m.opt.timestep))
        self._queue: deque[tuple[int, np.ndarray]] = deque()
        self._held_pen: np.ndarray | None = None
        self.steps = 0
        mujoco.mj_forward(m, self.d)

    @property
    def time(self) -> float:
        return float(self.d.time)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mujoco.mj_forward(self.m, self.d)

    def set_hand(self, qpos16: np.ndarray) -> None:
        self.d.qpos[self.qadr] = qpos16
        self.d.qvel[self.vadr] = 0.0
        self.d.ctrl[self.act] = qpos16
        mujoco.mj_forward(self.m, self.d)

    def set_ctrl(self, ctrl16: np.ndarray) -> None:
        ctrl16 = np.asarray(ctrl16, dtype=np.float64).copy()
        if self.delay_steps == 0:
            self.d.ctrl[self.act] = ctrl16
        else:
            self._queue.append((self.steps + self.delay_steps, ctrl16))

    def step_control(self) -> None:
        for _ in range(self.decimation):
            while self._queue and self._queue[0][0] <= self.steps:
                self.d.ctrl[self.act] = self._queue.popleft()[1]
            mujoco.mj_step(self.m, self.d)
            if self._held_pen is not None:
                self._pin_pen()
            self.steps += 1
        # As mjlab: kinematics refreshed once after the decimation loop.
        mujoco.mj_forward(self.m, self.d)

    def qpos(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def palm_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.xpos[self.palm].copy(), self.d.xquat[self.palm].copy()

    def pen_local(self) -> np.ndarray:
        """True pen pose in the palm frame (7: position, wxyz)."""
        palm_pos, palm_quat = self.palm_pose()
        return pose_in_frame(palm_pos, palm_quat, self.d.xpos[self.pen], self.d.xquat[self.pen])

    def place_pen(self, pose_local: np.ndarray) -> None:
        """Put the pen at a palm-frame pose and keep it there at rest, as the
        operator's fingers would, until release_pen()."""
        pos, quat = pose_from_frame(*self.palm_pose(), pose_local)
        self._held_pen = np.concatenate((pos, quat / np.linalg.norm(quat)))
        self._pin_pen()
        mujoco.mj_forward(self.m, self.d)

    def release_pen(self) -> None:
        self._held_pen = None

    def _pin_pen(self) -> None:
        self.d.qpos[self.pen_qadr:self.pen_qadr + 7] = self._held_pen
        self.d.qvel[self.pen_vadr:self.pen_vadr + 6] = 0.0

    def unstable(self) -> bool:
        return bool(not np.isfinite(self.d.qpos).all() or np.abs(self.d.qvel).max() > 1e4)


class SimOrcaHand(BaseHand):
    """orca_core's view of the twin. The wrist is not in the training scene: its
    command is kept and read back."""

    def __init__(self, twin: Twin, joint_map: JointMap):
        version, name = V1_RIGHT
        super().__init__(model_version=version, model_name=name)
        self.twin = twin
        self.map = joint_map
        self._extra = {j: 0.0 for j in self.config.joint_ids if j not in joint_map.orca_ids}

    def _get_joint_positions(self) -> OrcaJointPositions:
        joints = self.map.to_orca(self.twin.qpos())
        joints.update(self._extra)
        return OrcaJointPositions.from_dict(joints)

    def _set_joint_positions(self, joint_pos: OrcaJointPositions) -> bool:
        joints = joint_pos.as_dict()
        for j in self._extra:
            if joints.get(j) is not None:
                self._extra[j] = float(joints[j])
        current = self.map.to_orca(self.twin.qpos())
        merged = {j: current[j] if joints.get(j) is None else joints[j] for j in self.map.orca_ids}
        self.twin.set_ctrl(self.map.from_orca(merged))
        return True


class SimPenCamera:
    """Pen pose samples as ObjectPoseCameraModel produces them on the training side.

    ``sample()`` is called once per control tick after the physics step, like the
    observation term: the true pose enters the history and the returned frame is
    ``age`` ticks old. ``training=False`` keeps the latency only.
    """

    def __init__(self, twin: Twin, model: dict, *, training: bool, seed: int = 0):
        self.twin = twin
        self.rng = np.random.default_rng(seed)
        self.lag = int(model["lag_ticks"][0])
        if model["lag_ticks"][0] != model["lag_ticks"][1]:
            raise ValueError("the runtime twin models a constant camera lag")
        self.training = training
        self.model = model
        self.max_age = self.lag + int(model["dropout_max_ticks"])
        self.history: deque[np.ndarray] = deque(maxlen=self.max_age + 1)
        self.age = 0
        self.holding = False
        self.hold_ticks = 0
        self.bias_pos = np.zeros(3)
        self.bias_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if training:
            self.bias_pos = self.rng.uniform(*model["bias_position_m"], size=3)
            self.bias_quat = quat_unique(quat_from_axis_angle(
                self.rng.normal(size=3), self.rng.uniform(*model["bias_orientation_rad"])))

    def reset(self) -> None:
        self.history.clear()
        self.age = 0
        self.holding = False
        self.hold_ticks = 0

    def _measure(self) -> np.ndarray:
        pose = self.twin.pen_local()
        pose[:3] += self.bias_pos
        pose[3:] = quat_mul(self.bias_quat, pose[3:])
        if self.training:
            pose[:3] += self.rng.uniform(*self.model["noise_position_m"], size=3)
            pose[3:] = quat_mul(quat_from_axis_angle(
                self.rng.normal(size=3), self.rng.uniform(*self.model["noise_orientation_rad"])), pose[3:])
        pose[3:] = quat_unique(pose[3:])
        return pose

    def sample(self) -> tuple[np.ndarray, int]:
        """(palm-frame pen pose, its age in ticks)."""
        self.history.appendleft(self._measure())
        if self.training and self.model["dropout_enter_prob"] > 0:
            roll = self.rng.random()
            capped = self.hold_ticks >= self.model["dropout_max_ticks"]
            if self.holding:
                self.holding = roll < self.model["dropout_continue_prob"] and not capped
            else:
                self.holding = roll < self.model["dropout_enter_prob"]
            self.hold_ticks = self.hold_ticks + 1 if self.holding else 0
        aged = self.age + 1
        age = aged if self.holding else min(self.lag, aged)
        self.age = max(min(age, len(self.history) - 1), 0)
        return self.history[self.age].copy(), self.age
