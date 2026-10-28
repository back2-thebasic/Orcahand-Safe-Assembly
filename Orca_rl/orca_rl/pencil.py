"""Spin a pencil about the vertical axis on the palm-up ORCA hand.

The pencil lies across the palm and fingers; the task is to keep turning it
about the world z axis (counter-clockwise seen from above by default) without
dropping it. There are no goals and no success bonus: the reward is the yaw
the pencil actually gains, so an episode's return is about the number of
radians it was turned.

Three things about the reward are deliberate, and ``pencil_checks.py`` tests
each of them with a scripted policy:

1. **Yaw is measured on the pencil's axis, not on its angular velocity.**
   It is the turn of the axis's horizontal projection. A pencil rolling about
   its own long axis -- the cheapest motion there is for a round pencil on a
   palm -- leaves the axis where it is and earns nothing, and neither does a
   pencil stood on end and twirled.

2. **Only forward progress is capped, and the excess carries over.** Each
   step pays at most ``max_spin_rate * dt``; backward yaw is always billed in
   full. Capping both directions would let a policy creep forward slowly
   (paid in full) and snap back fast (billed only up to the cap): net
   rotation zero, net reward positive. Forward yaw over the cap is carried,
   up to ``spin_carry_rad``, and spent against the next steps -- above all
   the backward half of a vibration. Without the carry a pencil that only
   shivers is charged for every shiver: measured under random actions, the
   net yaw was -0.15 rad and a plain per-step cap paid -3.54. Either way the
   return can never exceed the net rotation.

3. **Off-hand motion is billed on return.** While the pencil is not held, or
   tilted past ``max_tilt_deg``, the reference yaw is frozen. When it comes
   back, the whole change since the freeze is settled at once -- otherwise a
   policy could pay itself for the forward half of a toss and never be
   charged for the way back.

Two options, off by default, go after where pencil1 and pencil2 got stuck:
both turned the pencil ~100 degrees from wherever it started, then parked it
somewhere that costs nothing -- pencil1 in a fist, pencil2 stood up in a gap
between the fingers, where yaw does not count. ``max_stand_deg`` ends the
episode, with the drop penalty, once the pencil tilts that far.
``stall_start_prob`` restarts that share of episodes from a state the policy
itself stalled in, so that the step it is missing -- getting past its own
parking spot -- is where a large part of its experience starts.

Shares the hand, the action mapping, the fingertip features and the physics
randomization with ``CubeReorientContinuous``; see ``task.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from orca_sim.task_envs import OrcaHandRightCubeOrientation
from orca_rl.task import ENV_KWARGS_FILE, CubeReorientContinuous

# The cube task's pose with the fingers brought together and the wrist bent
# back. In the cube pose the fingertips splay ~3 cm apart: of 6 initial yaws
# tried, the pencil fell through the gaps at 60, 90 and 120 degrees. With
# these it stayed on the hand at all 12 yaws tried (3 s settle each), resting
# 3-4 degrees off level. The wrist servo sags under the hand's weight,
# -0.30 -> -0.21, which leaves the palm about level.
#
# Not every yaw is stable for long: lying along the fingers (~15 and ~165
# degrees) with the hand idle, the pencil slips into a gap between them in
# 4-8 s, so the zero policy drops ~10% of 10 s episodes. Pressing the fingers
# tighter makes them collide with each other instead.
PENCIL_HAND_POSE = {
    "right_wrist": -0.30,
    "right_index_abd": -0.15,
    "right_middle_abd": 0.0,
    "right_ring_abd": -0.05,
    "right_pinky_abd": 0.08,
}


SCENE_TEMPLATE = Path(__file__).with_name("scenes") / "scene_right_pencil_spin_v1.xml"


def pencil_scene_path() -> Path:
    """The pencil scene, its includes pointed at the installed orca_sim.

    The scene ships with orca_rl so that orca_sim stays the unmodified
    upstream package. MuJoCo resolves an include relative to the file that
    includes it, so the template names orca_sim's hand and floor by absolute
    path, and the filled-in copy goes to the temp directory. The name carries
    a hash of the content; the write is an atomic rename, so parallel envs
    starting together never read half a file.
    """
    from orca_sim.versions import PACKAGE_ROOT

    text = SCENE_TEMPLATE.read_text().replace("{orca_sim}", PACKAGE_ROOT.as_posix())
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / "orca_rl" / f"scene_right_pencil_spin_v1_{digest}.xml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return path


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PencilSpin(OrcaHandRightCubeOrientation):
    """Continuous pencil spinning about the world z axis.

    Observation (90,):
        hand qpos (17) | pencil pos (3) | pencil quat (4)
        hand qvel (17) | pencil linvel (3) | pencil angvel, body frame (3)
        pencil axis, world (3) | pencil angvel, world (3)
        servo targets in [-1, 1] (17)
        fingertip offsets from the pencil centre (15) | per-finger contact (5)

    Action (17,), always in [-1, 1]; a delta on the previous servo target in
    the default relative mode, exactly as in the cube task.
    """

    # The cube task's helpers only touch the hand, the controller and
    # `_cube_body_id` / `_cube_geom_ids`, which here name the pencil.
    _setup_fingers = CubeReorientContinuous._setup_fingers
    _apply_randomization = CubeReorientContinuous._apply_randomization
    _settle_cube = CubeReorientContinuous._settle_cube
    _target_from_action = CubeReorientContinuous._target_from_action

    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = "v1",
        *,
        max_episode_steps: int = 1000,
        # --- reward ---
        spin_direction: str = "ccw",  # seen from above: "ccw" (+z) | "cw"
        max_spin_rate: float = 1.0,   # rad/s; forward yaw above this is not paid
        spin_carry_rad: float = 0.2,  # ...except this much, carried to later steps
        spin_coef: float = 1.0,
        drop_penalty: float = 2.0,
        pose_penalty: float = 0.0,    # per rad^2 of joint-angle distance from the settled pose
        action_rate_penalty: float = 0.002,
        # --- what counts as holding the pencil ---
        max_tilt_deg: float = 45.0,
        max_stand_deg: float | None = None,  # tilted past this ends the episode, as a drop
        in_hand_height: float = 0.13,
        drop_height: float = 0.12,
        # --- actions ---
        action_mode: str = "relative",  # "relative" | "absolute"
        action_scale: float = 0.15,
        # --- domain randomization (same knobs as the cube task) ---
        randomize_physics: bool = False,
        randomization: dict[str, float] | None = None,
        # --- reset ---
        reset_settle_steps: int = 100,
        initial_yaw: str = "random",  # "random" | "nominal"
        pencil_pos_xy_jitter: float | tuple[float, float] = 0.005,
        # Resume this share of resets from a state where an earlier episode
        # stalled: held, but less than `stall_turn_deg` turned over the last
        # `stall_window_steps`. Each env keeps its own buffer.
        stall_start_prob: float = 0.0,
        stall_buffer_size: int = 64,
        stall_window_steps: int = 100,
        stall_turn_deg: float = 10.0,
    ) -> None:
        if spin_direction not in {"ccw", "cw"}:
            raise ValueError("spin_direction must be 'ccw' or 'cw'")
        if action_mode not in {"relative", "absolute"}:
            raise ValueError("action_mode must be 'relative' or 'absolute'")
        if initial_yaw not in {"random", "nominal"}:
            raise ValueError("initial_yaw must be 'random' or 'nominal'")
        if version != "v1":
            raise ValueError("there is only a v1 pencil scene (the physical hand is v1)")
        if max_stand_deg is not None and max_stand_deg < max_tilt_deg:
            raise ValueError("max_stand_deg below max_tilt_deg would end episodes the reward still pays for")
        if not 0.0 <= stall_start_prob <= 1.0:
            raise ValueError("stall_start_prob must be in [0, 1]")

        self.spin_direction = spin_direction
        self._spin_sign = 1.0 if spin_direction == "ccw" else -1.0
        self.max_spin_rate = float(max_spin_rate)
        self.spin_carry_rad = float(spin_carry_rad)
        self.spin_coef = float(spin_coef)
        self.drop_penalty = float(drop_penalty)
        self.pose_penalty = float(pose_penalty)
        self.action_rate_penalty = float(action_rate_penalty)
        self.max_tilt_rad = float(np.deg2rad(max_tilt_deg))
        self.max_stand_rad = None if max_stand_deg is None else float(np.deg2rad(max_stand_deg))
        self.in_hand_height = float(in_hand_height)
        self.action_mode = action_mode
        self.action_scale = float(action_scale)
        self.randomize_physics = bool(randomize_physics)
        self.randomization = {
            "cube_mass": 0.30,        # the pencil's; names shared with task.py
            "cube_friction": 0.30,
            "joint_damping": 0.30,
            "actuator_gain": 0.20,
            "obs_noise_rad": 0.005,
            **(randomization or {}),
        }
        self.reset_settle_steps = int(reset_settle_steps)
        self.max_reset_attempts = 10
        self.max_reset_tilt_rad = float(np.deg2rad(15.0))
        self.initial_yaw = initial_yaw
        self._obs_noise_rad = 0.0
        self.stall_start_prob = float(stall_start_prob)
        self.stall_buffer_size = int(stall_buffer_size)
        self.stall_window_steps = int(stall_window_steps)
        self.stall_turn_rad = float(np.deg2rad(stall_turn_deg))
        # Everything mj_step reads, so a resumed state steps exactly as the
        # original would have.
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._stall_states: list[tuple[np.ndarray, np.ndarray]] = []  # (state, rest pose)
        self._stall_slot = 0
        self._stall_start = False

        super().__init__(
            render_mode=render_mode,
            version=version,
            # Absolute, so orca_sim's lookup under scenes/<version>/ returns it as is.
            scene_file=str(pencil_scene_path()),
            cube_joint_name="pencil_freejoint",
            cube_body_name="pencil",
            hand_pose_by_joint=PENCIL_HAND_POSE,
            cube_pos_xy_jitter=pencil_pos_xy_jitter,
            max_episode_steps=max_episode_steps,
            drop_height=drop_height,
        )
        self.version = "v1"  # the base class names it after the scene's folder
        self.dt = float(self.model.opt.timestep * self.frame_skip)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self._ctrl_center = 0.5 * (self.action_high + self.action_low)
        self._ctrl_halfspan = 0.5 * (self.action_high - self.action_low)

        m = self.model
        self._cube_geom_ids = {
            g for g in range(m.ngeom)
            if int(m.geom_bodyid[g]) == self._cube_body_id
            and (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))
        }
        self._setup_fingers()
        # Holding means touching the palm or a finger. The tower behind the
        # palm is part of the hand too, but a pencil parked on it is not held.
        self._palm_body_id = m.body("right_palm").id

        self._nominal = {
            "body_mass": m.body_mass.copy(),
            "geom_friction": m.geom_friction.copy(),
            "dof_damping": m.dof_damping.copy(),
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
        }

        self._prev_target = np.zeros(m.nu, dtype=np.float64)
        self._rest_q = np.zeros(m.nu, dtype=np.float64)
        self._reset_tracking()

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64
        )

    # ------------------------------------------------------------ pencil state

    def _axis(self) -> np.ndarray:
        """Unit vector from the eraser to the tip, world frame."""
        return self.data.xmat[self._cube_body_id].reshape(3, 3)[:, 0].copy()

    def _yaw(self) -> float:
        axis = self._axis()
        return float(np.arctan2(axis[1], axis[0]))

    def _tilt_rad(self) -> float:
        return float(np.arcsin(np.clip(abs(self._axis()[2]), 0.0, 1.0)))

    def _stood_up(self) -> bool:
        return self.max_stand_rad is not None and self._tilt_rad() > self.max_stand_rad

    def _angvel_world(self) -> np.ndarray:
        # A free joint's rotational qvel is in the body frame.
        rot = self.data.xmat[self._cube_body_id].reshape(3, 3)
        return rot @ self._cube_qvel()[3:]

    def _contacts(self) -> tuple[bool, np.ndarray]:
        """Whether the palm or any finger touches the pencil, and which fingers."""
        held = False
        fingers = np.zeros(len(self._finger_body_sets))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self._cube_geom_ids:
                other = g2
            elif g2 in self._cube_geom_ids:
                other = g1
            else:
                continue
            body = int(self.model.geom_bodyid[other])
            if body == self._palm_body_id:
                held = True
            for k, bodies in enumerate(self._finger_body_sets):
                if body in bodies:
                    fingers[k] = 1.0
                    held = True
        return held, fingers

    def _in_hand(self) -> bool:
        if float(self.data.xpos[self._cube_body_id, 2]) <= self.in_hand_height:
            return False
        return self._contacts()[0]

    def _resolve_initial_cube_quat(self, options: dict[str, Any]) -> np.ndarray:
        if "cube_quat" in options:
            return self._normalize_quat(np.asarray(options["cube_quat"], dtype=np.float64))
        yaw = self.np_random.uniform(0.0, 2.0 * np.pi) if self.initial_yaw == "random" else 0.0
        return self._quat_from_axis_angle(self.WORLD_UP, float(yaw))

    # ------------------------------------------------------------------ spaces

    def _get_obs(self) -> np.ndarray:
        base = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
        if not hasattr(self, "_tip_geom_ids"):
            return base  # during the parents' __init__
        if self._obs_noise_rad > 0.0:
            n_hand = self._cube_qpos_adr
            base[:n_hand] += self.np_random.normal(scale=self._obs_noise_rad, size=n_hand)
        centre = self.data.xpos[self._cube_body_id]
        tips = np.concatenate([self.data.geom_xpos[g] - centre for g in self._tip_geom_ids])
        return np.concatenate([
            base,
            self._axis(),
            self._angvel_world(),
            (self._prev_target - self._ctrl_center) / self._ctrl_halfspan,
            tips,
            self._contacts()[1],
        ])

    def _get_info(self) -> dict[str, Any]:
        steps = max(int(self._elapsed_steps), 1)
        return {
            "yaw_deg": float(np.degrees(self._yaw())),
            "tilt_deg": float(np.degrees(self._tilt_rad())),
            "pencil_pos": self._cube_pos(),
            "pencil_height": float(self.data.xpos[self._cube_body_id, 2]),
            "in_hand": self._in_hand(),
            "dropped": self._cube_dropped(),
            "stood_up": self._stood_up(),
            "stall_start": self._stall_start,
            "elapsed_steps": self._elapsed_steps,
            # Net yaw in the commanded direction over the steps that counted,
            # and the part of it the reward paid for (the rest was over the cap).
            "turned_rad": self._turned,
            "turned_revs": self._turned / (2.0 * np.pi),
            "paid_rad": self._paid,
            "in_hand_frac": self._steps_in_hand / steps,
        }

    # ------------------------------------------------------------------- gym API

    def _reset_tracking(self) -> None:
        self._yaw_ref = self._yaw()
        self._ref_age = 0
        self._turned = 0.0
        self._paid = 0.0
        self._carry = 0.0
        self._steps_in_hand = 0
        self._turned_history: deque[float] = deque(maxlen=self.stall_window_steps + 1)
        self._stall_saved = False

    def _save_stall(self) -> None:
        state = np.empty(mujoco.mj_stateSize(self.model, self._state_spec))
        mujoco.mj_getState(self.model, self.data, state, self._state_spec)
        entry = (state, self._rest_q.copy())
        if len(self._stall_states) < self.stall_buffer_size:
            self._stall_states.append(entry)
        else:
            self._stall_states[self._stall_slot] = entry
            self._stall_slot = (self._stall_slot + 1) % self.stall_buffer_size
        self._stall_saved = True

    def _resume_stall(self) -> None:
        state, rest_q = self._stall_states[int(self.np_random.integers(len(self._stall_states)))]
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state, self._state_spec)
        mujoco.mj_forward(self.model, self.data)
        self._elapsed_steps = 0
        # The episode the state came from charged the pose against its own
        # settled hand; keep doing so, or a stalled pose would cost nothing.
        self._rest_q = rest_q.copy()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        self._apply_randomization()
        options = {} if options is None else options
        # Short-circuits before drawing from np_random unless there is a
        # buffer, so with the option off a seed resets exactly as it always did.
        self._stall_start = bool(
            self._stall_states and not options
            and self.np_random.random() < self.stall_start_prob
        )
        if self._stall_start:
            self._resume_stall()
        else:
            self._fresh_reset(seed, options)
        # Continue from the targets the hand held while settling; re-deriving
        # them from the sagged qpos would let every joint go slack on step one.
        self._prev_target = np.asarray(self.data.ctrl, dtype=np.float64).copy()
        self._reset_tracking()
        return self._get_obs(), self._get_info()

    def _fresh_reset(self, seed: int | None, options: dict[str, Any]) -> None:
        pinned = "cube_quat" in options and "cube_pos" in options
        for attempt in range(self.max_reset_attempts):
            # Only the first attempt is seeded; reseeding every retry would
            # redraw the same failing spawn.
            super().reset(seed=seed if attempt == 0 else None, options=options)
            # The pencil drops 8 mm onto the palm and the wrist sags ~0.1 rad
            # under the hand's weight. After 100 steps the yaw drifts 0.6
            # degrees on average over the next 2 s (max 2.5); after 60 a pencil
            # can still be rolling about its axis at 26 rad/s.
            self._settle_cube()
            # Some spawns are not a start at all: around 45 degrees of yaw with
            # the pencil shifted toward the thumb it spawns inside the thumb
            # and is thrown to the floor while settling, and an episode that
            # ends on step one teaches nothing. Measured over 400 random
            # resets, 5.8% needed a redraw and none needed more than two.
            if pinned or (self._in_hand() and self._tilt_rad() <= self.max_reset_tilt_rad):
                break
        # Where the hand settled, not where it was commanded: the wrist sags
        # 0.09 rad under the hand's weight, and charging an idle hand for
        # gravity would make doing nothing cost something.
        self._rest_q = self.data.qpos[self._actuator_qpos_indices].copy()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = self._target_from_action(action)
        rate = (target - self._prev_target) / np.maximum(self._ctrl_halfspan, 1e-8)
        self._prev_target = target

        self.data.ctrl[:] = target
        time_before = self.data.time
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._elapsed_steps += 1
        self._ref_age += 1

        # On a NaN/Inf in the state MuJoCo logs a warning and resets mjData to
        # the model defaults: hand and pencil teleport to the XML pose and the
        # clock restarts. Nothing measured on this step is real, so end the
        # episode here, unscored. Seen once in the 3M-step smoke run and twice
        # in pencil1's 30M.
        # (The warning counters are cleared by that same reset; the clock is not
        # something this env ever winds back.)
        if self.data.time < time_before:
            info = self._get_info()
            info["spin_this_step"] = 0.0
            info["sim_unstable"] = True
            return self._get_obs(), 0.0, False, True, info

        in_hand = self._in_hand()
        dropped = self._cube_dropped()
        self._steps_in_hand += int(in_hand)

        # Yaw gained since the last step that counted. `_ref_age` is how many
        # steps that spans, so the cap stays a cap on the *rate*. The carry is
        # never negative, so nothing here pays more than the yaw gained.
        spin = paid = 0.0
        if in_hand and self._tilt_rad() <= self.max_tilt_rad:
            yaw = self._yaw()
            spin = self._spin_sign * _wrap(yaw - self._yaw_ref)
            gained = spin + self._carry
            paid = min(gained, self.max_spin_rate * self.dt * self._ref_age)
            self._carry = min(gained - paid, self.spin_carry_rad)
            self._yaw_ref = yaw
            self._ref_age = 0
            self._turned += spin
            self._paid += paid

        reward = self.spin_coef * paid
        reward -= self.action_rate_penalty * float(np.sum(rate**2))
        # Off by default. pencil1 (without it) turned the pencil ~90 degrees in
        # the first 2.5 s of an episode, then closed the hand into a fist around
        # it -- fingers flexed 1.0 rad on average, 75% of servo targets on a
        # rail -- and held still for the rest: clamped earns 0 but cannot drop,
        # and regrasping to keep turning can. This makes the fist cost
        # something. results/2026-10-14-pencil1.md.
        if self.pose_penalty:
            q = self.data.qpos[self._actuator_qpos_indices]
            reward -= self.pose_penalty * float(np.sum((q - self._rest_q) ** 2))
        # Off by default. pencil2 (pose penalty on) stopped making fists and
        # instead let one end of the pencil slip into a gap between the fingers
        # and stand there, 67 degrees up at the median: past max_tilt_deg it
        # earns nothing and costs nothing, the same free parking spot as the
        # fist. results/2026-10-26-pencil2.md.
        stood_up = self._stood_up()
        if dropped or stood_up:
            reward -= self.drop_penalty

        terminated = bool(dropped or stood_up)
        truncated = bool(self._elapsed_steps >= self.max_episode_steps)

        # Remember the first stall of each fresh episode. Resumed episodes do
        # not add to the buffer, so it stays a sample of where the policy gets
        # stuck from an ordinary start, not of where it drifts after many
        # resumes.
        if self.stall_start_prob > 0.0 and not self._stall_start and not self._stall_saved:
            self._turned_history.append(self._turned)
            history = self._turned_history
            if (not terminated and in_hand and len(history) == history.maxlen
                    and history[-1] - history[0] < self.stall_turn_rad):
                self._save_stall()

        info = self._get_info()
        info["spin_this_step"] = spin
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info


def saved_pencil_kwargs(model_path: str | Path) -> dict[str, Any]:
    """The env_kwargs.json pencil_train.py wrote next to (or above) a model."""
    path = Path(model_path)
    for run_dir in (path.parent, path.parent.parent):
        saved = run_dir / ENV_KWARGS_FILE
        if saved.exists():
            kwargs = json.loads(saved.read_text())
            kwargs.pop("task", None)
            return kwargs
    raise FileNotFoundError(f"no {ENV_KWARGS_FILE} next to {model_path}")
