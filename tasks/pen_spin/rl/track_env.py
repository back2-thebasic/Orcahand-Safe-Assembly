"""Residual reference tracking for pen spinning on the ORCA hand.

The policy does not drive the fingers from scratch: every control step it adds
a residual to the reference finger angles of a retargeted demonstration, and is
paid for keeping the pen (and the fingers) on the reference. This follows
Wuji's PenSpin task in wuji-mjlab (``SingleHandResidualEMAAction`` and the
rewards in ``mdp/rewards.py``), cut down to one clip and CPU MuJoCo:

- action: residual = clip(a) * action_scale rad, smoothed by an EMA
  (alpha 0.5), added to the reference angles of the frame being stepped to;
- reward (each term times dt, Wuji's weights): pen position (2, sigma 5 cm),
  pen axis points 10 cm either side of the centre (8, sigma 4 cm), link
  positions (1, sigma 2.5 cm), joint angles (1, sigma 0.1 of the joint's
  range), action rate and acceleration (-0.1), squared torque over the limit
  (-1), finger links touching another finger or the palm (-1 each);
- the episode ends when the pen centre is 5 cm off the reference or its axis
  60 degrees off, and is truncated at the end of the clip;
- reset: reference state initialization at a start frame drawn from 0.5 s bins,
  80% in proportion to where recent episodes failed and 20% uniformly; with
  probability hold_prob it instead starts at frame 0 and holds that frame for
  hold_s seconds first, the way a deployment starts.

The pen's velocity is in the observation. Without sim2real it is the
simulator's; with sim2real it is the finite difference of the camera poses the
policy is given, as a deployment would have to compute it. A deployment must
use exactly this estimator: every control tick, the frame on hand minus the
previous tick's, over 0.02 s, so zero while no new frame arrives (no
timestamp correction).

sim2real (SIM2REAL below) puts the real hand between the policy and the
simulator: each servo receives its target a few milliseconds late and holds it
in its own, slightly miscalibrated, encoder frame; the joint angles and speeds
read back are noisy; and the pen pose comes from a camera model (calibration
offset per episode, jitter per tick, a tick or two of pipeline latency, bursts
of dropped frames) as in Wuji's ObjectPoseCameraModel. Rewards and
terminations always use the true state.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
# The retargeted, optimized demonstration 003 (scene.xml and reference.npz), kept
# here since tasks/pen_spin/results/ was removed upstream. A relative `run` is
# taken from this directory, so runs/*/env_kwargs.json work from any checkout.
DEFAULT_RUN = Path("reference") / "optimized003-final"


def resolve_run(run: str | Path) -> Path:
    run = Path(run)
    return run if run.is_absolute() else HERE / run
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
AXIS_POINT_M = 0.10
# randomize_physics: (low, high) factors on the nominal values, drawn log-uniformly
# at every reset. Wuji's hand2 config spans pen mass 0.4-1.6x, friction 0.35-1.4
# (absolute) and kp 0.9-1.1x; these are a little wider on kp, as the ORCA
# servos are not identified.
RANDOMIZATION = dict(friction=(0.5, 1.5), pen_mass=(0.5, 2.0), kp=(0.8, 1.2), damping=(0.5, 2.0))
# sim2real: what the real hand adds between the policy and the simulator, drawn
# per episode (latencies, calibration errors) or per control tick (noise), all
# uniform. Magnitudes follow Wuji's hand2 PenSpin config (pen_spin_env_cfg.py:
# joint noise and the camera model, whose jitter bounds are ~4x the measured
# PnP noise); the ORCA hardware is unmeasured, so latencies span a range.
SIM2REAL = dict(
    cmd_delay_ms=(0.0, 40.0),      # servo command latency, per episode (2 ms resolution)
    joint_bias=0.01,               # rad per joint, per episode: encoder calibration
    joint_noise=0.015,             # rad per joint, per tick
    joint_vel_noise=0.1,           # rad/s per joint, per tick
    pen_pos_bias=0.005,            # m per axis, per episode: camera-to-hand calibration
    pen_rot_bias=0.03,             # rad about a random axis, per episode
    pen_pos_noise=0.002,           # m per axis, per tick
    pen_rot_noise=0.015,           # rad about a random axis, per tick
    pen_lag_steps=(1, 2),          # camera pipeline latency in control ticks, per episode
    pen_dropout=(0.03, 0.78, 40),  # missed-frame bursts: P(start), P(go on), longest (ticks)
    # Joint speeds: "noisy" is the simulator's plus joint_vel_noise (as trained up
    # to track003_8); "diff" is what a hand that reports positions only gives: the
    # change of the reported angles over one tick, zero on the first tick.
    joint_vel="noisy",
    joint_vel_lowpass=1.0,         # with "diff": weight of the new difference in a first-order low-pass (1 = none)
)
# disturb: Wuji's PenSpin step events as the group's dr policies use them
# (wuji_mjlab/tasks/pen_spin/mdp/events.py: apply_object_disturbance_wrench and
# apply_link_random_force, values from policies/clip002_004_dr_scratch env.yaml).
# The pen gets intermittent decaying pulses of force and torque after a warmup;
# each fingertip gets its own random kicks at a random point on the link. The
# strength `disturb` scales both (1 = their full curriculum).
DISTURB = dict(
    pen_grav_mult=(1.0, 2.0),      # linear acceleration of a pulse, in g
    pen_angular_accel=375.0,       # rad/s^2 at the top of pen_grav_mult, proportionally below
    pen_active_s=(0.2, 0.4),       # pulse length; it decays to pen_decay_to by the end
    pen_silence_s=(0.5, 1.2),
    pen_decay_to=0.02,
    pen_warmup_s=1.0,              # no pushes in an episode's first second
    tip_grav_mult=(0.0, 4.0),      # kick force in multiples of the link's weight
    tip_force_scale=(1.2, 0.5, 0.5, 0.5, 0.5),  # thumb, index, middle, ring, pinky
    tip_offset_radius=0.9,         # application point within this share of the link's bounding radius
    tip_duration_s=(0.1, 0.3),
    tip_cooldown_s=(0.5, 1.5),
)


def unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / max(np.linalg.norm(v), 1e-12)


def small_rotation(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    """A wxyz quaternion turning by U(-max_angle, max_angle) about a uniformly random axis."""
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    half = 0.5 * rng.uniform(-max_angle, max_angle)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by the wxyz quaternion q (both may carry a leading batch axis)."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def axis_points(pose: np.ndarray) -> np.ndarray:
    """The two points AXIS_POINT_M either side of the pen centre on its axis, (..., 2, 3)."""
    axis = quat_rotate(pose[..., 3:], np.array([1.0, 0.0, 0.0]))
    centre = pose[..., None, :3]
    return centre + np.stack([axis, -axis], axis=-2) * AXIS_POINT_M


def resample(ref_q: np.ndarray, ref_obj: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Play the reference `factor` times faster: joint angles and pen position
    interpolated linearly, the pen's quaternion by normalized lerp (the steps
    are under 2 degrees, where it matches slerp)."""
    n = int(np.floor((len(ref_q) - 1) / factor + 1e-9)) + 1
    src = np.arange(n) * factor
    i0 = np.minimum(np.floor(src).astype(int), len(ref_q) - 2)
    u = (src - i0)[:, None]
    q = (1 - u) * ref_q[i0] + u * ref_q[i0 + 1]
    pos = (1 - u) * ref_obj[i0, :3] + u * ref_obj[i0 + 1, :3]
    qa, qb = ref_obj[i0, 3:], ref_obj[i0 + 1, 3:].copy()
    qb[np.sum(qa * qb, axis=1) < 0] *= -1
    quat = (1 - u) * qa + u * qb
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return q, np.concatenate([pos, quat], axis=1)


class PenTrack(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        run: str | Path = DEFAULT_RUN,
        action_scale: float = 0.4,
        ema_alpha: float = 0.5,
        hold_prob: float = 0.15,
        hold_s: tuple[float, float] = (0.5, 3.0),
        adaptive_sampling: bool = True,
        uniform_mix: float = 0.2,
        bin_s: float = 0.5,
        lookahead: int = 5,
        pos_fail_m: float = 0.05,
        axis_fail_deg: float = 60.0,
        w_pos: float = 2.0,
        w_rot: float = 8.0,
        w_link: float = 1.0,
        w_joint: float = 1.0,
        w_action_rate: float = 0.1,
        w_torque: float = 1.0,
        w_collision: float = 1.0,
        start_frame: int | None = None,
        time_scale: float | None = None,
        world_angvel: bool = False,
        randomize_physics: bool = False,
        sim2real: bool = False,
        sim2real_cfg: dict | None = None,
        disturb: float = 0.0,
        render_mode: str | None = None,
    ):
        run = resolve_run(run)
        self.model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        ref = np.load(run / "reference.npz")
        self.ref_q = ref["reference_qpos"].astype(np.float64)
        self.ref_obj = ref["object_pose_w"].astype(np.float64)
        link_names = [str(n) for n in ref["link_names"]]
        self.dt = float(ref["frame_dt"])
        # time_scale: how many times slower than the human demonstration to play
        # the clip; the reference records its own (optimized003-final is 3x).
        source_scale = float(run_info(run).get("time_scale", 1.0))
        self.time_scale = source_scale if time_scale is None else float(time_scale)
        if abs(self.time_scale - source_scale) > 1e-9:
            self.ref_q, self.ref_obj = resample(self.ref_q, self.ref_obj, source_scale / self.time_scale)
        self.n_frames = len(self.ref_q)
        m = self.model
        self.frame_skip = int(round(self.dt / m.opt.timestep))
        assert abs(self.frame_skip * m.opt.timestep - self.dt) < 1e-9

        joints = m.actuator_trnid[:, 0]
        self._qadr = m.jnt_qposadr[joints]
        self._vadr = m.jnt_dofadr[joints]
        self._lo, self._hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self._range = self._hi - self._lo
        self._force_limit = np.abs(m.actuator_forcerange[:, 1]).copy()
        pen = m.joint("pen_free")
        self._pq, self._pv = pen.qposadr[0], pen.dofadr[0]
        self._pen_body = m.body("pen").id

        # Link tracking uses the moving hand links; the tower and palm are fixed.
        # Their reference positions come from forward kinematics of the
        # reference angles (identical to the clip's link_pos_w, and still right
        # after resampling).
        tracked = [n for n in link_names if n not in ("right_tower", "right_palm", "pen")]
        self._link_bodies = np.array([m.body(n).id for n in tracked])
        self._tip_bodies = np.array([m.body(f"right_{f}_{'dp' if f == 'thumb' else 'ip'}").id for f in FINGERS])
        self._ref_link_idx = np.arange(len(tracked))
        self._ref_tip_idx = np.array([tracked.index(m.body(b).name) for b in self._tip_bodies])
        self.ref_link = np.empty((self.n_frames, len(tracked), 3))
        for k in range(self.n_frames):
            self.data.qpos[self._qadr] = self.ref_q[k]
            mujoco.mj_kinematics(m, self.data)
            self.ref_link[k] = self.data.xpos[self._link_bodies]
        # Which finger each body belongs to (-1 the pen and world; 5 the palm and tower).
        self._finger_of_body = np.full(m.nbody, -1)
        for b in range(m.nbody):
            name = m.body(b).name
            for f, finger in enumerate(FINGERS):
                if name.startswith(f"right_{finger}_"):
                    self._finger_of_body[b] = f
            if name in ("right_palm", "right_tower"):
                self._finger_of_body[b] = 5
        self._centre = self.ref_obj[:, :3].mean(0)
        self._ref_pts = axis_points(self.ref_obj)

        self.action_scale = float(action_scale)
        self.ema_alpha = float(ema_alpha)
        self.hold_prob = float(hold_prob)
        self.hold_s = hold_s
        self.adaptive_sampling = bool(adaptive_sampling)
        self.uniform_mix = float(uniform_mix)
        self.lookahead = int(lookahead)
        self.pos_fail_m = float(pos_fail_m)
        self.cos_axis_fail = float(np.cos(np.radians(axis_fail_deg)))
        self.w = dict(pos=w_pos, rot=w_rot, link=w_link, joint=w_joint,
                      action_rate=w_action_rate, torque=w_torque, collision=w_collision)
        self.start_frame = start_frame
        self.world_angvel = bool(world_angvel)
        self.randomize_physics = bool(randomize_physics)
        self._nominal = dict(
            friction=m.geom_friction[:, 0].copy(),
            pen_mass=float(m.body_mass[self._pen_body]),
            pen_inertia=m.body_inertia[self._pen_body].copy(),
            kp=m.actuator_gainprm[:, 0].copy(),
            kp_bias=m.actuator_biasprm[:, 1].copy(),
            damping=m.dof_damping.copy(),
        )
        self.physics = {}
        self.sim2real = bool(sim2real)
        unknown = set(sim2real_cfg or {}) - set(SIM2REAL)
        if unknown:
            raise ValueError(f"unknown sim2real_cfg keys {sorted(unknown)}")
        self.s2r = {**SIM2REAL, **(sim2real_cfg or {})}
        if self.s2r["joint_vel"] not in ("noisy", "diff"):
            raise ValueError(f"sim2real_cfg joint_vel must be 'noisy' or 'diff', not {self.s2r['joint_vel']!r}")
        self.disturb = float(disturb)
        self._gravity = float(np.linalg.norm(m.opt.gravity))
        self._tip_radius = DISTURB["tip_offset_radius"] * np.array(
            [m.geom_rbound[m.geom_bodyid == b].max() for b in self._tip_bodies])
        self._reset_disturb()
        self._t = 0
        self._substep_ms = 1000.0 * m.opt.timestep
        max_delay = int(round(self.s2r["cmd_delay_ms"][1] / self._substep_ms)) if self.sim2real else 0
        # Targets sent so far, newest last: enough to reach back the longest delay.
        self._cmd_len = max_delay // self.frame_skip + 2
        self._cam_len = int(self.s2r["pen_lag_steps"][1]) + int(self.s2r["pen_dropout"][2]) + 1
        self._kin = mujoco.MjData(m)  # forward kinematics of the reported joint angles
        self._draw_sim2real()
        self.render_mode = render_mode
        self._renderer = None

        self._bin_frames = max(1, int(round(bin_s / self.dt)))
        self._n_bins = int(np.ceil((self.n_frames - 1) / self._bin_frames))
        self._bin_score = np.zeros(self._n_bins)

        n = m.nu
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        obs_dim = len(self._obs_template())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

    # ------------------------------------------------------------------ helpers
    def _obs_template(self) -> np.ndarray:
        self._k, self._hold, self._ema = 0, 0, np.zeros(self.model.nu)
        self._a1 = np.zeros(self.model.nu)
        self._a2 = np.zeros(self.model.nu)
        self._set_reference_state(0)
        return self._obs()

    def _frame(self, offset: int = 0) -> int:
        """Reference frame `offset` steps ahead (frame 0 while holding)."""
        if self._hold > 0:
            return 0
        return min(self._k + offset, self.n_frames - 1)

    def _pen_pose(self) -> np.ndarray:
        return self.data.qpos[self._pq:self._pq + 7].copy()

    def _set_reference_state(self, k: int) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self._qadr] = self.ref_q[k]
        d.qpos[self._pq:self._pq + 7] = self.ref_obj[k]
        d.ctrl[:] = self.ref_q[k] - self._joint_bias
        if self._hold == 0 and k + 1 < self.n_frames:
            d.qvel[self._vadr] = (self.ref_q[k + 1] - self.ref_q[k]) / self.dt
            d.qvel[self._pv:self._pv + 3] = (self.ref_obj[k + 1, :3] - self.ref_obj[k, :3]) / self.dt
            w = np.zeros(3)
            mujoco.mju_subQuat(w, self.ref_obj[k + 1, 3:], self.ref_obj[k, 3:])
            d.qvel[self._pv + 3:self._pv + 6] = w / self.dt
        mujoco.mj_forward(m, d)
        # The servos have been holding this frame's targets.
        self._cmds = deque([self.ref_q[k].copy() for _ in range(self._cmd_len)], maxlen=self._cmd_len)
        self._sense(reset=True)

    # ------------------------------------------------------------- sim2real
    def _draw_sim2real(self) -> None:
        """This episode's latencies and calibration errors (none without sim2real)."""
        c, nu = self.s2r, self.model.nu
        self._delay_sub, self._pen_lag = 0, 0
        self._joint_bias = np.zeros(nu)
        self._pen_bias_pos, self._pen_bias_quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        if not self.sim2real:
            return
        rng = self.np_random
        lo, hi = (int(round(ms / self._substep_ms)) for ms in c["cmd_delay_ms"])
        self._delay_sub = int(rng.integers(lo, hi + 1))
        self._pen_lag = int(rng.integers(c["pen_lag_steps"][0], c["pen_lag_steps"][1] + 1))
        self._joint_bias = rng.uniform(-c["joint_bias"], c["joint_bias"], nu)
        self._pen_bias_pos = rng.uniform(-c["pen_pos_bias"], c["pen_pos_bias"], 3)
        self._pen_bias_quat = small_rotation(rng, c["pen_rot_bias"])

    def _actuate(self) -> None:
        """One control tick of physics. Each servo gets the target it was sent
        _delay_sub substeps ago (so for part of the tick still the one before),
        and holds the angle at which its biased encoder reads that target."""
        m, d, fs = self.model, self.data, self.frame_skip
        whole, part = divmod(self._delay_sub, fs)
        if part:
            d.ctrl[:] = self._cmds[-2 - whole] - self._joint_bias
            mujoco.mj_step(m, d, nstep=part)
        d.ctrl[:] = self._cmds[-1 - whole] - self._joint_bias
        mujoco.mj_step(m, d, nstep=fs - part)

    # ---------------------------------------------------------- disturbances
    def _reset_disturb(self) -> None:
        """No wrench at the start of an episode; each fingertip's clock starts at
        a random phase so the five do not fire together (as Wuji's reset). Draws
        nothing without disturbances, so seeded runs are unchanged."""
        n = len(self._tip_bodies)
        self._pen_active, self._pen_left, self._pen_rate = False, 0.0, 1.0
        self._pen_wrench = np.zeros(6)
        self._tip_active = np.zeros(n, bool)
        self._tip_left = np.zeros(n)
        self._tip_wrench = np.zeros((n, 6))
        self._tip_interval = np.zeros(n)
        if self.disturb > 0:
            self._tip_interval = self.np_random.random(n) * DISTURB["tip_cooldown_s"][1]

    def set_disturb(self, strength: float) -> None:
        """Change the disturbance strength (a training curriculum); switching
        it on mid-episode starts the fingertips' clocks at random phases."""
        was_off = self.disturb <= 0
        self.disturb = float(strength)
        if was_off and self.disturb > 0 and not self._tip_active.any():
            self._tip_interval = self.np_random.random(len(self._tip_bodies)) * DISTURB["tip_cooldown_s"][1]

    def _disturb_tick(self) -> None:
        """This tick's external wrenches (world frame, at each body's centre of
        mass), held over the tick's substeps."""
        c, s, rng, m, d, dt, g = DISTURB, self.disturb, self.np_random, self.model, self.data, self.dt, self._gravity
        pb = self._pen_body
        if self._t * dt < c["pen_warmup_s"]:
            d.xfrc_applied[pb] = 0.0
        else:
            self._pen_left -= dt
            expired = self._pen_left <= 0.0
            if self._pen_active:
                self._pen_wrench *= self._pen_rate
            if expired and not self._pen_active:
                lo, hi = c["pen_grav_mult"]
                mult = rng.uniform(lo, hi)
                force = unit_vector(rng) * mult * g * s * m.body_mass[pb]
                accel = unit_vector(rng) * (mult / hi) * c["pen_angular_accel"] * s
                rot = d.ximat[pb].reshape(3, 3)  # principal axes of inertia -> world
                torque = rot @ (m.body_inertia[pb] * (rot.T @ accel))
                self._pen_wrench = np.concatenate([force, torque])
                active = rng.uniform(*c["pen_active_s"])
                self._pen_left = active
                self._pen_rate = c["pen_decay_to"] ** (dt / max(active, dt))
                self._pen_active = True
            elif expired:
                self._pen_wrench[:] = 0.0
                self._pen_active = False
                self._pen_left = rng.uniform(*c["pen_silence_s"])
            d.xfrc_applied[pb] = self._pen_wrench
        for i, b in enumerate(self._tip_bodies):
            if self._tip_active[i]:
                self._tip_left[i] -= dt
                if self._tip_left[i] <= 0.0:
                    self._tip_wrench[i] = 0.0
                    self._tip_active[i] = False
                    self._tip_interval[i] = rng.uniform(*c["tip_cooldown_s"])
            if not self._tip_active[i]:
                self._tip_interval[i] -= dt
                if self._tip_interval[i] <= 0.0:
                    force = (unit_vector(rng) * rng.uniform(*c["tip_grav_mult"]) * s * g
                             * m.body_mass[b] * c["tip_force_scale"][i])
                    offset = unit_vector(rng) * rng.random() ** (1.0 / 3.0) * self._tip_radius[i]
                    offset = d.ximat[b].reshape(3, 3) @ offset
                    self._tip_wrench[i] = np.concatenate([force, np.cross(offset, force)])
                    self._tip_left[i] = rng.uniform(*c["tip_duration_s"])
                    self._tip_active[i] = True
                    self._tip_interval[i] = rng.uniform(*c["tip_cooldown_s"])
            d.xfrc_applied[b] = self._tip_wrench[i]

    def _camera_frame(self) -> np.ndarray:
        """The pen pose a camera would report now: through this episode's
        calibration offset, plus this tick's jitter."""
        c, rng = self.s2r, self.np_random
        pose = self._pen_pose()
        pos = pose[:3] + self._pen_bias_pos + rng.uniform(-c["pen_pos_noise"], c["pen_pos_noise"], 3)
        biased, quat = np.empty(4), np.empty(4)
        mujoco.mju_mulQuat(biased, self._pen_bias_quat, pose[3:])
        mujoco.mju_mulQuat(quat, small_rotation(rng, c["pen_rot_noise"]), biased)
        return np.concatenate([pos, quat])

    def _camera_tick(self, reset: bool) -> np.ndarray:
        """The frame the control loop has on hand this tick (Wuji's
        ObjectPoseCameraModel): normally the one _pen_lag ticks old; during a
        dropout burst the last one received, one tick older each tick."""
        frame = self._camera_frame()
        if reset:
            self._cam = deque([frame], maxlen=self._cam_len)
            self._cam_age, self._cam_holding, self._cam_hold_ticks = 0, False, 0
            return frame
        self._cam.append(frame)
        enter, go_on, longest = self.s2r["pen_dropout"]
        roll = self.np_random.random()
        if self._cam_holding:
            holding = roll < go_on and self._cam_hold_ticks < longest
        else:
            holding = roll < enter
        self._cam_hold_ticks = self._cam_hold_ticks + 1 if holding else 0
        self._cam_holding = holding
        # A stream only gets older while it holds, and a new frame is never
        # older than the one already on hand.
        age = self._cam_age + 1 if holding else min(self._pen_lag, self._cam_age + 1)
        self._cam_age = min(age, len(self._cam) - 1)
        return self._cam[-1 - self._cam_age]

    def _sense(self, reset: bool = False) -> None:
        """What the policy reads this tick, into self._meas: the true state, or
        with sim2real what the encoders and the camera report."""
        d, pv = self.data, self._pv
        q, qd = d.qpos[self._qadr].copy(), d.qvel[self._vadr].copy()
        if not self.sim2real:
            # A free joint's angular velocity is in the pen's own frame, which turns
            # with the unobserved roll about the shaft; world_angvel rotates it into
            # the world frame. Models trained before 2026-11-09 saw the pen frame.
            angvel = d.qvel[pv + 3:pv + 6].copy()
            if self.world_angvel:
                angvel = d.xmat[self._pen_body].reshape(3, 3) @ angvel
            self._meas = dict(q=q, qd=qd, pose=self._pen_pose(), linvel=d.qvel[pv:pv + 3].copy(),
                              angvel=angvel, tips=d.xpos[self._tip_bodies].copy())
            return
        c, rng = self.s2r, self.np_random
        q = q + self._joint_bias + rng.uniform(-c["joint_noise"], c["joint_noise"], q.shape)
        if c["joint_vel"] == "diff":
            if reset:
                qd = np.zeros_like(q)
            else:
                lp = c["joint_vel_lowpass"]
                qd = lp * (q - self._meas["q"]) / self.dt + (1.0 - lp) * self._meas["qd"]
        else:
            qd = qd + rng.uniform(-c["joint_vel_noise"], c["joint_vel_noise"], qd.shape)
        pose = self._camera_tick(reset)
        linvel, angvel = np.zeros(3), np.zeros(3)
        if not reset:
            # The pen's velocity as a deployment gets it: differencing camera poses.
            prev = self._meas["pose"]
            linvel = (pose[:3] - prev[:3]) / self.dt
            mujoco.mju_subQuat(angvel, pose[3:], prev[3:])  # prev * quat(angvel) = pose
            angvel /= self.dt
            if self.world_angvel:
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, prev[3:])
                angvel = rot.reshape(3, 3) @ angvel
        # Fingertips by forward kinematics of the reported angles.
        self._kin.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, self._kin)
        self._meas = dict(q=q, qd=qd, pose=pose, linvel=linvel, angvel=angvel,
                          tips=self._kin.xpos[self._tip_bodies].copy())

    def _obs(self) -> np.ndarray:
        s = self._meas
        q, pose = s["q"], s["pose"]
        pts = axis_points(pose)
        k1, kl = self._frame(1), self._frame(self.lookahead)
        ref_tips = self.ref_link[k1, self._ref_tip_idx]
        return np.concatenate([
            q, 0.25 * s["qd"],
            pose[:3] - self._centre, (pts - self._centre).ravel(),
            s["linvel"], s["angvel"],
            self.ref_q[k1], self.ref_q[k1] - q,
            self.ref_obj[k1, :3] - self._centre, (self._ref_pts[k1] - pts).ravel(),
            self.ref_q[kl] - self.ref_q[k1], (self._ref_pts[kl] - self._ref_pts[k1]).ravel(),
            (ref_tips - s["tips"]).ravel(),
            self._a1, self._ema,
            [self._k / (self.n_frames - 1), float(self._hold > 0)],
        ]).astype(np.float32)

    def _finger_contacts(self) -> int:
        """Finger links touching a different finger or the palm (each link once)."""
        d, fb = self.data, self._finger_of_body
        touching = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            f1, f2 = fb[b1], fb[b2]
            if f1 < 0 or f2 < 0 or f1 == f2:
                continue
            if f1 < 5:
                touching.add(b1)
            if f2 < 5:
                touching.add(b2)
        return len(touching)

    def _pick_start(self) -> tuple[int, int]:
        """(start frame, hold steps)."""
        rng = self.np_random
        if self.start_frame is not None:
            return self.start_frame, 0
        if rng.random() < self.hold_prob:
            return 0, int(round(rng.uniform(*self.hold_s) / self.dt))
        if self.adaptive_sampling and self._bin_score.sum() > 0:
            p = self._bin_score / self._bin_score.sum()
            p = (1 - self.uniform_mix) * p + self.uniform_mix / self._n_bins
            b = rng.choice(self._n_bins, p=p)
        else:
            b = rng.integers(self._n_bins)
        lo = b * self._bin_frames
        hi = min(lo + self._bin_frames, self.n_frames - 1)
        return int(rng.integers(lo, hi)), 0

    def _draw_physics(self) -> None:
        """Redraw this episode's physics from RANDOMIZATION (log-uniform factors
        on the nominal values): one contact-friction factor for every geom (a
        contact takes the larger of its two geoms' friction, so scaling only the
        pen's would do nothing below 1), the pen's mass and inertia, and each
        servo's kp and each finger joint's damping separately."""
        m, nominal, rng = self.model, self._nominal, self.np_random

        def draw(key, size=None):
            lo, hi = RANDOMIZATION[key]
            return np.exp(rng.uniform(np.log(lo), np.log(hi), size))

        friction, mass = draw("friction"), draw("pen_mass")
        kp, damping = draw("kp", m.nu), draw("damping", m.nu)
        m.geom_friction[:, 0] = nominal["friction"] * friction
        m.body_mass[self._pen_body] = nominal["pen_mass"] * mass
        m.body_inertia[self._pen_body] = nominal["pen_inertia"] * mass
        m.actuator_gainprm[:, 0] = nominal["kp"] * kp
        m.actuator_biasprm[:, 1] = nominal["kp_bias"] * kp
        m.dof_damping[:] = nominal["damping"]
        m.dof_damping[self._vadr] *= damping
        self.physics = dict(friction=float(friction), pen_mass=float(mass),
                            kp_mean=float(kp.mean()), damping_mean=float(damping.mean()))

    def _record_outcome(self, failed: bool) -> None:
        """Every bin the episode passed through counts as a success, the one it
        ended in as this episode's outcome (as Wuji's _record_bin_outcomes).
        Scoring only the last bin left every bin but the clip's last at
        "failed", which flattened the sampler to nearly uniform."""
        b0 = min(self._start // self._bin_frames, self._n_bins - 1)
        b1 = min(self._k // self._bin_frames, self._n_bins - 1)
        self._bin_score[b0:b1] *= 0.9
        self._bin_score[b1] = 0.9 * self._bin_score[b1] + 0.1 * float(failed)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._k, self._hold = self._pick_start()
        self._start = self._k
        self._ema = np.zeros(self.model.nu)
        self._a1 = np.zeros(self.model.nu)
        self._a2 = np.zeros(self.model.nu)
        if self.randomize_physics:
            self._draw_physics()
        self._draw_sim2real()
        self._set_reference_state(self._k)
        self._reset_disturb()
        self._t = 0
        self._stats = dict(pos_err=0.0, axis_err=0.0, steps=0)
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._ema = self.ema_alpha * a * self.action_scale + (1 - self.ema_alpha) * self._ema
        if self._hold > 0:
            self._hold -= 1
        else:
            self._k = min(self._k + 1, self.n_frames - 1)
        k = self._frame()
        self._cmds.append(np.clip(self.ref_q[k] + self._ema, self._lo, self._hi))
        if self.disturb > 0:
            self._disturb_tick()
        self._actuate()
        self._t += 1

        pose = self._pen_pose()
        pos_err = float(np.linalg.norm(pose[:3] - self.ref_obj[k, :3]))
        pts = axis_points(pose)
        pts_err = float(np.linalg.norm(pts - self._ref_pts[k], axis=-1).mean())
        axis = quat_rotate(pose[3:], np.array([1.0, 0.0, 0.0]))
        ref_axis = quat_rotate(self.ref_obj[k, 3:], np.array([1.0, 0.0, 0.0]))
        cos_axis = float(axis @ ref_axis)
        link_err = np.linalg.norm(d.xpos[self._link_bodies] - self.ref_link[k, self._ref_link_idx], axis=-1)
        q = d.qpos[self._qadr]
        joint_err = np.abs(q - self.ref_q[k]) / self._range
        rate = a - self._a1
        accel = a - 2 * self._a1 + self._a2
        torque = d.actuator_force / self._force_limit
        contacts = self._finger_contacts()

        terms = dict(
            pos=self.w["pos"] * np.exp(-(pos_err / 0.05) ** 2),
            rot=self.w["rot"] * np.exp(-(pts_err / 0.04) ** 2),
            link=self.w["link"] * float(np.exp(-(link_err / 0.025) ** 2).mean()),
            joint=self.w["joint"] * float(np.exp(-(joint_err / 0.1) ** 2).mean()),
            action_rate=-self.w["action_rate"] * float(np.sum(rate**2 + accel**2)),
            torque=-self.w["torque"] * float(np.sum(torque**2)),
            collision=-self.w["collision"] * contacts,
        )
        reward = self.dt * sum(terms.values())
        self._a2, self._a1 = self._a1, a

        unstable = not np.isfinite(d.qpos).all() or np.abs(d.qvel).max() > 80.0
        failed = unstable or pos_err > self.pos_fail_m or cos_axis < self.cos_axis_fail
        truncated = (not failed) and self._hold == 0 and self._k >= self.n_frames - 1
        self._stats["pos_err"] += pos_err
        self._stats["axis_err"] += float(np.degrees(np.arccos(np.clip(cos_axis, -1, 1))))
        self._stats["steps"] += 1

        info = {f"r_{n}": v for n, v in terms.items()}
        if failed or truncated:
            if self.adaptive_sampling:
                self._record_outcome(failed)
            n = max(self._stats["steps"], 1)
            info.update(
                failed=float(failed),
                start_frame=self._start,
                end_frame=self._k,
                frames_tracked=self._k - self._start,
                mean_pos_err_cm=100 * self._stats["pos_err"] / n,
                mean_axis_err_deg=self._stats["axis_err"] / n,
            )
        if unstable:
            obs = np.zeros(self.observation_space.shape, np.float32)
        else:
            self._sense()
            obs = self._obs()
        return obs, float(reward), bool(failed), bool(truncated), info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._camera = mujoco.MjvCamera()
            self._camera.lookat[:] = self._centre
            self._camera.distance, self._camera.azimuth, self._camera.elevation = 0.45, 90.0, -55.0
        self._renderer.update_scene(self.data, self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def run_info(run: str | Path = DEFAULT_RUN) -> dict:
    """The reference's metadata, for logs."""
    ref = np.load(resolve_run(run) / "reference.npz")
    return json.loads(str(ref["metadata"])) if "metadata" in ref.files else {}
