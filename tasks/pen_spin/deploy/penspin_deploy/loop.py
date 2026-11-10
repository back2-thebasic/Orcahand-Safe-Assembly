"""The control loop: one observe -> infer -> command step per tick, 50 Hz.

    EASE    the hand moves slowly from wherever it is to the clip's first pose
    FROZEN  nothing is commanded; the pen is placed into the hand
    HOLD    the policy runs on the clip's first frame (the training warmup)
    POLICY  the clip plays; the reference advances with wall-clock time
    DONE    the clip is over; the last target stays latched

Entering HOLD resets the action EMA, the velocity estimate and the last action;
HOLD -> POLICY keeps them, as the training warmup hands over to the clip.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from penspin_deploy.bundle import Bundle, Clip
from penspin_deploy.control import ActionMap, VelocityEstimate, build_observation
from penspin_deploy.geometry import shaft_angle_deg

EASE, FROZEN, HOLD, POLICY, DONE = "ease", "frozen", "hold", "policy", "done"
STATE_CODES = {EASE: 0, FROZEN: 1, HOLD: 2, POLICY: 3, DONE: 4}
# Training ends an episode beyond these (object_position_error /
# object_shaft_direction_error); a start outside them is out of distribution.
MAX_POS_ERR_M = 0.05
MAX_SHAFT_ERR_DEG = 60.0


class WallClock:
    """Absolute deadlines: sleeping dt - elapsed drifts, a deadline does not."""

    def __init__(self, dt: float):
        self.dt = dt
        self.t0 = time.perf_counter()
        self.ticks = 0
        self.overruns = 0

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def wait(self) -> None:
        self.ticks += 1
        deadline = self.t0 + self.ticks * self.dt
        slack = deadline - time.perf_counter()
        if slack < 0:
            self.overruns += 1
            if slack < -self.dt:  # more than a whole tick behind: drop the backlog
                self.ticks = int((time.perf_counter() - self.t0) / self.dt)
            return
        if slack > 0.001:
            time.sleep(slack - 0.0005)
        while time.perf_counter() < deadline:
            pass


class TwinClock:
    """Simulated time: waiting a tick is stepping the twin one control period."""

    def __init__(self, twin, dt: float, *, realtime: bool, viewer=None):
        self.twin, self.dt = twin, dt
        self.wall = WallClock(dt) if realtime else None
        self.viewer = viewer
        self.overruns = 0

    def now(self) -> float:
        return self.twin.time

    def wait(self) -> None:
        self.twin.step_control()
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
        if self.wall is not None:
            self.wall.wait()
            self.overruns = self.wall.overruns


class KeyboardOperator:
    """Enter advances FROZEN -> HOLD -> POLICY; q (then Enter) quits."""

    interactive = True

    def __init__(self):
        self._keys: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in sys.stdin:
            self._keys.put(line.strip().lower())

    def poll(self, controller: "Controller") -> str | None:
        try:
            key = self._keys.get_nowait()
        except queue.Empty:
            return None
        return "quit" if key == "q" else "next"


class ScriptedOperator:
    """Headless runs: place the pen once the hand is at the start pose, hold, start."""

    interactive = False

    def __init__(self, *, hold_s: float, place_pen=None):
        self.hold_s = hold_s
        self.place_pen = place_pen
        self.placed = False

    def poll(self, controller: "Controller") -> str | None:
        if controller.state == FROZEN:
            if self.place_pen is not None and not self.placed:
                self.place_pen(controller.clip.object_pose_local[0])
                self.placed = True
                return None  # let the camera see it for a tick
            return "next"
        if controller.state == HOLD and controller.time_in_state() >= self.hold_s - 1e-9:
            return "next"
        return None


@dataclass
class Log:
    """Per-tick record of the HOLD/POLICY ticks."""

    rows: dict[str, list] = field(default_factory=lambda: {
        k: [] for k in ("t", "state", "frame", "qpos", "qvel", "pen_obs", "pen_age",
                        "raw", "target", "compute_ms", "pen_true")})

    def add(self, **values) -> None:
        for key, rows in self.rows.items():
            rows.append(values[key])

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(rows, dtype=np.float64) for key, rows in self.rows.items()}


class Controller:
    def __init__(self, bundle: Bundle, clip: Clip, hand, clock, operator, *,
                 policy=None, pen_source=None, pen_truth=None, on_hold=None,
                 ease_s: float = 2.0, hold_last_frame_s: float = 0.0, verbose: bool = True):
        """``policy=None`` replays the reference joints without a policy.
        ``pen_source=None`` feeds the reference pen pose to the policy (no camera).
        ``pen_truth`` (twin only) gives the true pen pose for scoring.
        ``on_hold`` runs as the policy takes over (twin: the operator lets go)."""
        self.bundle, self.clip, self.hand = bundle, clip, hand
        self.clock, self.operator = clock, operator
        self.policy, self.pen_source, self.pen_truth = policy, pen_source, pen_truth
        self.on_hold = on_hold
        self.ease_s, self.hold_last_frame_s = ease_s, hold_last_frame_s
        self.verbose = verbose
        self.action = ActionMap(bundle)
        self.velocity = VelocityEstimate(bundle.num_joints, bundle.ctrl_dt)
        self.last_raw = np.zeros(bundle.num_joints, dtype=np.float32)
        self.state = EASE
        self.state_t0 = clock.now()
        self.clip_t0 = 0.0
        self.prev_t: float | None = None
        self.log = Log()
        self.latest_pen: tuple[np.ndarray, int] | None = None
        self.dropped_at: float | None = None
        self.err_sum, self.err_n = 0.0, 0

    def say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def time_in_state(self) -> float:
        return self.clock.now() - self.state_t0

    def enter(self, state: str) -> None:
        if state == HOLD:
            if self.on_hold is not None:
                self.on_hold()
            q = self.hand.read()
            self.action.reset()
            self.velocity.reset(q)
            self.last_raw[:] = 0.0
            self.prev_t = self.clock.now()
        if state == POLICY:
            self.clip_t0 = self.clock.now()
        self.state, self.state_t0 = state, self.clock.now()
        self.say(f"  -> {state}")

    def placement_problem(self) -> str | None:
        if self.pen_source is None:
            return None
        if self.latest_pen is None:
            return "the camera has not seen the pen"
        pose, _ = self.latest_pen
        ref = self.clip.object_pose_local[0]
        pos_err = float(np.linalg.norm(pose[:3] - ref[:3]))
        shaft = shaft_angle_deg(pose[3:], ref[3:])
        if shaft > 180.0 - MAX_SHAFT_ERR_DEG:
            return "the pen is reversed"
        if shaft > MAX_SHAFT_ERR_DEG or pos_err > MAX_POS_ERR_M:
            return f"pen is {pos_err * 1000:.0f} mm / {shaft:.0f} deg from the start pose"
        return None

    def run(self) -> dict:
        q_start = self.hand.read()
        q_ref0 = self.clip.reference_qpos[0].astype(np.float64)
        self.say(f"  -> {EASE} ({self.ease_s:.1f} s to the clip's first pose)")
        try:
            while self.state != DONE:
                action = self.operator.poll(self)
                if action == "quit":
                    self.say("  quit")
                    break
                if self.state == EASE:
                    s = min(self.time_in_state() / self.ease_s, 1.0) if self.ease_s > 0 else 1.0
                    s = s * s * (3.0 - 2.0 * s)
                    self.hand.command(self.action.clamp(q_start + (q_ref0 - q_start) * s))
                    if s >= 1.0:
                        self.enter(FROZEN)
                        if self.operator.interactive:
                            self.say("     place the pen, then Enter: the policy holds the first frame")
                    self.clock.wait()
                    continue
                pen_sample = None
                if self.state == FROZEN:
                    # The camera streams while the pen is placed; the latest sample gates the start.
                    if self.pen_source is not None:
                        self.latest_pen = pen_sample = self.pen_source.sample()
                    if action == "next":
                        problem = self.placement_problem()
                        if problem:
                            self.say(f"     not starting: {problem}")
                        else:
                            # The policy acts on this very tick: a frozen hand loses the pen.
                            self.enter(HOLD)
                            action = None
                            if self.operator.interactive:
                                self.say("     Enter again starts the clip")
                    if self.state == FROZEN:
                        self.clock.wait()
                        continue
                if self.state == HOLD and action == "next":
                    self.enter(POLICY)
                self.tick(pen_sample)
                self.clock.wait()
        except KeyboardInterrupt:
            self.say("  interrupted")
        return self.summary()

    def tick(self, pen_sample: tuple[np.ndarray, int] | None = None) -> None:
        """``pen_sample``: this tick's camera sample, if it was already taken."""
        t = self.clock.now()
        started = time.perf_counter()
        q = self.hand.read()
        qd = self.velocity.update(q, None if self.prev_t is None else t - self.prev_t)
        self.prev_t = t
        if self.state == HOLD:
            frame = 0
        else:
            elapsed = t - self.clip_t0
            frame = self.clip.frame_at(elapsed)
            end_s = (self.clip.num_frames - 1) / self.clip.fps
            if elapsed >= end_s + self.hold_last_frame_s - 1e-9:
                self.state = DONE
        if self.pen_source is None:
            pen, age = self.clip.object_pose_local[frame].astype(np.float64), 0
        elif pen_sample is not None:
            pen, age = pen_sample
        else:
            pen, age = self.pen_source.sample()
        raw = np.zeros(self.bundle.num_joints, dtype=np.float32)
        if self.policy is None:
            target = self.clip.reference_qpos[frame].astype(np.float64)
        else:
            obs = build_observation(self.bundle, self.clip, frame, q, qd, pen, self.last_raw)
            raw = self.policy.infer(obs)
            self.last_raw = raw
            target = self.action(raw, self.clip.reference_qpos[frame])
        if not np.isfinite(target).all():
            raise FloatingPointError("non-finite joint target")
        target = self.action.clamp(target)
        self.hand.command(target)
        compute_ms = (time.perf_counter() - started) * 1000.0
        truth = None if self.pen_truth is None else self.pen_truth()
        if truth is not None and self.state != HOLD:
            self.score(t, truth, self.clip.object_pose_local[frame])
        self.log.add(t=t, state=STATE_CODES[self.state], frame=frame, qpos=q, qvel=qd,
                     pen_obs=pen, pen_age=age, raw=raw, target=target, compute_ms=compute_ms,
                     pen_true=truth if truth is not None else np.full(7, np.nan))

    def score(self, t: float, truth: np.ndarray, ref: np.ndarray) -> None:
        pos_err = float(np.linalg.norm(truth[:3] - ref[:3]))
        if self.dropped_at is None and (pos_err > MAX_POS_ERR_M or not np.isfinite(pos_err)
                                        or shaft_angle_deg(truth[3:], ref[3:]) > MAX_SHAFT_ERR_DEG):
            self.dropped_at = t - self.clip_t0
            self.say(f"     pen lost {self.dropped_at:.2f} s into the clip")
        if self.dropped_at is None:
            self.err_sum += pos_err
            self.err_n += 1

    def summary(self) -> dict:
        ms = np.array(self.log.rows["compute_ms"], dtype=float)
        out = {
            "ticks": len(ms),
            "compute_ms_p50": float(np.nanmedian(ms)) if len(ms) else float("nan"),
            "compute_ms_max": float(np.nanmax(ms)) if len(ms) else float("nan"),
            "overruns": int(getattr(self.clock, "overruns", 0)),
            "completed": self.state == DONE,
        }
        if self.pen_truth is not None:
            out["pen_lost_at_s"] = self.dropped_at
            out["mean_pos_err_cm"] = 100.0 * self.err_sum / max(self.err_n, 1)
        return out
