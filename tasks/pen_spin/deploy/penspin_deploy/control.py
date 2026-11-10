"""Observation assembly and action processing: the training-side math, in numpy.

    obs = [motion command (73) | q (16) | qdot * 0.25 (16) | pen axis points (6) | last raw action (16)]
    target = reference_q[k] + EMA(clip(raw * scale))

Joint values are in the training model's convention (MuJoCo qpos, radians,
policy joint order). Conversion to orca_core degrees happens in joint_map.
"""

from __future__ import annotations

import numpy as np

from penspin_deploy.bundle import Bundle, Clip


def build_observation(
    bundle: Bundle,
    clip: Clip,
    frame: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    pen_local: np.ndarray,
    last_raw: np.ndarray,
) -> np.ndarray:
    obs = np.concatenate(
        (
            clip.command[frame],
            np.asarray(qpos, dtype=np.float32),
            np.asarray(qvel, dtype=np.float32) * bundle.joint_vel_scale,
            bundle.pen_observation(pen_local),
            np.asarray(last_raw, dtype=np.float32),
        )
    ).astype(np.float32)
    if obs.shape != (bundle.obs_dim,) or not np.isfinite(obs).all():
        raise FloatingPointError(f"observation must be finite and {bundle.obs_dim}-D, got {obs.shape}")
    return obs


class ActionMap:
    """SingleHandResidualEMAAction without the training-only mask and encoder bias."""

    def __init__(self, bundle: Bundle):
        self.scale = bundle.action_scale
        self.lo, self.hi = bundle.residual_clip
        self.alpha = bundle.ema_alpha
        self.ctrl_lo, self.ctrl_hi = bundle.ctrl_range[:, 0], bundle.ctrl_range[:, 1]
        self.processed = np.zeros(bundle.num_joints, dtype=np.float32)

    def reset(self) -> None:
        self.processed[:] = 0.0

    def __call__(self, raw: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        residual = np.clip(np.asarray(raw, dtype=np.float32) * self.scale, self.lo, self.hi)
        self.processed = (self.alpha * residual + (1.0 - self.alpha) * self.processed).astype(np.float32)
        return reference_qpos + self.processed

    def clamp(self, target: np.ndarray) -> np.ndarray:
        """The training actuators' ctrlrange: what a position actuator would have done."""
        return np.clip(target, self.ctrl_lo, self.ctrl_hi)


class VelocityEstimate:
    """Finite-difference joint velocity through a first-order low-pass.

    Training observes exact qvel; the hand reports positions only. The default
    100 Hz corner at 50 Hz is Wuji's choice: nearly a pass-through.
    """

    def __init__(self, dim: int, dt: float, cutoff_hz: float = 100.0):
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        self.alpha = dt / (dt + rc)
        self.dt = dt
        self.prev_q: np.ndarray | None = None
        self.value = np.zeros(dim)

    def reset(self, q: np.ndarray) -> None:
        self.prev_q = np.asarray(q, dtype=np.float64).copy()
        self.value[:] = 0.0

    def update(self, q: np.ndarray, dt: float | None = None) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if self.prev_q is None:
            self.reset(q)
            return self.value.copy()
        raw = (q - self.prev_q) / max(dt or self.dt, 1e-6)
        self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        self.prev_q = q.copy()
        return self.value.copy()
