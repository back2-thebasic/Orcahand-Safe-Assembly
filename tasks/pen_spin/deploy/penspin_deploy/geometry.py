"""Quaternion helpers, scalar-first (w, x, y, z) like MuJoCo and mjlab."""

from __future__ import annotations

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate((q[..., :1], -q[..., 1:]), axis=-1)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return np.concatenate(([np.cos(angle / 2)], np.sin(angle / 2) * axis))


def quat_unique(q: np.ndarray) -> np.ndarray:
    """The hemisphere with w >= 0, as mjlab's quat_unique."""
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., :1] < 0, -q, q)


def pose_in_frame(frame_pos, frame_quat, pos, quat) -> np.ndarray:
    """(pos, quat) expressed in the frame (frame_pos, frame_quat), as one 7-vector."""
    inv = quat_conj(frame_quat)
    return np.concatenate(
        (quat_apply(inv, np.asarray(pos) - np.asarray(frame_pos)), quat_mul(inv, quat))
    )


def pose_from_frame(frame_pos, frame_quat, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pose_in_frame: a frame-local 7-vector back to world (pos, quat)."""
    local = np.asarray(local, dtype=np.float64)
    return (
        np.asarray(frame_pos) + quat_apply(frame_quat, local[:3]),
        quat_mul(frame_quat, local[3:7]),
    )


def bounded_axis_points(pose_local: np.ndarray, radius: float, tanh_scale: float) -> np.ndarray:
    """The pen's +-radius shaft points in the palm frame, squashed by tanh: 6 values.

    Matches bounded_object_pose_to_points_wrist_local on the training side.
    """
    pose_local = np.asarray(pose_local, dtype=np.float64)
    template = np.array([[radius, 0.0, 0.0], [-radius, 0.0, 0.0]])
    points = pose_local[:3] + quat_apply(np.broadcast_to(pose_local[3:7], (2, 4)), template)
    return (tanh_scale * np.tanh(points / tanh_scale)).reshape(-1)


def shaft_angle_deg(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Angle between the two poses' pen shafts (+X), as the training termination measures it."""
    a = quat_apply(quat_a, np.array([1.0, 0.0, 0.0]))
    b = quat_apply(quat_b, np.array([1.0, 0.0, 0.0]))
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
