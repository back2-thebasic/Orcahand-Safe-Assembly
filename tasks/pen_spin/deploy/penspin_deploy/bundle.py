"""The deploy bundle written by export_bundle.py: contract, clips, ONNX policy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from penspin_deploy.geometry import bounded_axis_points

SCHEMA = "orcahand_pen_spin_deploy:1"


@dataclass(frozen=True)
class Clip:
    name: str
    command: np.ndarray            # (T, 73) motion command term, frame by frame
    reference_qpos: np.ndarray     # (T, 16) joint reference the residual is added to
    object_pose_local: np.ndarray  # (T, 7) reference pen pose in the palm frame
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.command.shape[0])

    def frame_at(self, elapsed_s: float) -> int:
        """Clip frame ``elapsed_s`` after the clip started, held at the last frame.

        Wall-clock anchored: rounding elapsed time instead of counting ticks keeps
        a slow tick from accumulating into a lag behind the reference.
        """
        return int(min(max(round(elapsed_s * self.fps), 0), self.num_frames - 1))


class Bundle:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.contract = json.loads((self.path / "contract.json").read_text())
        if self.contract.get("schema") != SCHEMA:
            raise ValueError(f"{self.path}: not a {SCHEMA} bundle")
        c = self.contract
        self.ctrl_dt = float(c["ctrl_dt"])
        self.joint_names: list[str] = list(c["joints"]["names"])
        self.orca_ids: list[str] = list(c["joints"]["orca_ids"])
        self.qpos0 = np.asarray(c["joints"]["qpos0_rad"], dtype=np.float64)
        self.ctrl_range = np.asarray(c["joints"]["ctrl_range_rad"], dtype=np.float64)
        self.num_joints = len(self.joint_names)
        obs = c["observation"]
        self.obs_layout = [(name, int(width)) for name, width in obs["layout"]]
        self.obs_dim = sum(width for _, width in self.obs_layout)
        self.joint_vel_scale = float(obs["joint_vel_scale"])
        self.axis_radius = float(obs["axis_point_radius_m"])
        self.tanh_scale = float(obs["object_position_tanh_scale_m"])
        act = c["action"]
        self.action_scale = float(act["scale"])
        self.residual_clip = (float(act["residual_clip"][0]), float(act["residual_clip"][1]))
        self.ema_alpha = float(act["ema_alpha"])
        self.palm_pose_w = np.asarray(c["palm"]["pose_w"], dtype=np.float64)
        self.first_frame_hold_s = float(c["first_frame_hold_s"])

    @property
    def clip_names(self) -> list[str]:
        return list(self.contract["clips"])

    def clip(self, name: str) -> Clip:
        if name not in self.contract["clips"]:
            raise KeyError(f"clip {name!r} not in bundle; have {self.clip_names}")
        with np.load(self.path / "clips" / f"{name}.npz") as data:
            clip = Clip(
                name=name,
                command=data["command"],
                reference_qpos=data["reference_qpos"],
                object_pose_local=data["object_pose_local"],
                fps=float(data["fps"]),
            )
        if abs(clip.fps * self.ctrl_dt - 1.0) > 1e-6:
            raise ValueError(f"clip {name} at {clip.fps} fps does not advance one frame per tick")
        return clip

    def parity(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.path / "parity" / f"{name}.npz") as data:
            return {k: data[k] for k in data.files}

    def policy(self) -> "OnnxPolicy":
        return OnnxPolicy(self.path / "policy.onnx", self.obs_dim, self.num_joints)

    def pen_observation(self, pose_local: np.ndarray) -> np.ndarray:
        return bounded_axis_points(pose_local, self.axis_radius, self.tanh_scale)


class OnnxPolicy:
    def __init__(self, path: Path, obs_dim: int, action_dim: int):
        # One thread: the loop is latency-bound, and the camera process wants the cores.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        (inp,), (out,) = self.session.get_inputs(), self.session.get_outputs()
        if inp.shape[-1] != obs_dim or out.shape[-1] != action_dim:
            raise ValueError(
                f"{path.name}: IO {inp.shape}->{out.shape}, contract says {obs_dim}->{action_dim}"
            )
        self._inp, self._out = inp.name, out.name
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.infer(np.zeros(obs_dim, dtype=np.float32))  # first run pays the allocations

    def infer(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim)
        return self.session.run([self._out], {self._inp: x})[0].reshape(self.action_dim)
