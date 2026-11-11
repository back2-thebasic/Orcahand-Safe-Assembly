"""The camera pen pose from perception/observer.py, as the controller's pen source."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))

from pen_perception.bridge import PORT, PoseReceiver  # noqa: E402


class CameraPen:
    """``sample()``: the newest palm-frame pen pose and its age in control ticks,
    None until the observer has published one."""

    def __init__(self, ctrl_dt: float, host: str = "localhost", port: int = PORT):
        self.ctrl_dt = ctrl_dt
        self.receiver = PoseReceiver(host, port)

    def sample(self) -> tuple[np.ndarray, int] | None:
        latest = self.receiver.latest()
        if latest is None:
            return None
        pose, t_capture = latest
        return pose, max(0, round((time.monotonic() - t_capture) / self.ctrl_dt))

    def close(self) -> None:
        self.receiver.close()
