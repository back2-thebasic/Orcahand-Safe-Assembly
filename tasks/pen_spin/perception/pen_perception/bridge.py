"""ZMQ link: observer (PUB, binds) -> deploy runner (SUB, connects).

One JSON message per camera frame:
  {"seq": int, "palm_fixed": bool, "latency_s": float,
   "pen": {"pose": [x, y, z, qw, qx, qy, qz], "tags": int, "faces": int, "reproj_px": float}}
``pen`` is absent when the frame gave no pose (or the palm frame is not fixed yet).
``pose`` is the palm-frame pen pose in the policy convention. ``latency_s`` is capture
-> publish on the observer; the receiver dates the frame as its arrival minus that,
which needs no clock shared between the two processes.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import zmq

PORT = 5555


class PosePublisher:
    def __init__(self, port: int = PORT):
        self._sock = zmq.Context.instance().socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 4)
        self._sock.bind(f"tcp://*:{port}")
        self.seq = 0

    def send(self, t_capture: float, palm_fixed: bool, pose7=None, pen_info: dict | None = None) -> None:
        msg = {"seq": self.seq, "palm_fixed": bool(palm_fixed), "latency_s": time.monotonic() - t_capture}
        if pose7 is not None:
            msg["pen"] = {"pose": [float(v) for v in pose7], **(pen_info or {})}
        self.seq += 1
        try:
            self._sock.send_string(json.dumps(msg), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self._sock.close(linger=0)


class PoseReceiver:
    """Latest valid pen pose and its capture time (this process's time.monotonic)."""

    def __init__(self, host: str = "localhost", port: int = PORT):
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._lock = threading.Lock()
        self._pose: np.ndarray | None = None
        self._t_capture: float | None = None
        self.messages = self.poses = self.rejected = 0
        self.palm_fixed = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                raw = self._sock.recv_string()
            except zmq.Again:
                continue
            arrived = time.monotonic()
            try:
                msg = json.loads(raw)
                self._on_msg(msg, arrived)
            except (ValueError, KeyError, TypeError):
                self.rejected += 1

    def _on_msg(self, msg: dict, arrived: float) -> None:
        self.messages += 1
        self.palm_fixed = bool(msg["palm_fixed"])
        if not self.palm_fixed or "pen" not in msg:
            return
        pose = np.asarray(msg["pen"]["pose"], dtype=np.float64)
        latency = float(msg["latency_s"])
        # A single corrupt frame must never reach the policy observation.
        if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isfinite(latency):
            self.rejected += 1
            return
        norm = np.linalg.norm(pose[3:])
        if norm < 1e-6:
            self.rejected += 1
            return
        pose[3:] /= norm
        with self._lock:
            self._pose, self._t_capture = pose, arrived - max(latency, 0.0)
            self.poses += 1

    def latest(self) -> tuple[np.ndarray, float] | None:
        """(pose7, capture time) of the newest valid pose, None before the first."""
        with self._lock:
            return None if self._pose is None else (self._pose.copy(), self._t_capture)

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self._sock.close(linger=0)
