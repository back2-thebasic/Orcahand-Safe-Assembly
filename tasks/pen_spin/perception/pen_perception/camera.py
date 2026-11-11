"""Camera intrinsics (config/camera.yaml) and frame sources.

A source yields (gray uint8 image, capture time on time.monotonic). Vendor SDK
cameras plug in as another generator with the same contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import CONFIG

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray        # (3,3)
    dist: np.ndarray     # OpenCV order: k1 k2 p1 p2 k3
    width: int
    height: int
    calibrated: bool
    capture: dict

    @classmethod
    def load(cls, path: Path | None = None) -> "CameraModel":
        cfg = yaml.safe_load(Path(path or CONFIG / "camera.yaml").read_text())
        i = cfg["intrinsics"]
        K = np.array([[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]])
        d = cfg["distortion"]
        dist = np.array([d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]], dtype=np.float64)
        return cls(K=K, dist=dist, width=int(cfg["width"]), height=int(cfg["height"]),
                   calibrated=bool(cfg["calibrated"]), capture=cfg.get("capture") or {})

    def check_frame(self, gray: np.ndarray) -> None:
        if gray.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame is {gray.shape[1]}x{gray.shape[0]}, the intrinsics are for "
                             f"{self.width}x{self.height}")


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def open_source(spec: str, camera: CameraModel):
    """``device:N`` (OpenCV camera), a video file, or a directory of images."""
    if spec.startswith("device:"):
        return _device_frames(int(spec.split(":", 1)[1]), camera)
    path = Path(spec)
    if path.is_dir():
        return _image_frames(path)
    if path.is_file():
        return _video_frames(path)
    raise FileNotFoundError(f"no such source: {spec}")


def _device_frames(index: int, camera: CameraModel):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera device {index}")
    # Queued frames would be dated when read, not when exposed.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    if "fps" in camera.capture:
        cap.set(cv2.CAP_PROP_FPS, float(camera.capture["fps"]))
    if "exposure" in camera.capture:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(camera.capture["exposure"]))
    try:
        while True:
            ok, frame = cap.read()
            t = time.monotonic()
            if not ok:
                break
            yield _gray(frame), t
    finally:
        cap.release()


def _video_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield _gray(frame), time.monotonic()
    finally:
        cap.release()


def _image_frames(folder: Path):
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"unreadable image {path}")
        yield gray, time.monotonic()
