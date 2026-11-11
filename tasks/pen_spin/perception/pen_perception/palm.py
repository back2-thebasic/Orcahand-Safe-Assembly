# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
# Modified for the OrcaHand pen-spin deploy: the tag pose in the palm frame comes from
# config/wrist_tag.yaml, and poses leave in the palm frame and the policy pen frame.
"""Camera -> palm frame through an AprilTag rigidly mounted relative to the palm.

Tag frame (OpenCV marker convention): origin at the tag centre, +X to the right and
+Y up as the tag is printed, +Z out of the printed face.
Palm frame: the training scene's ``robot/right_palm_link``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import CONFIG
from .math3d import compose, invert, matrix_from_quat, mean_rotation, quat_from_matrix
from .pen_pose import POLICY_IN_TAG, PenPose


@dataclass(frozen=True)
class WristTag:
    tag_id: int
    size_m: float               # black border edge
    R_palm_tag: np.ndarray      # (3,3)
    t_palm_tag: np.ndarray      # (3,)
    measured: bool

    @classmethod
    def load(cls, path: Path | None = None) -> "WristTag":
        cfg = yaml.safe_load(Path(path or CONFIG / "wrist_tag.yaml").read_text())
        return cls(tag_id=int(cfg["tag_id"]), size_m=float(cfg["size_m"]),
                   R_palm_tag=matrix_from_quat(np.asarray(cfg["quat_in_palm"], dtype=np.float64)),
                   t_palm_tag=np.asarray(cfg["pos_in_palm"], dtype=np.float64),
                   measured=bool(cfg["measured"]))

    def object_points(self) -> np.ndarray:
        """Corners in the tag frame, in the detector's order (TL, TR, BR, BL)."""
        h = self.size_m / 2.0
        return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


@lru_cache(maxsize=1)
def _apriltag_detectors():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    coarse = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return coarse, cv2.aruco.ArucoDetector(dictionary, params)


def detect_wrist_tag(gray, K, dist, tag: WristTag):
    """(R, t) of the tag in the camera frame, or None."""
    coarse, precise = _apriltag_detectors()
    corners, ids, _ = coarse.detectMarkers(gray)
    if ids is None or tag.tag_id not in ids.flatten():
        return None
    quad = corners[list(ids.flatten()).index(tag.tag_id)][0]
    # Coarse corners jump by 1-3 px on a still tag (~5 deg IPPE jumps): refine the
    # AprilTag edges in a crop around it, located anew every frame.
    pad = max(8.0, 0.15 * np.max(np.ptp(quad, axis=0)))
    lo = np.maximum(np.floor(quad.min(axis=0) - pad).astype(int), 0)
    hi = np.minimum(np.ceil(quad.max(axis=0) + pad).astype(int), gray.shape[::-1])
    refined, refined_ids, _ = precise.detectMarkers(np.ascontiguousarray(gray[lo[1]:hi[1], lo[0]:hi[0]]))
    if refined_ids is None or tag.tag_id not in refined_ids.flatten():
        return None
    quad = refined[list(refined_ids.flatten()).index(tag.tag_id)][0].astype(np.float64) + lo
    ok, rvec, tvec = cv2.solvePnP(tag.object_points(), quad, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


class PalmFrame:
    """Camera -> palm, from the wrist tag averaged over the first N detections.

    Held fixed afterwards: re-estimating it every frame would add the tag's own
    detection noise to every pen pose. ``reset()`` resamples.
    """

    def __init__(self, tag: WristTag, sample_frames: int = 100):
        if sample_frames < 1:
            raise ValueError("sample_frames must be positive")
        self.tag, self.sample_frames = tag, sample_frames
        self._R: list[np.ndarray] = []
        self._t: list[np.ndarray] = []
        self.T_palm_cam: tuple[np.ndarray, np.ndarray] | None = None
        self.T_cam_tag: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def fixed(self) -> bool:
        return self.T_palm_cam is not None

    @property
    def samples(self) -> int:
        return len(self._t)

    def reset(self) -> None:
        self._R.clear()
        self._t.clear()
        self.T_palm_cam = self.T_cam_tag = None

    def add(self, R_cam_tag: np.ndarray, t_cam_tag: np.ndarray) -> None:
        if self.fixed:
            return
        self._R.append(R_cam_tag)
        self._t.append(t_cam_tag)
        if len(self._R) >= self.sample_frames:
            self.T_cam_tag = (mean_rotation(self._R), np.median(np.array(self._t), axis=0))
            self.T_palm_cam = compose((self.tag.R_palm_tag, self.tag.t_palm_tag), invert(self.T_cam_tag))

    def pen_in_palm(self, pose: PenPose) -> np.ndarray:
        """Pen pose as the policy reads it: palm-frame [pos, quat_wxyz], +X toward A_top."""
        R, t = compose(self.T_palm_cam, (pose.rotation @ POLICY_IN_TAG, pose.tvec))
        return np.concatenate((t, quat_from_matrix(R)))
