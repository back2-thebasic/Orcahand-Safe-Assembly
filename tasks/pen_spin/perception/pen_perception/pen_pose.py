# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
# Modified for the OrcaHand pen-spin deploy: codebook builder folded in, layout read from
# hardware/tags/pen_tags.json, tag-frame -> policy-frame rotation added.
"""Pen pose from its 18 ArUco tags: detection -> multi-tag PnP -> rigid pen pose.

The pen uses its own codebook (SOURCE_1X2_18). IDs are local pattern indices, not a
predefined ArUco dictionary; their 3D corners come from ``pen_tags.json``. The printed
tags have a black outer ring, a white inner ring and 4x4 data, and the codebook stores
black data as 1, so detectInvertedMarker must be set. Consistent tags constrain one
rigid-body PnP solve; unrelated tags are dropped by whole-marker consensus if the
all-tag fit fails. A single small tag does not give a reliable out-of-plane rotation.

Two pen frames, both with the origin at the shaft midpoint:
  tag frame (pen_tags.json):        +Z toward A_top. PnP solves in this frame.
  policy frame (pen_tracking.xml):  +X toward A_top. Everything downstream uses this.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from . import PEN_TAGS
from .math3d import matrix_from_quat
from .visibility import visibility_failure

DICTIONARY_NAME = "SOURCE_1X2_18"
# Policy frame in the tag frame: R_policy = R_tag @ POLICY_IN_TAG (a -90 deg turn about Y).
POLICY_IN_TAG = matrix_from_quat(np.array([2**-0.5, 0.0, -(2**-0.5), 0.0]))


def make_dictionary(bits):
    raw = np.asarray(bits)
    if raw.ndim != 3 or raw.shape[1:] != (4, 4) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("source codebook must contain binary 4x4 matrices")
    matrices = raw.astype(np.uint8)
    distances = [np.count_nonzero(a != np.rot90(b, rotation))
                 for i, a in enumerate(matrices) for j, b in enumerate(matrices)
                 for rotation in range(4) if i != j or rotation != 0]
    if not distances or min(distances) == 0:
        raise ValueError("source patterns have duplicate codes or ambiguous rotations")
    byte_list = np.concatenate([cv2.aruco.Dictionary_getByteListFromBits(b) for b in matrices])
    # Exact matching: do not silently repair or substitute source patterns.
    return cv2.aruco.Dictionary(byte_list, 4, 0)


@dataclass(frozen=True)
class PenPose:
    """One solved pose, in the camera frame and the pen TAG frame (+Z toward A_top)."""

    rotation: np.ndarray      # (3,3)
    tvec: np.ndarray          # (3,) pen origin in the camera frame, metres
    rvec: np.ndarray          # (3,) Rodrigues, as solvePnP returned it
    tag_ids: tuple[int, ...]  # tags that contributed
    faces: int                # distinct head faces among them
    reproj_px: float          # RMS reprojection error over the used corners
    coplanar: bool            # every contributing tag lay on one face
    ambiguous: bool = False   # several visible planar solutions fit similarly

    @property
    def num_tags(self) -> int:
        return len(self.tag_ids)


class PenTagLayout:
    """Tag corners and head/shaft geometry in the pen tag frame."""

    def __init__(self, path=None):
        self.path = path or PEN_TAGS
        with open(self.path) as f:
            raw = json.load(f)
        self.custom_dictionary = raw["custom_dictionary"]
        if raw["dictionary"] != DICTIONARY_NAME or self.custom_dictionary["marker_size"] != 4:
            raise ValueError("unsupported source dictionary format")
        patterns = self.custom_dictionary["patterns"]
        if [p["id"] for p in patterns] != list(range(len(patterns))):
            raise ValueError("source pattern IDs must be contiguous local indices")
        if {str(p["id"]) for p in patterns} != set(raw["tags"]):
            raise ValueError("source codebook and tag geometry IDs differ")
        self.dictionary = make_dictionary([p["bits"] for p in patterns])
        self.inverted = bool(raw["inverted_markers"])
        self.tag_size = float(raw["tag_size_m"])
        self.error_correction_rate = float(raw["error_correction_rate"])
        self.shaft_half_length = float(raw["shaft_half_length_m"])
        self.shaft_radius = float(raw["shaft_radius_m"])
        self.box_centers = {k: np.asarray(v, dtype=np.float64) for k, v in raw["box_centers_m"].items()}
        self.box_half_sizes = {k: np.asarray(v, dtype=np.float64) for k, v in raw["box_half_sizes_m"].items()}
        # id -> (4,3) corners, in cv2.aruco's corner order
        self.corners: dict[int, np.ndarray] = {
            int(k): np.asarray(v["corners_m"], dtype=np.float64) for k, v in raw["tags"].items()}
        self.face: dict[int, str] = {int(k): f'{v["box"]}{v["face_normal"]}' for k, v in raw["tags"].items()}
        self.face_normal: dict[int, str] = {int(k): v["face_normal"] for k, v in raw["tags"].items()}
        for tid, c in self.corners.items():
            if c.shape != (4, 3):
                raise ValueError(f"tag {tid}: corners must be (4,3), got {c.shape}")

    def __len__(self) -> int:
        return len(self.corners)

    def outward_normals_for(self, ids):
        normals = np.zeros((len(ids), 3))
        for row, tid in enumerate(ids):
            face = self.face_normal[tid]
            normals[row, "XYZ".index(face[1])] = 1 if face[0] == "+" else -1
        return normals


class PenPoseEstimator:
    """Detect the pen's tags in a grayscale image and solve one rigid-body pose.

    ``camera_matrix`` (3,3) and ``dist_coeffs`` are the usual OpenCV intrinsics.
    """

    def __init__(self, camera_matrix, dist_coeffs, *, layout: PenTagLayout | None = None,
                 min_tags: int = 2, max_reproj_px: float = 4.0):
        self.layout = layout or PenTagLayout()
        self.K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        self.min_tags = int(min_tags)
        self.max_reproj_px = float(max_reproj_px)

        params = cv2.aruco.DetectorParameters()
        # The tags are small, oblique and can sit near the image edge. Relax the
        # geometry/threshold search without accepting more corrupt bits.
        params.minMarkerPerimeterRate = 0.015
        params.minDistanceToBorder = 1
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 5
        params.detectInvertedMarker = self.layout.inverted
        params.errorCorrectionRate = self.layout.error_correction_rate
        # Blurred cell boundaries: decode a larger canonical patch and vote on cell
        # centres only. Bit matching stays strict.
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.30
        # The white ring of a small inverted marker is close to its data bits, so a
        # SUBPIX window gets pulled by them; fit corners from the border contour.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self._detector = cv2.aruco.ArucoDetector(self.layout.dictionary, params)
        fallback_params = self._detector.getDetectorParameters()
        fallback_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._contour_fallback = cv2.aruco.ArucoDetector(self.layout.dictionary, fallback_params)
        self.contour_fallbacks = 0
        self._last_rvec: np.ndarray | None = None
        self._last_tvec: np.ndarray | None = None
        self._last_inlier_ids: tuple[int, ...] = ()
        self.rejected_reproj = 0
        self.rejected_visibility = 0
        self.rejected_appearance = 0
        self.coplanar_solves = 0
        self.last_rejection = ""
        self.last_ignored_ids: tuple[int, ...] = ()
        self.last_detection = ([], [])
        self.last_inlier_indices: tuple[int, ...] = ()
        self._missed = 0

    def detect(self, gray: np.ndarray, roi=None):
        """Tags in the ROI, with corner coordinates in the full image."""
        x, y, width, height = roi if roi is not None else (0, 0, gray.shape[1], gray.shape[0])
        image = np.ascontiguousarray(gray[y:y + height, x:x + width])
        try:
            corners, ids, _ = self._detector.detectMarkers(image)
        except cv2.error as exc:
            # Contour refinement can assert on a degenerate edge: retry this frame only.
            if "_interpolate2Dline" not in str(exc) or "nContours.size() >= 2" not in str(exc):
                raise
            self.contour_fallbacks += 1
            corners, ids, _ = self._contour_fallback.detectMarkers(image)
        if ids is None:
            return [], []
        keep_ids, keep_px = [], []
        for quad, tid in zip(corners, ids.flatten(), strict=True):
            t = int(tid)
            if t in self.layout.corners:
                full_quad = quad[0].astype(np.float64) + np.array([x, y])
                if self.layout.inverted and not self._source_border_matches(gray, full_quad, t):
                    self.rejected_appearance += 1
                    continue
                keep_ids.append(t)
                keep_px.append(full_quad)
        return keep_ids, keep_px

    def _source_border_matches(self, gray, quad, tid):
        """The pen's tags have a WHITE inner ring and dark logical-one cells.

        detectInvertedMarker accepts both polarities; a conventional black-border
        pattern elsewhere in the scene must not pass as one of the pen's tags.
        """
        target = np.float32([[0, 0], [48, 0], [48, 48], [0, 48]])
        homography = cv2.getPerspectiveTransform(np.float32(quad), target)
        patch = cv2.warpPerspective(gray, homography, (48, 48))
        cells = patch.reshape(6, 8, 6, 8).transpose(0, 2, 1, 3)
        values = np.median(cells[:, :, 2:6, 2:6], axis=(2, 3))
        border = np.concatenate((values[0], values[-1], values[1:-1, 0], values[1:-1, -1]))
        bits = np.asarray(self.layout.custom_dictionary["patterns"][tid]["bits"], dtype=bool)
        white, black = np.median(border), np.median(values[1:5, 1:5][bits])
        return bool(white - black >= 15.0 and np.mean(border > (white + black) / 2.0) >= 0.8)

    def _candidates(self, obj, img, coplanar):
        """Cold hypotheses every frame, plus a warm solve; no pose averaging."""
        seeds = []
        if coplanar:
            # IPPE expects an exact XY plane: map the face into one, then convert
            # BOTH solutions back into the pen frame.
            origin = obj.mean(axis=0)
            _, _, vt = np.linalg.svd(obj - origin, full_matrices=False)
            basis = np.column_stack((vt[0], vt[1], np.cross(vt[0], vt[1])))
            plane = np.ascontiguousarray((obj - origin) @ basis)
            plane[:, 2] = 0.0
            result = cv2.solvePnPGeneric(plane, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
            if result[0]:
                for rv, tv in zip(result[1], result[2], strict=True):
                    rotation = cv2.Rodrigues(rv)[0] @ basis.T
                    seeds.append((cv2.Rodrigues(rotation)[0], (tv.reshape(3) - rotation @ origin).reshape(3, 1)))
        result = cv2.solvePnPGeneric(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_SQPNP)
        if result[0]:
            seeds.extend(zip(result[1], result[2], strict=True))
        candidates = []
        for rv, tv in seeds:
            candidates.append((rv.copy(), tv.copy()))
            try:
                ok, refined_r, refined_t = cv2.solvePnP(
                    obj, img, self.K, self.dist, rvec=rv.copy(), tvec=tv.copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                continue
            if ok:
                candidates.append((refined_r, refined_t))
        if self._last_rvec is not None:
            # Only this image's solve, seeded by the last pose, is a candidate:
            # returning the previous pose itself would stick.
            try:
                ok, rv, tv = cv2.solvePnP(
                    obj, img, self.K, self.dist, rvec=self._last_rvec.copy(),
                    tvec=self._last_tvec.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    candidates.append((rv, tv))
            except cv2.error:
                pass
        return candidates

    def _fit(self, ids, quads):
        """Fit a selected marker set, keeping both visible planar branches."""
        obj = np.concatenate([self.layout.corners[t] for t in ids], axis=0)
        img = np.concatenate(quads, axis=0)
        singular = np.linalg.svd(obj - obj.mean(axis=0), compute_uv=False)
        coplanar = singular[-1] < 1e-6
        try:
            candidates = self._candidates(obj, img, coplanar)
        except cv2.error:
            candidates = []
        visible, failures = [], []
        for rvec, tvec in candidates:
            rotation = cv2.Rodrigues(rvec)[0]
            failure = visibility_failure(self.layout, ids, rotation, tvec)
            if failure:
                self.rejected_visibility += 1
                failures.append(failure)
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            if np.isfinite(err):
                visible.append((err, rvec, tvec, rotation))
        visible.sort(key=lambda candidate: candidate[0])
        return visible, failures, coplanar

    def _consensus_support(self, ids, img, obj, rv, tv, rotation):
        """Score every current marker; duplicated IDs count once."""
        projected = cv2.projectPoints(obj, rv, tv, self.K, self.dist)[0]
        errors = np.sqrt(np.mean(np.sum((projected.reshape(-1, 4, 2) - img) ** 2, axis=2), axis=1))
        chosen = {}
        for i in np.argsort(errors):
            if not np.isfinite(errors[i]) or errors[i] > self.max_reproj_px or ids[i] in chosen:
                continue
            if not visibility_failure(self.layout, [ids[i]], rotation, tv):
                chosen[ids[i]] = int(i)
        indices = tuple(sorted(chosen.values()))
        error = float(np.mean(errors[list(indices)])) if indices else float("inf")
        return indices, error

    def _refit_consensus(self, ids, quads, supports):
        for indices in sorted(supports, key=lambda s: (-len(s), supports[s])):
            visible, _, coplanar = self._fit([ids[i] for i in indices], [quads[i] for i in indices])
            if visible and visible[0][0] <= self.max_reproj_px:
                return indices, visible, coplanar
        return None

    def _consensus(self, ids, quads):
        """Recover the majority rigid object when stray detections contribute tags.

        Two-marker hypotheses scored on all corners of every marker; at least three
        distinct IDs and a strict majority are required.
        """
        required = max(3, self.min_tags, len(ids) // 2 + 1)
        if len(set(ids)) < required:
            return None
        obj = np.concatenate([self.layout.corners[t] for t in ids])
        img = np.asarray(quads)
        # Fast path: last frame's inlier IDs select this frame's corners for a fresh solve.
        if self._last_rvec is not None and self._last_inlier_ids:
            projected = cv2.projectPoints(obj, self._last_rvec, self._last_tvec,
                                          self.K, self.dist)[0].reshape(-1, 4, 2)
            distances = np.sum((projected - img) ** 2, axis=(1, 2))
            chosen = {}
            for i in np.argsort(distances):
                if ids[i] in self._last_inlier_ids and ids[i] not in chosen:
                    chosen[ids[i]] = int(i)
            if len(chosen) >= required:
                indices = sorted(chosen.values())
                seeds, _, _ = self._fit([ids[i] for i in indices], [quads[i] for i in indices])
                supports = {}
                for _, rv, tv, rotation in seeds:
                    support, error = self._consensus_support(ids, img, obj, rv, tv, rotation)
                    if len(support) >= required:
                        supports[support] = min(supports.get(support, float("inf")), error)
                result = self._refit_consensus(ids, quads, supports)
                if result is not None:
                    return result
        supports = {}
        for a, b in combinations(range(len(ids)), 2):
            if ids[a] == ids[b]:
                continue
            seeds, _, _ = self._fit([ids[a], ids[b]], [quads[a], quads[b]])
            for _, rv, tv, rotation in seeds:
                indices, error = self._consensus_support(ids, img, obj, rv, tv, rotation)
                if len(indices) >= required:
                    supports[indices] = min(supports.get(indices, float("inf")), error)
        return self._refit_consensus(ids, quads, supports)

    def estimate(self, gray: np.ndarray, roi=None) -> PenPose | None:
        """A reprojection-consistent, physically visible rigid-body pose, or None."""
        self.last_rejection = ""
        self.last_ignored_ids = ()
        self.last_inlier_indices = ()
        ids, quads = self.detect(gray, roi)
        self.last_detection = (ids, quads)
        if len(ids) < self.min_tags:
            self.last_rejection = "not enough tags"
            self._missed += 1
            if self._missed > 12:
                self.reset()
            return None
        self._missed = 0
        inlier_indices = tuple(range(len(ids)))
        visible, failures, coplanar = self._fit(ids, quads)
        if not visible or visible[0][0] > self.max_reproj_px:
            consensus = self._consensus(ids, quads)
            if consensus is not None:
                indices, visible, coplanar = consensus
                inlier_indices = indices
                self.last_ignored_ids = tuple(t for i, t in enumerate(ids) if i not in indices)
                ids = [ids[i] for i in indices]
        if not visible:
            self.last_rejection = failures[0] if failures else "PnP failed"
            self.reset()
            return None
        best = visible[0]
        if best[0] > self.max_reproj_px:
            self.rejected_reproj += 1
            self.last_rejection = f"reprojection {best[0]:.1f}px"
            self.reset()
            return None

        # A subpixel residual difference can be corner noise: among similarly good
        # visible hypotheses the last rotation breaks the tie, never more.
        near = [c for c in visible if c[0] <= min(best[0] + 0.15, self.max_reproj_px)]
        ambiguous = coplanar and any(
            np.trace(candidate[3] @ best[3].T) < 1 + 2 * np.cos(np.radians(5)) for candidate in near)
        if self._last_rvec is not None:
            previous = cv2.Rodrigues(self._last_rvec)[0]

            def continuity(candidate):
                angle = np.arccos(np.clip((np.trace(candidate[3] @ previous.T) - 1) / 2, -1, 1))
                shift = np.linalg.norm(candidate[2].reshape(3) - self._last_tvec.reshape(3))
                return angle + shift / max(np.linalg.norm(self._last_tvec), 0.1)

            best = min(near, key=continuity)
        err, rvec, tvec, rotation = best
        if coplanar:
            self.coplanar_solves += 1
        self._last_rvec, self._last_tvec = rvec.copy(), tvec.copy()
        self._last_inlier_ids = tuple(ids)
        self.last_inlier_indices = inlier_indices
        return PenPose(rotation=rotation, tvec=tvec.reshape(3).copy(), rvec=rvec.reshape(3).copy(),
                       tag_ids=tuple(ids), faces=len({self.layout.face[t] for t in ids}),
                       reproj_px=err, coplanar=coplanar, ambiguous=ambiguous)

    def reset(self) -> None:
        """Drop the pose seed (after a long gap in detections)."""
        self._last_rvec = self._last_tvec = None
        self._last_inlier_ids = ()
        self._missed = 0
