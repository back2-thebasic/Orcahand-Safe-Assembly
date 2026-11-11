"""End-to-end check of the pose pipeline on rendered images with known ground truth.

The textured training pen (models/pen/assets/pen_tracking.xml) is rendered by MuJoCo
along a clip's reference pen poses, from several viewpoints; a wrist AprilTag is drawn
in at a known pose. The observer pipeline (pen PnP, palm frame from the tag, tag frame
-> policy frame) must return the clip's palm-frame pen pose.

    python perception/tools/synthetic_check.py [--clip clip_002] [--save DIR]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pen_perception import PEN_SPIN  # noqa: E402
from pen_perception.math3d import compose, invert, matrix_from_quat, rotation_angle_deg  # noqa: E402
from pen_perception.palm import PalmFrame, WristTag, detect_wrist_tag  # noqa: E402
from pen_perception.pen_pose import PenPoseEstimator  # noqa: E402

BUNDLE = PEN_SPIN / "policies/clip002_004_dr_scratch/deploy"
PEN_XML = PEN_SPIN / "models/pen/assets/pen_tracking.xml"
W, H, FOCAL = 1280, 1024, 1970.0
FOVY_DEG = float(np.degrees(2 * np.arctan(H / 2 / FOCAL)))
K = np.array([[FOCAL, 0, (W - 1) / 2], [0, FOCAL, (H - 1) / 2], [0, 0, 1]])
DIST = np.zeros(5)
# Viewpoints around the pen: (azimuth deg, elevation deg, distance m).
VIEWS = {"front": (0, 50, 0.38), "side": (90, 45, 0.38), "back-high": (200, 65, 0.42), "low": (300, 30, 0.40)}
BACKGROUND = 110
PALM_SAMPLES = 10
AXIS_RADIUS = 0.1


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera axes (x right, y up, looking along -z) as a 3x3, columns x y z."""
    z = (eye - target) / np.linalg.norm(eye - target)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def build_model(target: np.ndarray) -> tuple[mujoco.MjModel, dict]:
    root = ET.parse(PEN_XML).getroot()
    root.find("compiler").set("texturedir", str((PEN_XML.parent / "../textures").resolve()))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(W), offheight=str(H))
    ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.4 0.4 0.4", specular="0 0 0")
    world = root.find("worldbody")
    ET.SubElement(world, "light", directional="true", pos="0 0 2", dir="0.2 0.3 -1", diffuse="0.35 0.35 0.35")
    cams = {}
    for name, (az, el, dist) in VIEWS.items():
        a, e = np.radians(az), np.radians(el)
        eye = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        axes = look_at(eye, target)
        ET.SubElement(world, "camera", name=name, pos=" ".join(map(str, eye)), fovy=str(FOVY_DEG),
                      xyaxes=" ".join(map(str, np.concatenate((axes[:, 0], axes[:, 1])))))
        cams[name] = eye
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), cams


def tag_artwork(tag: WristTag, cell: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """The printed tag with a one-cell white quiet zone, and its black-edge corners (px)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag.tag_id, 8 * cell, borderBits=1)
    art = cv2.copyMakeBorder(marker, cell, cell, cell, cell, cv2.BORDER_CONSTANT, value=255)
    lo, hi = cell - 0.5, 9 * cell - 0.5
    return art, np.float32([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])


def project(T_cam_x, points: np.ndarray) -> np.ndarray:
    R, t = T_cam_x
    return cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, K, DIST)[0].reshape(-1, 2)


def compose_frame(rgb, seg, tag_img, rng, noise: float) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pen = seg[..., 0] >= 0
    out = np.full_like(gray, BACKGROUND)
    art, mask = tag_img
    out = out * (1 - mask) + art * mask
    out[pen] = gray[pen]
    out += rng.normal(0.0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", default="clip_002")
    p.add_argument("--step", type=int, default=4, help="use every Nth clip frame")
    p.add_argument("--noise", type=float, default=2.0, help="image noise, gray levels (sigma)")
    p.add_argument("--save", type=Path, default=None, help="write the annotated first frame of each view here")
    args = p.parse_args()

    import json
    contract = json.loads((BUNDLE / "contract.json").read_text())
    palm_w = np.asarray(contract["palm"]["pose_w"], dtype=np.float64)
    T_w_palm = (matrix_from_quat(palm_w[3:]), palm_w[:3])
    local = np.load(BUNDLE / "clips" / f"{args.clip}.npz")["object_pose_local"][:: args.step].astype(np.float64)
    target = compose(T_w_palm, (np.eye(3), local[:, :3].mean(axis=0)))[1]

    # A wrist tag placed where every view sees it, below the pen and facing up.
    z = np.array([0.15, -0.1, 1.0])
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    T_w_tag = (np.column_stack((x, np.cross(z, x), z)), target + np.array([0.0, 0.0, -0.07]))
    R_palm_tag, t_palm_tag = compose(invert(T_w_palm), T_w_tag)
    tag = WristTag(tag_id=0, size_m=0.0504, R_palm_tag=R_palm_tag, t_palm_tag=t_palm_tag, measured=True)
    art, art_corners = tag_artwork(tag)

    model, _ = build_model(target)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    rng = np.random.default_rng(0)
    print(f"{args.clip}: {len(local)} poses x {len(VIEWS)} views, {W}x{H} f={FOCAL:.0f}px, noise {args.noise}")
    print(f"{'view':10s} {'palm frame':>18s} {'poses':>7s} {'pos mm':>16s} {'shaft deg':>16s} "
          f"{'axis pts mm':>12s} {'flips':>5s}")
    print(f"{'':10s} {'(mm / deg err)':>18s} {'':>7s} {'median   p95':>16s} {'median   p95':>16s} {'max':>12s}")
    all_ok = True
    for view in VIEWS:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view)
        mujoco.mj_forward(model, data)
        T_w_cam = (data.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), data.cam_xpos[cam_id].copy())
        T_cam_tag = compose(invert(T_w_cam), T_w_tag)
        corners = project(T_cam_tag, tag.object_points()).astype(np.float32)
        homography = cv2.getPerspectiveTransform(art_corners, corners)
        tag_img = (cv2.warpPerspective(art.astype(np.float32), homography, (W, H)),
                   cv2.warpPerspective(np.ones_like(art, np.float32), homography, (W, H)))

        # The palm frame is fixed before the pen is placed.
        no_pen = np.full((H, W, 2), -1)
        palm = PalmFrame(tag, PALM_SAMPLES)
        while not palm.fixed:
            found = detect_wrist_tag(compose_frame(np.zeros((H, W, 3), np.uint8), no_pen, tag_img, rng, args.noise),
                                     K, DIST, tag)
            if found is None:
                raise RuntimeError(f"{view}: wrist tag not detected")
            palm.add(*found)
        estimator = PenPoseEstimator(K, DIST)
        pos_err, shaft_err, point_err, flips, missing = [], [], [], 0, 0
        for i, pose_local in enumerate(local):
            R_w, pos_w = compose(T_w_palm, (matrix_from_quat(pose_local[3:]), pose_local[:3]))
            quat_w = np.zeros(4)
            mujoco.mju_mat2Quat(quat_w, R_w.reshape(-1))
            data.qpos[:7] = np.concatenate((pos_w, quat_w))
            mujoco.mj_forward(model, data)
            renderer.disable_segmentation_rendering()
            renderer.update_scene(data, camera=view)
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=view)
            seg = renderer.render().copy()
            gray = compose_frame(rgb, seg, tag_img, rng, args.noise)
            if args.save is not None and i == 0:
                args.save.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save / f"{view}.png"), gray)
            pose = estimator.estimate(gray)
            if pose is None:
                missing += 1
                continue
            est = palm.pen_in_palm(pose)
            truth_axis = matrix_from_quat(pose_local[3:])[:, 0]
            est_axis = matrix_from_quat(est[3:])[:, 0]
            angle = float(np.degrees(np.arccos(np.clip(truth_axis @ est_axis, -1, 1))))
            flips += angle > 90.0
            pos_err.append(1000 * np.linalg.norm(est[:3] - pose_local[:3]))
            shaft_err.append(angle)
            point_err.append(1000 * max(np.linalg.norm(est[:3] + s * AXIS_RADIUS * est_axis
                                                       - pose_local[:3] - s * AXIS_RADIUS * truth_axis)
                                        for s in (1, -1)))
        T_palm_cam_true = compose(invert(T_w_palm), T_w_cam)
        frame_mm = 1000 * np.linalg.norm(palm.T_palm_cam[1] - T_palm_cam_true[1])
        frame_deg = rotation_angle_deg(palm.T_palm_cam[0], T_palm_cam_true[0])
        n = len(local) - missing
        if n:
            print(f"{view:10s} {frame_mm:8.2f} / {frame_deg:5.2f}  {n:3d}/{len(local):<3d} "
                  f"{np.median(pos_err):7.2f} {np.percentile(pos_err, 95):7.2f}  "
                  f"{np.median(shaft_err):7.2f} {np.percentile(shaft_err, 95):7.2f}  "
                  f"{max(point_err):11.2f} {flips:5d}")
        else:
            print(f"{view:10s} {frame_mm:8.2f} / {frame_deg:5.2f}    0/{len(local)}")
        all_ok &= n > 0 and flips == 0
    renderer.close()
    print("OK" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
