"""Camera -> palm-frame pen pose, published over ZMQ for penspin_deploy (--pen camera).

The first --palm-frames wrist-tag detections fix the camera -> palm transform; after
that every frame is published, with or without a pen pose. Nothing is smoothed, held
or gated here: the receiver keeps the last valid pose and ages it.

    python perception/observer.py --source device:0 --preview
    python perception/observer.py --source recording.mp4 --preview

Preview keys: q quits, w resamples the palm frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pen_perception.bridge import PORT, PosePublisher
from pen_perception.camera import CameraModel, open_source
from pen_perception.palm import PalmFrame, WristTag, detect_wrist_tag
from pen_perception.pen_pose import PenPoseEstimator, PenTagLayout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="device:N, a video file, or a directory of images")
    p.add_argument("--camera", type=Path, default=None, help="intrinsics (default: config/camera.yaml)")
    p.add_argument("--wrist-tag", type=Path, default=None, help="tag pose in the palm (default: config/wrist_tag.yaml)")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--palm-frames", type=int, default=100, help="wrist-tag detections averaged into the palm frame")
    p.add_argument("--min-faces", type=int, default=1,
                   help="drop poses from fewer head faces (2 drops the weaker single-face solves)")
    p.add_argument("--allow-placeholders", action="store_true",
                   help="run with an uncalibrated camera or unmeasured wrist tag (bench work only)")
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


def pen_wireframe(layout: PenTagLayout):
    """Head boxes and shaft axis in the pen tag frame, from the numbers the solver uses."""
    pts, edges = [], []
    for box in ("A_top", "A_bottom"):
        half, center, base = layout.box_half_sizes[box], layout.box_centers[box], len(pts)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(center + np.array([sx, sy, sz]) * half)
        for a, b in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                     (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
            edges.append((base + a, base + b))
    base = len(pts)
    pts += [[0, 0, -layout.shaft_half_length], [0, 0, layout.shaft_half_length]]
    edges.append((base, base + 1))
    return np.asarray(pts, dtype=np.float64), edges


def draw_pen(vis, pose, camera: CameraModel, wireframe) -> None:
    pts, edges = wireframe
    colour = (0, 165, 255) if pose.coplanar else (0, 255, 0)
    uv = cv2.projectPoints(pts, pose.rvec, pose.tvec, camera.K, camera.dist)[0].reshape(-1, 2)
    if np.isfinite(uv).all():
        for a, b in edges:
            cv2.line(vis, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), colour, 1)
        # A_top end of the shaft: the policy's +X.
        cv2.circle(vis, tuple(uv[-1].astype(int)), 5, (0, 0, 255), -1)


def draw_tag(vis, T_cam_tag, camera: CameraModel, tag: WristTag, fixed: bool) -> None:
    R, t = T_cam_tag
    if t[2] <= 0:
        return
    rvec = cv2.Rodrigues(R)[0]
    uv = cv2.projectPoints(tag.object_points(), rvec, t, camera.K, camera.dist)[0].reshape(4, 2)
    if np.isfinite(uv).all():
        cv2.polylines(vis, [uv.astype(int)], True, (255, 255, 0) if fixed else (0, 255, 255), 2)
        cv2.drawFrameAxes(vis, camera.K, camera.dist, rvec, t, tag.size_m, 2)


def text(vis, line: str, row: int, colour) -> None:
    cv2.putText(vis, line, (10, 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, line, (10, 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)


def main() -> int:
    args = parse_args()
    camera = CameraModel.load(args.camera)
    tag = WristTag.load(args.wrist_tag)
    placeholders = [name for name, ok in (("camera intrinsics", camera.calibrated),
                                          ("wrist tag pose", tag.measured)) if not ok]
    if placeholders and not args.allow_placeholders:
        print(f"[observer] placeholder {' and '.join(placeholders)}: calibrate / measure first, "
              "or pass --allow-placeholders for bench work", file=sys.stderr)
        return 2
    for name in placeholders:
        print(f"[observer] WARNING: placeholder {name}; palm-frame poses are not meaningful")

    estimator = PenPoseEstimator(camera.K, camera.dist)
    palm = PalmFrame(tag, args.palm_frames)
    wireframe = pen_wireframe(estimator.layout) if args.preview else None
    publisher = PosePublisher(args.port)
    print(f"[observer] {args.source} -> tcp://*:{args.port}  ({len(estimator.layout)} pen tags, "
          f"wrist tag {tag.tag_id} {tag.size_m * 1000:.1f} mm)")

    frames = poses = 0
    total_frames = total_poses = 0
    t_report = time.monotonic()
    try:
        for gray, t_capture in open_source(args.source, camera):
            camera.check_frame(gray)
            frames += 1
            total_frames += 1
            live_tag = None
            if not palm.fixed:
                live_tag = detect_wrist_tag(gray, camera.K, camera.dist, tag)
                if live_tag is not None:
                    palm.add(*live_tag)
                    if palm.fixed:
                        print(f"[observer] palm frame fixed from {palm.samples} detections")
            pose = estimator.estimate(gray) if palm.fixed else None
            if pose is not None and pose.faces < args.min_faces:
                pose = None
            if pose is None:
                publisher.send(t_capture, palm.fixed)
            else:
                poses += 1
                total_poses += 1
                publisher.send(t_capture, True, palm.pen_in_palm(pose),
                               {"tags": pose.num_tags, "faces": pose.faces, "reproj_px": round(pose.reproj_px, 3)})

            if args.preview:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if palm.fixed:
                    draw_tag(vis, palm.T_cam_tag, camera, tag, fixed=True)
                    text(vis, "palm frame FIXED (w resamples)", 2, (0, 255, 0))
                else:
                    if live_tag is not None:
                        draw_tag(vis, live_tag, camera, tag, fixed=False)
                    text(vis, f"sampling palm frame {palm.samples}/{palm.sample_frames}: "
                              f"{'tag seen' if live_tag is not None else f'show wrist tag {tag.tag_id}'}",
                         2, (0, 255, 255))
                ids, quads = estimator.last_detection if palm.fixed else estimator.detect(gray)
                for tid, quad in zip(ids, quads):
                    cv2.polylines(vis, [quad.astype(int)], True, (0, 200, 255), 1)
                    cv2.putText(vis, str(tid), tuple(quad[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 200, 255), 1)
                if pose is not None:
                    draw_pen(vis, pose, camera, wireframe)
                    text(vis, f"{pose.num_tags} tags / {pose.faces} faces  reproj {pose.reproj_px:.2f} px"
                              f"{'  PLANAR AMBIGUOUS' if pose.ambiguous else '  coplanar' if pose.coplanar else ''}",
                         1, (0, 165, 255) if pose.coplanar else (0, 255, 0))
                elif palm.fixed:
                    text(vis, f"no pose: {estimator.last_rejection or 'rejected'}", 1, (0, 0, 255))
                cv2.imshow("pen observer", vis)
                key = cv2.pollKey() & 0xFF
                if key == ord("q"):
                    break
                if key == ord("w"):
                    palm.reset()
                    print("[observer] resampling the palm frame")

            now = time.monotonic()
            if now - t_report >= 2.0:
                print(f"[observer] {frames / (now - t_report):5.1f} fps  poses {100 * poses / max(frames, 1):3.0f}%  "
                      f"rejects: reproj {estimator.rejected_reproj}  visibility {estimator.rejected_visibility}  "
                      f"appearance {estimator.rejected_appearance}", flush=True)
                frames = poses = 0
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[observer] {total_frames} frames, {total_poses} poses")
        publisher.close()
        if args.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
