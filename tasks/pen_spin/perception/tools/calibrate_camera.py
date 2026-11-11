"""Camera intrinsics from a ChArUco board; writes config/camera.yaml.

    # print the board (check the printed square size with a ruler: pass it as --square-mm)
    python perception/tools/calibrate_camera.py --make-board board.png
    # live: Space keeps the current view (vary distance and tilt, cover the corners), Enter solves
    python perception/tools/calibrate_camera.py --source device:0
    # or a folder of board images / a video
    python perception/tools/calibrate_camera.py --source calib_images/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pen_perception import CONFIG  # noqa: E402
from pen_perception.camera import CameraModel, open_source  # noqa: E402

MIN_CORNERS = 12
MIN_VIEWS = 15


def board(args) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_mm / 1000,
                                  args.marker_mm / 1000, dictionary)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="device:N, a video file, or a directory of images")
    p.add_argument("--make-board", type=Path, help="write a printable board image (A4 at 300 dpi) and exit")
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=5)
    p.add_argument("--square-mm", type=float, default=35.0)
    p.add_argument("--marker-mm", type=float, default=26.0)
    p.add_argument("--camera", type=Path, default=CONFIG / "camera.yaml", help="file to update")
    p.add_argument("--every", type=int, default=10, help="video/folder sources: try every Nth frame")
    args = p.parse_args()
    charuco = board(args)

    if args.make_board:
        px_per_mm = 300 / 25.4
        size = (round(args.squares_x * args.square_mm * px_per_mm), round(args.squares_y * args.square_mm * px_per_mm))
        cv2.imwrite(str(args.make_board), charuco.generateImage(size, marginSize=round(10 * px_per_mm)))
        print(f"{args.make_board}: print at 300 dpi, 100% scale; squares should measure {args.square_mm} mm")
        return 0
    if not args.source:
        p.error("--source or --make-board")

    camera = CameraModel.load(args.camera)
    detector = cv2.aruco.CharucoDetector(charuco)
    live = args.source.startswith("device:")
    obj_points, img_points, size = [], [], None
    for i, (gray, _) in enumerate(open_source(args.source, camera)):
        size = gray.shape[::-1]
        if not live and i % args.every:
            continue
        corners, ids, _, _ = detector.detectBoard(gray)
        n = 0 if ids is None else len(ids)
        keep = n >= MIN_CORNERS
        if live:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if n:
                cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
            cv2.putText(vis, f"{len(obj_points)} views kept, {n} corners  (Space keep, Enter solve, q quit)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if keep else (0, 0, 255), 2)
            cv2.imshow("calibrate", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 1
            if key in (10, 13):
                break
            keep = keep and key == ord(" ")
        if keep:
            obj, img = charuco.matchImagePoints(corners, ids)
            obj_points.append(obj)
            img_points.append(img)
    if live:
        cv2.destroyAllWindows()
    if len(obj_points) < MIN_VIEWS:
        print(f"only {len(obj_points)} usable views, need {MIN_VIEWS}", file=sys.stderr)
        return 1

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    dist = dist.reshape(-1)
    print(f"{len(obj_points)} views, {size[0]}x{size[1]}, RMS reprojection {rms:.3f} px")
    print(f"fx {K[0, 0]:.2f}  fy {K[1, 1]:.2f}  cx {K[0, 2]:.2f}  cy {K[1, 2]:.2f}  dist {np.round(dist[:5], 5)}")
    cfg = yaml.safe_load(args.camera.read_text())
    cfg.update(calibrated=True, width=int(size[0]), height=int(size[1]),
               intrinsics={k: float(v) for k, v in zip(("fx", "fy", "cx", "cy"),
                                                       (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))},
               distortion={k: float(v) for k, v in zip(("k1", "k2", "p1", "p2", "k3"), dist[:5])})
    args.camera.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.camera}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
