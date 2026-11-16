# OrcaHand v1 Pen Spin Deployment

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## 0. Set up the terminal paths

Run this block once in each new terminal. `mkdir -p` creates the log directory and preserves existing logs.

```bash
ORCA_ROOT=/Users/su/Code/Orcahand
ORCA_PY="$ORCA_ROOT/.scratch/deploy-venv/bin/python"
ORCA_SCRIPTS="$ORCA_ROOT/orca_core/scripts"
ORCA_CFG="$ORCA_ROOT/orca_core/orca_core/models/v1/orcahand-right/config.yaml"
PEN_TASK="$ORCA_ROOT/tasks/pen_spin"
PEN_BUNDLE="$PEN_TASK/policies/clip002_004_dr_scratch/deploy"
PEN_LOGS="$PEN_TASK/deploy/logs"
PEN_CAMERA=device:0
mkdir -p "$PEN_LOGS"
```

## 1. Verify ONNX and run the simulation

The first command compares ONNX output with the reference results saved during export. Look for `PASS`. The second runs `clip_002` in MuJoCo. Check that `completed` is `true` and `pen_lost_at_s` is `null`. Run both before connecting the physical hand.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy.checks parity "$PEN_BUNDLE"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand twin --pen-model training
```

## 2. Check communication, calibrate the hand, and test the neutral pose

Power the OrcaHand, connect it over USB, and keep the hand empty. The command below tries to connect to the motor bus, prints the result, and disconnects. Continue when you see `Connection successful`.

```bash
"$ORCA_PY" - "$ORCA_CFG" <<'PY'
import sys
from orca_core import OrcaHand

hand = OrcaHand(config_path=sys.argv[1])
ok, message = hand.connect(interactive=False)
print(message)
if ok:
    hand.disconnect()
raise SystemExit(0 if ok else 1)
PY
```

If the tendons on a newly assembled hand have yet to be tensioned, run the command below. The script moves the motors. Follow the assembly procedure to tension the tendons and press `Ctrl+C` when finished.

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/tension.py" "$ORCA_CFG"
```

If the hand needs calibration, run the following command to measure joint travel. It saves the results to `calibration.yaml` alongside `config.yaml`.

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/calibrate.py" "$ORCA_CFG"
```

After calibration, run the following command to move the fingers and wrist to their neutral pose. Watch for smooth motion, sticking, or persistent unusual noise.

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/neutral.py" "$ORCA_CFG"
```

## 3. Replay `clip_002` with an empty hand

`--replay` sends the demonstration's joint targets directly so you can check finger directions, the starting grasp, and clearance around the mount. `--ease-s 5` gives the hand five seconds to reach the starting pose. The rest plays at the demonstration's original speed. `--wrist-deg` defaults to `0` degrees. If you use a different angle, use the same value for replay, wrist tag measurement, and pen spinning.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand real --replay \
  --orca-config "$ORCA_CFG" --ease-s 5 \
  --log "$PEN_LOGS/replay_002_$(date +%Y%m%d_%H%M%S).npz"
```

The program enables the motors at startup. On the first run, press Enter once and wait for `frozen`. Check the starting pose, then type `q` and press Enter to exit.

Once the starting pose looks right, run the command again with the hand empty. At `frozen`, press Enter a second time to hold the first frame, then a third time to play the full clip. Per-frame logging starts in `hold`.

## 4. Calibrate the camera and wrist tag

Print the [ChArUco board](calibration/charuco-7x5-35mm.png) and [AprilTag 36h11 ID 0 wrist tag](calibration/apriltag-36h11-id0-50.4mm.png) at 100% scale. Mount the ChArUco print on a flat, rigid board and check that the squares are about 35 mm and the markers inside them about 26 mm across. Calibrate using the camera and resolution you will use for tracking.

```bash
"$ORCA_PY" "$PEN_TASK/perception/tools/calibrate_camera.py" \
  --source "$PEN_CAMERA"
```

Keep the camera fixed and move the board. One way to collect views is to take five at different tilts in the center, two in each of the four corners of the image, and two at different distances. Hold the board still, wait until the preview shows at least `12 corners`, and press Space. Check that `views kept` increases by one. Once you have at least 15 views, press Enter to save the calibration to [camera.yaml](../perception/config/camera.yaml).

Mount the wrist tag where the camera can see it with the palm facing up. Keep the white border. If you attach it to the xArm end-effector mount, use the same OrcaHand wrist angle for measurement and pen spinning.

```bash
nano "$PEN_TASK/perception/config/wrist_tag.yaml"
```

Keep `tag_id` set to `0`. Enter the measured outer edge length of the black border in `size_m`. Enter the tag center's three offsets relative to the training model's `right_palm_link` in `pos_in_palm`. Use meters for all lengths.

Use the assembly CAD or measurements to determine the tag's mounting orientation. Enter it in `quat_in_palm` in `[w, x, y, z]` order. On the printed tag, right, up, and out of the page are `+X`, `+Y`, and `+Z`, respectively. After checking the dimensions and orientation, set `measured: true`. Press `Ctrl+O` and Enter to save, then `Ctrl+X` to exit.

## 5. Verify pen tracking on its own

Apply the labels to both ends of the pen using the [label faces and orientations](../hardware/tags/texture_faces.csv). In the first terminal, start the observer. It opens the camera and publishes the pen pose in the palm frame on local port `5555`.

```bash
"$ORCA_PY" "$PEN_TASK/perception/observer.py" --source "$PEN_CAMERA" --preview
```

Wait for `palm frame FIXED`. Hold the pen in your own hand and move it through the starting grasp and the range of the spin. Check that the outline follows the pen, the two ends keep their identities, and tracking recovers after occlusion. If the camera or palm moves, hold it still again and press `w` in the preview to resample the wrist tag.

In a second terminal, run step 0 and then the command below to read poses. Keep the OrcaHand still and move the pen about 20 mm along a known palm axis. Check the magnitude and sign of the coordinate change. Use at least one measured position to check the origin offset. The `A_top` direction is a unit vector pointing from the tail of the pen to its tip.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" - <<'PY'
import time
import zmq
from penspin_deploy.geometry import quat_apply

with zmq.Context() as ctx, ctx.socket(zmq.SUB) as sub:
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    sub.connect("tcp://127.0.0.1:5555")
    try:
        while True:
            try:
                msg = sub.recv_json()
            except zmq.Again:
                print("No messages from the observer")
                continue
            pen = msg.get("pen")
            if not msg["palm_fixed"]:
                print("Still sampling the wrist tag")
            elif pen is None:
                print("No pen pose in this frame")
            else:
                pose = pen["pose"]
                center = [round(v * 1000, 1) for v in pose[:3]]
                axis = quat_apply(pose[3:], [1, 0, 0]).round(2).tolist()
                print(f"Pen center (mm) {center}  A_top direction {axis}")
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
PY
```

Press `Ctrl+C` to stop reading poses. Leave the observer running in the first terminal. The reader displays about four frames per second, so a brief dropout can fall between two printed samples.

## 6. Verify that a pose timeout stops the run

On the real hand in camera mode, the last valid pen pose can be used for up to 150 ms by default. If no valid pose is available or it has expired, pressing Enter in `frozen` prints `not starting` and leaves the hand in `frozen`. If the pose expires during `hold` or during the spin, the controller prints `stopped: pen pose timeout`, stops sending new joint targets, and ends the run. The last valid pose can bridge a short dropout until the time limit. Its age keeps increasing even if the observer is still sending messages without a pen pose.

Before testing with the pen, run the following check on your computer. A simulated camera and hand verify that expired poses prevent startup, brief dropouts can recover, and timeouts stop the controller during both `hold` and the spin. Look for `PASS`.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy.checks pen-timeout "$PEN_BUNDLE"
```

Step 7 enables this check automatically. Use `--pen-timeout-ms` to adjust the initial 150 ms limit based on the camera behavior you observed in step 5. An automatic stop records `abort_reason` in `SUMMARY` and in the file saved by `--log`. On exit, the controller attempts to disable motor torque, so the pen may fall.

## 7. Hold the first frame, then try a full spin

After the timeout check passes, leave the observer running in the first terminal. Run the following command in the second terminal. The real OrcaHand uses the camera's pen pose to run `policy.onnx` from the deployment bundle. If you used a nonzero wrist angle in steps 3 and 4, pass the same angle with `--wrist-deg`.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand real --pen camera --observer localhost:5555 \
  --orca-config "$ORCA_CFG" --ease-s 5 \
  --log "$PEN_LOGS/hold_002_$(date +%Y%m%d_%H%M%S).npz"
```

1. Press Enter once to move the empty hand to the starting grasp. Wait for `frozen`.
2. Place the pen in the starting grasp and press Enter a second time. At `hold`, the ONNX policy adjusts the joints to hold the first frame. For the first run, watch for a few seconds, then type `q` and press Enter to exit. The controller attempts to disable torque on exit. A soft pad beneath the hand can catch the pen.
3. Once the hold succeeds, rerun the command with the log prefix changed to `spin_002_`. At `hold`, check that the pen is secure and press Enter a third time to advance the reference trajectory. After `clip_002` succeeds consistently, try `--clip clip_004`.

Logs go to `tasks/pen_spin/deploy/logs/`. Timestamps distinguish runs. If the pen drops, compare the video with `pen_age` in 20 ms control cycles, measured joint angles `qpos`, and commanded targets `target`. `completed: true` means the reference trajectory reached the end. Check the hand or the video to see whether it held the pen throughout.
