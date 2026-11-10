"""Run the pen-spin policy on the OrcaHand, or on its MuJoCo twin.

From tasks/pen_spin/deploy, with the deploy venv:

  # the twin, headless and as fast as it computes: policy, camera latency, scored
  python -m penspin_deploy BUNDLE --clip clip_002

  # the twin through the training camera model, 20 seeds
  python -m penspin_deploy BUNDLE --clip clip_002 --pen-model training --episodes 20

  # watch it (macOS: mjpython instead of python); real time
  mjpython -m penspin_deploy BUNDLE --clip clip_002 --viewer

  # orca_core's MockOrcaHand: the orca_core path and the 50 Hz timing, no physics
  python -m penspin_deploy BUNDLE --clip clip_002 --hand mock --pen reference

  # the real hand, no camera yet: replay the reference joints, or the policy
  # with the reference pen pose in its observation
  python -m penspin_deploy BUNDLE --clip clip_002 --hand real --replay
  python -m penspin_deploy BUNDLE --clip clip_002 --hand real --pen reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from penspin_deploy.bundle import Bundle
from penspin_deploy.hand import OrcaCoreHand, connect_orca
from penspin_deploy.joint_map import JointMap
from penspin_deploy.loop import Controller, KeyboardOperator, ScriptedOperator, TwinClock, WallClock

# Where the twin parks the pen until it is placed: on the floor, clear of the hand.
PARKED_PEN_POS = (0.4, 0.4, 0.02)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", type=Path, help="deploy bundle directory (export_bundle.py)")
    p.add_argument("--clip", required=True)
    p.add_argument("--hand", choices=("twin", "mock", "real"), default="twin")
    p.add_argument("--pen", choices=("camera", "reference"), default="camera",
                   help="pen pose in the observation: the (twin's) camera, or the clip's reference")
    p.add_argument("--pen-model", choices=("lag", "training"), default="lag",
                   help="twin camera: one tick of latency only, or the full training camera model")
    p.add_argument("--replay", action="store_true", help="command the reference joints, no policy")
    p.add_argument("--ease-s", type=float, default=2.0, help="move to the start pose over this long")
    p.add_argument("--hold-s", type=float, default=None,
                   help="scripted runs: first-frame hold before the clip (default: the training play hold)")
    p.add_argument("--hold-last-frame-s", type=float, default=0.0,
                   help="keep running the policy on the last frame this long after the clip")
    p.add_argument("--joint-map", type=Path, default=None, help="JSON per-joint sign/offset overrides")
    p.add_argument("--wrist-deg", type=float, default=0.0, help="orca_core wrist angle held throughout")
    p.add_argument("--orca-config", type=Path, default=None, help="orca_core config.yaml (default: v1 right)")
    p.add_argument("--actuation-delay-ms", type=float, default=0.0,
                   help="twin: commands reach the actuators this much later (multiples of 5 ms)")
    p.add_argument("--realtime", action="store_true", help="twin: pace at 50 Hz wall clock")
    p.add_argument("--viewer", action="store_true", help="twin: MuJoCo viewer (implies --realtime)")
    p.add_argument("--interactive", action="store_true", help="twin: Enter to start, as on the hand")
    p.add_argument("--episodes", type=int, default=1, help="twin: repeat with camera seeds 0..N-1")
    p.add_argument("--log", type=Path, default=None, help="save the per-tick record (.npz)")
    args = p.parse_args()
    if args.hand != "twin" and args.pen == "camera" and not args.replay:
        p.error("no camera pipeline on the hand yet: use --pen reference or --replay")
    if args.hand != "twin" and (args.viewer or args.episodes != 1 or args.actuation_delay_ms):
        p.error("--viewer, --episodes and --actuation-delay-ms are twin options")
    return args


def report_roms(hand: OrcaCoreHand, bundle: Bundle) -> None:
    problems = hand.rom_problems(bundle)
    if problems:
        print("  training joint ranges vs orca_core ROMs (targets are clamped to both):")
        for line in problems:
            print("    " + line)


def run_twin(args, bundle: Bundle, clip, joint_map: JointMap, seed: int) -> tuple[dict, Controller]:
    from orca_core.joint_position import OrcaJointPositions

    from penspin_deploy.sim_twin import SimOrcaHand, SimPenCamera, Twin

    twin = Twin(bundle, actuation_delay_s=args.actuation_delay_ms / 1000.0)
    sim_hand = SimOrcaHand(twin, joint_map)
    neutral = sim_hand.config.clamp_joint_positions(
        OrcaJointPositions.from_dict(sim_hand.config.neutral_position)).as_dict()
    twin.d.qpos[twin.pen_qadr:twin.pen_qadr + 3] = PARKED_PEN_POS
    twin.set_hand(np.clip(joint_map.from_orca(neutral), bundle.ctrl_range[:, 0], bundle.ctrl_range[:, 1]))
    hand = OrcaCoreHand(sim_hand, joint_map, wrist_deg=args.wrist_deg)
    if seed == 0:
        report_roms(hand, bundle)
    camera = None
    if args.pen == "camera":
        camera = SimPenCamera(twin, bundle.contract["camera_model"],
                              training=args.pen_model == "training", seed=seed)
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(twin.m, twin.d)
    clock = TwinClock(twin, bundle.ctrl_dt, realtime=args.realtime or args.viewer, viewer=viewer)
    hold_s = bundle.first_frame_hold_s if args.hold_s is None else args.hold_s
    operator = (KeyboardOperator() if args.interactive
                else ScriptedOperator(hold_s=hold_s, place_pen=twin.place_pen))
    if args.interactive:
        # The operator's "placing the pen" is the twin teleporting it on the first Enter.
        operator = _PlacingKeyboard(operator, twin)
    controller = Controller(
        bundle, clip, hand, clock, operator,
        policy=None if args.replay else bundle.policy(),
        pen_source=camera, pen_truth=twin.pen_local, on_hold=twin.release_pen,
        ease_s=args.ease_s, hold_last_frame_s=args.hold_last_frame_s,
        verbose=args.episodes == 1,
    )
    summary = controller.run()
    summary["unstable"] = twin.unstable()
    if viewer is not None:
        viewer.close()
    return summary, controller


class _PlacingKeyboard:
    interactive = True

    def __init__(self, keyboard: KeyboardOperator, twin):
        self.keyboard, self.twin, self.placed = keyboard, twin, False

    def poll(self, controller):
        key = self.keyboard.poll(controller)
        if key == "next" and controller.state == "frozen" and not self.placed:
            self.twin.place_pen(controller.clip.object_pose_local[0])
            self.placed = True
            controller.say("     pen placed; Enter again holds the first frame")
            return None
        return key


def run_orca(args, bundle: Bundle, clip, joint_map: JointMap) -> tuple[dict, Controller]:
    raw = connect_orca(mock=args.hand == "mock", config_path=args.orca_config)
    hand = OrcaCoreHand(raw, joint_map, wrist_deg=args.wrist_deg)
    report_roms(hand, bundle)
    try:
        if args.hand == "real":
            print(f"  the hand will move to the first pose of {clip.name} over {args.ease_s:.1f} s")
            input("  clear the workspace, then Enter (Ctrl+C aborts) ")
            operator = KeyboardOperator()
        else:
            operator = ScriptedOperator(hold_s=bundle.first_frame_hold_s if args.hold_s is None else args.hold_s)
        controller = Controller(
            bundle, clip, hand, WallClock(bundle.ctrl_dt), operator,
            policy=None if args.replay else bundle.policy(), pen_source=None,
            ease_s=args.ease_s, hold_last_frame_s=args.hold_last_frame_s,
        )
        return controller.run(), controller
    finally:
        hand.close()


def main() -> None:
    args = parse_args()
    bundle = Bundle(args.bundle)
    clip = bundle.clip(args.clip)
    joint_map = JointMap.from_file(bundle, args.joint_map)
    mode = "replay" if args.replay else "policy"
    pen = args.pen if args.pen == "reference" else f"camera ({args.pen_model})"
    print(f"{bundle.contract['task']}  {clip.name} ({clip.num_frames} frames)  hand={args.hand}  {mode}  pen={pen}")

    if args.hand == "twin":
        results = []
        for seed in range(args.episodes):
            summary, controller = run_twin(args, bundle, clip, joint_map, seed)
            results.append(summary)
            if args.episodes > 1:
                lost = summary["pen_lost_at_s"]
                print(f"  seed {seed:3d}: {'held' if lost is None else f'lost at {lost:5.2f} s'}"
                      f"  mean error {summary['mean_pos_err_cm']:.2f} cm")
        if args.episodes > 1:
            held = [r for r in results if r["pen_lost_at_s"] is None]
            print(f"SUMMARY held {len(held)}/{len(results)}"
                  f"  mean error (held) {np.mean([r['mean_pos_err_cm'] for r in held]) if held else float('nan'):.2f} cm")
        else:
            print("SUMMARY " + json.dumps(summary))
    else:
        summary, controller = run_orca(args, bundle, clip, joint_map)
        print("SUMMARY " + json.dumps(summary))
    if args.log is not None:
        np.savez_compressed(args.log, **controller.log.arrays())
        print(f"log: {args.log}")


if __name__ == "__main__":
    main()
