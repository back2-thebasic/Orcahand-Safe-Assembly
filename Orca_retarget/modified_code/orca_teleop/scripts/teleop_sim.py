"""Teleoperate the simulated ORCA hand via the same ingress + retargeter
stack used for the physical robot. The only difference from
``pipeline.run()`` is the sink: instead of streaming joint targets to an
``OrcaHand``, ``SimSink`` steps an ``orca_sim`` Gymnasium env at its native
render rate.

Examples:
    # Robot machine: start the pipeline against the sim and wait for a
    # remote publisher to connect over gRPC.
    python scripts/teleop_sim.py --env right

    # Laptop: one-command teleop with a local MediaPipe webcam publisher.
    python scripts/teleop_sim.py --env right --local --show-video
"""

import argparse
import logging
from pathlib import Path

from orca_teleop.ingress.server import DEFAULT_PORT
from orca_teleop.pipeline import run, run_local
from orca_teleop.sim import OrcaHandSimSink


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orca Hand teleoperation pipeline against orca_sim."
    )
    parser.add_argument("--model_path", default=None, help="OrcaHand model directory")
    parser.add_argument("--urdf_path", default=None, help="Hand URDF file")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"gRPC port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--env",
        default="right",
        choices=["left", "right"],
        help="orca_sim env variant (default: right)",
    )
    parser.add_argument(
        "--version", default=None, help="orca_sim embodiment version, e.g. 'v1' or 'v2'"
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array"],
        help="MuJoCo render mode (default: human)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also launch a local MediaPipe publisher for one-command teleop",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Hand to track in the local publisher (default: right)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7, help="MediaPipe confidence (default: 0.7)"
    )
    parser.add_argument("--show-video", action="store_true", help="Show webcam feed with landmarks")
    parser.add_argument('--record-video', default=None, help='Optional new webcam .mp4 path; requires --local')
    parser.add_argument(
        "--retargeter",
        default="adaptive_analytical",
        choices=["rmsprop", "adaptive_analytical"],
        help="Retargeter backend (default: adaptive_analytical)",
    )
    parser.add_argument(
        "--retarget-config",
        default=None,
        help="YAML config for --retargeter adaptive_analytical",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--safety", action="store_true", help="Enable v1 right-hand simulation safety")
    parser.add_argument("--safety-log", default=None, help="New JSONL log path (must not exist)")
    parser.add_argument("--collision-safety", action="store_true", help="Enable independent CBF-QP layer (robot v1 right)")
    parser.add_argument("--collision-monitor", action="store_true", help="Log collisions without modifying commands")
    parser.add_argument("--collision-safety-config", default=None, help="Collision pair and QP YAML configuration")
    parser.add_argument("--collision-safety-log", default=None,
                        help="New per-frame JSONL log path; use 'none' to disable saving")
    parser.add_argument("--diagnostic-log", default=None,
                        help="New JSONL path: contacts, joint tracking, limits and replay targets")
    parser.add_argument("--sim-joint-limits", action="store_true",
                        help="Use loaded MuJoCo joint/control bounds for adaptive retargeting")
    parser.add_argument("--landmark-log", default=None,
                        help="Record consumed MediaPipe 3D inputs including calibration to a new JSONL")
    args = parser.parse_args()
    if args.landmark_log or args.record_video:
        parser.error("This five-file runtime bundle does not include recording modules; omit --landmark-log and --record-video")
    if args.landmark_log:
        if args.retargeter != "adaptive_analytical":
            parser.error("--landmark-log requires adaptive_analytical")
        if Path(args.landmark_log).exists():
            parser.error("--landmark-log must be a new file")
        if args.diagnostic_log and Path(args.landmark_log).resolve() == Path(args.diagnostic_log).resolve():
            parser.error("Input and diagnostic logs must use different files")
    if args.sim_joint_limits and args.retargeter != "adaptive_analytical":
        parser.error("--sim-joint-limits requires adaptive_analytical")
    if args.record_video:
        video_path = Path(args.record_video)
        if not args.local:
            parser.error('--record-video requires --local')
        if video_path.suffix.lower() != '.mp4' or video_path.exists() or video_path.with_suffix('.timing.json').exists():
            parser.error('--record-video needs a new .mp4 path and unused .timing.json path')
    if args.collision_safety or args.collision_monitor:
        if args.env != "right" or args.version != "v1":
            parser.error("collision modes require --env right --version v1")
        if args.safety or (args.collision_safety and args.collision_monitor):
            parser.error("choose only one of --safety, --collision-safety, --collision-monitor")
    if args.safety and (args.env != "right" or args.version != "v1"):
        parser.error("--safety requires --env right --version v1")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sink = OrcaHandSimSink(
        env_name=args.env,
        version=args.version,
        render_mode=args.render_mode,
        safety=args.safety,
        safety_log=args.safety_log,
        collision_safety=args.collision_safety,
        collision_monitor=args.collision_monitor,
        collision_safety_config=args.collision_safety_config,
        collision_safety_log=args.collision_safety_log,
        sim_joint_limits=args.sim_joint_limits,
        diagnostic_log=args.diagnostic_log,
        diagnostic_context={"arguments": vars(args), "retarget_config_text":
            Path(args.retarget_config).read_text() if args.diagnostic_log and args.retarget_config else None},
    )

    sink.landmark_log = args.landmark_log

    if args.local:
        run_local(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            handedness=args.hand,
            confidence=args.confidence,
            show_video=args.show_video,
            record_video=args.record_video,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
        )
    else:
        run(
            model_path=args.model_path,
            urdf_path=args.urdf_path,
            port=args.port,
            sink=sink,
            retargeter_backend=args.retargeter,
            retargeter_config_path=args.retarget_config,
        )


if __name__ == "__main__":
    main()
