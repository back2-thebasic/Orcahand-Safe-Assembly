"""Export a trained OrcaHand pen-spin checkpoint as a self-contained deploy bundle.

Runs on the training side (torch + mjlab + wuji_mjlab), from the wuji-mjlab repo:

    cd tasks/reorient/wuji-mjlab
    WARP_CACHE_PATH=... ../../../.scratch/mjlab-venv/bin/python \
        ../../pen_spin/deploy/export_bundle.py CKPT OUT_DIR [--clips clip_002,clip_004]

The bundle is everything the torch-free runtime (penspin_deploy) needs:

    policy.onnx            normalizer + actor mean, obs -> raw action
    contract.json          control rate, action processing, obs layout, joint table
    scene.mjb              compiled training scene (hand + pen) for the sim twin
    clips/<clip>.npz       per-frame motion command and reference, from the training command
    parity/<clip>.npz      a play-config rollout of the training env: what the runtime must reproduce
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.lab_api.math import subtract_frame_transforms
from wuji_mjlab.assets.robots.orcahand_v1.orcahand_v1_cfg import (
    ORCAHAND_V1_JOINT_NAMES,
    ORCAHAND_V1_WRIST_TAG_POS,
    ORCAHAND_V1_WRIST_TAG_ROT,
)
from wuji_mjlab.rl import onnx_export
from wuji_mjlab.tasks.pen_spin.config.orcahand_v1.env_cfgs import ORCAHAND_V1_FIXED_WRIST_POSE_W
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

REPO = Path(__file__).resolve().parents[2] / "reorient" / "wuji-mjlab"
RECIPE = str(REPO / "src/wuji_mjlab/tasks/pen_spin/data/recipe/orcahand_v1_continuous_single_axis__{}.json")
PALM_BODY = "right_palm_link"
PEN_BODY = "cube"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_env(task: str, clip: str):
    override = [f"env.commands.motion.recipe_path={RECIPE.format(clip)}"]
    cfg, agent_cfg = prepare_task_cfgs(task, override, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    return env, agent_cfg


def export_onnx(ckpt: Path, agent_cfg: dict, out: Path) -> None:
    """The standalone exporter's model build, with the agent config taken from the task."""
    state_dict = onnx_export._resolve_export_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True)
    )
    arch = onnx_export._build_architecture(agent_cfg, state_dict)
    norm_state = onnx_export._extract_normalizer_state_dict(state_dict)
    if not norm_state:
        raise ValueError("checkpoint has no observation normalizer")
    normalizer = onnx_export.EmpiricalNormalization(arch["num_actor_obs"])
    normalizer.load_state_dict(norm_state)
    actor = onnx_export.MLP(
        input_dim=arch["num_actor_obs"],
        output_dim=arch["actor_output_dim"],
        hidden_dims=arch["hidden_dims"],
        activation=arch["activation"],
    )
    actor.load_state_dict(onnx_export._extract_actor_mlp_state_dict(state_dict))
    if arch["state_dependent_std"]:
        actor = torch.nn.Sequential(
            actor,
            onnx_export.HeteroscedasticGaussianDistribution(
                arch["actor_output_dim"][1]
            ).as_deterministic_output_module(),
        )
    exporter = onnx_export._StandaloneExporter(normalizer, actor).eval()
    torch.onnx.export(
        exporter,
        torch.zeros(1, arch["num_actor_obs"]),
        str(out),
        export_params=True,
        opset_version=onnx_export._ONNX_OPSET_VERSION,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
        dynamo=False,
    )
    data = Path(str(out) + ".data")
    if data.exists():
        import onnx

        onnx.save(onnx.load(str(out), load_external_data=True), str(out), save_as_external_data=False)
        data.unlink()


def _restore_inertial_frames(m, compiled_iquat: np.ndarray) -> None:
    """Express each changed inertia in the body's compiled inertial frame.

    MuJoCo builds the body BVHs in the inertial frame at compile time. A new
    body_iquat rotates that frame under them and the midphase culls real contacts.
    """
    rotated = ~np.all(np.isclose(m.body_iquat, compiled_iquat, rtol=0.0, atol=1e-6), axis=1)
    for body in np.flatnonzero(rotated):
        new_rot, old_rot = np.zeros(9), np.zeros(9)
        mujoco.mju_quat2Mat(new_rot, m.body_iquat[body])
        mujoco.mju_quat2Mat(old_rot, compiled_iquat[body])
        new_rot, old_rot = new_rot.reshape(3, 3), old_rot.reshape(3, 3)
        tensor = old_rot.T @ new_rot @ np.diag(m.body_inertia[body]) @ new_rot.T @ old_rot
        off_diagonal = np.abs(tensor - np.diag(np.diag(tensor))).max()
        if off_diagonal > 1e-6 * np.abs(tensor).max():
            raise RuntimeError(f"body {m.body(body).name}: inertia is not diagonal in its compiled frame")
        m.body_inertia[body] = np.diag(tensor)
        m.body_iquat[body] = compiled_iquat[body]
        print(f"  scene: {m.body(body).name} inertia {np.diag(tensor)} kept in the compiled inertial frame")


def export_scene(env, out: Path) -> None:
    """Save the compiled scene with every model field startup events changed.

    mjlab applies startup events (here the pen inertia of the hardware pen) to the
    Warp model only; mj_model keeps the compiled values. Copy world 0 back.
    """
    m, w = env.sim.mj_model, env.sim.model
    compiled_iquat = m.body_iquat.copy()
    copied = []
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            dst = getattr(m, name)
            if not isinstance(dst, np.ndarray) or dst.size == 0:
                continue
            src = getattr(w, name)
        except Exception:
            continue
        src = src.cpu().numpy() if hasattr(src, "cpu") else np.asarray(src)
        if src.size != dst.size:
            if src.shape[0] != 1 or src[0].size != dst.size:
                continue
            src = src[0]
        src = src.reshape(dst.shape).astype(dst.dtype)
        # Warp holds float32: anything within its rounding is the compiled value.
        if not np.allclose(src, dst, rtol=1e-5, atol=1e-6):
            dst[...] = src
            copied.append(name)
    print(f"  scene: copied startup-event fields {copied}")
    if "body_iquat" in copied:
        _restore_inertial_frames(m, compiled_iquat)
    if any(name.startswith("geom_") for name in copied):
        # The body BVHs were built from the compiled geoms; stale ones drop contacts.
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        print("  scene: geometry changed after compile, midphase disabled")
    mujoco.mj_setConst(m, mujoco.MjData(m))
    mujoco.mj_saveModel(m, str(out), None)


def joint_table(env, motion) -> dict:
    m = env.sim.mj_model
    to_orca = {v: k.removeprefix("right_") for k, v in ORCAHAND_V1_JOINT_NAMES.items()}
    qpos0, ctrl_range = [], []
    for name in motion.joint_names:
        jid = m.joint(f"robot/{name}").id
        qpos0.append(float(m.qpos0[m.jnt_qposadr[jid]]))
        aid = m.actuator(f"robot/{name}_actuator").id
        if not m.actuator_ctrllimited[aid]:
            raise ValueError(f"actuator for {name} has no ctrlrange")
        ctrl_range.append([float(v) for v in m.actuator_ctrlrange[aid]])
    return {
        "names": list(motion.joint_names),
        "orca_ids": [to_orca[name] for name in motion.joint_names],
        "qpos0_rad": qpos0,
        "ctrl_range_rad": ctrl_range,
    }


def export_clip(motion, out: Path) -> int:
    start, end = 0, int(motion.trajectory_lengths[0])
    frames = motion._frames
    command = torch.cat(
        (
            frames["joint_pos"][start:end],
            motion._bounded_object_axis_points[start:end],
            frames["link_pos"][start:end].flatten(1),
        ),
        dim=-1,
    )
    np.savez_compressed(
        out,
        command=command.numpy().astype(np.float32),
        reference_qpos=frames["joint_pos"][start:end].numpy().astype(np.float32),
        object_pose_local=frames["object_pose"][start:end].numpy().astype(np.float32),
        fps=np.float64(1.0 / motion.group.frame_dt),
    )
    return end - start


def record_parity(env, agent_cfg, task: str, ckpt: Path, onnx_path: Path, out: Path) -> None:
    """Play-config rollout of the training env from the clip's first frame.

    Per control step, the state the policy observed (before its action) and what
    the env made of the action. Play config: no observation noise, no DR, pen pose
    without camera lag, 1 s first-frame hold.
    """
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(venv, asdict(agent_cfg), device="cpu")
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    motion = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")
    robot, pen = env.scene["robot"], env.scene["object"]
    joint_ids = robot.find_joints(motion.joint_names, preserve_order=True)[0]
    palm_id = robot.find_bodies((PALM_BODY,), preserve_order=True)[0][0]

    rec = {k: [] for k in ("obs", "raw", "target", "qpos", "qvel", "pen_local", "time_step",
                           "frame", "sim_qpos", "sim_qvel", "sim_ctrl")}
    obs = venv.get_observations()
    length = int(motion.trajectory_lengths[0])
    hold = int(-motion.time_steps[0])
    with torch.inference_mode():
        for _ in range(hold + length):
            pos, quat = subtract_frame_transforms(
                robot.data.body_link_pos_w[:, palm_id],
                robot.data.body_link_quat_w[:, palm_id],
                pen.data.root_link_pos_w,
                pen.data.root_link_quat_w,
            )
            rec["obs"].append(obs["policy"][0].clone())
            rec["qpos"].append(robot.data.joint_pos[0, joint_ids].clone())
            rec["qvel"].append(robot.data.joint_vel[0, joint_ids].clone())
            rec["pen_local"].append(torch.cat((pos[0], quat[0])))
            rec["time_step"].append(motion.time_steps[0].clone())
            rec["frame"].append(motion.frame_ids[0].clone())
            rec["sim_qpos"].append(env.sim.data.qpos[0].clone())
            rec["sim_qvel"].append(env.sim.data.qvel[0].clone())
            action = policy(obs)
            rec["raw"].append(action[0].clone())
            obs, _, dones, _ = venv.step(action)
            if bool(dones[0]):
                # reference_end: the env has already reset, the targets are gone.
                for values in rec.values():
                    del values[len(rec["target"]):]
                break
            rec["target"].append(action_term._targets[0].clone())
            rec["sim_ctrl"].append(env.sim.data.ctrl[0].clone())
    arrays = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    steps = len(arrays["raw"])

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_raw = np.concatenate(
        [session.run(None, {"obs": row[None].astype(np.float32)})[0] for row in arrays["obs"]]
    )
    err = float(np.abs(onnx_raw - arrays["raw"]).max())
    print(f"  parity: {steps} steps (hold {hold}), onnx vs torch max |diff| {err:.2e}")
    if err > 1e-4:
        raise RuntimeError(f"ONNX disagrees with the torch policy: {err}")
    np.savez_compressed(out, hold_steps=np.int64(hold), **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--task", default="OrcaHandV1_PenSpin_Continuous002004_DR")
    parser.add_argument("--clips", default="clip_002,clip_004")
    args = parser.parse_args()

    ckpt = args.checkpoint.resolve()
    out = args.out.resolve()
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "parity").mkdir(parents=True, exist_ok=True)
    clips = args.clips.split(",")

    contract = None
    for clip in clips:
        print(f"[{clip}]")
        env, agent_cfg = build_env(args.task, clip)
        motion = env.command_manager.get_term("motion")
        if contract is None:
            export_onnx(ckpt, asdict(agent_cfg), out / "policy.onnx")
            export_scene(env, out / "scene.mjb")
            act = env.cfg.actions["joint_pos"]
            cmd = env.cfg.commands["motion"]
            policy_terms = env.cfg.observations["policy"].terms
            train_cfg, _ = prepare_task_cfgs(args.task, [], play=False)
            camera = train_cfg.observations["policy"].terms["object_pose"].params
            contract = {
                "schema": "orcahand_pen_spin_deploy:1",
                "task": args.task,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha256(ckpt),
                "onnx_sha256": sha256(out / "policy.onnx"),
                "ctrl_dt": float(env.step_dt),
                "sim_timestep": float(env.cfg.sim.mujoco.timestep),
                "decimation": int(env.cfg.decimation),
                "action": {
                    "scale": float(act.scale),
                    "residual_clip": list(act.residual_clip),
                    "ema_alpha": float(act.alpha),
                },
                "observation": {
                    "layout": [
                        ["motion", int(motion.command.shape[-1])],
                        ["joint_pos", len(motion.joint_names)],
                        ["joint_vel", len(motion.joint_names)],
                        ["object_axis_points", 6],
                        ["last_action", len(motion.joint_names)],
                    ],
                    "joint_vel_scale": float(policy_terms["joint_vel"].scale),
                    "axis_point_radius_m": float(cmd.axis_point_radius_m),
                    "object_position_tanh_scale_m": float(cmd.object_position_tanh_scale_m),
                },
                "joints": joint_table(env, motion),
                "palm": {
                    "body": f"robot/{PALM_BODY}",
                    "pose_w": list(ORCAHAND_V1_FIXED_WRIST_POSE_W),
                    "wrist_tag_pos": list(ORCAHAND_V1_WRIST_TAG_POS),
                    "wrist_tag_quat": list(ORCAHAND_V1_WRIST_TAG_ROT),
                },
                "pen": {"body": f"object/{PEN_BODY}"},
                "camera_model": {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in camera.items()
                    if k not in ("command_name", "robot_cfg", "object_cfg")
                },
                "first_frame_hold_s": float(cmd.first_frame_hold_s[0]),
                "clips": {},
            }
        frames = export_clip(motion, out / "clips" / f"{clip}.npz")
        contract["clips"][clip] = {"frames": frames, "source": str(RECIPE.format(clip))}
        record_parity(env, agent_cfg, args.task, ckpt, out / "policy.onnx", out / "parity" / f"{clip}.npz")
        env.close()

    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"bundle written to {out}")


if __name__ == "__main__":
    main()
