"""Film one or two tracking policies side by side with the reference, from frame 0.

    python compare_video.py runs/track003_1/final_model.zip runs/track003_2/final_model.zip \
        --labels "track003_1" "track003_2 (smoother)" --out videos/track003_1_vs_2.mp4
    python compare_video.py runs/track003_10/final_model.zip --labels track003_10 \
        --sim2real on --joint-vel diff --disturb 1 --seed 3 --out pushes.mp4

Each policy runs in its own environment (with its own env_kwargs), holds frame 0
for --hold-s seconds and then follows the clip; the right panel is the
reference pose, forced. With --disturb the pushes on the pen (red) and the
kicks on the fingertips (orange) are drawn as arrows from each body's centre of
mass along the force, 3 cm plus 15 cm per newton long (the force only; the
pushes decay to 2% over each pulse). With the same --seed, sim2real and --joint-vel,
both policies get the same draws and the same pushes.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import evaluate as ev
from track_env import SIM2REAL, PenTrack


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    for i, text in enumerate(lines):
        draw.text((10, 8 + 16 * i), text, fill=(255, 255, 255))
    return np.asarray(im)


def draw_pushes(scene: mujoco.MjvScene, env: PenTrack) -> None:
    """Arrows for the external forces acting now, from each body's centre of mass."""
    bodies = [(env._pen_body, (0.9, 0.1, 0.1, 1.0))] + [(b, (1.0, 0.55, 0.0, 1.0)) for b in env._tip_bodies]
    for body, rgba in bodies:
        force = env.data.xfrc_applied[body, :3]
        size = np.linalg.norm(force)
        if size == 0 or scene.ngeom >= scene.maxgeom:
            continue
        tail = env.data.xipos[body]
        head = tail + force / size * (0.03 + 0.15 * size)
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.005, tail, head)
        scene.ngeom += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="one or two models")
    p.add_argument("--labels", nargs="+", required=True, help="one per model")
    p.add_argument("--out", required=True)
    p.add_argument("--hold-s", type=float, default=1.0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--time-scale", type=float, default=None,
                   help="play the clip this many times slower than the demonstration, for both "
                        "policies (default: as each was trained; they must then agree)")
    p.add_argument("--friction", type=float, default=1.0,
                   help="scale every geom's contact friction by this, for both policies")
    p.add_argument("--pen-mass", type=float, default=1.0,
                   help="scale the pen's mass and inertia by this, for both policies")
    p.add_argument("--sim2real", choices=["on", "off"], default=None,
                   help="servo delay, sensor noise and the camera model for both policies "
                        "(default: as each was trained)")
    p.add_argument("--seed", type=int, default=0,
                   help="reset seed for both policies: with sim2real, both get the same delay and calibration "
                        "draw; the per-tick noise and the pushes match only if both use the same joint_vel "
                        "(see --joint-vel)")
    p.add_argument("--joint-vel", choices=["noisy", "diff"], default=None,
                   help="with sim2real: joint speeds from the simulator plus noise or differenced from the "
                        "reported angles (default: as each was trained; set it to give both the same pushes)")
    p.add_argument("--disturb", type=float, default=0.0, metavar="S",
                   help="pushes on the pen and kicks on the fingertips at S times full strength (track_env.DISTURB)")
    args = p.parse_args()
    if not 1 <= len(args.models) <= 2 or len(args.labels) != len(args.models):
        p.error("give one or two models and one label per model")

    runs = []
    for path in args.models:
        act, kw = ev.build_actor(argparse.Namespace(policy="model", model=path))
        if args.time_scale is not None:
            kw = {**kw, "time_scale": args.time_scale}
        if args.sim2real is not None:
            kw = {**kw, "sim2real": args.sim2real == "on"}
        if args.joint_vel is not None:
            if not kw.get("sim2real"):
                p.error("--joint-vel acts through sim2real; add --sim2real on")
            kw = {**kw, "sim2real_cfg": {**kw.get("sim2real_cfg", SIM2REAL), "joint_vel": args.joint_vel}}
        env = PenTrack(**{**kw, "adaptive_sampling": False, "hold_prob": 0.0, "start_frame": 0,
                          "randomize_physics": False, "disturb": args.disturb})
        obs, _ = env.reset(seed=args.seed)
        env.model.geom_friction[:, 0] *= args.friction
        env.model.body_mass[env._pen_body] *= args.pen_mass
        env.model.body_inertia[env._pen_body] *= args.pen_mass
        env._hold = int(round(args.hold_s / env.dt))
        env._set_reference_state(0)
        runs.append(dict(env=env, act=act, obs=env._obs(), done=False, status="",
                         renderer=mujoco.Renderer(env.model, 480, 480)))
    if len({r["env"].n_frames for r in runs}) != 1:
        p.error("the two policies play the clip at different speeds; pass --time-scale")
    m = runs[0]["env"].model
    n_frames, dt = runs[0]["env"].n_frames, runs[0]["env"].dt
    hold = int(round(args.hold_s / dt))
    ref_data, ref_renderer = mujoco.MjData(m), mujoco.Renderer(m, 480, 480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = runs[0]["env"]._centre
    cam.distance, cam.azimuth, cam.elevation = 0.42, 90.0, -55.0

    frames, step = [], 0
    while not all(r["done"] for r in runs):
        step += 1
        panels = []
        for r, name in zip(runs, args.labels):
            env = r["env"]
            if not r["done"]:
                r["obs"], _, term, trunc, info = env.step(r["act"](r["obs"]))
                err = np.linalg.norm(env._pen_pose()[:3] - env.ref_obj[env._frame(), :3])
                r["status"] = f"pen error {100 * err:.1f} cm"
                if term or trunc:
                    r["done"] = True
                    r["status"] = "FAILED (pen >5 cm or >60 deg off)" if term else "clip completed"
                pushes = []
                if args.disturb > 0:
                    if np.any(env.data.xfrc_applied[env._pen_body, :3]):
                        pushes.append(f"push on the pen {np.linalg.norm(env.data.xfrc_applied[env._pen_body, :3]):.2f} N")
                    if env._tip_active.any():
                        n = int(env._tip_active.sum())
                        pushes.append(f"kick on {n} fingertip{'s' if n > 1 else ''}")
                r["pushes"] = ", ".join(pushes)
                r["renderer"].update_scene(env.data, cam)
                if args.disturb > 0:
                    draw_pushes(r["renderer"].scene, env)
                r["last"] = r["renderer"].render()
            panels.append(label(r["last"], [name, r["status"]] + ([r["pushes"]] if r.get("pushes") else [])))
        # The reference keeps its own clock, so it goes on after a policy fails.
        env = runs[0]["env"]
        k = 0 if step <= hold else min(step - hold, n_frames - 1)
        ref_data.qpos[env._qadr] = env.ref_q[k]
        ref_data.qpos[env._pq:env._pq + 7] = env.ref_obj[k]
        mujoco.mj_forward(m, ref_data)
        ref_renderer.update_scene(ref_data, cam)
        phase = "holding frame 0" if step <= hold else f"frame {k}/{n_frames - 1}"
        lines = ["reference (forced pose)", f"t={step * dt:.2f} s, {phase}"]
        if args.friction != 1 or args.pen_mass != 1:
            lines.append(f"policies: friction x{args.friction:g}, pen mass x{args.pen_mass:g}")
        if args.sim2real is not None:
            lines.append(f"policies: servo delay + sensor noise {args.sim2real}")
        if args.disturb > 0:
            lines.append(f"policies: pushes and kicks x{args.disturb:g} (arrows)")
        if env.sim2real:
            lines.append(f"servo delay {2 * env._delay_sub} ms, camera lag {env._pen_lag} ticks")
        panels.append(label(ref_renderer.render(), lines))
        frames.append(np.concatenate(panels, axis=1))

    import imageio.v2 as imageio
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
