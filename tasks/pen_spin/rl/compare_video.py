"""Film two tracking policies side by side with the reference, from frame 0.

    python compare_video.py runs/track003_1/final_model.zip runs/track003_2/final_model.zip \
        --labels "track003_1" "track003_2 (smoother)" --out videos/track003_1_vs_2.mp4

Each policy runs in its own environment (with its own env_kwargs), holds frame 0
for --hold-s seconds and then follows the clip; the right panel is the
reference pose, forced.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import evaluate as ev
from track_env import PenTrack


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    for i, text in enumerate(lines):
        draw.text((10, 8 + 16 * i), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs=2)
    p.add_argument("--labels", nargs=2, required=True)
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
                   help="reset seed for both policies: with sim2real, both get the same delay and noise draw")
    args = p.parse_args()

    runs = []
    for path in args.models:
        act, kw = ev.build_actor(argparse.Namespace(policy="model", model=path))
        if args.time_scale is not None:
            kw = {**kw, "time_scale": args.time_scale}
        if args.sim2real is not None:
            kw = {**kw, "sim2real": args.sim2real == "on"}
        env = PenTrack(**{**kw, "adaptive_sampling": False, "hold_prob": 0.0, "start_frame": 0,
                          "randomize_physics": False})
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
                r["renderer"].update_scene(env.data, cam)
                r["last"] = r["renderer"].render()
            panels.append(label(r["last"], [name, r["status"]]))
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
        if env.sim2real:
            lines.append(f"servo delay {2 * env._delay_sub} ms, camera lag {env._pen_lag} ticks")
        panels.append(label(ref_renderer.render(), lines))
        frames.append(np.concatenate(panels, axis=1))

    import imageio.v2 as imageio
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
