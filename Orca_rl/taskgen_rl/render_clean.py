"""Text-free demo video of a taskgen policy, 1280x720: the trained policy only,
several reset seeds back to back. Copy into the orca-agent-taskgen checkout
(with the patches applied) and run from its root:

    .venv/bin/python render_clean.py --model runs_taskgen/place_rel_s0/final_model.zip --out place.mp4 \
        --look fixed --lookat 0.02 -0.005 0.23 --azimuth 135 --elevation -35 --distance 0.36

For pick-and-place the target region is drawn as a green disc (the scene's own
marker does not render).
"""
import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

sys.path.insert(0, "rl")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wrappers import make_env
from evaluate_bundle import run_config_for, stats_path_for

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--fps", type=int, default=25, help="25 = 1/4 speed (one frame per 10 ms step)")
p.add_argument("--hold", type=int, default=20, help="frames kept after each success")
p.add_argument("--azimuth", type=float, default=150)
p.add_argument("--elevation", type=float, default=-15)
p.add_argument("--distance", type=float, default=0.45)
p.add_argument("--look", choices=["object", "hand", "fixed"], default="object")
p.add_argument("--lookat", type=float, nargs=3, default=None)
args = p.parse_args()

mp = Path(args.model)
cfg = run_config_for(mp)
bundle = cfg["bundle"]
if not Path(bundle).exists():
    bundle = str(Path("examples/generated") / Path(bundle).name)
rel = cfg.get("relative_hand")
model = PPO.load(str(mp), device="cpu")
norm = VecNormalize.load(str(stats_path_for(mp)), DummyVecEnv([lambda: make_env(bundle, rel)]))
norm.training = False

env = make_env(bundle, rel)
base = env.unwrapped
W, H = 1280, 720
base.model.vis.global_.offwidth, base.model.vis.global_.offheight = W, H
renderer = mujoco.Renderer(base.model, H, W)
frames = []
for seed in args.seeds:
    obs, info = env.reset(seed=seed)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    look = None
    for t in range(base.max_episode_steps):
        a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
        obs, r, te, tr, info = env.step(a[0])
        if args.look == "fixed":
            target = np.asarray(args.lookat)
        elif args.look == "hand":
            target = base._tip_points().mean(0)
        else:
            target = base._object_pos()
        look = target.copy() if look is None else 0.85 * look + 0.15 * target
        cam.lookat[:] = look
        renderer.update_scene(base.data, cam)
        if base.family == "pick_place":
            scene = renderer.scene
            geom = scene.geoms[scene.ngeom]
            radius = float(base.task["success"]["target_radius"])
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, 0.0008, 0.0]),
                                base._target_position - np.array([0, 0, 0.0215]), np.eye(3).flatten(),
                                np.array([0.15, 0.8, 0.25, 0.55], dtype=np.float32))
            scene.ngeom += 1
        frames.append(renderer.render())
        if te or tr:
            break
    print("seed", seed, "success", info["is_success"], "steps", t + 1)
    frames.extend([frames[-1]] * args.hold)
renderer.close()
imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
print(args.out, len(frames), "frames")
