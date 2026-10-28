"""Evaluate a pencil-spinning policy, and optionally film it.

    python -m orca_rl.pencil_eval --policy zero
    python -m orca_rl.pencil_eval --model runs/pencil1/final_model.zip --episodes 50
    python -m orca_rl.pencil_eval --model runs/pencil1/final_model.zip --episodes 3 --video pencil1.mp4

Revolutions count the net yaw of the pencil's axis in the commanded direction,
only while it is held and within ``max_tilt_deg`` of level -- the same
quantity the reward pays for, before the rate cap. Reference points: doing
nothing turns 0.00 rev; the scripted oracle in ``pencil_checks.py`` at the
1 rad/s cap turns 1.59 rev in 10 s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from orca_rl.pencil import PencilSpin, saved_pencil_kwargs
from orca_rl.task import resolve_stats_path


def build_actor(args):
    """(act, env_kwargs) for --policy; a model brings its own env settings."""
    if args.policy == "random":
        return (lambda env, obs: env.action_space.sample()), {}
    if args.policy == "zero":
        return (lambda env, obs: np.zeros(env.action_space.shape, dtype=np.float32)), {}

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(args.model, device="cpu")
    env_kwargs = saved_pencil_kwargs(args.model)
    stats = resolve_stats_path(Path(args.model))
    normalizer = None
    if stats.exists():
        normalizer = VecNormalize.load(str(stats), DummyVecEnv([lambda: PencilSpin(**env_kwargs)]))
        normalizer.training = False
        normalizer.norm_reward = False
    else:
        print(f"warning: {stats} not found -- evaluating WITHOUT obs normalization; "
              "the numbers below are meaningless for a model trained with it")

    def act(env, obs):
        x = obs[None, :]
        if normalizer is not None:
            x = normalizer.normalize_obs(x)
        action, _ = model.predict(x, deterministic=args.deterministic)
        return action[0]

    return act, env_kwargs


def make_camera(env):
    import mujoco

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    # Fixed on the palm, steep enough to read the pencil's heading.
    cam.azimuth, cam.elevation, cam.distance = 200.0, -55.0, 0.34
    cam.lookat[:] = [0.16, 0.045, 0.15]
    return cam


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--policy", default="model", choices=["model", "random", "zero"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--randomize-physics", action="store_true")
    p.add_argument("--video", default=None, metavar="OUT.mp4",
                   help="also render every episode into this file (keep --episodes small)")
    p.add_argument("--fps", type=int, default=50, help="50 = half speed")
    p.add_argument("--clean", action="store_true",
                   help="demo footage: 1280x720 and no text, like demo_videos/")
    args = p.parse_args()
    if args.policy == "model" and args.model is None:
        p.error("--model is required unless --policy is random or zero")

    act, env_kwargs = build_actor(args)
    # Always from an ordinary reset: a stall buffer would make the episodes
    # depend on the ones before them and on the policy's own failures.
    env_kwargs = {**env_kwargs, "max_episode_steps": args.max_episode_steps,
                  "randomize_physics": args.randomize_physics, "stall_start_prob": 0.0}
    env = PencilSpin(**env_kwargs)
    label = args.policy if args.policy != "model" else str(Path(args.model).parent)

    renderer = frames = None
    if args.video:
        import mujoco
        from orca_rl.record import _overlay

        width, height = (1280, 720) if args.clean else (640, 480)
        env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width)
        env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height)
        renderer, frames = mujoco.Renderer(env.model, height=height, width=width), []

    returns, revs, drops, stood, lengths, held = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total = 0.0
        if renderer is not None:
            cam = make_camera(env)
        while True:
            obs, reward, terminated, truncated, info = env.step(act(env, obs))
            total += reward
            if renderer is not None:
                renderer.update_scene(env.data, camera=cam)
                frame = renderer.render()
                if not args.clean:
                    t = info["elapsed_steps"] * env.dt
                    top = f"{label}   ep {ep + 1}/{args.episodes}   t={t:5.2f}s"
                    bottom = (f"turned {info['turned_revs']:+.2f} rev   "
                              f"{np.degrees(info['turned_rad']) / max(t, 1e-9):+5.0f} deg/s   "
                              f"{'held' if info['in_hand'] else 'NOT held'}")
                    ended = "DROPPED" if info["dropped"] else "STOOD UP" if info["stood_up"] else None
                    frame = _overlay(frame, top, bottom, (230, 40, 30) if ended else None, ended)
                frames.append(frame)
            if terminated or truncated:
                break
        returns.append(total)
        revs.append(info["turned_revs"])
        drops.append(bool(info["dropped"]))
        stood.append(bool(info["stood_up"]) and not info["dropped"])
        lengths.append(info["elapsed_steps"])
        held.append(info["in_hand_frac"])
    env.close()

    revs, lengths = np.asarray(revs), np.asarray(lengths)
    seconds = lengths.sum() * env.dt
    print(f"\n=== {label}  ({args.episodes} episodes x {args.max_episode_steps * env.dt:.0f} s, "
          f"{env.spin_direction}) ===")
    print(f"  revolutions / episode : {revs.mean():+.2f}  (max {revs.max():+.2f}, min {revs.min():+.2f})")
    print(f"  spin rate             : {np.degrees(2 * np.pi * revs.sum() / seconds):+.1f} deg/s "
          f"over all episode time, drops included")
    print(f"  drop rate             : {100 * np.mean(drops):.0f}%")
    if env.max_stand_rad is not None:
        print(f"  stood up (ended)      : {100 * np.mean(stood):.0f}%  "
              f"(past {np.degrees(env.max_stand_rad):.0f} deg)")
    print(f"  mean episode length   : {lengths.mean():.0f} / {args.max_episode_steps}")
    print(f"  fraction of time held : {100 * np.mean(held):.0f}%")
    print(f"  mean return           : {np.mean(returns):.2f} +- {np.std(returns):.2f}")
    print("\n  reference: zero ~0.00 rev, ~10% drop | oracle at the 1 rad/s cap 1.59 rev")

    if renderer is not None:
        import imageio.v2 as imageio

        renderer.close()
        imageio.mimsave(args.video, frames, fps=args.fps)
        print(f"wrote {args.video}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
