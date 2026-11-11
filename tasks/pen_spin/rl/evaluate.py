"""Evaluate a tracking policy, or the bare reference (zero residual), and optionally film it.

    python evaluate.py --model runs/track003_1/final_model.zip
    python evaluate.py --policy zero
    python evaluate.py --model runs/track003_1/final_model.zip --video track003_1.mp4

The main test runs the clip the way a deployment would: hold frame 0 for 1 s,
then follow the whole reference. It also runs --starts episodes from random
start frames (reference state initialization, as in training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from track_env import PenTrack, quat_rotate


def build_actor(args):
    if args.policy == "zero":
        return (lambda obs: np.zeros(16)), {}
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_path = Path(args.model)
    run_dir = model_path.parent if model_path.parent.name != "checkpoints" else model_path.parent.parent
    env_kwargs = json.loads((run_dir / "env_kwargs.json").read_text())
    stats = model_path.parent / "vecnormalize.pkl"
    if model_path.parent.name == "checkpoints":
        stats = model_path.parent / model_path.name.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
    model = PPO.load(str(model_path), device="cpu")
    norm = VecNormalize.load(str(stats), DummyVecEnv([lambda: PenTrack(**env_kwargs)]))
    norm.training = False

    def act(obs):
        a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
        return a[0]

    return act, env_kwargs


def axis_of(pose):
    return quat_rotate(pose[3:], np.array([1.0, 0.0, 0.0]))


def high_freq_share(x, dt, cutoff_hz=5.0):
    """Share of the power of the per-step change of x above cutoff_hz (the jitter measure used on pencil7_26)."""
    v = np.diff(np.asarray(x), axis=0)
    if len(v) < 8:
        return float("nan")
    power = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), dt)
    return float(power[f > cutoff_hz].sum() / power.sum())


def high_freq_rms(x, dt, cutoff_hz=5.0):
    """RMS (rad/s) of the velocity of x above cutoff_hz: how fast it shakes, in absolute terms."""
    v = np.diff(np.asarray(x), axis=0) / dt
    if len(v) < 8:
        return float("nan")
    spectrum = np.fft.rfft(v - v.mean(0), axis=0)
    spectrum[np.fft.rfftfreq(len(v), dt) <= cutoff_hz] = 0
    return float(np.sqrt((np.fft.irfft(spectrum, n=len(v), axis=0) ** 2).mean()))


def run_episode(env, act, start_frame, hold_steps, frames=None, ref_data=None, renderers=None, traces=None,
                seed=0):
    env.start_frame = start_frame
    obs, _ = env.reset(seed=seed)
    env._hold = hold_steps
    env._set_reference_state(start_frame)
    obs = env._obs()
    swept, ref_swept, prev, prev_ref = 0.0, 0.0, axis_of(env._pen_pose()), axis_of(env.ref_obj[start_frame])
    while True:
        obs, r, term, trunc, info = env.step(act(obs))
        ax, ref_ax = axis_of(env._pen_pose()), axis_of(env.ref_obj[env._frame()])
        swept += np.degrees(np.arccos(np.clip(ax @ prev, -1, 1)))
        ref_swept += np.degrees(np.arccos(np.clip(ref_ax @ prev_ref, -1, 1)))
        prev, prev_ref = ax, ref_ax
        if traces is not None:
            traces["target"].append(env.data.ctrl.copy())
            traces["joint"].append(env.data.qpos[env._qadr].copy())
            traces["reference"].append(env.ref_q[env._frame()].copy())
        if frames is not None:
            frames.append(render_pair(env, ref_data, renderers, info))
        if term or trunc:
            return info, swept, ref_swept


def render_pair(env, ref_data, renderers, info):
    m = env.model
    k = env._frame()
    ref_data.qpos[env._qadr] = env.ref_q[k]
    ref_data.qpos[env._pq:env._pq + 7] = env.ref_obj[k]
    mujoco.mj_forward(m, ref_data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = env._centre
    cam.distance, cam.azimuth, cam.elevation = 0.42, 90.0, -55.0
    out = []
    for data, renderer in ((env.data, renderers[0]), (ref_data, renderers[1])):
        renderer.update_scene(data, cam)
        out.append(renderer.render())
    return np.concatenate(out, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model")
    p.add_argument("--policy", choices=["model", "zero"], default="model")
    p.add_argument("--starts", type=int, default=50, help="episodes from random start frames")
    p.add_argument("--hold-s", type=float, default=1.0)
    p.add_argument("--video")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--time-scale", type=float, default=None,
                   help="play the clip this many times slower than the demonstration (default: as trained)")
    p.add_argument("--random-physics", type=int, default=0, metavar="N",
                   help="also run N full clips from frame 0, each with freshly randomized physics "
                        "(seeds 10000+K .. 10000+K+N-1, K = --seed-offset)")
    p.add_argument("--sim2real", choices=["on", "off"], default=None,
                   help="servo delay, sensor noise and the camera model (track_env.SIM2REAL) in every "
                        "test (default: as the policy was trained)")
    p.add_argument("--s2r-clips", type=int, default=40, metavar="N",
                   help="with sim2real on, also run N full clips from frame 0 (nominal physics), "
                        "seeds K..K+N-1 (K = --seed-offset), each with its own delay, lag, bias and noise draw")
    p.add_argument("--seed-offset", type=int, default=0, metavar="K",
                   help="start the --s2r-clips and --random-physics seeds K later, for draws beyond "
                        "the standard ones (the same K gives every policy the same draws)")
    p.add_argument("--joint-vel", choices=["noisy", "diff"], default=None,
                   help="with sim2real: joint speeds from the simulator plus noise ('noisy') or differenced "
                        "from the reported angles ('diff'), as a hand that reports positions only; "
                        "default: as the policy was trained (env_kwargs.json)")
    p.add_argument("--joint-vel-lowpass", type=float, default=None, metavar="W",
                   help="with --joint-vel diff: low-pass weight of the new difference (1 = none)")
    p.add_argument("--joint-noise", type=float, default=None, metavar="RAD",
                   help="with sim2real: per-tick encoder noise bound instead of the trained one")
    p.add_argument("--disturb", type=float, default=0.0, metavar="S",
                   help="pushes on the pen and kicks on the fingertips in every test, at S times "
                        "the group's full strength (track_env.DISTURB)")
    args = p.parse_args()
    if args.policy == "model" and not args.model:
        p.error("--model is required unless --policy zero")
    act, env_kwargs = build_actor(args)
    if args.time_scale is not None:
        env_kwargs = {**env_kwargs, "time_scale": args.time_scale}
    if args.sim2real is not None:
        env_kwargs = {**env_kwargs, "sim2real": args.sim2real == "on"}
    overrides = {k: v for k, v in (("joint_vel", args.joint_vel), ("joint_vel_lowpass", args.joint_vel_lowpass),
                                   ("joint_noise", args.joint_noise)) if v is not None}
    if overrides and not env_kwargs.get("sim2real"):
        p.error("--joint-vel, --joint-vel-lowpass and --joint-noise act through sim2real; add --sim2real on")
    if args.joint_vel_lowpass is not None and args.joint_vel != "diff":
        p.error("--joint-vel-lowpass filters differenced joint speeds; add --joint-vel diff")
    if overrides:
        env_kwargs = {**env_kwargs, "sim2real_cfg": {**env_kwargs.get("sim2real_cfg", {}), **overrides}}
    env_kwargs = {**env_kwargs, "disturb": args.disturb}
    print(f"sim2real (delay, noise, camera model) {'on' if env_kwargs.get('sim2real') else 'off'}"
          + (f"; {overrides}" if overrides else "")
          + (f"; disturbances x{args.disturb:g}" if args.disturb else ""))
    env = PenTrack(**{**env_kwargs, "adaptive_sampling": False, "hold_prob": 0.0, "randomize_physics": False})
    T = env.n_frames

    hold = int(round(args.hold_s / env.dt))
    frames = [] if args.video else None
    ref_data = mujoco.MjData(env.model) if args.video else None
    renderers = [mujoco.Renderer(env.model, 480, 560) for _ in range(2)] if args.video else None
    traces = {"target": [], "joint": [], "reference": []}
    info, swept, ref_swept = run_episode(env, act, 0, hold, frames, ref_data, renderers, traces)
    print(f"full clip from frame 0 (hold {args.hold_s:g} s): tracked {info['frames_tracked']}/{T - 1} frames "
          f"({100 * info['frames_tracked'] / (T - 1):.0f}%, {info['frames_tracked'] * env.dt:.2f} s), "
          f"{'FAILED' if info['failed'] else 'completed'}; pen axis swept {swept:.0f} deg "
          f"(reference over the same frames {ref_swept:.0f} deg); mean error "
          f"{info['mean_pos_err_cm']:.2f} cm / {info['mean_axis_err_deg']:.1f} deg")
    print("  jitter (power of the per-step change above 5 Hz): servo targets "
          f"{100 * high_freq_share(traces['target'], env.dt):.0f}%, joint angles "
          f"{100 * high_freq_share(traces['joint'], env.dt):.0f}%, reference angles "
          f"{100 * high_freq_share(traces['reference'], env.dt):.0f}%")
    joint_speed = np.sqrt(((np.diff(np.asarray(traces["reference"]), axis=0) / env.dt) ** 2).mean())
    print(f"  shaking (velocity above 5 Hz, RMS): joint angles {high_freq_rms(traces['joint'], env.dt):.3f} rad/s, "
          f"servo targets {high_freq_rms(traces['target'], env.dt):.3f} rad/s; the reference moves at "
          f"{joint_speed:.3f} rad/s RMS in all")

    if env.sim2real:
        print(f"  (that clip's draw: servo delay {2 * env._delay_sub} ms, camera lag {env._pen_lag} ticks)")
        rows = []
        for i in range(args.s2r_clips):
            tr = {"target": [], "joint": [], "reference": []}
            info, _, _ = run_episode(env, act, 0, hold, traces=tr, seed=args.seed_offset + i)
            rows.append((2 * env._delay_sub, env._pen_lag, 1 - info["failed"], info["frames_tracked"] / (T - 1),
                         info["mean_pos_err_cm"], info["mean_axis_err_deg"], high_freq_rms(tr["joint"], env.dt)))
        if rows:
            r = np.array(rows)
            print(f"{args.s2r_clips} full clips, each with its own sim2real draw: {100 * r[:, 2].mean():.0f}% "
                  f"completed, {100 * r[:, 3].mean():.0f}% of the clip tracked on average; mean error "
                  f"{r[:, 4].mean():.2f} cm / {r[:, 5].mean():.1f} deg; joint shaking {np.nanmean(r[:, 6]):.3f} rad/s")
            for lo, hi in ((0, 12), (14, 26), (28, 40)):
                sel = (r[:, 0] >= lo) & (r[:, 0] <= hi)
                if sel.any():
                    print(f"  servo delay {lo:2d}-{hi:2d} ms ({sel.sum():2d} clips): {100 * r[sel, 2].mean():.0f}% "
                          f"completed, error {r[sel, 4].mean():.2f} cm / {r[sel, 5].mean():.1f} deg")

    rng = np.random.default_rng(1234)
    rows = []
    for i in range(args.starts):
        s = int(rng.integers(0, T - 50))
        # Each start its own seed: with sim2real on, its own delay and noise draw.
        info, swept, ref_swept = run_episode(env, act, s, 0, seed=1000 + i)
        rows.append((info["frames_tracked"] / (T - 1 - s), info["failed"], info["frames_tracked"] * env.dt,
                     info["mean_pos_err_cm"], info["mean_axis_err_deg"]))
    if rows:
        r = np.array(rows)
        print(f"{args.starts} random starts: {100 * r[:, 0].mean():.0f}% of the remaining clip tracked on average, "
              f"{100 * (1 - r[:, 1].mean()):.0f}% reach the end; {r[:, 2].mean():.2f} s per episode; mean error "
              f"{r[:, 3].mean():.2f} cm / {r[:, 4].mean():.1f} deg")

    if args.random_physics:
        renv = PenTrack(**{**env_kwargs, "adaptive_sampling": False, "hold_prob": 0.0, "randomize_physics": True})
        rows = []
        for i in range(args.random_physics):
            info, _, _ = run_episode(renv, act, 0, hold, seed=10_000 + args.seed_offset + i)
            rows.append((1 - info["failed"], info["frames_tracked"] / (T - 1), info["mean_pos_err_cm"],
                         info["mean_axis_err_deg"]))
        r = np.array(rows)
        print(f"{args.random_physics} full clips with randomized physics (track_env.RANDOMIZATION): "
              f"{100 * r[:, 0].mean():.0f}% completed, {100 * r[:, 1].mean():.0f}% of the clip tracked on average; "
              f"mean error {r[:, 2].mean():.2f} cm / {r[:, 3].mean():.1f} deg")

    if args.video:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=args.fps, macro_block_size=1)
        print(f"wrote {args.video} ({len(frames)} frames; left: policy, free pen; right: reference, forced pose)")


if __name__ == "__main__":
    main()
