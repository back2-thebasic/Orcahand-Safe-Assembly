"""Train a residual tracking policy for one pen-spin reference with SB3 PPO.

    python train.py --name track003_1 --timesteps 20_000_000
    python train.py --name track003_2 --resume runs/track003_1/final_model.zip

Runs go to rl/runs/<name>/ (final_model.zip, vecnormalize.pkl, env_kwargs.json,
checkpoints/, tensorboard logs).
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from track_env import DEFAULT_RUN, SIM2REAL, PenTrack

HERE = Path(__file__).resolve().parent
EPISODE_KEYS = ("failed", "start_frame", "frames_tracked", "mean_pos_err_cm", "mean_axis_err_deg")
TERMS = ("pos", "rot", "link", "joint", "action_rate", "torque", "collision")


def make_env(env_kwargs: dict, seed: int):
    def thunk():
        env = Monitor(PenTrack(**env_kwargs), info_keywords=EPISODE_KEYS)
        env.reset(seed=seed)
        return env
    return thunk


def build_vec_env(n_envs: int, seed: int, env_kwargs: dict, stats: Path | None, gamma: float):
    fns = [make_env(env_kwargs, seed + i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    if stats is not None and stats.exists():
        venv = VecNormalize.load(str(stats), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=gamma, clip_obs=10.0)
    return venv


class TrackMetrics(BaseCallback):
    """Episode outcomes and per-term rewards (per step, before dt) into the logs."""

    def _on_rollout_start(self) -> None:
        self._terms = {t: [] for t in TERMS}
        self._episodes = {k: [] for k in EPISODE_KEYS}
        self._from_zero = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for t in TERMS:
                self._terms[t].append(info[f"r_{t}"])
            if "episode" in info:
                for k in EPISODE_KEYS:
                    self._episodes[k].append(info[k])
                if info["start_frame"] == 0:
                    self._from_zero.append(info["frames_tracked"])
        return True

    def _on_rollout_end(self) -> None:
        for t, v in self._terms.items():
            if v:
                self.logger.record(f"reward_terms/{t}", float(np.mean(v)))
        for k, v in self._episodes.items():
            if v:
                self.logger.record(f"track/{k}", float(np.mean(v)))
        if self._from_zero:
            self.logger.record("track/frames_from_frame0", float(np.mean(self._from_zero)))


class DisturbCurriculum(BaseCallback):
    """Ramp the disturbance strength linearly with this run's steps: 0 before
    `start`, `peak` from `full` on (by steps only, as the group's
    linear_step_curriculum; a success-rate curriculum would be skewed by the
    adaptive start sampling)."""

    def __init__(self, peak: float, start: int, full: int):
        super().__init__()
        self.peak, self.start, self.full = peak, start, max(full, start + 1)
        self.value = None

    def _on_training_start(self) -> None:
        self.t0 = self.model.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        t = self.model.num_timesteps - self.t0
        value = self.peak * float(np.clip((t - self.start) / (self.full - self.start), 0.0, 1.0))
        if value != self.value:
            self.training_env.env_method("set_disturb", value)
            self.value = value
        self.logger.record("track/disturb", value)


class ClampLogStd(BaseCallback):
    """Keep the policy's exploration noise between min and max std."""

    def __init__(self, lo: float, hi: float):
        super().__init__()
        self.lo, self.hi = np.log(lo), np.log(hi)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        log_std = self.model.policy.log_std
        log_std.data.clamp_(self.lo, self.hi)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--reference", default=str(DEFAULT_RUN),
                   help="directory with scene.xml and reference.npz (relative: to this directory)")
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", default=None, metavar="MODEL.zip")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2560)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--init-std", type=float, default=0.5)
    p.add_argument("--min-std", type=float, default=0.2)
    p.add_argument("--max-std", type=float, default=1.0)
    p.add_argument("--action-scale", type=float, default=0.4)
    p.add_argument("--ema-alpha", type=float, default=0.5)
    p.add_argument("--hold-prob", type=float, default=0.15)
    p.add_argument("--w-action-rate", type=float, default=0.1)
    p.add_argument("--time-scale", type=float, default=None,
                   help="play the clip this many times slower than the human demonstration "
                        "(default: as the reference is stored; optimized003-final is 3)")
    p.add_argument("--randomize-physics", action="store_true",
                   help="redraw contact friction, pen mass, servo kp and joint damping at every reset "
                        "(ranges in track_env.RANDOMIZATION)")
    p.add_argument("--sim2real", action="store_true",
                   help="servo command delay, encoder bias and noise, and a camera model for the pen "
                        "(latency, jitter, calibration offset, dropped frames); see track_env.SIM2REAL")
    p.add_argument("--disturb", type=float, default=0.0, metavar="S",
                   help="pushes on the pen and kicks on the fingertips up to S times the group's full "
                        "strength (track_env.DISTURB), ramped in by --disturb-ramp")
    p.add_argument("--disturb-ramp", type=float, nargs=2, default=(0.05, 0.6), metavar=("START", "FULL"),
                   help="shares of --timesteps at which the disturbance ramp starts and reaches S")
    p.add_argument("--pen-angvel-frame", choices=["world", "pen"], default="world",
                   help="frame of the pen's angular velocity in the observation; track003_1-3 "
                        "were trained with 'pen' (the free joint's own, which turns with the roll)")
    p.add_argument("--checkpoint-every", type=int, default=2_000_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = HERE / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(
        run=args.reference,
        action_scale=args.action_scale,
        ema_alpha=args.ema_alpha,
        hold_prob=args.hold_prob,
        w_action_rate=args.w_action_rate,
        time_scale=args.time_scale,
        world_angvel=args.pen_angvel_frame == "world",
        randomize_physics=args.randomize_physics,
        sim2real=args.sim2real,
    )
    if args.sim2real:
        # Record the magnitudes trained with, so evaluation reproduces them even if SIM2REAL changes.
        env_kwargs["sim2real_cfg"] = dict(SIM2REAL)
    (run_dir / "env_kwargs.json").write_text(json.dumps(env_kwargs, indent=2))
    (run_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    resume = Path(args.resume) if args.resume else None
    stats = None
    if resume:
        stats = resume.parent / "vecnormalize.pkl"
        if resume.parent.name == "checkpoints":
            stats = resume.parent / resume.name.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
        if not stats.exists():
            raise SystemExit(f"--resume: {stats} not found (the observation statistics the model was trained with)")
    venv = build_vec_env(args.n_envs, args.seed, env_kwargs, stats, args.gamma)
    venv.gamma = args.gamma

    if resume:
        # PPO.load keeps every hyperparameter saved in the zip unless overridden here.
        model = PPO.load(str(resume), env=venv, device="cpu",
                         custom_objects=dict(learning_rate=args.lr, n_steps=args.n_steps,
                                             batch_size=args.batch_size, n_epochs=args.n_epochs,
                                             gamma=args.gamma, gae_lambda=args.gae_lambda,
                                             clip_range=args.clip_range, ent_coef=args.ent_coef))
        model.tensorboard_log = str(run_dir / "tb")
        model.set_random_seed(args.seed)
        print(f"resumed from {resume} ({model.num_timesteps:,} steps)")
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
            learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_range=args.clip_range, ent_coef=args.ent_coef, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 512, 256]),
                               log_std_init=float(np.log(args.init_std))),
            tensorboard_log=str(run_dir / "tb"), seed=args.seed, device="cpu", verbose=1,
        )

    callbacks = [
        TrackMetrics(),
        ClampLogStd(args.min_std, args.max_std),
        *([DisturbCurriculum(args.disturb, int(args.disturb_ramp[0] * args.timesteps),
                             int(args.disturb_ramp[1] * args.timesteps))] if args.disturb > 0 else []),
        CheckpointCallback(save_freq=max(args.checkpoint_every // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"), name_prefix="ppo",
                           save_vecnormalize=True),
    ]
    print(f"obs {venv.observation_space.shape} act {venv.action_space.shape} envs {args.n_envs}")

    def _stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=resume is None, progress_bar=False)
    except KeyboardInterrupt:
        print("\ninterrupted -- saving what we have")
    model.save(run_dir / "final_model")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    venv.close()
    print(f"saved {run_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
