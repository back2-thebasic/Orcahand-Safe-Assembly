"""PPO for spinning a pencil about the vertical axis.

    python -m orca_rl.pencil_checks                      # first, every time the reward changes
    python -m orca_rl.pencil_train --name pencil1 --timesteps 30_000_000 --n-envs 10

Same PPO setup as ``train.py`` (the cube task) and the same layout under
``runs/<name>/``: ``final_model.zip``, ``vecnormalize.pkl``, ``checkpoints/``,
``tb/`` and ``env_kwargs.json``.

The numbers to watch are under ``task/`` in TensorBoard, not ``ep_rew_mean``:
``task/revs_per_episode`` is the progress metric, ``task/drop_rate`` the cost.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from orca_rl.pencil import PencilSpin
from orca_rl.task import ENV_KWARGS_FILE, resolve_stats_path
from orca_rl.train import (
    ClampLogStdCallback,
    EpochsUsedCallback,
    build_vec_env,
    resolve_tensorboard_dir,
)


class PencilMetricsCallback(BaseCallback):
    """Revolutions, spin rate and drops per episode, over a window of episodes."""

    def __init__(self, window: int = 40) -> None:
        super().__init__()
        self.window = window
        self._episodes: list[dict] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self._episodes.append(info)
        if len(self._episodes) < self.window:
            return True

        eps = self._episodes
        steps = np.array([e["elapsed_steps"] for e in eps], dtype=np.float64)
        turned = np.array([e["turned_rad"] for e in eps])
        paid = np.array([e["paid_rad"] for e in eps])
        dt = float(self.training_env.get_attr("dt", 0)[0])
        self.logger.record("task/revs_per_episode", float(np.mean(turned)) / (2 * np.pi))
        # Per second of episode, dropped or not: a policy that spins fast and
        # drops at once should not look better than one that spins steadily.
        self.logger.record("task/spin_rate_deg_s", float(np.degrees(turned.sum() / (steps.sum() * dt))))
        self.logger.record("task/paid_frac", float(paid.sum() / turned.sum()) if turned.sum() > 0 else 0.0)
        self.logger.record("task/drop_rate", float(np.mean([e["dropped"] for e in eps])))
        self.logger.record("task/episode_length", float(np.mean(steps)))
        self.logger.record("task/in_hand_frac", float(np.mean([e["in_hand_frac"] for e in eps])))
        # Episodes cut short because MuJoCo blew up and reset itself (see PencilSpin.step).
        self.logger.record("task/unstable_rate", float(np.mean([e.get("sim_unstable", False) for e in eps])))
        self._episodes = []
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="pencil")
    p.add_argument("--timesteps", type=int, default=30_000_000,
                   help="steps to train in THIS invocation; added on top with --resume")
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu | auto | cuda | mps")
    p.add_argument("--no-subproc", action="store_true", help="single process (easier to debug)")

    # PPO -- the cube task's tuned values; see train.py for why each differs from SB3's.
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--log-std-init", type=float, default=-0.5)
    p.add_argument("--max-log-std", type=float, default=0.0)

    # task
    p.add_argument("--version", default="v1", choices=["v1"],
                   help="only the v1 hand (the physical one) has a pencil scene")
    p.add_argument("--max-episode-steps", type=int, default=1000, help="1000 = 10 s")
    p.add_argument("--spin-direction", default="ccw", choices=["ccw", "cw"],
                   help="seen from above")
    p.add_argument("--max-spin-rate", type=float, default=1.0,
                   help="rad/s. Forward yaw beyond this rate is not paid, so flicking the "
                        "pencil into a free spin is worth no more than turning it steadily")
    p.add_argument("--spin-carry", type=float, default=0.2,
                   help="rad of over-cap forward yaw carried to later steps. 0 makes every "
                        "vibration of the pencil a net cost")
    p.add_argument("--drop-penalty", type=float, default=2.0,
                   help="a full episode at the cap is worth 10. Keep this well under it: "
                        "in the cube task drop_penalty > success_bonus taught the policy "
                        "to clamp the object and freeze")
    p.add_argument("--pose-penalty", type=float, default=0.0,
                   help="cost per rad^2 of joint-angle distance from the pose the hand settled "
                        "in. pencil1 (0) turned the pencil once, then clamped it in a fist for "
                        "the rest of the episode. 5e-4 makes that fist cost about half of what "
                        "turning at the cap pays")
    p.add_argument("--max-tilt-deg", type=float, default=45.0,
                   help="yaw only counts while the pencil is within this of level")
    p.add_argument("--action-mode", default="relative", choices=["relative", "absolute"])
    p.add_argument("--action-scale", type=float, default=0.15)
    p.add_argument("--randomize-physics", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000)
    p.add_argument("--resume", default=None, metavar="MODEL.zip",
                   help="continue from a saved model; its vecnormalize.pkl comes along")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        version=args.version,
        max_episode_steps=args.max_episode_steps,
        spin_direction=args.spin_direction,
        max_spin_rate=args.max_spin_rate,
        spin_carry_rad=args.spin_carry,
        drop_penalty=args.drop_penalty,
        pose_penalty=args.pose_penalty,
        max_tilt_deg=args.max_tilt_deg,
        action_mode=args.action_mode,
        action_scale=args.action_scale,
        randomize_physics=args.randomize_physics,
    )
    (run_dir / ENV_KWARGS_FILE).write_text(json.dumps({"task": "pencil_spin", **env_kwargs}, indent=2))

    resume_path = Path(args.resume) if args.resume else None
    stats_path = resolve_stats_path(resume_path) if resume_path else None
    if resume_path is not None and not resume_path.exists():
        raise SystemExit(f"--resume: {resume_path} not found")

    venv = build_vec_env(
        args.n_envs, args.seed, env_kwargs,
        subproc=not args.no_subproc,
        vecnormalize_path=stats_path,
        gamma=args.gamma,
        env_cls=PencilSpin,
    )
    if stats_path is not None and not stats_path.exists():
        print(f"WARNING: {stats_path} not found -- observation statistics restart from zero")

    if resume_path is not None:
        model = PPO.load(
            str(resume_path), env=venv, device=args.device,
            custom_objects=dict(
                learning_rate=args.lr, clip_range=args.clip_range, ent_coef=args.ent_coef,
                target_kl=args.target_kl, n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
            ),
        )
        model.tensorboard_log = resolve_tensorboard_dir(run_dir)
        print(f"resumed policy from {resume_path} ({model.num_timesteps:,} steps already trained)")
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
            learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               log_std_init=args.log_std_init),
            tensorboard_log=resolve_tensorboard_dir(run_dir),
            seed=args.seed, device=args.device, verbose=1,
        )

    callbacks = [
        PencilMetricsCallback(),
        EpochsUsedCallback(),
        ClampLogStdCallback(args.max_log_std),
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
        ),
    ]

    print(f"obs {venv.observation_space.shape}  act {venv.action_space.shape}  envs {args.n_envs}")
    print(f"rollout = {args.n_envs} x {args.n_steps} = {args.n_envs * args.n_steps} transitions")
    print(f"logging to {run_dir}")

    def _stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=resume_path is None,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\ninterrupted -- saving what we have")

    model.save(run_dir / "final_model")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    venv.close()
    print(f"saved {run_dir/'final_model.zip'} and {run_dir/'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
