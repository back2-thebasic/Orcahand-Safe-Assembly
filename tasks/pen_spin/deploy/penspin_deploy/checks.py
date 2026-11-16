"""Offline checks of the runtime against the training env, no hand needed.

    python -m penspin_deploy.checks parity BUNDLE
        Rebuild every observation of the bundled training rollout from its recorded
        state and compare: observation, ONNX action, joint target.
    python -m penspin_deploy.checks twin BUNDLE
        Drive the MuJoCo twin open loop with the rollout's actuator commands from
        the rollout's first state and report how far it drifts from the training sim.
    python -m penspin_deploy.checks pen-timeout BUNDLE
        Exercise the camera timeout with simulated pose updates and joint I/O.
"""

from __future__ import annotations

import argparse

import numpy as np

from penspin_deploy.bundle import Bundle
from penspin_deploy.control import ActionMap, build_observation
from penspin_deploy.loop import DEFAULT_PEN_TIMEOUT_MS, HOLD, POLICY, STATE_CODES, Controller


def check_parity(bundle: Bundle, clip_name: str) -> bool:
    clip = bundle.clip(clip_name)
    rec = bundle.parity(clip_name)
    policy = bundle.policy()
    action = ActionMap(bundle)
    steps = len(rec["raw"])
    obs_err = np.zeros(steps)
    raw_err = np.zeros(steps)
    target_err = np.zeros(steps)
    last_raw = np.zeros(bundle.num_joints, dtype=np.float32)
    for i in range(steps):
        obs = build_observation(
            bundle, clip, int(rec["frame"][i]), rec["qpos"][i], rec["qvel"][i],
            rec["pen_local"][i], last_raw,
        )
        obs_err[i] = np.abs(obs - rec["obs"][i]).max()
        raw = policy.infer(rec["obs"][i])
        raw_err[i] = np.abs(raw - rec["raw"][i]).max()
        target = action(rec["raw"][i], clip.reference_qpos[int(rec["frame"][i])])
        target_err[i] = np.abs(target - rec["target"][i]).max()
        last_raw = rec["raw"][i]
    worst = {
        "observation": (obs_err, 1e-4),
        "onnx action": (raw_err, 1e-4),
        "joint target": (target_err, 1e-5),
    }
    ok = True
    for what, (err, tol) in worst.items():
        good = err.max() <= tol
        ok &= good
        print(f"  {clip_name} {what:13s} max |diff| {err.max():.2e} at step {int(err.argmax()):4d}"
              f"  {'ok' if good else f'FAIL (tolerance {tol:.0e})'}")
    if obs_err.max() > 1e-4:
        layout = np.cumsum([0] + [w for _, w in bundle.obs_layout])
        i = int(obs_err.argmax())
        obs = build_observation(bundle, clip, int(rec["frame"][i]), rec["qpos"][i], rec["qvel"][i],
                                rec["pen_local"][i], rec["raw"][i - 1] if i else np.zeros(bundle.num_joints))
        for (name, _), a, b in zip(bundle.obs_layout, layout[:-1], layout[1:]):
            print(f"    {name:20s} {np.abs(obs[a:b] - rec['obs'][i][a:b]).max():.2e}")
    return ok


def check_twin(bundle: Bundle, clip_name: str) -> bool:
    from penspin_deploy.sim_twin import Twin

    clip = bundle.clip(clip_name)
    rec = bundle.parity(clip_name)
    twin = Twin(bundle)
    twin.set_state(rec["sim_qpos"][0], rec["sim_qvel"][0])
    steps = len(rec["target"])
    q_err = np.zeros(steps)
    pen_err = np.zeros(steps)
    for i in range(steps):
        q_err[i] = np.abs(twin.qpos() - rec["qpos"][i]).max()
        pen_err[i] = np.linalg.norm(twin.pen_local()[:3] - rec["pen_local"][i][:3])
        twin.d.ctrl[:] = rec["sim_ctrl"][i]
        twin.step_control()
    hold = int(rec["hold_steps"])
    print(f"  {clip_name} open-loop twin vs training sim, same ctrl sequence:")
    for upto in (1, 10, hold, hold + 50, steps):
        upto = min(upto, steps)
        print(f"    first {upto:4d} steps: joint max |diff| {np.degrees(q_err[:upto].max()):7.3f} deg, "
              f"pen position max diff {pen_err[:upto].max() * 1000:7.2f} mm")
    # Contact is chaotic and native MuJoCo makes up to 4 contacts per cylinder-mesh
    # pair where Warp makes 1, so long open-loop rollouts separate; the first ticks
    # do not. Closed loop is the real test.
    return bool(q_err[1] < 1e-3 and pen_err[1] < 1e-3 and pen_err[:10].max() < 2e-3)


def check_pen_timeout(bundle: Bundle, clip_name: str) -> bool:
    clip = bundle.clip(clip_name)
    policy = bundle.policy()
    timeout_ticks = DEFAULT_PEN_TIMEOUT_MS / (1000 * bundle.ctrl_dt)
    expired = timeout_ticks + 1
    short_gap = timeout_ticks / 2

    def simulate(ages, enter_at, quit_at):
        class Clock:
            tick = 0

            def now(self):
                return self.tick * bundle.ctrl_dt

            def wait(self):
                self.tick += 1

        clock = Clock()
        commands = []

        class Hand:
            def read(self):
                return clip.reference_qpos[0].copy()

            def command(self, _target):
                commands.append(clock.tick)

        class Camera:
            def sample(self):
                age = ages[clock.tick - 1]
                return None if age is None else (clip.object_pose_local[0], age)

        class Operator:
            interactive = True

            def poll(self, _controller):
                if clock.tick == quit_at:
                    return "quit"
                if clock.tick in enter_at:
                    return "next"
                return None

        controller = Controller(bundle, clip, Hand(), clock, Operator(), policy=policy,
                                pen_source=Camera(), ease_s=0, verbose=False,
                                pen_timeout_ms=DEFAULT_PEN_TIMEOUT_MS)
        summary = controller.run()
        return commands, controller.log.rows["state"], summary

    cases = (
        ("start rejects missing and expired poses, then accepts fresh pose",
         [None, expired, 0], {1, 2, 3}, [0, 3], [HOLD], None),
        ("hold survives a short gap and stops at timeout",
         [0, short_gap, 0, expired], {1}, [0, 1, 2, 3], [HOLD, HOLD, HOLD], "pen pose timeout"),
        ("policy stops at timeout",
         [0, 0, 0, expired], {1, 2}, [0, 1, 2, 3], [HOLD, POLICY, POLICY], "pen pose timeout"),
        ("policy stops if the palm frame is lost",
         [0, 0, None], {1, 2}, [0, 1, 2], [HOLD, POLICY], "no valid pen pose"),
    )
    ok = True
    for label, ages, enter_at, expected_commands, expected_states, reason in cases:
        commands, states, summary = simulate(ages, enter_at, quit_at=len(ages) + 1)
        actual_reason = summary.get("abort_reason")
        reason_ok = actual_reason is None if reason is None else (actual_reason or "").startswith(reason)
        good = (commands == expected_commands and states == [STATE_CODES[s] for s in expected_states]
                and reason_ok and not summary["completed"])
        print(f"  {clip_name} {label}: {'ok' if good else 'FAIL'}")
        if not good:
            print(f"    command ticks: {commands}, states: {states}, summary: {summary}")
        ok &= good
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    checks = {"parity": check_parity, "twin": check_twin, "pen-timeout": check_pen_timeout}
    parser.add_argument("check", choices=checks)
    parser.add_argument("bundle")
    parser.add_argument("--clips", default=None, help="comma-separated; default all bundled clips")
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    clips = args.clips.split(",") if args.clips else bundle.clip_names
    run = checks[args.check]
    results = [run(bundle, clip) for clip in clips]
    print("PASS" if all(results) else "FAIL")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
