"""Prove the pencil reward pays for spinning and nothing else.

    python -m orca_rl.pencil_checks

Scripted policies, most of which move the pencil directly (they are not
policies a hand could execute -- they are probes of the reward). Each line
asserts what an honest reward has to do with that motion. Run this before
training, and again after every change to the reward.
"""

from __future__ import annotations

import mujoco
import numpy as np

from orca_rl.pencil import PencilSpin

SEEDS = range(10)


def yaw_quat(yaw: float, tilt: float = 0.0) -> np.ndarray:
    """Pencil axis at `yaw`, raised `tilt` above level (rotation about local y)."""
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    qy = np.array([np.cos(-tilt / 2), 0.0, np.sin(-tilt / 2), 0.0])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def run(policy, seeds=SEEDS, steps: int = 1000, **env_kwargs) -> dict:
    env = PencilSpin(**env_kwargs)
    returns, revs, drops, stood, resumed, sym, step_cap = [], [], [], [], [], [], []
    for seed in seeds:
        _, info = env.reset(seed=seed)
        resumed.append(info["stall_start"])
        state = {"t": 0, "yaw": env._yaw(), "rest": env._cube_pos().copy()}
        total, total_sym, total_step = 0.0, 0.0, 0.0
        cap = env.max_spin_rate * env.dt
        for t in range(steps):
            state["t"] = t
            _, reward, terminated, truncated, info = env.step(policy(env, state))
            total += reward
            # What the two rejected designs would have paid: a cap on both
            # sides (ratchet check) and a per-step cap with no carry (shiver).
            total_sym += float(np.clip(info["spin_this_step"], -cap, cap))
            total_step += min(float(info["spin_this_step"]), cap)
            if terminated or truncated:
                break
        returns.append(total)
        revs.append(info["turned_revs"])
        drops.append(info["dropped"])
        stood.append(info["stood_up"])
        sym.append(total_sym)
        step_cap.append(total_step)
    env.close()
    return {"return": float(np.mean(returns)), "revs": float(np.mean(revs)),
            "drop": float(np.mean(drops)), "stood": float(np.mean(stood)),
            "resumed": float(np.mean(resumed)), "steps": int(info["elapsed_steps"]), "sym_cap": float(np.mean(sym)),
            "step_cap": float(np.mean(step_cap))}


def zero(env, s):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def random(env, s):
    return env.action_space.sample()


def fist(env, s):
    """Curl the four fingers shut and keep them there -- pencil1's end state."""
    m = env.model
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for a in range(m.nu):
        name = m.joint(int(m.actuator_trnid[a, 0])).name
        if name.endswith(("_mcp", "_pip")) and "thumb" not in name:
            action[a] = 1.0   # positive is flexion
    return action


def place(env, pos, quat, angvel_body=(0.0, 0.0, 0.0)):
    """Pose the pencil directly, hand left alone, then hold the hand still."""
    adr, vadr = env._cube_qpos_adr, env._cube_qvel_adr
    env.data.qpos[adr:adr + 3] = pos
    env.data.qpos[adr + 3:adr + 7] = quat
    env.data.qvel[vadr:vadr + 3] = 0.0
    env.data.qvel[vadr + 3:vadr + 6] = angvel_body
    mujoco.mj_forward(env.model, env.data)
    return zero(env, None)


def turning(rate: float):
    """Oracle: turn the pencil flat on the palm at `rate` rad/s (+ = ccw)."""
    def policy(env, s):
        s["yaw"] += rate * env.dt
        return place(env, s["rest"], yaw_quat(s["yaw"]), (0.0, 0.0, rate))
    return policy


def ratchet(env, s, slow=0.5, fast=4.0, period=45, back=5):
    """Creep forward 0.2 rad at `slow`, snap back 0.2 rad at `fast`: net zero."""
    phase = s["t"] % period
    rate = -fast if phase >= period - back else slow
    s["yaw"] += rate * env.dt
    return place(env, s["rest"], yaw_quat(s["yaw"]), (0.0, 0.0, rate))


def shiver(env, s, rate=0.5, amp=np.deg2rad(2.0), period=4):
    """Turn forward at `rate` while the heading shivers +-2 degrees.

    The shiver adds no net yaw, so it should cost nothing. 2 degrees per step
    at 100 Hz is far over the cap, which is exactly what a pencil pressed by
    servoing fingers does: under random actions 20% of steps exceed it.
    """
    s["yaw"] += rate * env.dt
    wobble = amp * (1.0 if (s["t"] % period) < period // 2 else -1.0)
    return place(env, s["rest"], yaw_quat(s["yaw"] + wobble), (0.0, 0.0, rate))


def roll(env, s, rate=20.0):
    """Spin the pencil about its own long axis, where it lies."""
    adr = env._cube_qpos_adr
    quat = env.data.qpos[adr + 3:adr + 7].copy()
    return place(env, env._cube_pos(), quat, (rate, 0.0, 0.0))


def standing(env, s, rate=1.0, tilt=np.deg2rad(60)):
    """Turn the pencil about the vertical while it stands 60 degrees up."""
    s["yaw"] += rate * env.dt
    lift = np.array([0.0, 0.0, 0.08 * np.sin(tilt)])
    return place(env, s["rest"] + lift, yaw_quat(s["yaw"], tilt), (0.0, 0.0, rate))


def stand_at(deg: float, rate: float = 0.0):
    def policy(env, s):
        return standing(env, s, rate=rate, tilt=np.deg2rad(deg))
    return policy


def toss(env, s, period=100):
    """Turn it forward 0.5 rad in hand, back 1.0 rad in the air, set it down.

    Net -0.5 rad a cycle. If yaw lost while the pencil is out of the hand went
    unbilled, this would collect +0.5 a cycle.
    """
    phase = s["t"] % period
    if phase < 50:                       # in hand, forward at 1 rad/s
        rate, pos = 1.0, s["rest"]
    elif phase < 80:                     # 5 cm up, back at 3.33 rad/s
        rate, pos = -1.0 / (30 * env.dt), s["rest"] + [0.0, 0.0, 0.05]
    else:                                # back on the palm, still
        rate, pos = 0.0, s["rest"]
    s["yaw"] += rate * env.dt
    return place(env, pos, yaw_quat(s["yaw"]), (0.0, 0.0, rate))


def env_drop() -> float:
    return PencilSpin().drop_penalty


def resume_is_exact() -> bool:
    """Save a state mid-episode, resume from it, replay the same actions."""
    env = PencilSpin(stall_start_prob=1.0, pose_penalty=5e-4)
    env.reset(seed=3)
    actions = np.random.default_rng(0).uniform(-0.5, 0.5, size=(400, 17)).astype(np.float32)
    for a in actions[:150]:
        env.step(a)
    env._stall_states.clear()
    env._save_stall()
    original = [env.step(a) for a in actions[150:]]
    _, info = env.reset(seed=4)
    fresh_count = info["stall_start"] and info["turned_rad"] == 0.0 and info["elapsed_steps"] == 0
    resumed = [env.step(a) for a in actions[150:]]
    env.close()
    same = all(np.array_equal(o[0], r[0]) for o, r in zip(original, resumed))
    return bool(fresh_count and same)


def main() -> None:
    rows = [
        ("do nothing", zero),
        ("random", random),
        ("oracle: ccw at the cap, 1 rad/s", turning(1.0)),
        ("oracle: ccw at 5x the cap", turning(5.0)),
        ("oracle: wrong way (cw), 1 rad/s", turning(-1.0)),
        ("ratchet: creep fwd, snap back", ratchet),
        ("0.5 rad/s with a +-2 deg shiver", shiver),
        ("roll about its own axis, 20 rad/s", roll),
        ("stand it 60 deg up and turn", standing),
        ("toss: fwd in hand, back in air", toss),
    ]
    results = {}
    print(f"{'policy (10 episodes of 10 s)':38s} {'return':>8s} {'revs':>7s} {'drop':>6s}")
    print("-" * 62)
    for name, policy in rows:
        r = run(policy)
        results[name] = r
        print(f"{name:38s} {r['return']:8.2f} {r['revs']:7.2f} {100 * r['drop']:5.0f}%")

    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'ok' if cond else 'FAIL'}] {msg}")

    print()
    z, o1, o5 = results["do nothing"], results["oracle: ccw at the cap, 1 rad/s"], results["oracle: ccw at 5x the cap"]
    check(abs(z["return"]) < 0.5, f"doing nothing earns ~0 ({z['return']:.2f})")
    check(o1["return"] > 8.0, f"turning at the cap for 10 s earns ~10 ({o1['return']:.2f})")
    check(o5["return"] < o1["return"] + 0.5,
          f"5x faster earns no more ({o5['return']:.2f} vs {o1['return']:.2f}); flicks do not pay")
    check(results["oracle: wrong way (cw), 1 rad/s"]["return"] < -8.0, "the wrong direction is charged in full")
    rat = results["ratchet: creep fwd, snap back"]
    check(rat["return"] < 0.2, f"ratcheting earns nothing ({rat['return']:.2f}); "
                               f"a cap on both sides would have paid {rat['sym_cap']:.2f}")
    shv, smooth = results["0.5 rad/s with a +-2 deg shiver"], 0.5 * 10.0
    check(shv["return"] > 0.9 * smooth,
          f"shivering costs nothing: {shv['return']:.2f} for 5 rad turned; "
          f"a per-step cap without the carry would have paid {shv['step_cap']:.2f}")
    check(abs(results["roll about its own axis, 20 rad/s"]["return"]) < 0.2, "rolling in place earns nothing")
    check(abs(results["stand it 60 deg up and turn"]["return"]) < 0.2, "turning a standing pencil earns nothing")
    check(results["toss: fwd in hand, back in air"]["return"] < 0.0, "yaw lost in the air is billed on landing")
    # The optional pose term (--pose-penalty). It has to leave the two honest
    # baselines alone -- idle and the oracle both keep the hand where it
    # settled -- and make pencil1's fist cost something.
    c = 5e-4
    print(f"\nwith --pose-penalty {c:g}:")
    pose_rows = [("do nothing", zero), ("oracle: ccw at the cap, 1 rad/s", turning(1.0)), ("close a fist and hold", fist)]
    pose = {name: run(policy, pose_penalty=c) for name, policy in pose_rows}
    plain_fist = run(fist)
    for name, r in pose.items():
        print(f"  {name:36s} {r['return']:8.2f} {r['revs']:7.2f} {100 * r['drop']:5.0f}%")
    check(abs(pose["do nothing"]["return"] - z["return"]) < 0.1,
          f"idling costs the same as without it ({pose['do nothing']['return']:.2f} vs {z['return']:.2f})")
    check(pose["oracle: ccw at the cap, 1 rad/s"]["return"] > 8.0, "turning at the cap still earns ~10")
    check(pose["close a fist and hold"]["return"] < plain_fist["return"] - 2.0,
          f"a fist costs {plain_fist['return'] - pose['close a fist and hold']['return']:.2f} per episode")

    # --max-stand-deg and --stall-start-prob (pencil3). Standing the pencil up
    # past the limit has to cost a drop; just under it, between max_tilt_deg
    # and the limit, it still earns nothing and costs nothing. Neither option
    # may move the honest baselines.
    stand = dict(max_stand_deg=60.0)
    print(f"\nwith --max-stand-deg 60:")
    stand_rows = [("do nothing", zero), ("oracle: ccw at the cap, 1 rad/s", turning(1.0)),
                  ("stand it 50 deg up and turn", stand_at(50.0, rate=1.0)),
                  ("stand it 70 deg up and hold", stand_at(70.0))]
    st = {name: run(policy, **stand) for name, policy in stand_rows}
    for name, r in st.items():
        print(f"  {name:36s} {r['return']:8.2f} {r['revs']:7.2f} {100 * r['drop']:5.0f}%"
              f"  stood up {100 * r['stood']:3.0f}%")
    check(abs(st["do nothing"]["return"] - z["return"]) < 0.1,
          f"idling costs the same ({st['do nothing']['return']:.2f} vs {z['return']:.2f}); "
          f"its slips into a finger gap now end as a stand ({100 * st['do nothing']['stood']:.0f}%) "
          f"instead of a drop")
    check(st["oracle: ccw at the cap, 1 rad/s"]["return"] > 8.0, "turning flat at the cap still earns ~10")
    s50 = st["stand it 50 deg up and turn"]
    check(abs(s50["return"]) < 0.2 and s50["stood"] == 0.0,
          f"50 degrees up still earns and costs nothing ({s50['return']:.2f})")
    s70 = st["stand it 70 deg up and hold"]
    check(s70["stood"] == 1.0 and s70["steps"] == 1 and abs(s70["return"] + env_drop()) < 0.05,
          f"70 degrees up ends the episode at once and costs the drop penalty ({s70['return']:.2f})")

    print("\nwith --stall-start-prob 0.5 (and --max-stand-deg 60):")
    resume = dict(stall_start_prob=0.5, **stand)
    many = range(30)
    rz = run(zero, seeds=many, **resume)
    ro = run(turning(1.0), seeds=many, **resume)
    print(f"  {'do nothing':36s} {rz['return']:8.2f} {rz['revs']:7.2f} {100 * rz['drop']:5.0f}%"
          f"  resumed {100 * rz['resumed']:3.0f}%")
    print(f"  {'oracle: ccw at the cap, 1 rad/s':36s} {ro['return']:8.2f} {ro['revs']:7.2f} "
          f"{100 * ro['drop']:5.0f}%  resumed {100 * ro['resumed']:3.0f}%")
    check(0.2 < rz["resumed"] < 0.8, f"an idle hand stalls, and about half the resets resume a stall "
                                     f"({100 * rz['resumed']:.0f}%)")
    check(abs(rz["return"] - z["return"]) < 0.5, f"idling from a resumed stall still earns ~0 ({rz['return']:.2f})")
    check(ro["resumed"] == 0.0 and ro["return"] > 8.0,
          "a policy that never stalls is never resumed, and still earns ~10")
    check(resume_is_exact(), "a resumed state steps exactly as the original episode did, "
                             "and its yaw is counted from zero")

    print("\nall checks passed" if ok else "\nCHECKS FAILED -- do not train on this reward")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
