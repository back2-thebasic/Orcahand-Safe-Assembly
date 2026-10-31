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
GAMMA = 0.995  # pencil_train.py's --gamma: what the policy weighs a cost by


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
    goals, over, discounted, stalled, stalled_s, charged_s, touch_s = [], 0, [], [], [], [], []
    max_tilt_deg = np.degrees(env.max_tilt_rad)
    for seed in seeds:
        _, info = env.reset(seed=seed)
        resumed.append(info["stall_start"])
        state = {"t": 0, "yaw": env._yaw(), "rest": env._cube_pos().copy()}
        total, total_sym, total_step, total_disc = 0.0, 0.0, 0.0, 0.0
        charged = 0  # steps the stall or the uncounted penalty would bill
        cap = env.max_spin_rate * env.dt
        peak = 0.0
        for t in range(steps):
            state["t"] = t
            _, reward, terminated, truncated, info = env.step(policy(env, state))
            total += reward
            total_disc += GAMMA**t * reward
            charged += int(not (info["in_hand"] and info["tilt_deg"] <= max_tilt_deg)
                           or info.get("stalled", False))
            # What the two rejected designs would have paid: a cap on both
            # sides (ratchet check) and a per-step cap with no carry (shiver).
            total_sym += float(np.clip(info["spin_this_step"], -cap, cap))
            total_step += min(float(info["spin_this_step"]), cap)
            # Goals are reached on paid yaw, and paid never runs ahead of the
            # net turn, so the bonuses so far can never cover more than the
            # furthest the pencil has been turned.
            peak = max(peak, info["turned_rad"])
            if env.goal_step_rad is not None:
                over += info["goals_reached"] * env.goal_step_rad > peak + 1e-9
            if terminated or truncated:
                break
        returns.append(total)
        discounted.append(total_disc)
        stalled.append(info["stalled_frac"])
        stalled_s.append(info["stalled_frac"] * info["elapsed_steps"] * env.dt)
        charged_s.append(charged * env.dt)
        touch_s.append(info["finger_touch_frac"] * info["elapsed_steps"] * env.dt)
        revs.append(info["turned_revs"])
        drops.append(info["dropped"])
        stood.append(info["stood_up"])
        goals.append(info["goals_reached"])
        sym.append(total_sym)
        step_cap.append(total_step)
    env.close()
    return {"return": float(np.mean(returns)), "revs": float(np.mean(revs)),
            "drop": float(np.mean(drops)), "stood": float(np.mean(stood)),
            "resumed": float(np.mean(resumed)), "goals": float(np.mean(goals)), "over": over, "steps": int(info["elapsed_steps"]), "sym_cap": float(np.mean(sym)),
            "step_cap": float(np.mean(step_cap)), "disc": float(np.mean(discounted)),
            "stalled": float(np.mean(stalled)), "stalled_s": float(np.mean(stalled_s)),
            "charged_s": float(np.mean(charged_s)), "touch_s": float(np.mean(touch_s))}


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


def sink(env, s, depth=0.125):
    """Hold the pencil level, sunk into the hand below the held line (0.13 m)."""
    pos = s["rest"].copy()
    pos[2] = depth
    return place(env, pos, yaw_quat(s["yaw"]))


def push_then_park(seconds: float = 2.0, rate: float = 1.0):
    """Turn at `rate` for `seconds`, then hold the pencil still where it got to."""
    def policy(env, s):
        if s["t"] * env.dt >= seconds:
            return place(env, s["rest"], yaw_quat(s["yaw"]))
        return turning(rate)(env, s)
    return policy


def drop_it(env, s):
    """Put the pencil down beside the hand, below the drop line, at once."""
    pos = s["rest"] + np.array([0.0, 0.3, 0.0])
    pos[2] = 0.05
    return place(env, pos, yaw_quat(s["yaw"]))


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

    # --uncounted-penalty and --goal-step-deg (pencil4), on top of pencil3's
    # options. Every way of earning has to keep its old verdict, with bonuses
    # only for net turn; every way of parking unscored now has to cost.
    g4 = dict(max_stand_deg=60.0, uncounted_penalty=0.5, goal_step_deg=45.0, goal_bonus=1.0)
    step = np.deg2rad(45.0)
    print("\nwith --uncounted-penalty 0.5 --goal-step-deg 45 --goal-bonus 1 (and --max-stand-deg 60):")
    g_rows = [("do nothing", zero), ("oracle: ccw at the cap, 1 rad/s", turning(1.0)),
              ("oracle: ccw at 5x the cap", turning(5.0)), ("oracle: wrong way (cw), 1 rad/s", turning(-1.0)),
              ("ratchet: creep fwd, snap back", ratchet), ("0.5 rad/s with a +-2 deg shiver", shiver),
              ("stand it 50 deg up and turn", stand_at(50.0, rate=1.0)),
              ("toss: fwd in hand, back in air", toss), ("sink it below the held line", sink)]
    g = {name: run(policy, **g4) for name, policy in g_rows}
    for name, r in g.items():
        print(f"  {name:36s} {r['return']:8.2f} {r['revs']:7.2f} {100 * r['drop']:5.0f}%  goals {r['goals']:5.1f}")
    gz, g1, g5 = g["do nothing"], g["oracle: ccw at the cap, 1 rad/s"], g["oracle: ccw at 5x the cap"]
    check(abs(gz["return"] - z["return"]) < 0.3,
          f"idling costs about the same ({gz['return']:.2f} vs {z['return']:.2f})")
    expected = o1["return"] + np.floor(o1["return"] / step)
    check(abs(g1["return"] - expected) < 0.5 and g1["goals"] == np.floor(o1["return"] / step),
          f"turning at the cap earns the spin plus a bonus every 45 degrees ({g1['return']:.2f}, "
          f"{g1['goals']:.0f} goals)")
    check(g5["return"] < g1["return"] + 0.5 and g5["goals"] <= g1["goals"],
          f"5x faster reaches no more goals ({g5['goals']:.0f}); the cap still holds")
    check(g["oracle: wrong way (cw), 1 rad/s"]["return"] < -8.0
          and g["oracle: wrong way (cw), 1 rad/s"]["goals"] == 0, "the wrong direction reaches no goals")
    check(g["ratchet: creep fwd, snap back"]["return"] < 0.2 and g["ratchet: creep fwd, snap back"]["goals"] == 0,
          "ratcheting reaches no goals")
    gs = g["0.5 rad/s with a +-2 deg shiver"]
    check(gs["return"] > 0.9 * (5.0 + np.floor(5.0 / step)), f"shivering still costs nothing ({gs['return']:.2f})")
    g50 = g["stand it 50 deg up and turn"]
    check(abs(g50["return"] + 5.0) < 0.2 and g50["goals"] == 0,
          f"standing 50 degrees up now costs 0.5/s ({g50['return']:.2f}) and reaches nothing")
    check(g["toss: fwd in hand, back in air"]["return"] < 0.0 and g["toss: fwd in hand, back in air"]["goals"] == 0,
          "tossing reaches no goals")
    gk = g["sink it below the held line"]
    check(abs(gk["return"] + 5.0) < 0.2, f"hiding the pencil below the held line costs 0.5/s ({gk['return']:.2f})")
    check(sum(r["over"] for r in g.values()) == 0,
          "at no step did the bonuses cover more turn than the pencil had made")

    # --stall-penalty (pencil5), on top of pencil4's options. Holding the
    # pencil still now costs what not counting it does, so doing nothing is
    # no longer ~0 -- that check is replaced by: parking still costs less
    # than dropping, as the policy weighs it (discounted), so dropping on
    # purpose never pays. Every way of turning keeps its pencil4 return.
    g5 = dict(g4, stall_penalty=0.5)
    print("\nwith --stall-penalty 0.5 (and pencil4's options):")
    p_rows = [("do nothing", zero), ("drop it at once", drop_it),
              ("oracle: ccw at the cap, 1 rad/s", turning(1.0)), ("oracle: ccw at 5x the cap", turning(5.0)),
              ("oracle: wrong way (cw), 1 rad/s", turning(-1.0)),
              ("ccw at 0.25 rad/s (14 deg/s)", turning(0.25)), ("ccw at 0.1 rad/s (6 deg/s)", turning(0.1)),
              ("push 2 s at the cap, then park", push_then_park(2.0)),
              ("ratchet: creep fwd, snap back", ratchet), ("0.5 rad/s with a +-2 deg shiver", shiver),
              ("stand it 50 deg up and turn", stand_at(50.0, rate=1.0)),
              ("toss: fwd in hand, back in air", toss), ("sink it below the held line", sink)]
    p = {name: run(policy, **g5) for name, policy in p_rows}
    park4 = run(push_then_park(2.0), **g4)
    print(f"  {'':36s} {'return':>8s} {'revs':>7s} {'drop':>6s}  {'goals':>5s}  {'stalled':>7s}  {'disc.':>6s}")
    for name, r in p.items():
        print(f"  {name:36s} {r['return']:8.2f} {r['revs']:7.2f} {100 * r['drop']:5.0f}%  {r['goals']:5.1f}"
              f"  {100 * r['stalled']:6.0f}%  {r['disc']:6.2f}")
    pz, pd, p1 = p["do nothing"], p["drop it at once"], p["oracle: ccw at the cap, 1 rad/s"]
    ref = PencilSpin()
    window_s = ref.stall_window_steps * ref.dt
    line_deg_s = np.degrees(ref.stall_turn_rad) / window_s
    forever = 0.5 * ref.dt / (1.0 - GAMMA)
    ref.close()
    check(abs(pz["return"] - (gz["return"] - 0.5 * pz["stalled_s"])) < 0.01 and pz["stalled"] > 0.8,
          f"doing nothing is now charged 0.5/s for the {pz['stalled_s']:.1f} s it sits stalled "
          f"({pz['return']:.2f}); the first {window_s:.0f} s are never a stall")
    check(pd["drop"] == 1.0 and pz["disc"] > pd["disc"] and forever < env_drop(),
          f"parking still beats dropping as the policy weighs it: {pz['disc']:.2f} vs {pd['disc']:.2f} "
          f"discounted, and parked forever costs {forever:.2f} < the drop penalty {env_drop():.0f}")
    check(abs(p1["return"] - g1["return"]) < 0.05 and p1["stalled"] == 0.0,
          f"turning at the cap is never stalled and earns what it did ({p1['return']:.2f})")
    check(p["oracle: ccw at 5x the cap"]["return"] < p1["return"] + 0.5, "5x faster still earns no more")
    pcw = p["oracle: wrong way (cw), 1 rad/s"]
    check(pcw["return"] < g["oracle: wrong way (cw), 1 rad/s"]["return"] - 3.0 and pcw["goals"] == 0,
          f"the wrong direction makes no new ground: it is charged and stalled ({pcw['return']:.2f})")
    p14, p6 = p["ccw at 0.25 rad/s (14 deg/s)"], p["ccw at 0.1 rad/s (6 deg/s)"]
    check(p14["stalled"] == 0.0 and p6["stalled"] > 0.8,
          f"the line is {line_deg_s:.0f} deg/s: 14 deg/s is never stalled, "
          f"6 deg/s nearly always ({100 * p6['stalled']:.0f}%)")
    pp = p["push 2 s at the cap, then park"]
    parked_s = 10.0 - 2.0 - window_s
    # Where the script sets the pencil down it does not always touch the
    # hand (2 of the 10 seeds); that time is billed as uncounted instead.
    check(abs((park4["return"] - pp["return"]) - 0.5 * pp["stalled_s"]) < 0.01
          and pp["charged_s"] > parked_s,
          f"pencil4's end state costs now: pushing 2 s and parking earns {pp['return']:.2f}, "
          f"{park4['return'] - pp['return']:.2f} less than under pencil4; from {window_s:.0f} s after it "
          f"stops every second is billed, stalled or uncounted ({pp['charged_s']:.1f} s of {parked_s + window_s:.0f})")
    prat = p["ratchet: creep fwd, snap back"]
    check(prat["return"] < -3.0 and prat["stalled"] > 0.8 and prat["goals"] == 0,
          f"a ratchet covers no new ground after its first arc and is charged as a stall ({prat['return']:.2f})")
    check(abs(p["0.5 rad/s with a +-2 deg shiver"]["return"] - gs["return"]) < 0.05,
          "shivering while turning still costs nothing")
    check(abs(p["stand it 50 deg up and turn"]["return"] - g50["return"]) < 0.05
          and abs(p["sink it below the held line"]["return"] - gk["return"]) < 0.05,
          "unscored time pays the uncounted penalty once, not a stall on top")
    check(p["toss: fwd in hand, back in air"]["return"] < 0.0, "tossing still loses")
    check(sum(r["over"] for r in p.values()) == 0,
          "at no step did the bonuses cover more turn than the pencil had made")

    # --finger-touch-penalty (pencil6), on top of pencil5's options. It bills
    # fingers touching each other and nothing else, and spinning with the
    # fingers touching the whole time must still beat parking, or the policy
    # could give up turning to keep its fingers apart.
    g6 = dict(g5, finger_touch_penalty=1.0)
    print("\nwith --finger-touch-penalty 1 (and pencil5's options):")
    f_rows = [("do nothing", zero), ("oracle: ccw at the cap, 1 rad/s", turning(1.0)),
              ("fist: four fingers curled shut", fist)]
    f = {name: run(policy, **g6) for name, policy in f_rows}
    fist5 = run(fist, **g5)
    print(f"  {'':36s} {'return':>8s} {'revs':>7s} {'drop':>6s}  {'touching':>8s}  {'disc.':>6s}")
    for name, r in f.items():
        print(f"  {name:36s} {r['return']:8.2f} {r['revs']:7.2f} {100 * r['drop']:5.0f}%"
              f"  {r['touch_s']:7.1f}s  {r['disc']:6.2f}")
    fz, f1, ff = f["do nothing"], f["oracle: ccw at the cap, 1 rad/s"], f["fist: four fingers curled shut"]
    check(fz["touch_s"] == 0.0 and f1["touch_s"] == 0.0
          and abs(fz["return"] - pz["return"]) < 1e-9 and abs(f1["return"] - p1["return"]) < 1e-9,
          "a still hand never has two fingers touching: doing nothing and the oracle earn exactly what they did")
    check(abs((fist5["return"] - ff["return"]) - 1.0 * ff["touch_s"]) < 0.01 and ff["touch_s"] > 8.0,
          f"a fist is billed 1/s for the {ff['touch_s']:.1f} s its fingers touch "
          f"({fist5['return']:.2f} -> {ff['return']:.2f})")
    always = 1.0 * ref.dt * (1.0 - GAMMA**1000) / (1.0 - GAMMA)
    check(p1["return"] - 10.0 > pz["return"] and p1["disc"] - always > pz["disc"],
          f"turning at the cap with fingers touching the whole 10 s still beats parking: "
          f"{p1['disc'] - always:.2f} vs {pz['disc']:.2f} discounted")

    print("\nall checks passed" if ok else "\nCHECKS FAILED -- do not train on this reward")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
