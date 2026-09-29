"""Side-by-side videos of a trained policy vs the scripted controller on a taskgen bundle."""
import sys, argparse
from pathlib import Path
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "rl")
from wrappers import make_env
from evaluate_bundle import run_config_for, stats_path_for

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
p.add_argument("--every", type=int, default=1, help="render every Nth control step")
p.add_argument("--fps", type=int, default=25)
p.add_argument("--max-steps", type=int, default=450)
p.add_argument("--tail", type=int, default=25, help="frames to hold after an episode ends")
p.add_argument("--azimuth", type=float, default=150)
p.add_argument("--elevation", type=float, default=-12)
p.add_argument("--distance", type=float, default=0.42)
p.add_argument("--track", choices=["object", "hand"], default="object")
p.add_argument("--title", default="")
args = p.parse_args()

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
mp = Path(args.model)
cfg = run_config_for(mp)
bundle, rel = cfg["bundle"], cfg.get("relative_hand")
if not Path(bundle).exists():
    bundle = str(Path("examples/generated") / Path(bundle).name)
model = PPO.load(str(mp), device="cpu")
norm = VecNormalize.load(str(stats_path_for(mp)), DummyVecEnv([lambda: make_env(bundle, rel)]))
norm.training = False

W, H = 480, 400
try:
    font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 17)
    big = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 20)
except OSError:
    font = big = ImageFont.load_default()
STAGES = {0: "张开", 1: "半握", 2: "握拳", 3: "张开"}


def rollout(policy: str, seed: int):
    env = make_env(bundle, rel) if policy == "RL" else make_env(bundle, None)
    base = env.unwrapped
    renderer = mujoco.Renderer(base.model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    obs, info = env.reset(seed=seed)
    look = None
    frames, done_at = [], None
    for t in range(args.max_steps):
        if policy == "RL":
            a, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=True)
            a = a[0]
        else:
            a = base.scripted_action()
        if done_at is None:
            obs, r, te, tr, info = env.step(a)
            if te or tr:
                done_at = t + 1
        if t % args.every and done_at is None:
            continue
        target = base._object_pos() if args.track == "object" else base._tip_points().mean(0)
        look = target.copy() if look is None else 0.85 * look + 0.15 * target
        cam.lookat[:] = look
        renderer.update_scene(base.data, cam)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        steps = done_at or (t + 1)
        d.rectangle([0, 0, W, 58], fill=(0, 0, 0))
        d.text((8, 4), f"{'RL 策略' if policy == 'RL' else '脚本控制器'}   t = {steps * 0.01:.2f} s  ({steps} 步)", font=big, fill=(255, 255, 255))
        if base.family == "gesture":
            gi = info["gesture_index"]
            err = base._normalized_hand_error(base._gesture_target())
            tol = float(base.task["success"]["gesture_tolerance"])
            line = f"目标：{STAGES.get(gi, gi)}（第 {gi}/3 段）  误差 {err:.3f}（容差 {tol}）"
            color = (120, 255, 120) if err <= tol else (255, 200, 90)
        else:
            rise = 100 * (base._object_pos()[2] - base._initial_object_position[2])
            state = "成功" if info["is_success"] else ("抓住" if info["grasped"] else "接近中")
            line = f"方块抬高 {rise:4.1f} cm   状态：{state}"
            color = (120, 255, 120) if info["is_success"] else (255, 255, 255)
        d.text((8, 32), line, font=font, fill=color)
        if done_at is not None:
            ok = info["is_success"]
            d.text((8, H - 30), "成功" if ok else "超时（1000 步）", font=big, fill=(120, 255, 120) if ok else (255, 110, 110))
        frames.append(np.asarray(img))
        if done_at is not None and len(frames) >= 1:
            frames.extend([frames[-1]] * args.tail)
            break
    renderer.close()
    env.close()
    return frames


out = []
for seed in args.seeds:
    left, right = rollout("RL", seed), rollout("scripted", seed)
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))
    for a, b in zip(left, right):
        frame = np.concatenate([a, np.full((H, 6, 3), 255, np.uint8), b], axis=1)
        if args.title:
            bar = Image.new("RGB", (frame.shape[1], 34), (30, 30, 30))
            ImageDraw.Draw(bar).text((8, 5), f"{args.title}   (reset seed {seed})", font=big, fill=(255, 255, 255))
            frame = np.concatenate([np.asarray(bar), frame], axis=0)
        out.append(frame)
imageio.mimwrite(args.out, out, fps=args.fps, quality=8, macro_block_size=2)
Image.fromarray(out[len(out) // 4]).save(Path(args.out).with_suffix(".png"))
print(args.out, len(out), "frames")
