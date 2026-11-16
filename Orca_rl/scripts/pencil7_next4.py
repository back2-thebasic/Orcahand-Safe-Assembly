"""After each segment of the fourth pencil7 loop: widen, retune, go back, or stop.

    pencil7_next4.py STATE_JSON NAME

NAME is the segment just trained; runs/NAME_eval_rand.txt is its 200-episode
deterministic eval (10 s, seed 1234) at the scale it trained at. STATE_JSON
holds what the rules need from earlier segments and is updated in place.
Prints "NEXT <scale> <finger-touch-penalty> <model to resume from>", or
"DONE <scale> <finger-touch-penalty> <model>" once a model is smooth at scale 1.

pencil7_next3.py's rules, with four changes after the overnight run
(results/2026-10-31-pencil7.md):

- Smooth at scale 1 ends the loop: those are the ranges the randomization was
  built with, and scale 2 was only a stress test. MAX_SCALE is 1.
- The bars are judged on 200 episodes, not 100: pencil7_3 missed three of them
  by one point each on 100 episodes and met all four on 200 others.
- Plateau: after PLATEAU segments at a scale without a new best, widen from
  the best if it turns >= 2.7 revolutions and ends <= 10% early.
- Too costly (under 2.0 revolutions) goes back to the model this segment
  resumed from at the scale halfway back, not all the way back.
- The finger charge doubles only against a segment at the same scale (added
  after pencil7_8, the first at 0.75, was compared with pencil7_3 at 0.5:
  touching rises when the scale widens, and fell by itself overnight, 17% ->
  11% from pencil7_2 to pencil7_3).

Going back to a model repeats no segment, because the loop trains every
segment with its own seed (pencil7_7 replayed pencil7_5 bit for bit).
"""
import json
import re
import sys
from pathlib import Path

STEP, MAX_SCALE, MAX_FINGER, PLATEAU = 0.25, 1.0, 2.0, 3


def read(path: str) -> dict:
    text = Path(path).read_text()
    def pct(label: str) -> float:
        return float(re.search(label + r"\s*:\s*([\d.]+)%", text).group(1))
    return {
        "revs": float(re.search(r"revolutions / episode : ([+-][\d.]+)", text).group(1)),
        "early": pct("drop rate") + pct(r"stood up \(ended\)"),
        "stalled": pct("held but stalled"), "touch": pct("fingers touching"),
    }


def model_of(eval_txt: str) -> str:
    return str(Path(eval_txt).with_name(Path(eval_txt).name.replace("_eval_rand.txt", "")) / "final_model.zip")


def score(r: dict) -> float:
    return r["revs"] * (1.0 - r["early"] / 100.0)


def smooth(r: dict) -> bool:
    return r["revs"] >= 2.7 and r["early"] <= 5 and r["stalled"] <= 2 and r["touch"] <= 10


def main() -> None:
    state_path, name = Path(sys.argv[1]), sys.argv[2]
    s = json.loads(state_path.read_text())
    eval_txt, model = f"runs/{name}_eval_rand.txt", f"runs/{name}/final_model.zip"
    r, before = read(eval_txt), read(s["prev_eval"])
    old_best = read(s["best"]) if s["best"] else None
    if old_best is None or score(r) > score(old_best):
        s["best"], s["since_best"] = eval_txt, 0
    else:
        s["since_best"] += 1
    best = read(s["best"])
    scale, finger = s["scale"], s["finger"]

    verdict, new_finger = "NEXT", finger
    if smooth(r):
        if scale >= MAX_SCALE:
            verdict, new_scale, resume = "DONE", scale, model
        else:
            new_scale, resume = min(scale + STEP, MAX_SCALE), model
    elif r["revs"] < 2.0:
        new_scale = (s["prev_scale"] + scale) / 2 if s["prev_scale"] < scale else scale / 2
        new_finger, resume = s["prev_finger"], s["resume"]
    elif s["since_best"] >= PLATEAU and scale < MAX_SCALE and best["revs"] >= 2.7 and best["early"] <= 10:
        new_scale, resume = min(scale + STEP, MAX_SCALE), model_of(s["best"])
    elif old_best is not None and (r["revs"] < old_best["revs"] - 0.3 or r["early"] > old_best["early"] + 10):
        new_scale, resume = scale, model_of(s["best"])
    elif r["touch"] > 10:
        if s["prev_scale"] == scale and before["touch"] - r["touch"] < 5:
            new_finger = min(2 * finger, MAX_FINGER)
        new_scale, resume = scale, model
    else:
        new_scale, resume = scale, model

    if new_scale != scale:
        s["best"], s["since_best"] = None, 0
    s.update(prev_scale=scale, prev_finger=finger, prev_eval=eval_txt,
             scale=new_scale, finger=new_finger, resume=resume)
    state_path.write_text(json.dumps(s, indent=1) + "\n")
    print(f"{verdict} {new_scale:g} {new_finger:g} {resume}")


if __name__ == "__main__":
    main()
