"""After each pencil7 segment (third loop): pencil7_next2.py's rules, plus going back to the best model.

    pencil7_next3.py EVAL_TXT SCALE FINGER RESUMED_FROM PREV_EVAL_TXT PREV_SCALE PREV_FINGER BEST_EVAL_TXT

BEST_EVAL_TXT is the eval of the best segment so far at this scale ("-" for
none). Prints "NEXT <scale> <finger-touch-penalty> <model to resume from>
<best eval at the next scale, or ->".

pencil7_4 trained at the same settings as pencil7_3 and drifted: 3.16 -> 2.61
revolutions, 6% -> 22% ending early at scale 0.5. pencil7_next2.py then
doubled the finger charge on top of the drifted model. Now, after the
too-costly check:

  worse than the best at this scale (revolutions down by more than 0.3, or
  ending early up by more than 10 points) -> back to the best model, same settings

"Best" is the most revolutions times the share of episodes that ran the full
10 s. The other rules are pencil7_next2.py's.
"""
import re
import sys
from pathlib import Path

STEP, MAX_SCALE, MAX_FINGER = 0.25, 2.0, 2.0


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


def main() -> None:
    eval_txt, resumed_from, prev_txt, best_txt = sys.argv[1], sys.argv[4], sys.argv[5], sys.argv[8]
    scale, finger, prev_scale, prev_finger = (float(sys.argv[i]) for i in (2, 3, 6, 7))
    r, before = read(eval_txt), read(prev_txt)
    best = read(best_txt) if best_txt != "-" else None
    if best is None or score(r) > score(best):
        new_best = eval_txt
    else:
        new_best = best_txt
    model = model_of(eval_txt)
    if r["revs"] >= 2.7 and r["early"] <= 5 and r["stalled"] <= 2 and r["touch"] <= 10:
        print(f"NEXT {min(scale + STEP, MAX_SCALE):g} {finger:g} {model} -")
    elif r["revs"] < 2.0:
        if (scale, finger) == (prev_scale, prev_finger):
            prev_scale = scale / 2
        print(f"NEXT {prev_scale:g} {prev_finger:g} {resumed_from} {new_best if prev_scale == scale else '-'}")
    elif best is not None and (r["revs"] < best["revs"] - 0.3 or r["early"] > best["early"] + 10):
        print(f"NEXT {scale:g} {finger:g} {model_of(best_txt)} {best_txt}")
    elif r["touch"] > 10:
        if before["touch"] - r["touch"] < 5:
            finger = min(2 * finger, MAX_FINGER)
        print(f"NEXT {scale:g} {finger:g} {model} {new_best}")
    else:
        print(f"NEXT {scale:g} {finger:g} {model} {new_best}")


if __name__ == "__main__":
    main()
