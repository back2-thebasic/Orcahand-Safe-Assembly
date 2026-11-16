"""After each pencil6 segment (second loop): smooth yet? If not, what next.

    pencil6_next2.py EVAL_TXT FINGER DROP RESUMED_FROM PREV_EVAL_TXT PREV_FINGER PREV_DROP

Prints "SMOOTH" or "NEXT <finger-touch-penalty> <drop-penalty> <model to resume from>".

Smooth, over the 100-episode deterministic eval (10 s, seed 1234):
  revolutions >= 2.7          (85% of the 3.18 the 2 rad/s cap allows)
  dropped + stood up <= 5%
  held but stalled <= 2%
  fingers touching <= 10%
(The first loop also asked for held >= 95%. Dropped: pencil6's not-held time is
contact chatter -- 73% of it in gaps of 10-30 ms, the pencil rising a median
1 mm -- not the pencil being thrown. Hops are checked by hand at the end.)

One knob at a time:
  revolutions < 2.0  -> too costly: back to the model this segment resumed from,
                        with the previous segment's penalties (halve the finger
                        penalty if those were the same)
  touching > 10%     -> finger penalty: double it (at most 3) if touching fell
                        by under 5 points since the previous segment
  ended early > 5%   -> drop penalty: +1 (at most 6) if ending early fell by
                        under 2 points since the previous segment
  otherwise          -> same penalties, resume from this segment's model
"""
import re
import sys
from pathlib import Path

MAX_FINGER, MAX_DROP = 3.0, 6.0


def read(path: str) -> dict:
    text = Path(path).read_text()
    def pct(label: str) -> float:
        return float(re.search(label + r"\s*:\s*([\d.]+)%", text).group(1))
    r = {
        "revs": float(re.search(r"revolutions / episode : ([+-][\d.]+)", text).group(1)),
        "stalled": pct("held but stalled"), "touch": pct("fingers touching"),
    }
    r["early"] = pct("drop rate") + pct(r"stood up \(ended\)")
    return r


def main() -> None:
    eval_txt, resumed_from, prev_txt = sys.argv[1], sys.argv[4], sys.argv[5]
    finger, drop, prev_finger, prev_drop = (float(sys.argv[i]) for i in (2, 3, 6, 7))
    r, before = read(eval_txt), read(prev_txt)
    model = str(Path(eval_txt).with_name(Path(eval_txt).name.replace("_eval.txt", "")) / "final_model.zip")
    if r["revs"] >= 2.7 and r["early"] <= 5 and r["stalled"] <= 2 and r["touch"] <= 10:
        print("SMOOTH")
    elif r["revs"] < 2.0:
        if (finger, drop) == (prev_finger, prev_drop):
            prev_finger = finger / 2
        print(f"NEXT {prev_finger:g} {prev_drop:g} {resumed_from}")
    elif r["touch"] > 10:
        if before["touch"] - r["touch"] < 5:
            finger = min(2 * finger, MAX_FINGER)
        print(f"NEXT {finger:g} {drop:g} {model}")
    elif r["early"] > 5:
        if before["early"] - r["early"] < 2:
            drop = min(drop + 1, MAX_DROP)
        print(f"NEXT {finger:g} {drop:g} {model}")
    else:
        print(f"NEXT {finger:g} {drop:g} {model}")


if __name__ == "__main__":
    main()
