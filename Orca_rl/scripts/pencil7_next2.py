"""After each pencil7 segment (second loop): widen the randomization, raise the finger charge, or back off.

    pencil7_next2.py EVAL_TXT SCALE FINGER RESUMED_FROM PREV_EVAL_TXT PREV_SCALE PREV_FINGER

EVAL_TXT is the 100-episode deterministic eval (10 s, seed 1234) of the segment
just trained, with --randomize-physics at the scale it trained at; PREV_EVAL_TXT
the same for the segment before. Prints "NEXT <scale> <finger-touch-penalty>
<model to resume from>".

Good at a scale: revolutions >= 2.7, dropped + stood up <= 5%, held but
stalled <= 2%, fingers touching <= 10% -- the bar pencil6_2 cleared on nominal
physics. One knob at a time, in this order (the first loop, pencil7_next.py,
had only the scale; at the same scale pencil7_2 traded ending early, 21% ->
9%, for fingers touching, 10% -> 17%, so the scale alone could stall):

  good                -> widen by STEP (at most MAX_SCALE)
  revolutions < 2.0   -> too costly: back to the model this segment resumed from,
                         with the previous segment's scale and charge (half the
                         scale if those were the same)
  touching > 10%      -> double the finger charge (at most MAX_FINGER) if
                         touching fell by under 5 points since the previous segment
  otherwise           -> same settings, resume from this model

MAX_FINGER is 2, not pencil6's 3: turning at the cap with the fingers touching
the whole episode is worth 3.94 - 1.98 * charge discounted against -0.60 for
parking (pencil_checks.py), so at 3 parking would win.
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


def main() -> None:
    eval_txt, resumed_from, prev_txt = sys.argv[1], sys.argv[4], sys.argv[5]
    scale, finger, prev_scale, prev_finger = (float(sys.argv[i]) for i in (2, 3, 6, 7))
    r, before = read(eval_txt), read(prev_txt)
    model = str(Path(eval_txt).with_name(Path(eval_txt).name.replace("_eval_rand.txt", "")) / "final_model.zip")
    if r["revs"] >= 2.7 and r["early"] <= 5 and r["stalled"] <= 2 and r["touch"] <= 10:
        print(f"NEXT {min(scale + STEP, MAX_SCALE):g} {finger:g} {model}")
    elif r["revs"] < 2.0:
        if (scale, finger) == (prev_scale, prev_finger):
            prev_scale = scale / 2
        print(f"NEXT {prev_scale:g} {prev_finger:g} {resumed_from}")
    elif r["touch"] > 10:
        if before["touch"] - r["touch"] < 5:
            finger = min(2 * finger, MAX_FINGER)
        print(f"NEXT {scale:g} {finger:g} {model}")
    else:
        print(f"NEXT {scale:g} {finger:g} {model}")


if __name__ == "__main__":
    main()
