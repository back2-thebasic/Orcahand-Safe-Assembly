"""After each pencil7 segment: widen the physics randomization, keep it, or back off.

    pencil7_next.py EVAL_TXT SCALE RESUMED_FROM PREV_SCALE

EVAL_TXT is the 100-episode deterministic eval (10 s, seed 1234) of the segment
just trained, with --randomize-physics at the scale it trained at. Prints
"NEXT <scale> <model to resume from>".

Good at a scale means the same bar pencil6_2 cleared on nominal physics:
  revolutions >= 2.7, dropped + stood up <= 5%, held but stalled <= 2%,
  fingers touching <= 10%.

  good                -> widen by STEP (at most MAX_SCALE), resume from this model
  revolutions < 2.0   -> too wide: back to the model this segment resumed from,
                         at the scale before (half this one if it was the same)
  otherwise           -> same scale, resume from this model
"""
import re
import sys
from pathlib import Path

STEP, MAX_SCALE = 0.25, 2.0


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
    eval_txt, scale, resumed_from, prev_scale = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    r = read(eval_txt)
    model = str(Path(eval_txt).with_name(Path(eval_txt).name.replace("_eval_rand.txt", "")) / "final_model.zip")
    if r["revs"] >= 2.7 and r["early"] <= 5 and r["stalled"] <= 2 and r["touch"] <= 10:
        print(f"NEXT {min(scale + STEP, MAX_SCALE):g} {model}")
    elif r["revs"] < 2.0:
        back = prev_scale if prev_scale < scale else scale / 2
        print(f"NEXT {back:g} {resumed_from}")
    else:
        print(f"NEXT {scale:g} {model}")


if __name__ == "__main__":
    main()
