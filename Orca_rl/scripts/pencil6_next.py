"""After each pencil6 segment: smooth yet? If not, which model and penalty next.

    pencil6_next.py EVAL_TXT PENALTY RESUMED_FROM [PREVIOUS_EVAL_TXT]

Prints "SMOOTH" or "NEXT <penalty> <model to resume from>".

Smooth, over the 100-episode deterministic eval (10 s, seed 1234):
  revolutions >= 2.7      (85% of the 3.18 the 2 rad/s cap allows; pencil5_s1_fast: 2.98)
  dropped + stood up <= 5%
  held but stalled <= 2%
  fingers touching <= 10% (pencil5_s1_fast: 69%)
  held >= 95%             (pencil5_s1_fast: 91%: the pencil is in the air 9% of the time)

Not smooth:
  revolutions < 2.0 -> the penalty cost too much speed: go back to the model this
                       segment resumed from, halve the penalty
  touching > 10% and down by under 5 points since the previous segment
                    -> double the penalty (at most 3: at 3/s, spinning with the
                       fingers touching the whole time still beats parking)
  otherwise         -> same penalty, resume from this segment's model
"""
import re
import sys
from pathlib import Path

MAX_PENALTY = 3.0


def read(path: str) -> dict:
    text = Path(path).read_text()
    def pct(label: str) -> float:
        return float(re.search(label + r"\s*:\s*([\d.]+)%", text).group(1))
    return {
        "revs": float(re.search(r"revolutions / episode : ([+-][\d.]+)", text).group(1)),
        "drop": pct("drop rate"), "stood": pct(r"stood up \(ended\)"),
        "held": pct("fraction of time held"), "stalled": pct("held but stalled"),
        "touch": pct("fingers touching"),
    }


def main() -> None:
    eval_txt, penalty, resumed_from = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    r = read(eval_txt)
    model = str(Path(eval_txt).with_name(Path(eval_txt).name.replace("_eval.txt", "")) / "final_model.zip")
    if (r["revs"] >= 2.7 and r["drop"] + r["stood"] <= 5 and r["stalled"] <= 2
            and r["touch"] <= 10 and r["held"] >= 95):
        print("SMOOTH")
        return
    if r["revs"] < 2.0:
        print(f"NEXT {penalty / 2:g} {resumed_from}")
        return
    if len(sys.argv) > 4 and r["touch"] > 10:
        before = read(sys.argv[4])["touch"]
        if before - r["touch"] < 5:
            print(f"NEXT {min(2 * penalty, MAX_PENALTY):g} {model}")
            return
    print(f"NEXT {penalty:g} {model}")


if __name__ == "__main__":
    main()
