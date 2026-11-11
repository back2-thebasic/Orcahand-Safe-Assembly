#!/bin/bash
# The standard evaluation of one run, the same seeds every time, all six tests
# in parallel (do not run it beside a training run: it takes every core).
#
#   bash eval_suite.sh track003_10
#   PY=/path/to/python bash eval_suite.sh track003_10   # a relative PY is taken from the current directory
#
# Outputs go to runs/stage_eval/<run>/, and a summary is printed.
#   deploy           delay and noise, joint speeds differenced from the angles (as on the hand)
#   deploy_dist1     the same with the full pen pushes and fingertip kicks (track_env.DISTURB)
#   deploy_dist0.5   ... at half strength
#   s2r_noisy        delay and noise, joint speeds from the simulator plus noise
#   s2r_noisy_dist1  ... with the full disturbances
#   clean            no delay, noise or disturbances
set -u
RUN=${1:?usage: bash eval_suite.sh <run>}
PY=${PY:-python}
case $PY in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")" ;; esac
cd "$(dirname "$0")" || exit 1
command -v "$PY" > /dev/null || { echo "no interpreter '$PY'; set PY=/path/to/python" >&2; exit 1; }
M=runs/$RUN/final_model.zip
O=runs/stage_eval/$RUN
[ -f "$M" ] || { echo "no $M" >&2; exit 1; }
mkdir -p "$O"
"$PY" evaluate.py --model "$M" --sim2real on --joint-vel diff --starts 100 --s2r-clips 100 --random-physics 200 > "$O/deploy.txt" 2>&1 &
"$PY" evaluate.py --model "$M" --sim2real on --joint-vel diff --disturb 1.0 --starts 0 --s2r-clips 200 --random-physics 200 > "$O/deploy_dist1.txt" 2>&1 &
"$PY" evaluate.py --model "$M" --sim2real on --joint-vel diff --disturb 0.5 --starts 0 --s2r-clips 100 --random-physics 100 > "$O/deploy_dist0.5.txt" 2>&1 &
"$PY" evaluate.py --model "$M" --sim2real on --joint-vel noisy --starts 100 --s2r-clips 100 --random-physics 200 > "$O/s2r_noisy.txt" 2>&1 &
"$PY" evaluate.py --model "$M" --sim2real on --joint-vel noisy --disturb 1.0 --starts 0 --s2r-clips 200 --random-physics 200 > "$O/s2r_noisy_dist1.txt" 2>&1 &
"$PY" evaluate.py --model "$M" --sim2real off --starts 100 --random-physics 200 > "$O/clean.txt" 2>&1 &
wait
echo "== $RUN ($(date))"
for f in deploy deploy_dist1 deploy_dist0.5 s2r_noisy s2r_noisy_dist1 clean; do
  echo "-- $f"
  grep -E "^(full clip|[0-9]+ (full clips|random starts))|shaking|Error" "$O/$f.txt" | sed -E 's/;? ?(pen axis swept [0-9]+ deg \(reference over the same frames [0-9]+ deg\))//'
done
