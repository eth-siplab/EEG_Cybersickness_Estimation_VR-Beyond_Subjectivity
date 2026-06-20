#!/bin/sh
# Full leave-one-subject-out evaluation, then aggregate.
# Loops main.py over input-types x seeds x held-out subjects (each fold appends a
# JSON line to results/<input-type>.jsonl), then parses with parse_logs.py.
# Usage: PY=python sh evaluate.sh
PY=${PY:-python}
HERE=$(dirname "$0")
mkdir -p "$HERE/results"

INPUT_TYPES="power-spectral-difference power-spectral-no-kinematic power-spectral-no-eeg kinematic"
PATIENTS="0001 0002 0003 0005 0006 0007 1000 1001 1002 1003 1004 1101 1102"
SEEDS="10 20 40"

for it in $INPUT_TYPES; do
  out="$HERE/results/${it}.jsonl"
  : > "$out"
  for s in $SEEDS; do
    for p in $PATIENTS; do
      echo "[$(date +%T)] $it patient=$p seed=$s"
      $PY "$HERE/main.py" --patient "$p" --seed "$s" --input-type "$it" --out "$out"
    done
  done
done

$PY "$HERE/parse_logs.py" "$HERE"/results/*.jsonl
