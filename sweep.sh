#!/bin/sh
# Leave-one-subject-out validation sweep across the 13 patients.
# Usage: sh sweep.sh <input-type> <task> <out.jsonl> [seed ...]
# Reproduces the paper's leave-one-out protocol (one process per held-out subject).
PY=${PY:-python}
HERE=$(dirname "$0")

INPUT_TYPE=${1:-power-spectral-difference}
TASK=${2:-regression}
OUT=${3:-$HERE/results_${INPUT_TYPE}.jsonl}
shift 3 2>/dev/null
SEEDS=${*:-42}

PATIENTS="0001 0002 0003 0005 0006 0007 1000 1001 1002 1003 1004 1101 1102"

: > "$OUT"   # truncate
echo "sweep input=$INPUT_TYPE task=$TASK seeds=$SEEDS -> $OUT"
for seed in $SEEDS; do
  for p in $PATIENTS; do
    echo "[$(date +%T)] patient=$p seed=$seed"
    $PY "$HERE/main.py" --patient "$p" --seed "$seed" \
        --input-type "$INPUT_TYPE" --task "$TASK" --out "$OUT" \
        > "$HERE/logs_sweep/${INPUT_TYPE}_${p}_${seed}.log" 2>&1
  done
done
echo "[$(date +%T)] sweep done -> $OUT"
