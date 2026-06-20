#!/bin/sh
# Parallel leave-one-out sweep: 4 input-types x 13 patients x 3 seeds, N concurrent.
# Each fold is an isolated process writing its own result file (results_par/),
# so parallel appends can't interleave. Resumes: skips folds already done.
# Usage: sh parallel_sweep.sh [N_parallel]
cd /home/adhd/src/research/human-computer-interaction/eeg-cybersickness
PY=${PY:-python}
NPAR=${1:-4}
mkdir -p results_par logs_sweep

INPUT_TYPES="power-spectral-difference power-spectral-no-kinematic power-spectral-no-eeg kinematic"
PATIENTS="0001 0002 0003 0005 0006 0007 1000 1001 1002 1003 1004 1101 1102"
SEEDS="10 20 40"

# Emit "IT P S" for every fold not yet done.
gen_jobs() {
  for it in $INPUT_TYPES; do
    for p in $PATIENTS; do
      for s in $SEEDS; do
        out="results_par/${it}__${p}__${s}.jsonl"
        [ -s "$out" ] && continue           # resume: skip completed folds
        echo "$it $p $s"
      done
    done
  done
}

NJOBS=$(gen_jobs | wc -l)
echo "[$(date +%T)] launching $NJOBS folds, $NPAR-way parallel"

gen_jobs | xargs -P "$NPAR" -n 3 sh -c '
  it=$1; p=$2; s=$3
  out="results_par/${it}__${p}__${s}.jsonl"
  echo "[$(date +%T)] start  $it $p seed=$s"
  ${PY:-python} ./main.py \
      --patient "$p" --seed "$s" --input-type "$it" --out "$out" \
      > "logs_sweep/${it}__${p}__${s}.log" 2>&1
  echo "[$(date +%T)] done   $it $p seed=$s (exit $?)"
' sh

echo "[$(date +%T)] PARALLEL SWEEP DONE -- $(ls results_par/*.jsonl | wc -l) result files"
