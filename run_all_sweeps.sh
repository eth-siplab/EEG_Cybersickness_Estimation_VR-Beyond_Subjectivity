#!/bin/sh
# Full leave-one-out validation: the four input-types mapping to the paper's Table.
cd /home/adhd/src/research/human-computer-interaction/eeg-cybersickness
mkdir -p logs_sweep
sh sweep.sh power-spectral-difference   regression results_psd-imu.jsonl   42
sh sweep.sh power-spectral-no-kinematic regression results_psd-only.jsonl  42
sh sweep.sh power-spectral-no-eeg       regression results_imu-only.jsonl  42
sh sweep.sh kinematic                   regression results_kinematic.jsonl 42
echo "ALL SWEEPS DONE: $(date +%T)"
