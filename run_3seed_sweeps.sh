#!/bin/sh
# Publication-grade 3-seed leave-one-out sweep (paper seeds 10/20/40).
cd /home/adhd/src/research/human-computer-interaction/eeg-cybersickness
mkdir -p logs_sweep
SEEDS="10 20 40"
sh sweep.sh power-spectral-difference   regression results_psd-imu_3seed.jsonl   $SEEDS
sh sweep.sh power-spectral-no-kinematic regression results_psd-only_3seed.jsonl  $SEEDS
sh sweep.sh power-spectral-no-eeg       regression results_imu-only_3seed.jsonl  $SEEDS
sh sweep.sh kinematic                   regression results_kinematic_3seed.jsonl $SEEDS
echo "3-SEED SWEEPS DONE: $(date +%T)"
