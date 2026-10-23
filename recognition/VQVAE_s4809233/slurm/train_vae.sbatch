#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
for w in 0.1 0.01; do
    python train.py --model vae --kl-weight $w --out-dir $OUT --downsample $DOWNSAMPLE --epochs 50
done
