#!/bin/bash
#SBATCH --job-name=train_prior
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
python train.py --model prior --out-dir $OUT --vqvae-ckpt $OUT/vqvae/best.pt --epochs 100 --lr 3e-4
