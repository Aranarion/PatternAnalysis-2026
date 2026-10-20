#!/bin/bash
#SBATCH --job-name=experiments
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=outputs/logs/%x_%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
python experiments.py vae-sweep --out-dir outputs_experiments
python experiments.py regularisation --out-dir outputs_experiments --vqvae-ckpt $OUT/vqvae/best.pt
