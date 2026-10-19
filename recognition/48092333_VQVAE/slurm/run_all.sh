#!/bin/bash
set -eo pipefail
mkdir -p outputs/logs
vae=$(sbatch --parsable slurm/train_vae.sbatch)
vqvae=$(sbatch --parsable slurm/train_vqvae.sbatch)
prior=$(sbatch --parsable --dependency=afterok:$vqvae slurm/train_prior.sbatch)
predict=$(sbatch --parsable --dependency=afterok:$vae:$prior slurm/predict.sbatch)
echo "submitted vae=$vae vqvae=$vqvae prior=$prior predict=$predict"
