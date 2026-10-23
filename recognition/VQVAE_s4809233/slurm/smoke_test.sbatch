#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=outputs/logs/%x_%j.out
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
S=outputs/smoke
python dataset.py
for w in 0.1 0.01; do
    python train.py --model vae --kl-weight $w --out-dir $S --max-slices 256 --epochs 5 --downsample $DOWNSAMPLE
done
python train.py --model vqvae --out-dir $S --max-slices 256 --epochs 5 --downsample $DOWNSAMPLE
python train.py --model prior --out-dir $S --vqvae-ckpt $S/vqvae/best.pt --epochs 5
python predict.py --out-dir $S --num-samples 64 --batch-size 32
echo "SMOKE TEST PASSED"
