source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
export PYTHONUNBUFFERED=1
export OUT=${OUT:-outputs}
export DOWNSAMPLE=${DOWNSAMPLE:-4}
nvidia-smi --query-gpu=name,memory.total --format=csv
