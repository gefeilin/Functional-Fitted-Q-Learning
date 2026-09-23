#!/usr/bin/env bash
#SBATCH --job-name=functional-fqi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# Submit from the repository root with the training environment active.
# Adjust the resource directives above for your cluster.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this example from the repository root}"

export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

APPROXIMATOR="${1:-adafnn}"
TASK_INDEX="${2:-0}"
exec python scripts/run_functional_fqi.py --approximator "$APPROXIMATOR" --index "$TASK_INDEX"
