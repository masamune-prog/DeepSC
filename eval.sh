#!/bin/bash
#SBATCH --job-name=deepsc_diffusion
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Create log directory if it does not exist
mkdir -p logs

# Print job diagnostics
echo "=================================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "Start time:    $(date)"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo "=================================================="

# Navigate to project root
cd "${SLURM_SUBMIT_DIR}"

# Ensure CUDA runtime diagnostics are recorded
nvidia-smi

# Execute training inside the virtual environment managed by uv
# (64G RAM prevents OOM when unpickling the ~580k sentences)
PYTHONUNBUFFERED=1 uv run python -u diffusion_perf_ori.py --bert-score --channel Rayleigh --sampler ddim --epoch-seeds 10 20 30 40 50 --epochs 5

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
