#!/bin/bash
#SBATCH --job-name=s2omics_step1to5
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=72:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsolun@student.ethz.ch

set -euo pipefail

PROJECT_DIR="${HOME}/S2Omics"
INPUT_GLOB="${SCRATCH}/Data/NSCLC1/HES1/*.ndpi"
WORK_DIR="${SCRATCH}/S2Omics_outputs"
LOG_DIR="${PROJECT_DIR}/logs"

# Model settings
FOUNDATION_MODEL="uni"
CKPT_PATH="${PROJECT_DIR}/checkpoints/uni/"
DEVICE="cuda:0"
DOWN_SAMP_STEP=10
BATCH_SIZE=32
NUM_WORKERS=4
TARGET_LEVEL=0

# Step 4/5 settings
CLUSTERING_METHOD="kmeans"
N_CLUSTERS=20
RESOLUTION=1.0
TARGET_N_CLUSTERS=15

NUM_TASKS=10
TASK_ID="${SLURM_ARRAY_TASK_ID}"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"
cd "${PROJECT_DIR}"

module purge
module load stack/2024-06 gcc/12.2.0 python_gpu/3.11.6

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate s2omics
elif [[ -f "${HOME}/mambaforge/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/mambaforge/etc/profile.d/conda.sh"
  conda activate s2omics
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate s2omics
else
  echo "No conda installation detected under ${HOME}."
  echo "Continuing with the loaded Euler Python module."
fi

python run_batch.py \
  --input-glob "${INPUT_GLOB}" \
  --work-dir "${WORK_DIR}" \
  --target-level "${TARGET_LEVEL}" \
  --foundation-model "${FOUNDATION_MODEL}" \
  --ckpt-path "${CKPT_PATH}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --down-samp-step "${DOWN_SAMP_STEP}" \
  --clustering-method "${CLUSTERING_METHOD}" \
  --n-clusters "${N_CLUSTERS}" \
  --resolution "${RESOLUTION}" \
  --target-n-clusters "${TARGET_N_CLUSTERS}" \
  --num-workers "${NUM_WORKERS}" \
  --task-id "${TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --skip-existing