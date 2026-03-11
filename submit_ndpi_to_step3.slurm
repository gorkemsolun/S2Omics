#!/bin/bash
#SBATCH --job-name=s2omics_step3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

set -euo pipefail

ROOT_DIR="/scratch/"
PROJECT_DIR="${ROOT_DIR}/S2Omics"
INPUT_GLOB="${ROOT_DIR}/Data/NSCLC2/data/*.ndpi"
WORK_DIR="${ROOT_DIR}/output_root"

# Model settings
FOUNDATION_MODEL="uni"
CKPT_PATH="${PROJECT_DIR}/checkpoints/uni/"
DEVICE="cuda:0"
DOWN_SAMP_STEP=10
BATCH_SIZE=32
NUM_WORKERS=4
TARGET_LEVEL=0

# Total array tasks should match SBATCH array size.
NUM_TASKS=10
TASK_ID="${SLURM_ARRAY_TASK_ID}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# Load module and activate environment
module load miniforge3/25.3.0-3
source activate s2omics

python run_batch_ndpi_to_step3.py \
  --input-glob "${INPUT_GLOB}" \
  --work-dir "${WORK_DIR}" \
  --target-level "${TARGET_LEVEL}" \
  --foundation-model "${FOUNDATION_MODEL}" \
  --ckpt-path "${CKPT_PATH}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --down-samp-step "${DOWN_SAMP_STEP}" \
  --num-workers "${NUM_WORKERS}" \
  --task-id "${TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --skip-existing
