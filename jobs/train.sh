#!/bin/bash
# train.sh —

# ------- LSF resources ------
#BSUB -J train
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=128G]"
#BSUB -W 6:00
#BSUB -o logs/train.%J.out
#BSUB -e logs/train.%J.err

# --------------------------------

set -Eeuo pipefail
trap 'ec=$?; echo "[ERROR] line ${LINENO} status ${ec}" >&2' ERR


# --- Resolve repo root to the SUBMISSION directory, not the script folder ---
SUBMIT_DIR="$(pwd)"     # because -cwd is set by LSF to the submission dir (or %J_workdir)
echo "Submit dir: ${SUBMIT_DIR}"


echo "------------------------------------------------------------"
echo "JOB START: $(date)"
echo "JOBID     : ${LSB_JOBID:-local}  IDX=${LSB_JOBINDEX:-}"
echo "HOST      : $(hostname)"
echo "PWD       : $(pwd)"
echo "------------------------------------------------------------"

# ---- Modules / shell setup ----
module purge || true
module load anaconda3/latest || true
module load cuda/12.4.0 || true

# ---- Conda bootstrap ----
if ! base_dir="$(conda info --base 2>/dev/null)"; then
  base_dir="$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "${base_dir}/etc/profile.d/conda.sh"

# ---- Paths / env ----
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/geometric"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"

export PIP_CACHE_DIR CONDA_PKGS_DIRS PYTHONNOUSERSITE=1 TERM=xterm PYTHONUNBUFFERED=1
unset PYTHONPATH || true

echo "Activating conda env: ${ENV_PREFIX}"
conda activate "${ENV_PREFIX}" || { echo "[ERROR] conda activate failed"; exit 1; }
PYTHON="${ENV_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python"

# ---- Project paths ----
LOG_LEVEL="INFO"
DATA_FN="output/data/20251031_all_binding_db_genes.parquet"
OUTPUT_DIR="output/data/graph_dta"; mkdir -p "${OUTPUT_DIR}"
MAIN="src/train.py"
DATASET_NAME="All_binding_db_genes"
MODEL=0
MODEL_DIR="output/models"
RESULT_DIR="output/results"
TRAIN_BATCH_SIZE=512
TEST_BATCH_SIZE=512
LR=0.0005
NUM_EPOCHS=1000
LOG_INTERVAL=20

ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${ts}_${DATASET_NAME}_train.log"

[[ -f "${MAIN}" ]] || { echo "[ERROR] MAIN not found: ${MAIN} (PWD=$(pwd))"; exit 2; }

echo "Python     : $(command -v "${PYTHON}")"
echo "Main script: ${MAIN}"
echo "Data file  : ${DATA_FN}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Dataset name: ${DATASET_NAME}"
echo "Model      : ${MODEL}"
echo "Model dir  : ${MODEL_DIR}"
echo "Result dir : ${RESULT_DIR}"
echo "Train batch size: ${TRAIN_BATCH_SIZE}"
echo "Test batch size: ${TEST_BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Number of epochs: ${NUM_EPOCHS}"
echo "Log interval: ${LOG_INTERVAL}"
echo "Log file   : ${LOG_FILE}"
echo "------------------------------------------------------------"

set +e
"${PYTHON}" "${MAIN}" \
  --log_fn "${LOG_FILE}" \
  --log_level "${LOG_LEVEL}" \
  --data_fn "${DATA_FN}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --model "${MODEL}" \
  --model_dir "${MODEL_DIR}" \
  --result_dir "${RESULT_DIR}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --lr "${LR}" \
  --num_epochs "${NUM_EPOCHS}" \
  --log_interval "${LOG_INTERVAL}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
  echo "[OK] finished at $(date)"
else
  echo "[ERROR] exit code ${exit_code} at $(date)"
  exit ${exit_code}
fi

echo "JOB END: $(date)"
