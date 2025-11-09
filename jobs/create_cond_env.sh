#!/bin/bash
set -euo pipefail

# --- Modules / shell setup ---
module purge
module load anaconda3/latest
#
# You DO NOT need to load a CUDA module when using the 
# 'pytorch-cuda' package from Conda. Conda brings its own.
# This line is not needed for this script:
# module load cuda/12.4.0
#

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Paths (Your setup is perfect) ---
ENV_PREFIX="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/geometric"
PIP_CACHE_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.pip_cache"
CONDA_PKGS_DIRS="/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/pkgs"

# --- Keep installs off $HOME (Your setup is perfect) ---
mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# --- Create env if missing ---
# Using Python 3.10 for best compatibility with ML packages
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create --prefix "${ENV_PREFIX}" python=3.10 -y
fi

# --- Activate env ---
conda activate "${ENV_PREFIX}"

# --- Install All Packages (The Correct Conda Way) ---
#
# We install everything in ONE command from the correct channels.
#
# **CRITICAL CHANGE**: We are using 'pytorch-cuda=11.8'
# to match your system's NVIDIA DRIVER version.
#
conda install -y \
  -c pyg \
  -c pytorch \
  -c nvidia \
  -c conda-forge \
  'pytorch-geometric' \
  'rdkit' \
  'pytorch' \
  'torchvision' \
  'pytorch-cuda=11.8'

echo "---"
echo "Environment '${ENV_PREFIX}' is set up successfully."
echo "---"