#!/bin/bash
set -euo pipefail

# pick conda/mamba
if command -v mamba &>/dev/null; then
  CONDA_CMD=mamba
else
  CONDA_CMD=conda
fi

ENV_NAME="ribozyme"

# remove existing env if present (ignore failures)
if $CONDA_CMD env list | grep -q "^${ENV_NAME} "; then
  echo "Removing existing env '${ENV_NAME}'"
  $CONDA_CMD env remove -n "$ENV_NAME" || true
fi

echo "Creating env from environment.yml using $CONDA_CMD"
$CONDA_CMD env create -f environment.yml

# activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing package in editable mode"
pip install -e .

echo "Done. Run 'conda activate ${ENV_NAME}' (or 'mamba activate ${ENV_NAME}') in new shells."
