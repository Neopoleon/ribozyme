#!/usr/bin/env bash
set -euo pipefail

# Runs train_hydra over a small matrix of architectures and config files.
# Uses test=false so all artifacts are written to results/runs/<run_name>.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARCHES=(gin) # gat gcn
CONFIGS=(config_1 config_2 config_3 config_4) # config_0 

SUCCESS=0
FAILURES=()

for cfg in "${CONFIGS[@]}"; do
  for arch in "${ARCHES[@]}"; do
    echo "---- Running train_hydra --config-name ${cfg} model.architecture=${arch} test=false ----"
    if python -m train_hydra --config-name "${cfg}" model.architecture="${arch}" test=false; then
      ((SUCCESS += 1))  # use += to keep exit status zero under `set -e`
    else
      FAILURES+=("${cfg}:${arch}")
    fi
  done
done

TOTAL=$(( ${#CONFIGS[@]} * ${#ARCHES[@]} ))
FAIL_COUNT=${#FAILURES[@]}

echo
echo "Summary: succeeded=${SUCCESS} failed=${FAIL_COUNT} total=${TOTAL}"
if (( FAIL_COUNT > 0 )); then
  echo "Failures:"
  for entry in "${FAILURES[@]}"; do
    echo "  - ${entry}"
  done
fi
