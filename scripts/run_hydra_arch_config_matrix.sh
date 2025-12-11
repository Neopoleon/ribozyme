#!/usr/bin/env bash
set -euo pipefail

# Runs train_hydra across all remaining feature flag permutations for each
# architecture, skipping the five conditions already covered by config_0..config_4:
#   True_True_True_True
#   False_True_True_True
#   True_False_True_True
#   True_True_False_True
#   True_True_True_False
# Uses test=false so artifacts land in results/runs/<run_name>.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARCHES=(gcn gat gin)
RUNS_DIR="${REPO_ROOT}/results/runs"

# Keys are "<use_nucleotide>_<use_structure_annotation>_<use_pseudoknot>_<use_position_encoding>"
# with capitalized booleans to match the experiment_name formatting.
declare -A SKIP_KEYS=(
  ["True_True_True_True"]=1
  ["False_True_True_True"]=1
  ["True_False_True_True"]=1
  ["True_True_False_True"]=1
  ["True_True_True_False"]=1
)

bool_label() {
  [[ "$1" == "true" ]] && echo "True" || echo "False"
}

SUCCESS=0
SKIPPED=0
FAILURES=()

for arch in "${ARCHES[@]}"; do
  for nuc in true false; do
    for struct in true false; do
      for pseudo in true false; do
        for pos in true false; do
          nuc_lbl=$(bool_label "${nuc}")
          struct_lbl=$(bool_label "${struct}")
          pseudo_lbl=$(bool_label "${pseudo}")
          pos_lbl=$(bool_label "${pos}")
          key="${nuc_lbl}_${struct_lbl}_${pseudo_lbl}_${pos_lbl}"

          if [[ -n "${SKIP_KEYS[${key}]:-}" ]]; then
            ((SKIPPED += 1))
            echo "Skipping ${arch}:${key} (already run via config_0..config_4)"
            continue
          fi

          run_prefix="${arch}_${key}_"
          if compgen -G "${RUNS_DIR}/${run_prefix}*" > /dev/null; then
            ((SKIPPED += 1))
            echo "Skipping ${arch}:${key} (found existing run in ${RUNS_DIR})"
            continue
          fi

          echo "---- Running train_hydra --config-name config_0 model.architecture=${arch} features.use_nucleotide=${nuc} features.use_structure_annotation=${struct} features.use_pseudoknot=${pseudo} features.use_position_encoding=${pos} test=false ----"
          if python -m train_hydra \
            --config-name config_0 \
            model.architecture="${arch}" \
            features.use_nucleotide="${nuc}" \
            features.use_structure_annotation="${struct}" \
            features.use_pseudoknot="${pseudo}" \
            features.use_position_encoding="${pos}" \
            test=false; then
            ((SUCCESS += 1)) # use += to keep exit status zero under `set -e`
          else
            FAILURES+=("${arch}:${key}")
          fi
        done
      done
    done
  done
done

TOTAL=$(( 16 * ${#ARCHES[@]} - 5 * ${#ARCHES[@]} )) # 16 combos per arch minus 5 skipped combos
FAIL_COUNT=${#FAILURES[@]}

echo
echo "Summary: succeeded=${SUCCESS} failed=${FAIL_COUNT} skipped=${SKIPPED} remaining_expected=${TOTAL}"
if (( FAIL_COUNT > 0 )); then
  echo "Failures:"
  for entry in "${FAILURES[@]}"; do
    echo "  - ${entry}"
  done
fi
