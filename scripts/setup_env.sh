#!/bin/bash
set -e

# pick conda/mamba
if command -v mamba &>/dev/null; then
  CONDA_CMD=mamba
else
  CONDA_CMD=conda
fi

$CONDA_CMD env create -f environment.yml

# activate
eval "$(conda shell.bash hook)"
conda activate ribozyme

pip install -e .

# quick checks
python - << 'PY'
for m in ["numpy", "pandas", "Bio", "sklearn", "pydantic"]:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except ImportError:
        print(f"[FAIL] {m}")
try:
    import tqdm  # noqa: F401
    print("[OK] tqdm")
except ImportError:
    print("[FAIL] tqdm")
PY

if which cd-hit-est >/dev/null 2>&1; then
  echo "[OK] cd-hit-est"
else
  echo "[FAIL] cd-hit-est"
fi

echo "Done. run:  conda activate ribozyme or mamba activate ribozyme"
