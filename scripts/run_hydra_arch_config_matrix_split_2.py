#!/usr/bin/env python
"""Run the remaining Hydra architecture/config matrix in Python."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"

# Architectures to evaluate.
ARCHES = ["gcn", "gat", "gin"]

# Remaining boolean permutations (11 total) not covered by config_0..config_4.
REMAINING_KEYS = [
    "True_True_False_False",
    "True_False_True_False",
    "True_False_False_True",
    "True_False_False_False",
    "False_True_True_False",
    "False_True_False_True",
    "False_True_False_False",
    "False_False_True_True",
    "False_False_True_False",
    "False_False_False_True",
    "False_False_False_False",
]


def has_existing_run(arch: str, key: str) -> bool:
    pattern = f"{arch}_{key}_*"
    return any(RUNS_DIR.glob(pattern))


def run_combo(arch: str, key: str) -> bool | None:
    if has_existing_run(arch, key):
        print(f"Skipping {arch}:{key} (found existing run in {RUNS_DIR})")
        return None

    nuc_lbl, struct_lbl, pseudo_lbl, pos_lbl = key.split("_")
    flags = (nuc_lbl == "True", struct_lbl == "True", pseudo_lbl == "True", pos_lbl == "True")

    args = [
        sys.executable,
        "-m",
        "train_hydra",
        "--config-name",
        "config_split_2",
        f"model.architecture={arch}",
        f"features.use_nucleotide={str(flags[0]).lower()}",
        f"features.use_structure_annotation={str(flags[1]).lower()}",
        f"features.use_pseudoknot={str(flags[2]).lower()}",
        f"features.use_position_encoding={str(flags[3]).lower()}",
        "test=false",
    ]

    print(f"---- Running {' '.join(args)} ----")
    result = subprocess.run(args, cwd=REPO_ROOT)
    return result.returncode == 0


def main() -> int:
    success = 0
    skipped = 0
    failures: list[str] = []

    for arch in ARCHES:
        for key in REMAINING_KEYS:
            ran = run_combo(arch, key)
            if ran is None:
                skipped += 1
            elif ran:
                success += 1
            else:
                failures.append(f"{arch}:{key}")

    expected_total = len(ARCHES) * len(REMAINING_KEYS)

    fail_count = len(failures)
    print()
    print(
        f"Summary: succeeded={success} failed={fail_count} skipped={skipped} "
        f"remaining_expected={expected_total}"
    )
    if failures:
        print("Failures:")
        for entry in failures:
            print(f"  - {entry}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
