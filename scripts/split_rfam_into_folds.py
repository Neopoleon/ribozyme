#!/usr/bin/env python3
"""
Split all RFAM sequence files into 3 equal folds for parallel processing.

Usage:
    python scripts/split_rfam_into_folds.py \
        --input-dir data/unzipped/bpRNA_1m_90_bpseqFiles \
        --output-dir data/splits
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Split RFAM files into 3 folds')
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory containing .bpseq files')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory to save fold JSON files')

    args = parser.parse_args()

    # Get current working directory to compute relative paths
    cwd = Path.cwd()

    # Find all RFAM files and convert to relative paths
    rfam_files = sorted([
        str(f.resolve().relative_to(cwd)) for f in args.input_dir.glob('*RFAM*.bpseq')
    ])

    if not rfam_files:
        print(f'Error: No RFAM .bpseq files found in {args.input_dir}')
        return

    total = len(rfam_files)
    print(f'Found {total:,} RFAM sequence files')

    # Split into 3 roughly equal folds
    fold_size = total // 3
    remainder = total % 3

    folds = []
    start = 0
    for i in range(3):
        # Distribute remainder across first folds
        size = fold_size + (1 if i < remainder else 0)
        end = start + size
        folds.append(rfam_files[start:end])
        start = end

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save each fold
    for i, fold in enumerate(folds, 1):
        output_file = args.output_dir / f'fold{i}.json'
        with open(output_file, 'w') as f:
            json.dump(fold, f, indent=2)
        print(f'Fold {i}: {len(fold):,} files -> {output_file}')

    print(f'\nTotal files split: {sum(len(f) for f in folds):,}')
    print('Ready for parallel processing!')


if __name__ == '__main__':
    main()