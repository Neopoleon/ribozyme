#!/usr/bin/env python3
"""
Extract entries with null RNA types from fold1_labels.json
and create fold4.json with their file paths.

This script processes the labeled RNA data and identifies entries
where the RNA type could not be determined (null values), creating
a new fold split for additional scraping/processing.
"""

import json
from pathlib import Path


def extract_null_rna_type_paths():
    """Extract file paths from entries with null rna_type."""

    # Define paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "results" / "scrap_runs" / "fold1_labels.json"
    output_file = base_dir / "data" / "splits" / "fold4.json"

    print(f"Reading from: {input_file}")

    # Read the input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    # Extract file paths where rna_type is null
    null_rna_paths = [
        entry['file_path']
        for entry in data
        if entry.get('rna_type') is None
    ]

    print(f"Entries with null rna_type: {len(null_rna_paths)}")

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(null_rna_paths, f, indent=2)

    print(f"Saved to: {output_file}")
    print(f"\nFirst few entries:")
    for i, path in enumerate(null_rna_paths[:5], 1):
        print(f"  {i}. {path}")


if __name__ == "__main__":
    extract_null_rna_type_paths()
