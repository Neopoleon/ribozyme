#!/usr/bin/env python3
"""
Fetch RNA type labels from bprna.cgrb.oregonstate.edu with concurrent requests.

Usage:
    python scripts/fetch_rna_labels.py --input-dir data/unzipped/bpRNA_1m_90_bpseqFiles \
                                       --output results/rna_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Optional

import requests
from tqdm import tqdm


@dataclass
class RNALabel:
    """RNA label data."""
    bprna_id: str
    file_path: str
    rna_type: Optional[str]
    error: Optional[str] = None


class RateLimiter:
    """Thread-safe rate limiter."""

    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self.lock = Lock()
        self.last_request = 0.0

    def acquire(self):
        """Wait if needed to maintain rate limit."""
        with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request = time.time()


def extract_bprna_id(file_path: Path) -> str:
    """Extract bpRNA ID from filename."""
    return 'bpRNA_' + file_path.name.split('bpRNA_')[-1].split('.bpseq')[0]


def fetch_rna_type(file_path: Path, rate_limiter: RateLimiter, timeout: float = 30.0) -> RNALabel:
    """Fetch RNA type for a single file."""
    bprna_id = extract_bprna_id(file_path)

    try:
        rate_limiter.acquire()

        response = requests.get(
            'https://bprna.cgrb.oregonstate.edu/search.php',
            params={'query': bprna_id},
            headers={'User-Agent': 'RNA-Research-Bot/1.0'},
            timeout=timeout
        )
        response.raise_for_status()

        if 'RNA Type:</b>' not in response.text:
            return RNALabel(bprna_id, str(file_path), None, 'RNA Type field not found')

        rna_type = response.text.split('RNA Type:</b>')[1].split('<br>')[0].strip()
        return RNALabel(bprna_id, str(file_path), rna_type)

    except requests.Timeout:
        return RNALabel(bprna_id, str(file_path), None, 'Timeout')
    except requests.RequestException as e:
        return RNALabel(bprna_id, str(file_path), None, str(e))
    except Exception as e:
        return RNALabel(bprna_id, str(file_path), None, f'Error: {str(e)}')


def load_checkpoint(output_path: Path) -> set[str]:
    """Load completed IDs from existing JSON output file."""
    if not output_path.exists():
        return set()
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
            return {item['bprna_id'] for item in data}
    except Exception:
        return set()


def append_result(result: RNALabel, output_path: Path):
    """Append a single result to JSON output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    if output_path.exists():
        with open(output_path, 'r') as f:
            data = json.load(f)

    data.append(asdict(result))

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Fetch RNA type labels')
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory with .bpseq files')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON file')
    parser.add_argument('--max-workers', type=int, default=5, help='Concurrent workers (default: 5)')
    parser.add_argument('--rate', type=float, default=2.0, help='Requests per second (default: 2.0)')
    parser.add_argument('--timeout', type=float, default=30.0, help='Request timeout (default: 30.0)')

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f'Error: Directory not found: {args.input_dir}', file=sys.stderr)
        sys.exit(1)

    if args.output.suffix != '.json':
        print('Error: Output must be a .json file', file=sys.stderr)
        sys.exit(1)

    # Load checkpoint to skip completed files
    completed = load_checkpoint(args.output)
    all_files = list(args.input_dir.glob('*.bpseq'))

    if not all_files:
        print(f'Error: No .bpseq files found in {args.input_dir}', file=sys.stderr)
        sys.exit(1)

    # Filter out already completed files
    files = [f for f in all_files if extract_bprna_id(f) not in completed]

    print(f'Total files: {len(all_files)}')
    if completed:
        print(f'Already completed: {len(completed)} (resuming)')
    print(f'To process: {len(files)}')
    print(f'Workers: {args.max_workers}, Rate: {args.rate} req/s')

    if not files:
        print('All files already processed!')
        return

    # PARALLEL FETCHING with progress tracking
    rate_limiter = RateLimiter(args.rate)
    write_lock = Lock()  # Protect file writes
    stats = {'successful': 0, 'failed': 0, 'rna_types': {}}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(fetch_rna_type, f, rate_limiter, args.timeout): f
            for f in files
        }

        for future in tqdm(as_completed(futures), total=len(files), desc='Fetching'):
            result = future.result()

            # Save immediately (progress tracking)
            with write_lock:
                append_result(result, args.output)

            # Update stats
            if result.rna_type:
                stats['successful'] += 1
                stats['rna_types'][result.rna_type] = stats['rna_types'].get(result.rna_type, 0) + 1
            else:
                stats['failed'] += 1

    # Print summary
    total = stats['successful'] + stats['failed']
    print(f'\nCompleted: {total} files')
    print(f'Successful: {stats["successful"]} ({100*stats["successful"]/total:.1f}%)')
    print(f'Failed: {stats["failed"]} ({100*stats["failed"]/total:.1f}%)')
    print(f'Saved to: {args.output}')

    if stats['rna_types']:
        print('\nTop RNA types:')
        for rna_type, count in sorted(stats['rna_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f'  {rna_type}: {count}')


if __name__ == '__main__':
    main()