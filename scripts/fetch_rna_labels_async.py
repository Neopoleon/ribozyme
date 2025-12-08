#!/usr/bin/env python3
"""
Async version using aiohttp for much faster fetching.

Usage with directory:
    python scripts/fetch_rna_labels_async.py \
        --input-dir data/unzipped/bpRNA_1m_90_bpseqFiles \
        --output results/rna_labels.json

Usage with JSON file list:
    python scripts/fetch_rna_labels_async.py \
        --input-json data/splits/fold1.json \
        --output results/fold1_labels.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiohttp
import tqdm
import tqdm.asyncio


@dataclass
class RNALabel:
    """RNA label data."""
    bprna_id: str
    file_path: str
    rna_type: Optional[str]
    reference_name: Optional[str] = None
    error: Optional[str] = None


def extract_bprna_id(file_path: Path) -> str:
    """Extract bpRNA ID from filename."""
    return 'bpRNA_' + file_path.name.split('bpRNA_')[-1].split('.bpseq')[0]


async def fetch_rna_type(session: aiohttp.ClientSession, file_path: Path,
                         semaphore: asyncio.Semaphore, timeout: float = 20.0) -> RNALabel:
    """Fetch RNA type and reference name for a single file asynchronously."""
    bprna_id = extract_bprna_id(file_path)
    url = f'https://bprna.cgrb.oregonstate.edu/search.php?query={bprna_id}'

    async with semaphore:  # Limit concurrent requests
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                text = await response.text()

                # Extract RNA Type
                rna_type = None
                if 'RNA Type:</b>' in text:
                    rna_type = text.split('RNA Type:</b>')[1].split('<br>')[0].strip()

                # Extract Reference Name
                reference_name = None
                if 'Reference Name:</b>' in text:
                    reference_name = text.split('Reference Name:</b>')[1].split('<br>')[0].strip()

                # At least one field must be present, otherwise it's an error
                if not rna_type and not reference_name:
                    # Error case: neither field found
                    return RNALabel(bprna_id, str(file_path), None, None, 'RNA Type and Reference Name not found')

                # Success case: at least one field found, no error
                return RNALabel(bprna_id, str(file_path), rna_type, reference_name, None)
        except asyncio.TimeoutError:
            # Error case: timeout
            return RNALabel(bprna_id, str(file_path), None, None, 'Timeout')
        except Exception as e:
            # Error case: other failure
            return RNALabel(bprna_id, str(file_path), None, None, f'Failed: {e}')


def load_checkpoint(output_path: Path) -> dict[str, RNALabel]:
    """Load existing labels from JSON file."""
    if not output_path.exists():
        return {}

    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        results = {}
        for item in data:
            results[item['bprna_id']] = RNALabel(
                bprna_id=item['bprna_id'],
                file_path=item.get('file_path', ''),
                rna_type=item.get('rna_type'),
                reference_name=item.get('reference_name'),
                error=item.get('error'),
            )
        return results
    except Exception:
        return {}


def save_results(existing: dict[str, RNALabel], new_results: list[RNALabel],
                 output_path: Path):
    """Merge and save results."""
    merged = dict(existing)
    for r in new_results:
        merged[r.bprna_id] = r

    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = [asdict(merged[k]) for k in sorted(merged.keys())]

    with open(output_path, 'w') as f:
        json.dump(items, f, indent=2)


async def fetch_all(files: list[Path], max_concurrent: int, timeout: float,
                    existing: dict[str, RNALabel], output_path: Path):
    """Fetch all files with limited concurrency."""
    import time

    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()  # Prevent race conditions on file writes
    results = []

    start_time = time.time()

    # Use a single session for connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_rna_type(session, f, semaphore, timeout) for f in files]

        # Process with progress bar and incremental saves
        write_counter = 0
        WRITE_EVERY = 20  # Write every 20 results

        # Configure tqdm to update every 5 seconds with time estimates
        pbar = tqdm.tqdm(total=len(files), desc='Fetching',
                        mininterval=5.0,  # Update display every 5 seconds
                        unit='files')

        for coro in asyncio.as_completed(tasks):
            result = await coro

            # Use async lock to prevent race conditions
            async with write_lock:
                results.append(result)
                write_counter += 1

                # Update progress bar
                completed = len(results)
                pbar.n = completed
                elapsed = time.time() - start_time
                avg_time_per_file = elapsed / completed
                remaining = len(files) - completed
                eta_seconds = avg_time_per_file * remaining

                # Format ETA
                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                pbar.set_postfix({
                    'avg': f'{avg_time_per_file:.2f}s/file',
                    'ETA': f'{hours}h {minutes}m'
                })
                pbar.refresh()

                if write_counter >= WRITE_EVERY:
                    # Run sync I/O in thread pool to not block event loop
                    await asyncio.to_thread(save_results, existing, results, output_path)
                    write_counter = 0

        pbar.close()

    # Final save
    await asyncio.to_thread(save_results, existing, results, output_path)
    return results


async def main_async(args):
    """Main async function."""
    # Load files from either directory or JSON file list
    if args.input_json:
        # Load file paths from JSON
        if not args.input_json.exists():
            print(f'Error: JSON file not found: {args.input_json}', file=sys.stderr)
            sys.exit(1)

        with open(args.input_json, 'r') as f:
            file_paths = json.load(f)

        all_files = [Path(p) for p in file_paths]
        print(f'Loaded {len(all_files):,} file paths from {args.input_json}')
    else:
        # Load files from directory
        if not args.input_dir.exists():
            print(f'Error: Directory not found: {args.input_dir}', file=sys.stderr)
            sys.exit(1)

        all_files = list(args.input_dir.glob('*.bpseq'))
        if not all_files:
            print(f'Error: No .bpseq files found in {args.input_dir}', file=sys.stderr)
            sys.exit(1)

    # Load existing results
    existing = load_checkpoint(args.output)
    completed_ids = set(existing.keys())
    files = [f for f in all_files if extract_bprna_id(f) not in completed_ids]

    print(f'Total files: {len(all_files):,}')
    if completed_ids:
        print(f'Already completed: {len(completed_ids):,} (resuming)')
    print(f'To process: {len(files):,}')
    print(f'Max concurrent: {args.max_concurrent}, Timeout: {args.timeout}s')

    if not files:
        print('All files already processed!')
        return

    # Fetch all
    results = await fetch_all(files, args.max_concurrent, args.timeout, existing, args.output)

    # Print summary
    successes = sum(1 for r in results if r.rna_type)
    failures = len(results) - successes

    print(f'\nCompleted: {len(results):,} files')
    print(f'Successful: {successes:,} ({100*successes/len(results):.1f}%)')
    print(f'Failed: {failures:,}')
    print(f'Saved to: {args.output}')

    if successes > 0:
        rna_types = {}
        for r in results:
            if r.rna_type:
                rna_types[r.rna_type] = rna_types.get(r.rna_type, 0) + 1

        print('\nTop RNA types:')
        for rna_type, count in sorted(rna_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f'  {rna_type}: {count}')


def main():
    parser = argparse.ArgumentParser(description='Async RNA label fetching')

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', type=Path,
                             help='Directory containing .bpseq files')
    input_group.add_argument('--input-json', type=Path,
                             help='JSON file with list of file paths')

    parser.add_argument('--output', type=Path, required=True,
                        help='Output JSON file for results')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='Max concurrent requests (default: 10)')
    parser.add_argument('--timeout', type=float, default=20.0,
                        help='Request timeout in seconds (default: 20.0)')

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()