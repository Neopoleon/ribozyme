#!/usr/bin/env python3
"""
Fetch RNA type and reference name from bpRNA database asynchronously.

Usage:
    python scripts/fetch_rna_labels_async.py --input-dir data/unzipped/bpRNA_1m_90_bpseqFiles --output results/rna_labels.json
    python scripts/fetch_rna_labels_async.py --input-json data/splits/fold1.json --output results/scrap_runs/fold1_labels.json
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


@dataclass
class RNALabel:
    bprna_id: str
    file_path: str
    rna_type: Optional[str]
    reference_name: Optional[str] = None
    error: Optional[str] = None
    request_time_seconds: Optional[float] = None


def extract_bprna_id(file_path: Path) -> str:
    return 'bpRNA_' + file_path.name.split('bpRNA_')[-1].split('.bpseq')[0]


async def fetch_rna_type(session: aiohttp.ClientSession, file_path: Path,
                         semaphore: asyncio.Semaphore, timeout: float = 20.0) -> RNALabel:
    import time

    bprna_id = extract_bprna_id(file_path)
    url = f'https://bprna.cgrb.oregonstate.edu/search.php?query={bprna_id}'

    async with semaphore:
        request_start = time.time()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                text = await response.text()

                rna_type = None
                if 'RNA Type:</b>' in text:
                    rna_type = text.split('RNA Type:</b>')[1].split('<br>')[0].strip()

                reference_name = None
                if 'Reference Name:</b>' in text:
                    reference_name = text.split('Reference Name:</b>')[1].split('<br>')[0].strip()

                request_time = time.time() - request_start

                if not rna_type and not reference_name:
                    return RNALabel(bprna_id, str(file_path), None, None,
                                    'RNA Type and Reference Name not found', request_time)

                return RNALabel(bprna_id, str(file_path), rna_type, reference_name, None, request_time)
        except asyncio.TimeoutError:
            request_time = time.time() - request_start
            return RNALabel(bprna_id, str(file_path), None, None, 'Timeout', request_time)
        except Exception as e:
            request_time = time.time() - request_start
            return RNALabel(bprna_id, str(file_path), None, None, f'Failed: {e}', request_time)


def load_checkpoint(output_path: Path) -> dict[str, RNALabel]:
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
                request_time_seconds=item.get('request_time_seconds'),
            )
        return results
    except Exception:
        return {}


def save_results(existing: dict[str, RNALabel], new_results: list[RNALabel], output_path: Path):
    merged = dict(existing)
    for r in new_results:
        merged[r.bprna_id] = r

    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = [asdict(merged[k]) for k in sorted(merged.keys())]

    with open(output_path, 'w') as f:
        json.dump(items, f, indent=2)


def log_null_values(result: RNALabel, log_file: Path):
    if result.rna_type is None or result.reference_name is None:
        import datetime
        timestamp = datetime.datetime.now().isoformat()

        log_entry = {
            'timestamp': timestamp,
            'bprna_id': result.bprna_id,
            'file_path': result.file_path,
            'rna_type': result.rna_type,
            'reference_name': result.reference_name,
            'error': result.error,
            'request_time_seconds': result.request_time_seconds
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


async def fetch_all(files: list[Path], max_concurrent: int, timeout: float,
                    existing: dict[str, RNALabel], output_path: Path, log_file: Path):
    import time

    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()
    results = []
    null_rna_type_count = 0
    null_reference_name_count = 0
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_rna_type(session, f, semaphore, timeout) for f in files]
        write_counter = 0
        WRITE_EVERY = 1

        pbar = tqdm.tqdm(total=len(files), desc='Fetching', mininterval=5.0, unit='files')

        for coro in asyncio.as_completed(tasks):
            result = await coro

            async with write_lock:
                results.append(result)
                write_counter += 1

                if result.rna_type is None:
                    null_rna_type_count += 1
                if result.reference_name is None:
                    null_reference_name_count += 1

                await asyncio.to_thread(log_null_values, result, log_file)

                completed = len(results)
                pbar.n = completed
                elapsed = time.time() - start_time
                avg_time_per_file = elapsed / completed
                remaining = len(files) - completed
                eta_seconds = avg_time_per_file * remaining

                hours = int(eta_seconds // 3600)
                minutes = int((eta_seconds % 3600) // 60)
                pbar.set_postfix({
                    'avg': f'{avg_time_per_file:.2f}s/file',
                    'ETA': f'{hours}h {minutes}m',
                    'null_rna': null_rna_type_count,
                    'null_ref': null_reference_name_count
                })
                pbar.refresh()

                if write_counter >= WRITE_EVERY:
                    await asyncio.to_thread(save_results, existing, results, output_path)
                    write_counter = 0

        pbar.close()

    await asyncio.to_thread(save_results, existing, results, output_path)

    print(f'\nNull field statistics:')
    print(f'  Null rna_type: {null_rna_type_count}')
    print(f'  Null reference_name: {null_reference_name_count}')

    return results


async def main_async(args):
    if args.input_json:
        if not args.input_json.exists():
            print(f'Error: JSON file not found: {args.input_json}', file=sys.stderr)
            sys.exit(1)

        with open(args.input_json, 'r') as f:
            file_paths = json.load(f)

        all_files = [Path(p) for p in file_paths]
        print(f'Loaded {len(all_files):,} file paths from {args.input_json}')
    else:
        if not args.input_dir.exists():
            print(f'Error: Directory not found: {args.input_dir}', file=sys.stderr)
            sys.exit(1)

        all_files = list(args.input_dir.glob('*.bpseq'))
        if not all_files:
            print(f'Error: No .bpseq files found in {args.input_dir}', file=sys.stderr)
            sys.exit(1)

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

    log_file = Path.cwd() / 'null_values.log'
    print(f'Logging null values to: {log_file}')

    results = await fetch_all(files, args.max_concurrent, args.timeout, existing, args.output, log_file)

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

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', type=Path, help='Directory containing .bpseq files')
    input_group.add_argument('--input-json', type=Path, help='JSON file with list of file paths')

    parser.add_argument('--output', type=Path, required=True, help='Output JSON file for results')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Max concurrent requests (default: 10)')
    parser.add_argument('--timeout', type=float, default=20.0, help='Request timeout in seconds (default: 20.0)')

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
