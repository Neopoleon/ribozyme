# Split and Fetch bpRNA Labels

This guide explains how to split RFAM sequence files into 3 folds and fetch RNA type and reference name labels from bpRNA in parallel across multiple computers.

## Overview

1. **Split**: Divide 22,521 RFAM files into 3 equal folds (7,507 files each)
2. **Fetch**: Run label fetching on 3 different computers overnight to parallelize the slow bpRNA server requests

## Step 1: Split RFAM Files into 3 Folds

Run this command **once** to generate the fold JSON files:

```bash
python scripts/split_rfam_into_folds.py \
    --input-dir data/unzipped/bpRNA_1m_90_bpseqFiles \
    --output-dir data/splits
```

**Output:**
- `data/splits/fold1.json` - 7,507 files
- `data/splits/fold2.json` - 7,507 files
- `data/splits/fold3.json` - 7,507 files

**Note:** The split is deterministic (always produces the same output) and uses relative paths from the `ribozyme` directory.

## Step 2: Fetch Labels on 3 Computers

Copy the `data/splits/` folder and the bpRNA dataset to each computer, then run one command per computer:

### Computer 1 - Process Fold 1

```bash
python scripts/fetch_rna_labels_async.py \
    --input-json data/splits/fold1.json \
    --output results/fold1_labels.json \
    --max-concurrent 5 \
    --timeout 20.0
```

### Computer 2 - Process Fold 2

```bash
python scripts/fetch_rna_labels_async.py \
    --input-json data/splits/fold2.json \
    --output results/fold2_labels.json \
    --max-concurrent 5 \
    --timeout 20.0
```

### Computer 3 - Process Fold 3

```bash
python scripts/fetch_rna_labels_async.py \
    --input-json data/splits/fold3.json \
    --output results/fold3_labels.json \
    --max-concurrent 5 \
    --timeout 20.0
```

## Output Format

Each output JSON file contains an array of RNA labels:

```json
[
  {
    "bprna_id": "bpRNA_RFAM_10",
    "file_path": "data/unzipped/bpRNA_1m_90_bpseqFiles/bpRNA_RFAM_10.bpseq",
    "rna_type": "5S",
    "reference_name": "RF00001_AF033641.1_428-544",
    "error": null
  }
]
```

**Success case:** At least one of `rna_type` or `reference_name` is present, `error` is `null`

**Error case:** Both `rna_type` and `reference_name` are `null`, `error` contains the error message

## Features

- **Async fetching**: Uses `aiohttp` for concurrent requests
- **Checkpointing**: Saves progress every 20 results
- **Resumable**: Can restart if interrupted - already completed IDs are skipped
- **Rate limiting**: `--max-concurrent` limits simultaneous requests to avoid overwhelming the server

## Test Run

Before running the full overnight job, test with a small subset:

```bash
# Create test subset (5 files)
python -c "import json; data = json.load(open('data/splits/fold1.json')); json.dump(data[:5], open('data/splits/fold1_test.json', 'w'))"

# Test fetch
python scripts/fetch_rna_labels_async.py \
    --input-json data/splits/fold1_test.json \
    --output results/fold1_test_labels.json \
    --max-concurrent 5 \
    --timeout 20.0
```

## Merging Results (After All Folds Complete)

Once all 3 computers finish, merge the results:

```bash
python -c "
import json
fold1 = json.load(open('results/fold1_labels.json'))
fold2 = json.load(open('results/fold2_labels.json'))
fold3 = json.load(open('results/fold3_labels.json'))
merged = fold1 + fold2 + fold3
json.dump(merged, open('results/all_rfam_labels.json', 'w'), indent=2)
print(f'Merged {len(merged)} total labels')
"
```