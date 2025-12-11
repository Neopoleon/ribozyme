"""Memory-cached sequence dataset for faster training."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .label_encoder import LabelEncoder
from .parser import extract_rfid, get_meta_type, parse_st_file

# Vocab for nucleotide tokens
NUCLEOTIDE_VOCAB = ['A', 'U', 'G', 'C', 'N']
NUCLEOTIDE_TO_IDX = {char: idx for idx, char in enumerate(NUCLEOTIDE_VOCAB)}
PAD_TOKEN_ID = len(NUCLEOTIDE_VOCAB)
VOCAB_SIZE = len(NUCLEOTIDE_VOCAB) + 1


class CachedRNASequenceDataset(Dataset):
    """
    Dataset that pre-loads all sequences into memory for faster access.
    Eliminates file I/O during training at the cost of higher memory usage.
    """

    def __init__(
        self,
        fold_labels_path: str,
        rfam_types_path: str = 'rfam/rfam_types_full.pkl',
        st_files_dir: str = 'data/unzipped/bpRNA_1m_90_STAFILES',
        label_encoder: LabelEncoder | None = None,
    ) -> None:
        self.fold_labels_path = Path(fold_labels_path)
        self.rfam_types_path = Path(rfam_types_path)
        self.st_files_dir = Path(st_files_dir)

        # Load metadata tables
        with open(self.fold_labels_path, 'r') as f:
            self.fold_labels = json.load(f)

        with open(self.rfam_types_path, 'rb') as f:
            self.rfam_types = pickle.load(f)

        self.label_encoder = label_encoder or LabelEncoder()

        # Pre-load all sequences into memory
        self.samples: list[dict[str, Any]] = []
        print(f"[CachedRNASequenceDataset] Pre-loading sequences into memory...")

        for i, entry in enumerate(self.fold_labels):
            if i % 1000 == 0 and i > 0:
                print(f"  Loaded {i}/{len(self.fold_labels)} sequences...")

            bprna_id = entry['bprna_id']
            reference_name = entry['reference_name']
            rfid = extract_rfid(reference_name)
            meta_type = get_meta_type(rfid, self.rfam_types)

            if meta_type is None or meta_type not in self.label_encoder.type_to_idx:
                continue

            label = self.label_encoder.encode(meta_type)

            # Load and tokenize sequence NOW instead of in __getitem__
            st_file = self.st_files_dir / f"{bprna_id}.st"
            try:
                rna_data = parse_st_file(str(st_file))
                sequence = rna_data['sequence']

                # Pre-tokenize
                token_ids = torch.tensor(
                    [NUCLEOTIDE_TO_IDX.get(nt, NUCLEOTIDE_TO_IDX['N']) for nt in sequence],
                    dtype=torch.long,
                )

                self.samples.append({
                    'token_ids': token_ids,
                    'label': label,
                    'metadata': {
                        'bprna_id': bprna_id,
                        'rfid': rfid,
                        'meta_type': meta_type,
                    },
                })
            except Exception as e:
                print(f"Warning: Failed to load {bprna_id}: {e}")
                continue

        if not self.samples:
            raise RuntimeError("No valid RNA entries found for sequence dataset.")

        print(f"[CachedRNASequenceDataset] Loaded {len(self.samples)} sequences into memory")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fast access - data already in memory."""
        sample = self.samples[idx]
        return {
            'input_ids': sample['token_ids'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'length': sample['token_ids'].size(0),
            'metadata': sample['metadata'],
        }

    def label_counts(self) -> torch.Tensor:
        """Return counts per class to help derive class weights."""
        counts = torch.zeros(self.label_encoder.num_classes, dtype=torch.long)
        for sample in self.samples:
            counts[sample['label']] += 1
        return counts
