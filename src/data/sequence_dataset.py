"""Sequence-only dataset for RNA classification baselines."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .label_encoder import LabelEncoder
from .parser import extract_rfid, get_meta_type, parse_st_file

# Vocab for nucleotide tokens (Pad token appended implicitly)
NUCLEOTIDE_VOCAB = ['A', 'U', 'G', 'C', 'N']
NUCLEOTIDE_TO_IDX = {char: idx for idx, char in enumerate(NUCLEOTIDE_VOCAB)}
PAD_TOKEN_ID = len(NUCLEOTIDE_VOCAB)
VOCAB_SIZE = len(NUCLEOTIDE_VOCAB) + 1  # +1 for padding symbol


class RNASequenceDataset(Dataset):
    """
    Dataset that yields tokenized RNA sequences and integer labels.

    Each item is a dictionary with fields:
        - input_ids: LongTensor of nucleotide token ids
        - label: LongTensor scalar with encoded meta-type
        - length: Sequence length (int)
        - metadata: dict with identifiers, useful for debugging
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

        # Build index of valid samples upfront (saves recomputation later)
        self.samples: list[dict[str, Any]] = []
        for entry in self.fold_labels:
            bprna_id = entry['bprna_id']
            reference_name = entry['reference_name']
            rfid = extract_rfid(reference_name)
            meta_type = get_meta_type(rfid, self.rfam_types)
            if meta_type is None or meta_type not in self.label_encoder.type_to_idx:
                continue
            label = self.label_encoder.encode(meta_type)
            self.samples.append({
                'bprna_id': bprna_id,
                'rfid': rfid,
                'meta_type': meta_type,
                'label': label,
            })

        if not self.samples:
            raise RuntimeError("No valid RNA entries found for sequence dataset.")

        print(f"[RNASequenceDataset] Loaded {len(self.samples)} valid sequences")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        st_file = self.st_files_dir / f"{sample['bprna_id']}.st"
        rna_data = parse_st_file(str(st_file))
        sequence = rna_data['sequence']

        token_ids = torch.tensor(
            [NUCLEOTIDE_TO_IDX.get(nt, NUCLEOTIDE_TO_IDX['N']) for nt in sequence],
            dtype=torch.long,
        )

        return {
            'input_ids': token_ids,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'length': token_ids.size(0),
            'metadata': {
                'bprna_id': sample['bprna_id'],
                'rfid': sample['rfid'],
                'meta_type': sample['meta_type'],
            },
        }

    def label_counts(self) -> torch.Tensor:
        """Return counts per class to help derive class weights."""
        counts = torch.zeros(self.label_encoder.num_classes, dtype=torch.long)
        for sample in self.samples:
            counts[sample['label']] += 1
        return counts


def sequence_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """
    Collate function that pads variable-length sequences.

    Returns dict with:
        - input_ids: [B, T] padded with PAD token
        - attention_mask: [B, T] bool mask (True for valid tokens)
        - labels: [B] tensor of target classes
    """
    if not batch:
        raise ValueError("Empty batch passed to sequence_collate_fn")

    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = True
        labels[i] = item['label']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
