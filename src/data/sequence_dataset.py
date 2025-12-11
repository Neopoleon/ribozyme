"""Sequence-only dataset for RNA classification baselines.

This module provides the standard RNA sequence dataset with file I/O per sample.
For better performance, see cached_sequence_dataset.py which pre-loads all data.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .label_encoder import LabelEncoder
from .parser import extract_rfid, get_meta_type, parse_st_file

# RNA nucleotide vocabulary
NUCLEOTIDE_VOCAB: list[str] = ['A', 'U', 'G', 'C', 'N']
NUCLEOTIDE_TO_IDX: dict[str, int] = {char: idx for idx, char in enumerate(NUCLEOTIDE_VOCAB)}
PAD_TOKEN_ID: int = len(NUCLEOTIDE_VOCAB)
VOCAB_SIZE: int = len(NUCLEOTIDE_VOCAB) + 1  # +1 for padding token


class RNASequenceDataset(Dataset):
    """RNA sequence dataset with file I/O per sample.

    Loads sequences from disk on-demand during training. Memory efficient but slower
    than the cached version. Consider using CachedRNASequenceDataset for better
    performance if you have sufficient RAM.

    Args:
        fold_labels_path: Path to JSON file containing fold labels and sample metadata
        rfam_types_path: Path to pickle file with Rfam type classifications
        st_files_dir: Directory containing .st structure files
        label_encoder: Optional pre-configured label encoder; creates new one if None

    Returns:
        Each item is a dictionary with fields:
            - input_ids: LongTensor of nucleotide token ids [seq_len]
            - label: LongTensor scalar with encoded meta-type
            - length: Sequence length (int)
            - metadata: dict with identifiers for debugging
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

        # Build index of valid samples upfront (avoids recomputation in __getitem__)
        self.samples: list[dict[str, Any]] = []
        for entry in self.fold_labels:
            bprna_id = entry['bprna_id']
            reference_name = entry['reference_name']
            rfid = extract_rfid(reference_name)
            meta_type = get_meta_type(rfid, self.rfam_types)

            # Skip entries without valid meta type
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
            msg = "No valid RNA entries found in dataset"
            raise RuntimeError(msg)

        print(f"[RNASequenceDataset] Indexed {len(self.samples)} valid sequences")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and tokenize a sequence from disk.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, label, length, and metadata
        """
        sample = self.samples[idx]
        st_file = self.st_files_dir / f"{sample['bprna_id']}.st"
        rna_data = parse_st_file(str(st_file))
        sequence = rna_data['sequence']

        # Convert sequence to token IDs (unknown nucleotides → 'N')
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
        """Compute class distribution for weighted loss calculation.

        Returns:
            Tensor of shape [num_classes] with count for each class
        """
        counts = torch.zeros(self.label_encoder.num_classes, dtype=torch.long)
        for sample in self.samples:
            counts[sample['label']] += 1
        return counts


def sequence_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Collate function for batching variable-length RNA sequences.

    Pads sequences to the maximum length in the batch and creates attention masks.

    Args:
        batch: List of samples from RNASequenceDataset

    Returns:
        Dictionary containing:
            - input_ids: [batch_size, max_len] padded token IDs
            - attention_mask: [batch_size, max_len] bool mask (True for valid tokens)
            - labels: [batch_size] class labels

    Raises:
        ValueError: If batch is empty
    """
    if not batch:
        msg = "Empty batch passed to sequence_collate_fn"
        raise ValueError(msg)

    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)

    # Initialize padded tensors
    input_ids = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    # Fill in actual values
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
