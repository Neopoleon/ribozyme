"""PyTorch Geometric Dataset for RNA structures"""

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data, Dataset

from .graph_builder import rna_to_graph
from .label_encoder import LabelEncoder
from .parser import extract_rfid, get_meta_type, parse_st_file

if TYPE_CHECKING:
    from ..config import FeatureConfig


class RNADataset(Dataset):
    """
    PyTorch Geometric Dataset for RNA structures.

    Loads RNA structure files (.st format) and converts them to graph representations
    with RFAM meta-type labels for classification.
    """

    def __init__(
        self,
        root: str,
        fold_labels_path: str,
        rfam_types_path: str = 'rfam/rfam_types_full.pkl',
        st_files_dir: str = 'data/unzipped/bpRNA_1m_90_STAFILES',
        label_encoder: LabelEncoder | None = None,
        feature_config: 'FeatureConfig | None' = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Args:
            root: Root directory for dataset (for PyG caching)
            fold_labels_path: Path to fold labels JSON file
            rfam_types_path: Path to rfam_types_full.pkl
            st_files_dir: Directory containing .st structure files
            label_encoder: LabelEncoder instance. If None, creates default.
            feature_config: FeatureConfig instance. If None, uses all features.
            transform: Optional transform to apply to data objects
            pre_transform: Optional pre-transform
            pre_filter: Optional pre-filter
        """
        self.fold_labels_path = fold_labels_path
        self.rfam_types_path = rfam_types_path
        self.st_files_dir = Path(st_files_dir)
        self.feature_config = feature_config

        # Load metadata
        with open(fold_labels_path, 'r') as f:
            self.fold_labels = json.load(f)

        with open(rfam_types_path, 'rb') as f:
            self.rfam_types = pickle.load(f)

        # Initialize label encoder
        self.label_encoder = label_encoder or LabelEncoder()

        # Filter out entries with missing meta-types
        self.valid_indices = []
        for i, entry in enumerate(self.fold_labels):
            rfid = extract_rfid(entry['reference_name'])
            meta_type = get_meta_type(rfid, self.rfam_types)
            if meta_type is not None and meta_type in self.label_encoder.type_to_idx:
                self.valid_indices.append(i)

        print(f"Loaded {len(self.valid_indices)}/{len(self.fold_labels)} valid samples")

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> list[str]:
        """Required by PyG - list of raw files"""
        return []  # We handle file loading manually

    @property
    def processed_file_names(self) -> list[str]:
        """Required by PyG - list of processed files"""
        # We'll process on-the-fly for now (can add caching later)
        return []

    def download(self):
        """Required by PyG - download raw data (not needed for us)"""
        pass

    def process(self):
        """Required by PyG - process raw data (not needed for us)"""
        pass

    def len(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.valid_indices)

    def get(self, idx: int) -> Data:
        """
        Get a single graph data sample.

        Args:
            idx: Index of sample to retrieve

        Returns:
            PyG Data object with:
            - x: node features [num_nodes, feature_dim]
            - edge_index: edge indices [2, num_edges]
            - edge_attr: edge types [num_edges, 5] (5D one-hot)
            - y: label (integer)
            - metadata: dict with bprna_id, rfid, meta_type, etc.
        """
        # Get entry from valid indices
        entry_idx = self.valid_indices[idx]
        entry = self.fold_labels[entry_idx]

        # Extract metadata
        bprna_id = entry['bprna_id']
        reference_name = entry['reference_name']
        rfid = extract_rfid(reference_name)
        meta_type = get_meta_type(rfid, self.rfam_types)

        # Parse .st file
        st_file_path = self.st_files_dir / f"{bprna_id}.st"
        try:
            rna_data = parse_st_file(str(st_file_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse {st_file_path} for index {idx}: {e}"
            )

        # Convert to graph
        node_features, edge_index, edge_attr = rna_to_graph(
            sequence=rna_data['sequence'],
            dot_bracket=rna_data['dot_bracket'],
            structure=rna_data['structure'],
            pseudoknot=rna_data['pseudoknot'],
            feature_config=self.feature_config,
        )

        # Encode label
        assert meta_type is not None, f"meta_type should not be None for {bprna_id}"
        try:
            label = self.label_encoder.encode(meta_type)
        except KeyError:
            raise RuntimeError(
                f"Failed to encode meta_type '{meta_type}' for {bprna_id}"
            )

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
        )

        # Store metadata
        data.bprna_id = bprna_id
        data.rfid = rfid
        data.meta_type = meta_type
        data.sequence_length = len(rna_data['sequence'])

        return data

    def get_statistics(self) -> dict:
        """Compute dataset statistics"""
        label_counts = {}

        for i in range(len(self)):
            entry_idx = self.valid_indices[i]
            entry = self.fold_labels[entry_idx]
            rfid = extract_rfid(entry['reference_name'])
            meta_type = get_meta_type(rfid, self.rfam_types)

            if meta_type:
                label_counts[meta_type] = label_counts.get(meta_type, 0) + 1

        return {
            'num_samples': len(self),
            'num_classes': self.label_encoder.num_classes,
            'label_distribution': label_counts,
        }
