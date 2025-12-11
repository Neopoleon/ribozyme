"""Convert RNA structure to graph representation for GNN"""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..config import FeatureConfig


# Character to index mappings
NUCLEOTIDE_CHARS = ['A', 'U', 'G', 'C', 'N']
STRUCTURE_CHARS = ['E', 'S', 'H', 'I', 'M', 'B', 'X']
BRACKET_PAIRS = {
    '(': (')', 1),  # canonical base pairs
    '[': (']', 2),  # pseudoknot bracket type 1
    '{': ('}', 3),  # pseudoknot bracket type 2
    '<': ('>', 4),  # pseudoknot bracket type 3
}


def parse_dot_bracket(dot_bracket: str) -> list[tuple[int, int, int]]:
    """
    Parse dot-bracket notation to extract base pair edges.

    Args:
        dot_bracket: Dot-bracket string like ".((...))."

    Returns:
        List of (i, j, edge_type) tuples where:
        - i, j are paired positions
        - edge_type is 1 for (), 2 for [], 3 for {}, 4 for <>
    """
    edges = []
    stacks = {bracket: [] for bracket in BRACKET_PAIRS.keys()}

    for i, char in enumerate(dot_bracket):
        if char in BRACKET_PAIRS:
            # Opening bracket - push to stack
            stacks[char].append(i)
        elif char in [')', ']', '}', '>']:
            # Closing bracket - pop from corresponding stack
            opening = {')': '(', ']': '[', '}': '{', '>': '<'}[char]
            if stacks[opening]:
                j = stacks[opening].pop()
                edge_type = BRACKET_PAIRS[opening][1]
                edges.append((j, i, edge_type))
        # '.' means unpaired, skip

    return edges


def encode_nucleotide(nuc: str) -> np.ndarray:
    """One-hot encode nucleotide character (5 dims)"""
    encoding = np.zeros(len(NUCLEOTIDE_CHARS), dtype=np.float32)
    if nuc in NUCLEOTIDE_CHARS:
        encoding[NUCLEOTIDE_CHARS.index(nuc)] = 1.0
    else:
        # Unknown nucleotide maps to 'N'
        encoding[NUCLEOTIDE_CHARS.index('N')] = 1.0
    return encoding


def encode_structure(struct: str) -> np.ndarray:
    """One-hot encode structural annotation (7 dims)"""
    encoding = np.zeros(len(STRUCTURE_CHARS), dtype=np.float32)
    if struct in STRUCTURE_CHARS:
        encoding[STRUCTURE_CHARS.index(struct)] = 1.0
    else:
        # Unknown structure maps to 'X'
        encoding[STRUCTURE_CHARS.index('X')] = 1.0
    return encoding


def encode_pseudoknot(pk: str) -> float:
    """Encode pseudoknot indicator (1 dim, binary)"""
    return 1.0 if pk == 'K' else 0.0


def encode_edge_type_onehot(edge_type: int) -> np.ndarray:
    """
    One-hot encode edge type (5 dims).

    Edge types:
    - 0: Backbone edge
    - 1: Canonical base pair ()
    - 2: Pseudoknot bracket []
    - 3: Pseudoknot bracket {}
    - 4: Pseudoknot bracket <>
    """
    encoding = np.zeros(5, dtype=np.float32)
    encoding[edge_type] = 1.0
    return encoding


def rna_to_graph(
    sequence: str,
    dot_bracket: str,
    structure: str,
    pseudoknot: str,
    feature_config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert RNA structure information to graph representation.

    Args:
        sequence: Nucleotide sequence (A/U/G/C/N)
        dot_bracket: Dot-bracket notation for secondary structure
        structure: Structural annotation (E/S/H/I/M/B/X)
        pseudoknot: Pseudoknot annotation (N/K)
        feature_config: FeatureConfig instance to control which features to include.
                       If None, uses all features (backward compatible).

    Returns:
        Tuple of (node_features, edge_index, edge_attr):
        - node_features: [num_nodes, feature_dim] tensor (feature_dim depends on config)
        - edge_index: [2, num_edges] tensor
        - edge_attr: [num_edges, 5] tensor (5D one-hot encoded edge types)
    """
    seq_len = len(sequence)

    # Build node features
    node_features = []
    for i in range(seq_len):
        features = []

        # Add features based on configuration
        if feature_config is None or feature_config.use_nucleotide:
            features.append(encode_nucleotide(sequence[i]))  # 5 dims

        if feature_config is None or feature_config.use_structure_annotation:
            features.append(encode_structure(structure[i]))  # 7 dims

        if feature_config is None or feature_config.use_pseudoknot:
            features.append([encode_pseudoknot(pseudoknot[i])])  # 1 dim

        if feature_config is None or feature_config.use_position_encoding:
            position = i / max(seq_len - 1, 1)  # normalized [0, 1]
            features.append([position])  # 1 dim

        node_feat = np.concatenate(features)
        node_features.append(node_feat)

    node_features = torch.tensor(np.array(node_features), dtype=torch.float32)

    # Build edges
    edge_list = []
    edge_attrs = []

    # 1. Backbone edges (sequential connections)
    for i in range(seq_len - 1):
        # Undirected edges: add both directions
        edge_list.append([i, i + 1])
        edge_attrs.append(encode_edge_type_onehot(0))  # 5D one-hot

        edge_list.append([i + 1, i])
        edge_attrs.append(encode_edge_type_onehot(0))  # 5D one-hot

    # 2. Base-pair edges from dot-bracket notation (skip if only_backbone=True)
    if feature_config is None or not feature_config.only_backbone:
        base_pairs = parse_dot_bracket(dot_bracket)
        for i, j, edge_type in base_pairs:
            # Undirected edges: add both directions
            edge_list.append([i, j])
            edge_attrs.append(encode_edge_type_onehot(edge_type))  # 5D one-hot

            edge_list.append([j, i])
            edge_attrs.append(encode_edge_type_onehot(edge_type))  # 5D one-hot

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)
    else:
        # Handle case with no edges (shouldn't happen for RNA, but be safe)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float32)

    return node_features, edge_index, edge_attr
