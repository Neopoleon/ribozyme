"""Convert RNA structure to graph representation for GNN"""

import torch
import numpy as np
from typing import Dict, Tuple, List


# Character to index mappings
NUCLEOTIDE_CHARS = ['A', 'U', 'G', 'C', 'N']
STRUCTURE_CHARS = ['E', 'S', 'H', 'I', 'M', 'B', 'X']
BRACKET_PAIRS = {
    '(': (')', 1),  # canonical base pairs
    '[': (']', 2),  # pseudoknot bracket type 1
    '{': ('}', 3),  # pseudoknot bracket type 2
    '<': ('>', 4),  # pseudoknot bracket type 3
}


def parse_dot_bracket(dot_bracket: str) -> List[Tuple[int, int, int]]:
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


def rna_to_graph(
    sequence: str,
    dot_bracket: str,
    structure: str,
    pseudoknot: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert RNA structure information to graph representation.

    Args:
        sequence: Nucleotide sequence (A/U/G/C/N)
        dot_bracket: Dot-bracket notation for secondary structure
        structure: Structural annotation (E/S/H/I/M/B/X)
        pseudoknot: Pseudoknot annotation (N/K)

    Returns:
        Tuple of (node_features, edge_index, edge_attr):
        - node_features: [num_nodes, 14] tensor
        - edge_index: [2, num_edges] tensor
        - edge_attr: [num_edges, 1] tensor (edge type)
    """
    seq_len = len(sequence)

    # Build node features (14 dimensions per node)
    node_features = []
    for i in range(seq_len):
        nuc_encoding = encode_nucleotide(sequence[i])  # 5 dims
        struct_encoding = encode_structure(structure[i])  # 7 dims
        pk_encoding = encode_pseudoknot(pseudoknot[i])  # 1 dim
        position = i / max(seq_len - 1, 1)  # normalized position [0, 1], 1 dim

        node_feat = np.concatenate([
            nuc_encoding,
            struct_encoding,
            [pk_encoding],
            [position]
        ])
        node_features.append(node_feat)

    node_features = torch.tensor(np.array(node_features), dtype=torch.float32)

    # Build edges
    edge_list = []
    edge_types = []

    # 1. Backbone edges (sequential connections)
    for i in range(seq_len - 1):
        # Undirected edges: add both directions
        edge_list.append([i, i + 1])
        edge_types.append(0)  # backbone edge type
        edge_list.append([i + 1, i])
        edge_types.append(0)

    # 2. Base-pair edges from dot-bracket notation
    base_pairs = parse_dot_bracket(dot_bracket)
    for i, j, edge_type in base_pairs:
        # Undirected edges: add both directions
        edge_list.append([i, j])
        edge_types.append(edge_type)
        edge_list.append([j, i])
        edge_types.append(edge_type)

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)
    else:
        # Handle case with no edges (shouldn't happen for RNA, but be safe)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.long)

    return node_features, edge_index, edge_attr
