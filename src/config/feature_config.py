"""Feature configuration for RNA graph construction"""

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """
    Controls which node and edge features to include in the graph.

    Node Features (configurable):
    - use_nucleotide: 5D one-hot (A, U, G, C, N)
    - use_structure_annotation: 7D one-hot (E, S, H, I, M, B, X) - motif types
    - use_pseudoknot: 1D binary (0 or 1) - pseudoknot indicator
    - use_position_encoding: 1D float [0, 1] - normalized position in sequence

    Edge Features (always 5D one-hot):
    - Backbone edge (sequential connection)
    - Canonical base pair ()
    - Pseudoknot bracket []
    - Pseudoknot bracket {}
    - Pseudoknot bracket <>
    """

    # Node features
    use_nucleotide: bool = True
    use_structure_annotation: bool = True  # Motif annotation (E/S/H/I/M/B/X)
    use_pseudoknot: bool = True
    use_position_encoding: bool = True

    def get_node_feature_dim(self) -> int:
        """Calculate total node feature dimension based on enabled features"""
        return sum([
            5 if self.use_nucleotide else 0,
            7 if self.use_structure_annotation else 0,
            1 if self.use_pseudoknot else 0,
            1 if self.use_position_encoding else 0,
        ])

    def get_edge_feature_dim(self) -> int:
        """Edge features are always 5D one-hot encoded"""
        return 5

    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        if not any([
            self.use_nucleotide,
            self.use_structure_annotation,
            self.use_pseudoknot,
            self.use_position_encoding
        ]):
            raise ValueError("At least one node feature must be enabled")

    def __repr__(self) -> str:
        features = []
        if self.use_nucleotide:
            features.append("nucleotide(5D)")
        if self.use_structure_annotation:
            features.append("structure(7D)")
        if self.use_pseudoknot:
            features.append("pseudoknot(1D)")
        if self.use_position_encoding:
            features.append("position(1D)")

        return f"FeatureConfig(node_dim={self.get_node_feature_dim()}, features=[{', '.join(features)}])"
