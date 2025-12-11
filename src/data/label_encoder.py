"""Label encoding for RNA meta-types"""

import json

class LabelEncoder:
    """
    Encode RNA meta-type strings to integer labels for classification.

    Handles the 23 unique meta-type categories from RFAM.
    """

    def __init__(self, meta_types: list[str] | None = None):
        """
        Initialize label encoder.

        Args:
            meta_types: List of meta-type strings. If None, uses predefined 23 classes.
        """
        if meta_types is None:
            # Default 23 meta-type classes (sorted alphabetically)
            meta_types = [
                'Cis-reg;',
                'Cis-reg; IRES;',
                'Cis-reg; frameshift_element;',
                'Cis-reg; leader;',
                'Cis-reg; riboswitch;',
                # 'Cis-reg; thermoregulator;', # too few samples, removed
                'Gene;',
                'Gene; CRISPR;',
                'Gene; antisense;',
                'Gene; antitoxin;',
                # 'Gene; lncRNA;', # too few samples, removed
                'Gene; miRNA;',
                'Gene; rRNA;',
                'Gene; ribozyme;',
                'Gene; sRNA;',
                # 'Gene; snRNA;', # too few samples, removed
                # 'Gene; snRNA; snoRNA;', # too few samples, removed
                'Gene; snRNA; snoRNA; CD-box;',
                'Gene; snRNA; snoRNA; HACA-box;',
                'Gene; snRNA; snoRNA; scaRNA;',
                'Gene; snRNA; splicing;',
                'Gene; tRNA;',
                'Intron;',
            ]

        self.meta_types = meta_types
        self.type_to_idx = {t: i for i, t in enumerate(meta_types)}
        self.idx_to_type = {i: t for i, t in enumerate(meta_types)}
        self.num_classes = len(meta_types)

    def encode(self, meta_type: str) -> int:
        """
        Encode meta-type string to integer label.

        Args:
            meta_type: Meta-type string like "Gene; miRNA;"

        Returns:
            Integer label

        Raises:
            KeyError: If meta_type not in known types
        """
        if meta_type not in self.type_to_idx:
            raise KeyError(
                f"Unknown meta-type: '{meta_type}'. "
                f"Known types: {list(self.type_to_idx.keys())}"
            )
        return self.type_to_idx[meta_type]

    def decode(self, label: int) -> str:
        """
        Decode integer label back to meta-type string.

        Args:
            label: Integer label

        Returns:
            Meta-type string
        """
        if label not in self.idx_to_type:
            raise ValueError(f"Invalid label: {label}. Valid range: [0, {self.num_classes-1}]")
        return self.idx_to_type[label]

    def save(self, path: str):
        """Save label encoder configuration to JSON."""
        config = {
            'meta_types': self.meta_types,
            'num_classes': self.num_classes,
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LabelEncoder':
        """Load label encoder from JSON configuration."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(meta_types=config['meta_types'])

    def __repr__(self) -> str:
        return f"LabelEncoder(num_classes={self.num_classes})"
