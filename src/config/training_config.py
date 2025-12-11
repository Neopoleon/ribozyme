"""Training configuration with Hydra support"""

from dataclasses import dataclass
from pathlib import Path

from .feature_config import FeatureConfig


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str = 'gcn'  # 'gcn', 'gat', 'gin', 'gcn_edge', 'gat_edge', 'gin_edge'
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    pooling: str = 'mean'  # 'mean' or 'max'
    num_heads: int = 4  # Only used for GAT

    def __post_init__(self) -> None:
        """Validate model configuration"""
        valid_architectures = ['gcn', 'gat', 'gin', 'gcn_edge', 'gat_edge', 'gin_edge']
        if self.architecture not in valid_architectures:
            raise ValueError(
                f"Invalid architecture: {self.architecture}. "
                f"Choose from {valid_architectures}"
            )
        if self.pooling not in ['mean', 'max']:
            raise ValueError(f"Invalid pooling: {self.pooling}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive: {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive: {self.num_layers}")


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-5
    use_class_weights: bool = True
    patience: int = 20  # For learning rate scheduler
    early_stop_patience: int = 30  # Stop training if no improvement

    def __post_init__(self) -> None:
        """Validate training parameters"""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive: {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive: {self.lr}")


@dataclass
class DataConfig:
    """Data paths configuration"""
    train_path: str = 'data/splits/train_labels.json'
    val_path: str = 'data/splits/val_labels.json'
    test_path: str = 'data/splits/test_labels.json'
    rfam_types_path: str = 'rfam/rfam_types_full.pkl'
    st_files_dir: str = 'data/unzipped/bpRNA_1m_90_STAFILES'

    def validate(self) -> None:
        """Validate data paths exist"""
        for path_name in ['train_path', 'val_path', 'test_path', 'rfam_types_path']:
            path = Path(getattr(self, path_name))
            if not path.exists():
                raise FileNotFoundError(f"{path_name} not found: {path}")

        st_dir = Path(self.st_files_dir)
        if not st_dir.exists():
            raise FileNotFoundError(f"st_files_dir not found: {st_dir}")


@dataclass
class Config:
    """
    Complete Hydra configuration for RNA GNN training.

    This is the top-level config class that Hydra will instantiate.
    """
    experiment_name: str = "rna_gnn_experiment"
    features: FeatureConfig | None = None
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    data: DataConfig | None = None
    output_dir: str = 'results/runs'
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialize nested configs with defaults if not provided"""
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()

    def __repr__(self) -> str:
        assert self.model is not None
        assert self.training is not None
        assert self.features is not None
        assert self.data is not None
        assert self.features is not None

        return (
            f"Config(\n"
            f"  experiment={self.experiment_name}\n"
            f"  {self.features}\n"
            f"  model={self.model.architecture}, "
            f"hidden={self.model.hidden_dim}, "
            f"layers={self.model.num_layers}\n"
            f"  epochs={self.training.epochs}, "
            f"batch={self.training.batch_size}, "
            f"lr={self.training.lr}\n"
            f")"
        )
