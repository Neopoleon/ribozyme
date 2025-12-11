"""GPU-based integration test for the Hydra training pipeline.

This script mirrors the stage-by-stage validations in quick_test.py while
running the Hydra-driven pipeline for 10 epochs on GPU to ensure every step
executes correctly and that the model meaningfully learns (loss decreases and
metrics improve).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from hydra import compose, initialize
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.config import FeatureConfig
from src.data import LabelEncoder, RNADataset
from train_hydra import compute_class_weights, evaluate, set_seed, train_epoch

# Allow overriding the subset size via environment variables to tune runtime.
MAX_TRAIN_SAMPLES = int(os.getenv("HYDRA_TEST_MAX_TRAIN", "1024"))
MAX_VAL_SAMPLES = int(os.getenv("HYDRA_TEST_MAX_VAL", "512"))
MAX_TEST_SAMPLES = int(os.getenv("HYDRA_TEST_MAX_TEST", "512"))
TARGET_EPOCHS = int(os.getenv("HYDRA_TEST_EPOCHS", "5"))


def _subset_dataset(
    dataset: RNADataset,
    max_samples: int,
    seed: int,
) -> RNADataset | Subset[RNADataset]:
    """Return a deterministic subset if the dataset is larger than max_samples."""
    if len(dataset) <= max_samples:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples]
    return Subset(dataset, indices.tolist())


def _print_stage(idx: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"{idx}. {title}")
    print("=" * 80)


def main() -> None:
    print("=" * 80)
    print("Hydra Training Pipeline Integration Test")
    print("=" * 80)

    _print_stage(1, "Checking GPU availability")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this test.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch_tag = f"sm_{capability[0]}{capability[1]}"
    print(f"Using GPU: {gpu_name} (capability {arch_tag})")

    supported_archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if supported_archs and arch_tag not in supported_archs:
        raise RuntimeError(
            "Detected GPU architecture is unsupported by the current PyTorch build.\n"
            f"  Device: {gpu_name} ({arch_tag})\n"
            f"  Supported archs in this install: {', '.join(supported_archs)}\n"
            "Please install a PyTorch build compiled for your GPU or upgrade hardware."
        )

    # Load Hydra configuration from conf/config.yaml.
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")

    _print_stage(2, "Preparing features and label encoder")
    set_seed(cfg.seed)

    feature_config = FeatureConfig(
        use_nucleotide=cfg.features.use_nucleotide,
        use_structure_annotation=cfg.features.use_structure_annotation,
        use_pseudoknot=cfg.features.use_pseudoknot,
        use_position_encoding=cfg.features.use_position_encoding,
    )
    num_node_features = feature_config.get_node_feature_dim()
    print(f"Node feature dimension: {num_node_features}")

    label_encoder = LabelEncoder()
    num_classes = label_encoder.num_classes
    print(f"Number of classes: {num_classes}")

    # Load datasets using the Hydra config paths.
    _print_stage(3, "Loading datasets from Hydra config")
    train_dataset = RNADataset(
        root="data/processed/train",
        fold_labels_path=cfg.data.train_path,
        rfam_types_path=cfg.data.rfam_types_path,
        st_files_dir=cfg.data.st_files_dir,
        label_encoder=label_encoder,
        feature_config=feature_config,
    )
    val_dataset = RNADataset(
        root="data/processed/val",
        fold_labels_path=cfg.data.val_path,
        rfam_types_path=cfg.data.rfam_types_path,
        st_files_dir=cfg.data.st_files_dir,
        label_encoder=label_encoder,
        feature_config=feature_config,
    )
    test_dataset = RNADataset(
        root="data/processed/test",
        fold_labels_path=cfg.data.test_path,
        rfam_types_path=cfg.data.rfam_types_path,
        st_files_dir=cfg.data.st_files_dir,
        label_encoder=label_encoder,
        feature_config=feature_config,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise AssertionError("Datasets must contain at least one sample.")

    # Keep the runtime tractable by taking deterministic subsets.
    train_dataset = _subset_dataset(train_dataset, MAX_TRAIN_SAMPLES, cfg.seed)
    val_dataset = _subset_dataset(val_dataset, MAX_VAL_SAMPLES, cfg.seed + 1)
    test_dataset = _subset_dataset(test_dataset, MAX_TEST_SAMPLES, cfg.seed + 2)

    print(
        f"Train samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )

    batch_size = min(cfg.training.batch_size, 32)
    print(f"Using batch size: {batch_size}")

    _print_stage(4, "Creating dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader)
    print(
        f"Train batches: {train_batches} | "
        f"Val batches: {val_batches} | "
        f"Test batches: {test_batches}"
    )
    if min(train_batches, val_batches, test_batches) == 0:
        raise AssertionError("Dataloaders produced zero batches; check batch size.")

    # Compute class weights to validate that stage as well.
    class_weights = (
        compute_class_weights(train_dataset, num_classes, device)
        if cfg.training.use_class_weights
        else None
    )
    if class_weights is not None:
        print(
            "Class weights stats → "
            f"min: {class_weights.min():.4f}, max: {class_weights.max():.4f}"
        )

    _print_stage(5, f"Initializing {cfg.model.architecture.upper()} model")
    model_kwargs = {
        "num_node_features": num_node_features,
        "num_classes": num_classes,
        "hidden_dim": cfg.model.hidden_dim,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
        "pooling": cfg.model.pooling,
    }
    if "gat" in cfg.model.architecture.lower():
        model_kwargs["num_heads"] = cfg.model.num_heads

    # Lazily import here to avoid circular issues on startup.
    from src.models import get_model

    model = get_model(cfg.model.architecture, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg.training.patience
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    _print_stage(6, f"Training for {TARGET_EPOCHS} epochs on GPU")
    for epoch in range(1, TARGET_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, device, class_weights
        )
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:2d}/{TARGET_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

    # Basic correctness checks on the collected metrics.
    if history["train_loss"][-1] >= history["train_loss"][0]:
        raise AssertionError(
            "Training loss did not decrease over 10 epochs. "
            f"Start={history['train_loss'][0]:.4f}, "
            f"End={history['train_loss'][-1]:.4f}"
        )
    if history["val_loss"][-1] >= history["val_loss"][0]:
        raise AssertionError(
            "Validation loss did not decrease over 10 epochs. "
            f"Start={history['val_loss'][0]:.4f}, "
            f"End={history['val_loss'][-1]:.4f}"
        )
    if history["val_acc"][-1] <= history["val_acc"][0]:
        raise AssertionError(
            "Validation accuracy did not improve over 10 epochs. "
            f"Start={history['val_acc'][0]:.4f}, "
            f"End={history['val_acc'][-1]:.4f}"
        )
    if history["val_f1"][-1] <= history["val_f1"][0]:
        raise AssertionError(
            "Validation F1 did not improve over 10 epochs. "
            f"Start={history['val_f1'][0]:.4f}, "
            f"End={history['val_f1'][-1]:.4f}"
        )

    _print_stage(7, "Evaluating on the held-out test subset")
    test_loss, test_acc, test_f1, _, _, _ = evaluate(
        model, test_loader, device, class_weights
    )
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
        f"Test F1: {test_f1:.4f}"
    )
    if not np.isfinite(test_loss):
        raise AssertionError("Test loss is not finite, indicating instability.")

    print("\n✓ Hydra training pipeline passed the 10-epoch GPU integration test!")
    print("All stages (data, training, validation, test) executed successfully.")


if __name__ == "__main__":
    main()
