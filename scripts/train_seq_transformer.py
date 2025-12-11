"""Sequence-only Transformer baseline training script.

This is the standard single-GPU training script using file I/O per sample.
For better performance, consider using train_seq_transformer_fast.py which provides:
- Multi-GPU distributed training
- Memory-cached dataset (5-10x faster)
- Non-blocking transfers and persistent workers

Usage:
    python scripts/train_seq_transformer.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data import (
    LabelEncoder,
    PAD_TOKEN_ID,
    RNASequenceDataset,
    sequence_collate_fn,
    VOCAB_SIZE,
)
from src.models import RNASequenceTransformer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: RNASequenceDataset, device: torch.device) -> torch.Tensor:
    """Compute inverse frequency class weights for balanced loss.

    Args:
        dataset: Dataset with label_counts() method
        device: Target device for weights tensor

    Returns:
        Normalized class weights tensor [num_classes]
    """
    counts = dataset.label_counts().float()
    weights = 1.0 / (counts + 1e-6)  # Inverse frequency
    weights = weights / weights.sum() * len(counts)  # Normalize
    return weights.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """Run one epoch of training or evaluation.

    Args:
        model: Neural network model
        loader: DataLoader
        optimizer: Optimizer for training (None for evaluation)
        device: Target device (cuda or cpu)
        criterion: Loss function

    Returns:
        Tuple of (average_loss, accuracy, f1_score)
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, acc, f1


@hydra.main(config_path="../conf", config_name="seq_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function for single-GPU baseline.

    Args:
        cfg: Hydra configuration from conf/seq_baseline.yaml

    Note:
        For better performance with multi-GPU support, use train_seq_transformer_fast.py
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    project_root = Path(get_original_cwd())
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"seq_transformer_{run_timestamp}"
    output_dir = project_root / cfg.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    label_encoder = LabelEncoder()

    print("\nLoading datasets...")
    train_dataset = RNASequenceDataset(
        fold_labels_path=to_absolute_path(cfg.data.train_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
    )
    val_dataset = RNASequenceDataset(
        fold_labels_path=to_absolute_path(cfg.data.val_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
    )
    test_dataset = RNASequenceDataset(
        fold_labels_path=to_absolute_path(cfg.data.test_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=sequence_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    model = RNASequenceTransformer(
        vocab_size=VOCAB_SIZE,
        num_classes=label_encoder.num_classes,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        pad_token_id=PAD_TOKEN_ID,
        max_seq_len=cfg.model.max_seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=cfg.training.lr_patience, factor=0.5
    )

    class_weights = None
    if cfg.training.use_class_weights:
        class_weights = compute_class_weights(train_dataset, device)
        print("Class weights computed.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    best_state = None
    history: dict[str, list[float]] = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
    }

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, optimizer, device, criterion
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, None, device, criterion
        )

        scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }
            torch.save(best_state, output_dir / 'best_model.pt')

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state['model_state_dict'])

    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1 = run_epoch(
        model, test_loader, None, device, criterion
    )
    print(f"Test Loss {test_loss:.4f} | Acc {test_acc:.4f} | F1 {test_f1:.4f}")

    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    results = {
        'val_best_f1': best_val_f1,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
