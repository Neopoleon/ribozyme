"""Scaling Law Experiment: Train GNN models with varying training data fractions

This script runs a data scaling experiment to understand how model performance
degrades with reduced training data. It trains models (GIN/GAT/GCN) with all
features=True on different fractions of the training set while keeping val/test
sets fixed.

Usage:
    # Single run with 40% of training data, seed 0, default config (GIN)
    python train_scaling_law.py data_fraction=0.4 seed=0

    # Run GIN model on all fractions and seeds (Machine 1)
    python train_scaling_law.py --config-name=scaling_law_gin -m

    # Run GAT model on all fractions and seeds (Machine 2)
    python train_scaling_law.py --config-name=scaling_law_gat -m

    # Run GCN model on all fractions and seeds (Machine 3)
    python train_scaling_law.py --config-name=scaling_law_gcn -m

    # Custom sweep
    python train_scaling_law.py -m data_fraction=1.0,0.8,0.4,0.2,0.1 seed=0,1,2,3,4
"""

import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

try:
    import wandb
except Exception:
    wandb = None

from src.config import FeatureConfig
from src.data import LabelEncoder, RNADataset
from src.models import get_model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_subsample_dataset(
    dataset: RNADataset,
    fraction: float,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    """
    Stratified subsampling of dataset indices to maintain class distribution.

    Args:
        dataset: The RNADataset to subsample
        fraction: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        selected_indices: List of selected dataset indices
        class_distribution: Dict mapping class names to counts in the subsample
    """
    if fraction >= 1.0:
        return list(range(len(dataset))), {}

    # Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        all_labels.append(dataset[i].y.item())
    all_labels = np.array(all_labels)

    # Stratified split
    all_indices = np.arange(len(dataset))
    selected_indices, _ = train_test_split(
        all_indices,
        train_size=fraction,
        random_state=seed,
        stratify=all_labels,
    )

    # Compute class distribution in subsample
    selected_labels = all_labels[selected_indices]
    unique_labels, counts = np.unique(selected_labels, return_counts=True)
    class_distribution = {
        dataset.label_encoder.decode(int(label)): int(count)
        for label, count in zip(unique_labels, counts)
    }

    return selected_indices.tolist(), class_distribution


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        labels = batch.y.squeeze()

        loss = F.cross_entropy(out, labels, weight=class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, float, float, list[int], list[int], list[list[float]]]:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        labels = batch.y.squeeze()

        loss = F.cross_entropy(out, labels, weight=class_weights)
        total_loss += loss.item() * batch.num_graphs

        preds = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1, all_preds, all_labels, all_probs


def compute_class_weights(
    dataset: RNADataset,
    indices: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute class weights for handling imbalance (from subset of indices)"""
    label_counts = torch.zeros(num_classes)

    for idx in indices:
        label = dataset[idx].y.item()
        label_counts[label] += 1

    # Inverse frequency weighting
    weights = 1.0 / (label_counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return weights.to(device)


@hydra.main(version_base=None, config_path="conf", config_name="scaling_law")
def main(cfg: DictConfig) -> None:
    """Main scaling law experiment function"""

    # Setup output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fraction_str = f"{cfg.data_fraction:.2f}".replace(".", "p")
    model_tag = getattr(cfg.model, "architecture", "model")
    run_name = f"scaling_{model_tag}_frac{fraction_str}_seed{cfg.seed}_{run_timestamp}"
    project_root = Path(get_original_cwd())
    base_output_dir = project_root / cfg.output_dir
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"SCALING LAW EXPERIMENT: {cfg.data_fraction*100:.1f}% training data, seed {cfg.seed}")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # Set seed
    set_seed(cfg.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=False), f, indent=2)
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Optional W&B logging
    wandb_enabled = os.getenv("HYDRA_WANDB", "0").lower() in {"1", "true", "yes"}
    wandb_run = None
    if wandb_enabled and wandb is not None:
        try:
            wandb_config = OmegaConf.to_container(cfg, resolve=False)
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "ribozyme-scaling"),
                entity=os.getenv("WANDB_ENTITY"),
                mode=os.getenv("WANDB_MODE", "online"),
                name=run_name,
                dir=str(output_dir),
                config=wandb_config,
                tags=["scaling_law", f"fraction_{fraction_str}", f"seed_{cfg.seed}"],
            )
        except Exception as exc:
            print(f"W&B init failed: {exc}")

    # Feature config: all True for best-performing setup
    feature_config = FeatureConfig(
        use_nucleotide=True,
        use_structure_annotation=True,
        use_pseudoknot=True,
        use_position_encoding=True,
        only_backbone=False,
    )
    print(f"\n{feature_config}")
    num_node_features = feature_config.get_node_feature_dim()
    print(f"Node feature dimension: {num_node_features}")

    # Initialize label encoder
    label_encoder = LabelEncoder()
    num_classes = label_encoder.num_classes
    print(f"Number of classes: {num_classes}")

    # Load FULL datasets (we'll subsample training only)
    print("\nLoading datasets...")
    train_dataset_full = RNADataset(
        root=to_absolute_path('data/processed/train'),
        fold_labels_path=to_absolute_path(cfg.data.train_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
        feature_config=feature_config,
    )

    val_dataset = RNADataset(
        root=to_absolute_path('data/processed/val'),
        fold_labels_path=to_absolute_path(cfg.data.val_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
        feature_config=feature_config,
    )

    test_dataset = RNADataset(
        root=to_absolute_path('data/processed/test'),
        fold_labels_path=to_absolute_path(cfg.data.test_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
        feature_config=feature_config,
    )

    print(f"Full train dataset: {len(train_dataset_full)} samples")
    print(f"Val dataset: {len(val_dataset)} samples (FIXED)")
    print(f"Test dataset: {len(test_dataset)} samples (FIXED)")

    # Stratified subsample training data
    print(f"\nSubsampling training data to {cfg.data_fraction*100:.1f}%...")
    start_time = time.time()
    train_indices, class_dist = stratified_subsample_dataset(
        train_dataset_full,
        cfg.data_fraction,
        cfg.seed,
    )
    subsample_time = time.time() - start_time

    print(f"Selected {len(train_indices)} training samples ({subsample_time:.2f}s)")
    if class_dist:
        print("\nClass distribution in subsample:")
        for class_name, count in sorted(class_dist.items()):
            print(f"  {class_name}: {count}")

    # Create subset of training dataset
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # Compute class weights on the subsampled training data
    class_weights = None
    if cfg.training.use_class_weights:
        print("\nComputing class weights from subsampled training data...")
        class_weights = compute_class_weights(
            train_dataset_full, train_indices, num_classes, device
        )
        print(f"Class weights: min={class_weights.min():.3f}, "
              f"max={class_weights.max():.3f}")

    # Initialize model
    print(f"\nInitializing {cfg.model.architecture.upper()} model...")

    # Build model kwargs
    model_kwargs = {
        'num_node_features': num_node_features,
        'num_classes': num_classes,
        'hidden_dim': cfg.model.hidden_dim,
        'num_layers': cfg.model.num_layers,
        'dropout': cfg.model.dropout,
        'pooling': cfg.model.pooling,
    }

    # Add num_heads for GAT models
    if 'gat' in cfg.model.architecture.lower():
        model_kwargs['num_heads'] = cfg.model.get('num_heads', 4)

    model = get_model(
        cfg.model.architecture,
        **model_kwargs
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=cfg.training.patience,
    )

    # Training loop
    print("\nStarting training...")
    training_start_time = time.time()
    best_val_f1 = 0.0
    history: dict[str, list[float]] = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    epochs_without_improvement = 0
    best_state = None
    best_epoch = 0

    for epoch in range(1, cfg.training.epochs + 1):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )

        # Validate
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, device, class_weights
        )

        # Update scheduler
        scheduler.step(val_f1)

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/f1": train_f1,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/f1": val_f1,
                "lr": scheduler.optimizer.param_groups[0]["lr"],
            }, step=epoch)

        print(f"Epoch {epoch:3d}/{cfg.training.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            print(f"  → New best model (Val F1: {val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= cfg.training.early_stop_patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    training_time = time.time() - training_start_time
    print(f"\nTraining completed in {training_time:.2f}s ({training_time/60:.2f} min)")

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Load best model and evaluate
    print("\nEvaluating best model on test set...")
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, class_weights
    )

    # Calculate additional metrics
    test_precision = precision_score(
        test_labels, test_preds, average='weighted', zero_division=0
    )
    test_recall = recall_score(
        test_labels, test_preds, average='weighted', zero_division=0
    )

    # Calculate train-val gap (overfitting indicator)
    best_train_acc = history['train_acc'][best_epoch - 1]
    best_train_f1 = history['train_f1'][best_epoch - 1]
    best_val_acc = history['val_acc'][best_epoch - 1]
    train_val_acc_gap = best_train_acc - best_val_acc
    train_val_f1_gap = best_train_f1 - best_val_f1

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Data fraction: {cfg.data_fraction*100:.1f}% ({len(train_indices)} samples)")
    print(f"Seed: {cfg.seed}")
    print(f"Best epoch: {best_epoch}")
    print(f"Training time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"\nOverfitting Indicators (train-val gap at best epoch):")
    print(f"  Accuracy gap: {train_val_acc_gap:+.4f}")
    print(f"  F1 gap:       {train_val_f1_gap:+.4f}")
    print(f"{'='*80}")

    # Classification report
    target_names = [label_encoder.decode(i) for i in range(num_classes)]
    report = classification_report(
        test_labels, test_preds,
        target_names=target_names,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)

    # Save comprehensive results
    scaling_results = {
        'experiment': 'scaling_law',
        'data_fraction': cfg.data_fraction,
        'seed': cfg.seed,
        'num_train_samples': len(train_indices),
        'num_val_samples': len(val_dataset),
        'num_test_samples': len(test_dataset),
        'best_epoch': best_epoch,
        'training_time_seconds': training_time,
        'subsample_time_seconds': subsample_time,
        'test_metrics': {
            'accuracy': test_acc,
            'f1_weighted': test_f1,
            'precision_weighted': test_precision,
            'recall_weighted': test_recall,
            'loss': test_loss,
        },
        'overfitting_indicators': {
            'train_val_acc_gap': train_val_acc_gap,
            'train_val_f1_gap': train_val_f1_gap,
            'best_train_acc': best_train_acc,
            'best_train_f1': best_train_f1,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
        },
        'class_distribution': class_dist,
        'classification_report': report,
    }

    with open(output_dir / 'scaling_results.json', 'w') as f:
        json.dump(scaling_results, f, indent=2)

    # Save predictions and confusion matrix
    np.save(output_dir / 'test_predictions.npy', np.array(test_preds))
    np.save(output_dir / 'test_labels.npy', np.array(test_labels))
    np.save(output_dir / 'test_probabilities.npy', np.array(test_probs))

    cm = confusion_matrix(test_labels, test_preds)
    np.save(output_dir / 'confusion_matrix.npy', cm)

    # Save concise summary for easy aggregation
    summary = {
        'data_fraction': cfg.data_fraction,
        'seed': cfg.seed,
        'num_train_samples': len(train_indices),
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'train_val_acc_gap': train_val_acc_gap,
        'train_val_f1_gap': train_val_f1_gap,
        'training_time_seconds': training_time,
        'best_epoch': best_epoch,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        wandb.log({
            'test/accuracy': test_acc,
            'test/f1': test_f1,
            'test/precision': test_precision,
            'test/recall': test_recall,
            'test/loss': test_loss,
            'overfitting/train_val_acc_gap': train_val_acc_gap,
            'overfitting/train_val_f1_gap': train_val_f1_gap,
            'training_time_seconds': training_time,
        })
        wandb_run.finish()

    print(f"\n✓ Experiment complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
