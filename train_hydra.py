"""Training script for RNA GNN classifier with Hydra configuration"""

import copy
import json
import os
from datetime import datetime
# Fix OpenMP issue on Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader
try:
    import wandb  # type: ignore
except Exception:  # noqa: BLE001
    wandb = None

from src.config import Config, FeatureConfig
from src.data import LabelEncoder, RNADataset
from src.models import get_model
from visualize_results import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_top_predictions,
    plot_training_history,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        # Compute loss
        loss = F.cross_entropy(out, labels, weight=class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        # Track predictions
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

        # Compute loss
        loss = F.cross_entropy(out, labels, weight=class_weights)
        total_loss += loss.item() * batch.num_graphs

        # Track predictions
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
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute class weights for handling imbalance"""
    label_counts = torch.zeros(num_classes)

    for i in range(len(dataset)):
        label = dataset[i].y.item()
        label_counts[label] += 1

    # Inverse frequency weighting
    weights = 1.0 / (label_counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return weights.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    save_path: Path,
) -> None:
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)


@hydra.main(version_base=None, config_path="conf", config_name="config_0")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration"""

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_tag = (
        f"{cfg.features.use_nucleotide}_"
        f"{cfg.features.use_structure_annotation}_"
        f"{cfg.features.use_pseudoknot}_"
        f"{cfg.features.use_position_encoding}"
    )
    run_name = f"{cfg.model.architecture}_{feature_tag}_{run_timestamp}"
    project_root = Path(get_original_cwd())
    base_output_dir = project_root / cfg.output_dir
    output_dir = base_output_dir / run_name
    save_outputs = not cfg.test
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Convert DictConfig to structured config
    print("="*80)
    print("Configuration")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)

    # Set seed
    set_seed(cfg.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Output directory mirrors train.py style
    if save_outputs:
        print(f"Output directory: {output_dir}")
        # Save config snapshot (json + yaml). Avoid resolving interpolations that require
        # custom resolvers (e.g., ${datetime:...}) by keeping resolve=False.
        with open(output_dir / "config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=False), f, indent=2)
        with open(output_dir / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
    else:
        print("Test mode enabled: outputs will not be saved to disk.")
    # Optional Weights & Biases logging
    wandb_enabled = os.getenv("HYDRA_WANDB", "0").lower() in {"1", "true", "yes"}
    wandb_run = None
    if wandb_enabled:
        if wandb is None:
            print("HYDRA_WANDB set but wandb is not installed; skipping wandb logging.")
        else:
            try:
                wandb_config = OmegaConf.to_container(cfg, resolve=False)
                wandb_run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", "ribozyme"),
                    entity=os.getenv("WANDB_ENTITY"),
                    mode=os.getenv("WANDB_MODE", "online"),
                    name=run_name,
                    dir=str(output_dir if save_outputs else project_root / "outputs"),
                    config=wandb_config,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"HYDRA_WANDB set but wandb init failed: {exc}. Skipping wandb logging.")

    # Convert feature config from DictConfig
    feature_config = FeatureConfig(
        use_nucleotide=cfg.features.use_nucleotide,
        use_structure_annotation=cfg.features.use_structure_annotation,
        use_pseudoknot=cfg.features.use_pseudoknot,
        use_position_encoding=cfg.features.use_position_encoding,
        only_backbone=cfg.features.get('only_backbone', False),  # Backward compatible default
    )
    print(f"\n{feature_config}")
    num_node_features = feature_config.get_node_feature_dim()
    print(f"Node feature dimension: {num_node_features}")

    # Initialize label encoder
    label_encoder = LabelEncoder()
    num_classes = label_encoder.num_classes
    print(f"Number of classes: {num_classes}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = RNADataset(
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

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")

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

    # Compute class weights if enabled
    class_weights = None
    if cfg.training.use_class_weights:
        print("\nComputing class weights...")
        class_weights = compute_class_weights(train_dataset, num_classes, device)
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

    # Only add num_heads for GAT models
    if 'gat' in cfg.model.architecture.lower():
        model_kwargs['num_heads'] = cfg.model.num_heads

    model = get_model(
        cfg.model.architecture,
        **model_kwargs
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=cfg.training.patience,
    )

    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0.0
    history: dict[str, list[float]] = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    epochs_without_improvement = 0
    best_state = None
    best_metrics = None

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
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/f1": train_f1,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/f1": val_f1,
                    "lr": scheduler.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        print(f"Epoch {epoch:3d}/{cfg.training.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_metrics = {'val_f1': val_f1, 'val_acc': val_acc, 'val_loss': val_loss}
            if save_outputs:
                save_checkpoint(
                    model, optimizer, epoch,
                    best_metrics,
                    output_dir / 'best_model.pt'
                )
            else:
                best_state = copy.deepcopy(model.state_dict())
            print(f"  → New best model{' saved' if save_outputs else ''} (Val F1: {val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= cfg.training.early_stop_patience:
            print(f"\nEarly stopping after {epoch} epochs "
                  f"({epochs_without_improvement} epochs without improvement)")
            break

    # Save training history (json for compatibility with visualize_results.py)
    if save_outputs:
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    if save_outputs:
        checkpoint = torch.load(output_dir / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, class_weights
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")

    if wandb_run is not None:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/acc": test_acc,
                "test/f1": test_f1,
            }
        )

    # Generate detailed classification report
    print("\nClassification Report:")
    target_names = [label_encoder.decode(i) for i in range(num_classes)]
    report = classification_report(
        test_labels, test_preds,
        target_names=target_names,
        zero_division=0,
    )
    print(report)

    # Save test results and artifacts if not in test mode
    if save_outputs:
        test_results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'classification_report': report,
        }
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        np.save(output_dir / 'test_predictions.npy', np.array(test_preds))
        np.save(output_dir / 'test_labels.npy', np.array(test_labels))
        np.save(output_dir / 'test_probabilities.npy', np.array(test_probs))

        # Save text report
        with open(output_dir / 'test_results.txt', 'w') as f:
            f.write(f"Test Results\n")
            f.write(f"============\n\n")
            f.write(f"Loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {test_acc:.4f}\n")
            f.write(f"F1 Score: {test_f1:.4f}\n\n")
            f.write(f"Classification Report:\n")
            f.write(f"====================\n\n")
            f.write(report)

        # Compute and save confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        np.save(output_dir / 'confusion_matrix.npy', cm)

        # Generate visualizations into the run folder (mirrors visualize_results.py)
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        if (output_dir / 'history.json').exists():
            plot_training_history(output_dir / 'history.json', vis_dir / 'training_history.png')
        class_names = [label_encoder.decode(i) for i in range(num_classes)]
        plot_confusion_matrix(cm, class_names, vis_dir / 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_names, vis_dir / 'confusion_matrix_normalized.png', normalize=True)
        plot_per_class_metrics(test_labels, test_preds, class_names, vis_dir / 'per_class_metrics.png')
        plot_top_predictions(test_probs, test_labels, class_names, vis_dir / 'confidence_analysis.png')

        # Save summary text for quick reference
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(test_labels, test_preds)
        f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
        summary = f"""
Results Summary
===============

Overall Metrics:
  - Accuracy: {accuracy:.4f}
  - F1 Score (Macro): {f1_macro:.4f}
  - F1 Score (Weighted): {f1_weighted:.4f}

Visualizations saved to: {vis_dir}
  - training_history.png
  - confusion_matrix.png
  - confusion_matrix_normalized.png
  - per_class_metrics.png
  - confidence_analysis.png

For detailed per-class metrics, see: {output_dir}/test_results.txt
"""
        with open(vis_dir / 'summary.txt', 'w') as f:
            f.write(summary)
        print(summary)
        if wandb_run is not None:
            artifact = wandb.Artifact(
                name=f"{run_name}-artifacts",
                type="model",
                metadata={
                    "run_name": run_name,
                    "config": OmegaConf.to_container(cfg, resolve=False),
                },
            )
            for fname in [
                "config.json",
                "config.yaml",
                "history.json",
                "test_results.json",
                "test_results.txt",
                "test_predictions.npy",
                "test_labels.npy",
                "test_probabilities.npy",
                "confusion_matrix.npy",
                "best_model.pt",
            ]:
                fpath = output_dir / fname
                if fpath.exists():
                    artifact.add_file(str(fpath))
            for vis_fname in [
                "training_history.png",
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
                "per_class_metrics.png",
                "confidence_analysis.png",
                "summary.txt",
            ]:
                fpath = vis_dir / vis_fname
                if fpath.exists():
                    artifact.add_file(str(fpath), name=f"visualizations/{vis_fname}")
            wandb_run.log_artifact(artifact)
    else:
        print("Test mode: skipping artifact saving and visualizations.")

    print(f"\n✓ Training complete! Results saved to: {output_dir}" if save_outputs else "\n✓ Training complete (test mode, no artifacts saved).")
    if wandb_run is not None:
        wandb_run.finish()


# Register config with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


if __name__ == '__main__':
    main()
