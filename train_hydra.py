"""Training script for RNA GNN classifier with Hydra configuration"""

import os
# Fix OpenMP issue on Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader

from src.config import Config, FeatureConfig
from src.data import LabelEncoder, RNADataset
from src.models import get_model


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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration"""

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

    # Create output directory (Hydra handles this automatically)
    output_dir = Path.cwd()  # Hydra changes working directory
    print(f"Output directory: {output_dir}")

    # Convert feature config from DictConfig
    feature_config = FeatureConfig(
        use_nucleotide=cfg.features.use_nucleotide,
        use_structure_annotation=cfg.features.use_structure_annotation,
        use_pseudoknot=cfg.features.use_pseudoknot,
        use_position_encoding=cfg.features.use_position_encoding,
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
        root='data/processed/train',
        fold_labels_path=cfg.data.train_path,
        rfam_types_path=cfg.data.rfam_types_path,
        st_files_dir=cfg.data.st_files_dir,
        label_encoder=label_encoder,
        feature_config=feature_config,
    )

    val_dataset = RNADataset(
        root='data/processed/val',
        fold_labels_path=cfg.data.val_path,
        rfam_types_path=cfg.data.rfam_types_path,
        st_files_dir=cfg.data.st_files_dir,
        label_encoder=label_encoder,
        feature_config=feature_config,
    )

    test_dataset = RNADataset(
        root='data/processed/test',
        fold_labels_path=cfg.data.test_path,
        rfam_types_path=cfg.data.rfam_types_path,
        st_files_dir=cfg.data.st_files_dir,
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

        print(f"Epoch {epoch:3d}/{cfg.training.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch,
                {'val_f1': val_f1, 'val_acc': val_acc, 'val_loss': val_loss},
                output_dir / 'best_model.pt'
            )
            print(f"  → New best model saved (Val F1: {val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= cfg.training.early_stop_patience:
            print(f"\nEarly stopping after {epoch} epochs "
                  f"({epochs_without_improvement} epochs without improvement)")
            break

    # Save training history
    np.save(output_dir / 'history.npy', history)

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, class_weights
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")

    # Generate detailed classification report
    print("\nClassification Report:")
    target_names = [label_encoder.decode(i) for i in range(num_classes)]
    report = classification_report(
        test_labels, test_preds,
        target_names=target_names,
        zero_division=0,
    )
    print(report)

    # Save test results
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

    print(f"Training complete!. Results saved to: {output_dir}")


# Register config with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


if __name__ == '__main__':
    main()