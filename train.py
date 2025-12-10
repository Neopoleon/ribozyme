"""Training script for RNA GNN classifier"""

import os
# Fix OpenMP issue on Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

from src.data import RNADataset, LabelEncoder
from src.models import get_model


def train_epoch(model, loader, optimizer, device, class_weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        labels = batch.y.squeeze()

        # Compute loss
        if class_weights is not None:
            loss = F.cross_entropy(out, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(out, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        # Track predictions
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        labels = batch.y.squeeze()

        # Compute loss
        if class_weights is not None:
            loss = F.cross_entropy(out, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(out, labels)

        total_loss += loss.item() * batch.num_graphs

        # Track predictions
        preds = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1, all_preds, all_labels, all_probs


def compute_class_weights(dataset, num_classes, device):
    """Compute class weights for handling imbalance"""
    label_counts = torch.zeros(num_classes)

    for i in range(len(dataset)):
        label = dataset[i].y.item()
        label_counts[label] += 1

    # Inverse frequency weighting
    weights = 1.0 / (label_counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return weights.to(device)


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Output directory: {output_dir}")

    # Initialize label encoder
    label_encoder = LabelEncoder()
    print(f"Number of classes: {label_encoder.num_classes}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = RNADataset(
        root='data/processed/train',
        fold_labels_path='data/splits/train_labels.json',
        rfam_types_path='rfam/rfam_types_full.pkl',
        st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
        label_encoder=label_encoder,
    )

    val_dataset = RNADataset(
        root='data/processed/val',
        fold_labels_path='data/splits/val_labels.json',
        rfam_types_path='rfam/rfam_types_full.pkl',
        st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
        label_encoder=label_encoder,
    )

    test_dataset = RNADataset(
        root='data/processed/test',
        fold_labels_path='data/splits/test_labels.json',
        rfam_types_path='rfam/rfam_types_full.pkl',
        st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
        label_encoder=label_encoder,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Compute class weights if enabled
    class_weights = None
    if args.use_class_weights:
        print("\nComputing class weights...")
        class_weights = compute_class_weights(train_dataset, label_encoder.num_classes, device)
        print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")

    # Initialize model
    print(f"\nInitializing {args.model.upper()} model...")
    model = get_model(
        args.model,
        num_node_features=14,
        num_classes=label_encoder.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    for epoch in range(1, args.epochs + 1):
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

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                model, optimizer, epoch,
                {'val_f1': val_f1, 'val_acc': val_acc, 'val_loss': val_loss},
                output_dir / 'best_model.pt'
            )
            print(f"  → New best model saved (Val F1: {val_f1:.4f})")

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

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
    target_names = [label_encoder.decode(i) for i in range(label_encoder.num_classes)]
    report = classification_report(test_labels, test_preds, target_names=target_names, zero_division=0)
    print(report)

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'classification_report': report,
        'predictions': test_preds,
        'labels': test_labels,
        'probabilities': test_probs,
    }

    # Save as numpy arrays for analysis
    np.save(output_dir / 'test_predictions.npy', test_preds)
    np.save(output_dir / 'test_labels.npy', test_labels)
    np.save(output_dir / 'test_probabilities.npy', test_probs)

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

    print(f"\n✓ Training complete! Results saved to: {output_dir}")
    print(f"\nTo visualize results, run:")
    print(f"  python visualize_results.py {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNA GNN classifier')

    # Model args
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gin'],
                        help='Model architecture')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max'],
                        help='Graph pooling method')

    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights to handle imbalance')

    # Output args
    parser.add_argument('--output-dir', type=str, default='results/runs',
                        help='Output directory for results')

    args = parser.parse_args()
    main(args)
