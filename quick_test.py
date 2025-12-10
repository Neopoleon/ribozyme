"""Quick test of the training pipeline (CPU-friendly)"""

import os
# Fix OpenMP issue on Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
from torch_geometric.loader import DataLoader

from src.data import RNADataset, LabelEncoder
from src.models import get_model

print("="*80)
print("Quick Training Pipeline Test (CPU-Friendly)")
print("="*80)

# Set device
device = torch.device('cpu')
print(f"\n1. Using device: {device}")

# Initialize label encoder
label_encoder = LabelEncoder()
print(f"2. Number of classes: {label_encoder.num_classes}")

# Load small subset of train data
print("\n3. Loading datasets (this may take a minute)...")
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

print(f"   Train: {len(train_dataset)} samples")
print(f"   Val:   {len(val_dataset)} samples")

# Create dataloaders with small batch size
print("\n4. Creating dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Initialize small model
print("\n5. Initializing small GCN model...")
model = get_model(
    'gcn',
    num_node_features=14,
    num_classes=label_encoder.num_classes,
    hidden_dim=64,  # Smaller for CPU
    num_layers=2,    # Fewer layers for CPU
    dropout=0.3,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parameters: {num_params:,}")

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n6. Testing training for 2 epochs...")
print("   (This will take a few minutes on CPU)")

for epoch in range(1, 3):
    # Train
    model.train()
    train_loss = 0
    num_batches = 0

    for i, batch in enumerate(train_loader):
        if i >= 10:  # Only train on first 10 batches for quick test
            break

        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = torch.nn.functional.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

        if (i + 1) % 5 == 0:
            print(f"     Batch {i+1}/10: Loss = {loss.item():.4f}")

    avg_train_loss = train_loss / num_batches

    # Validate on first 5 batches
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:  # Only validate on first 5 batches
                break

            batch = batch.to(device)
            out = model(batch)
            loss = torch.nn.functional.cross_entropy(out, batch.y.squeeze())
            val_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == batch.y.squeeze()).sum().item()
            total += batch.num_graphs

    avg_val_loss = val_loss / 5
    val_acc = correct / total

    print(f"\n   Epoch {epoch}/2:")
    print(f"     Train Loss: {avg_train_loss:.4f}")
    print(f"     Val Loss:   {avg_val_loss:.4f}")
    print(f"     Val Acc:    {val_acc:.4f}")
    print()

print("="*80)
print("✓ Quick test complete!")
print("="*80)

print("\nThe training pipeline is working correctly!")
print("\nNext steps:")
print("  1. For full training, run: python train.py --epochs 100 --use-class-weights")
print("  2. For faster CPU training: python train.py --epochs 50 --hidden-dim 64 --num-layers 2")
print("  3. If you have a GPU: The script will automatically use it")
print("\nNote: Full training on CPU will take several hours.")
print("Consider using a GPU or reducing epochs/model size for CPU training.")
