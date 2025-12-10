# GNN Data Pipeline - Quick Start Guide

## What Was Built

A complete, production-ready data ingestion pipeline for training Graph Neural Networks on RNA structure classification with 23 RFAM meta-type categories.

## Files Created

### Core Modules (`src/data/`)
- **`parser.py`** - Parse .st files and extract RFAM metadata
- **`graph_builder.py`** - Convert RNA structures to graph representations
- **`label_encoder.py`** - Encode meta-type strings to integer labels
- **`rna_dataset.py`** - PyTorch Geometric Dataset class
- **`split_dataset.py`** - Create stratified train/val/test splits

### Data Files Generated
- **`data/splits/train_labels.json`** - 17,663 training samples
- **`data/splits/val_labels.json`** - 2,198 validation samples
- **`data/splits/test_labels.json`** - 2,227 test samples

### Documentation & Examples
- **`DATA_PIPELINE.md`** - Complete pipeline documentation
- **`test_data_pipeline.py`** - Test suite
- **`notebooks/data_pipeline_demo.ipynb`** - Interactive demo

## Quick Start

### 1. Test the Pipeline

```bash
python test_data_pipeline.py
```

This validates:
- Dataset loading for all splits ✓
- PyG DataLoader batching ✓
- Class distribution statistics ✓

### 2. Use in Your Code

```python
from src.data import RNADataset, LabelEncoder
from torch_geometric.loader import DataLoader

# Initialize
label_encoder = LabelEncoder()

# Load datasets
train_dataset = RNADataset(
    root='data/processed/train',
    fold_labels_path='data/splits/train_labels.json',
    rfam_types_path='rfam/rfam_types_full.pkl',
    st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
    label_encoder=label_encoder,
)

# Create loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Iterate
for batch in train_loader:
    # batch.x: [total_nodes, 14] node features
    # batch.edge_index: [2, total_edges]
    # batch.edge_attr: [total_edges, 1] edge types
    # batch.y: [batch_size] labels
    pass
```

### 3. Explore Interactively

Open `notebooks/data_pipeline_demo.ipynb` to:
- Load and inspect samples
- Visualize class distributions
- Analyze node and edge features
- See sequence length statistics

## Key Features

### Graph Representation
Each RNA becomes a graph where:
- **Nodes** = nucleotides (14-dim features)
- **Edges** = backbone connections + base pairs (typed)
- **Label** = RFAM meta-type (23 classes)

### Node Features (14 dimensions)
1. Nucleotide type (A/U/G/C/N) - 5 dims
2. Structural annotation (E/S/H/I/M/B/X) - 7 dims
3. Pseudoknot indicator - 1 dim
4. Position encoding - 1 dim

### Edge Types
- Type 0: Backbone (sequential)
- Type 1: Base pair `()`
- Type 2-4: Pseudoknots `[]{}` `<>`

### Dataset Statistics
- **Total**: 22,088 samples
- **Classes**: 23 RFAM meta-types
- **Split**: 80/10/10 (stratified)
- **Largest class**: Gene; miRNA (3,640 samples)
- **Smallest class**: Gene; lncRNA (15 samples)

## Next Steps

1. **Define GNN architecture** (GCN, GAT, GIN, etc.)
2. **Handle class imbalance** (weighted loss, focal loss)
3. **Train model** with your preferred framework
4. **Evaluate** on validation set
5. **Test** on held-out test set

## Example Training Loop

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class RNAGNN(torch.nn.Module):
    def __init__(self, num_node_features=14, num_classes=23):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        return self.fc(x)

# Train
model = RNAGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
```

## Documentation

- **Full docs**: See `DATA_PIPELINE.md`
- **Demo notebook**: See `notebooks/data_pipeline_demo.ipynb`
- **Module reference**: See docstrings in `src/data/`

## Success Metrics

✓ All 22,088 samples successfully loaded
✓ Stratified splits maintain class balance
✓ Graph conversion handles all edge types
✓ PyG DataLoader batching works correctly
✓ Test suite passes all checks

**The pipeline is ready for GNN training!** 🎉
