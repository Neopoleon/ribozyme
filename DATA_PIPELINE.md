# RNA GNN Data Pipeline

Complete data ingestion pipeline for training Graph Neural Networks on RNA structure classification.

## Overview

This pipeline converts RNA structure files (.st format) into graph representations suitable for PyTorch Geometric GNNs. Each RNA molecule becomes a graph where nucleotides are nodes and bonds (backbone + base pairs) are edges.

## Dataset Statistics

- **Total samples**: 22,088 RNA structures
- **Number of classes**: 23 RFAM meta-types
- **Train/Val/Test split**: 80/10/10 (stratified)
  - Train: 17,663 samples (78.4%)
  - Val: 2,198 samples (9.8%)
  - Test: 2,227 samples (9.9%)

## Graph Representation

### Node Features (14 dimensions per nucleotide)
1. **Nucleotide type** (5 dims, one-hot): A, U, G, C, N
2. **Structural annotation** (7 dims, one-hot): E, S, H, I, M, B, X
   - E = External loop
   - S = Stem
   - H = Hairpin loop
   - I = Interior loop
   - M = Multi-branch loop
   - B = Bulge
   - X = Unknown
3. **Pseudoknot indicator** (1 dim, binary): 0 or 1
4. **Position encoding** (1 dim): Normalized position [0, 1]

### Edge Types (encoded in edge_attr)
- **0**: Backbone edge (sequential nucleotide connection)
- **1**: Canonical base pair `()`
- **2**: Pseudoknot bracket `[]`
- **3**: Pseudoknot bracket `{}`
- **4**: Pseudoknot bracket `<>`

### Labels (23 classes)
Full hierarchical RFAM meta-types:
```
1.  Cis-reg;
2.  Cis-reg; IRES;
3.  Cis-reg; frameshift_element;
4.  Cis-reg; leader;
5.  Cis-reg; riboswitch;
6.  Cis-reg; thermoregulator;
7.  Gene;
8.  Gene; CRISPR;
9.  Gene; antisense;
10. Gene; antitoxin;
11. Gene; lncRNA;
12. Gene; miRNA;
13. Gene; rRNA;
14. Gene; ribozyme;
15. Gene; sRNA;
16. Gene; snRNA;
17. Gene; snRNA; snoRNA;
18. Gene; snRNA; snoRNA; CD-box;
19. Gene; snRNA; snoRNA; HACA-box;
20. Gene; snRNA; snoRNA; scaRNA;
21. Gene; snRNA; splicing;
22. Gene; tRNA;
23. Intron;
```

## Usage

### Basic Usage

```python
from src.data import RNADataset, LabelEncoder
from torch_geometric.loader import DataLoader

# Create label encoder
label_encoder = LabelEncoder()

# Create dataset
train_dataset = RNADataset(
    root='data/processed/train',
    fold_labels_path='data/splits/train_labels.json',
    rfam_types_path='rfam/rfam_types_full.pkl',
    st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
    label_encoder=label_encoder,
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

# Iterate over batches
for batch in train_loader:
    # batch.x: node features [total_nodes, 14]
    # batch.edge_index: edges [2, total_edges]
    # batch.edge_attr: edge types [total_edges, 1]
    # batch.y: labels [batch_size]
    # batch.batch: node-to-graph mapping [total_nodes]

    # Your GNN forward pass here
    pass
```

### Creating All Splits

```python
from src.data import RNADataset, LabelEncoder

label_encoder = LabelEncoder()

datasets = {}
for split in ['train', 'val', 'test']:
    datasets[split] = RNADataset(
        root=f'data/processed/{split}',
        fold_labels_path=f'data/splits/{split}_labels.json',
        rfam_types_path='rfam/rfam_types_full.pkl',
        st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
        label_encoder=label_encoder,
    )

train_dataset = datasets['train']
val_dataset = datasets['val']
test_dataset = datasets['test']
```

### Inspecting a Sample

```python
# Get first sample
data = train_dataset[0]

print(f"Nodes: {data.x.shape}")  # [num_nodes, 14]
print(f"Edges: {data.edge_index.shape}")  # [2, num_edges]
print(f"Label: {data.y.item()}")  # integer label
print(f"Meta-type: {label_encoder.decode(data.y.item())}")
print(f"BPRNA ID: {data.bprna_id}")
print(f"Sequence length: {data.sequence_length}")
```

## Module Reference

### `src/data/parser.py`
- `parse_st_file(file_path)`: Parse .st file to extract RNA data
- `extract_rfid(reference_name)`: Extract RFAM ID from reference name
- `load_rfam_types(path)`: Load RFAM type mappings
- `get_meta_type(rfid, rfam_types)`: Get meta-type for RFID

### `src/data/graph_builder.py`
- `rna_to_graph(sequence, dot_bracket, structure, pseudoknot)`: Convert RNA to graph
- `parse_dot_bracket(dot_bracket)`: Parse dot-bracket to extract base pairs
- `encode_nucleotide(nuc)`: One-hot encode nucleotide
- `encode_structure(struct)`: One-hot encode structural annotation

### `src/data/label_encoder.py`
- `LabelEncoder`: Encode/decode meta-type strings to integer labels
  - `encode(meta_type)`: String → integer
  - `decode(label)`: Integer → string
  - `save(path)` / `load(path)`: Persist encoder

### `src/data/rna_dataset.py`
- `RNADataset`: PyTorch Geometric Dataset class
  - Inherits from `torch_geometric.data.Dataset`
  - Loads and converts RNA structures to graphs on-the-fly
  - Compatible with PyG DataLoader

### `src/data/split_dataset.py`
- `create_stratified_splits()`: Create train/val/test splits with stratification
- `save_splits()`: Save splits to JSON files
- `print_split_statistics()`: Display class distribution

## Data Files

### Input Files
- `results/fold{1,2,3,4}_labels.json`: Original fold labels from RFAM API
- `rfam/rfam_types_full.pkl`: RFID → meta-type mapping
- `data/unzipped/bpRNA_1m_90_STAFILES/*.st`: RNA structure files

### Generated Files
- `data/splits/train_labels.json`: Training set labels (17,663 samples)
- `data/splits/val_labels.json`: Validation set labels (2,198 samples)
- `data/splits/test_labels.json`: Test set labels (2,227 samples)

## Testing

Run the test suite to verify the pipeline:

```bash
python test_data_pipeline.py
```

This will:
1. Test dataset creation for all splits
2. Test DataLoader batching
3. Display class distribution statistics

## Class Imbalance

Note: The dataset has class imbalance. The largest class (Gene; miRNA;) has 3,640 samples while the smallest (Gene; lncRNA;) has only 15 samples.

Consider using:
- Weighted loss functions
- Oversampling minority classes
- Focal loss
- Class weights in cross-entropy

## Next Steps

1. **Define your GNN architecture** (e.g., GCN, GAT, GIN)
2. **Create training script** with your model
3. **Handle class imbalance** with weighted loss
4. **Add evaluation metrics** (accuracy, F1, confusion matrix)
5. **Implement early stopping** and checkpointing
6. **Tune hyperparameters** (learning rate, hidden dims, etc.)

## Example GNN Training Loop

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class RNAGNN(torch.nn.Module):
    def __init__(self, num_node_features=14, num_classes=23):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classifier
        x = self.fc(x)
        return x

# Training
model = RNAGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
```

## Credits

Data source: bpRNA-1m dataset with RFAM annotations
