# RNA Classifier - Graph Neural Network

A complete machine learning pipeline for classifying RNA structures using Graph Neural Networks (GNNs).

## 🎯 What This Does

Classifies RNA molecules into **23 different types** based on their sequence and structure, including:
- Gene; miRNA
- Gene; ribozyme
- Cis-reg; riboswitch
- Gene; tRNA
- And 19 other RFAM meta-types

## 📊 Dataset

- **22,088 RNA structures** from the bpRNA-1m dataset
- **23 classes** from RFAM taxonomy
- **Stratified 80/10/10** train/val/test split
- Graphs with 14-dimensional node features per nucleotide

## 🚀 Quick Start

### 1. Quick Test (3 minutes)
```bash
python quick_test.py
```

### 2. Train a Model
```bash
# CPU (2-4 hours)
python train.py --epochs 50 --use-class-weights

# GPU (30-60 minutes)
python train.py --epochs 100 --hidden-dim 256 --batch-size 128 --use-class-weights
```

### 3. Visualize Results
```bash
python visualize_results.py results/runs/<your_run>/
```

## 📚 Documentation

- **[HOW_TO_RUN.md](HOW_TO_RUN.md)** - Start here! Complete guide to training and results
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training options and hyperparameters
- **[DATA_PIPELINE.md](DATA_PIPELINE.md)** - How the data pipeline works
- **[QUICKSTART.md](QUICKSTART.md)** - Data pipeline quick reference

## 🏗️ Project Structure

```
ribozyme/
├── src/
│   ├── data/                    # Data loading and processing
│   │   ├── parser.py           # Parse .st RNA structure files
│   │   ├── graph_builder.py    # Convert RNA to graphs
│   │   ├── label_encoder.py    # Encode meta-type labels
│   │   ├── rna_dataset.py      # PyTorch Geometric Dataset
│   │   └── split_dataset.py    # Create train/val/test splits
│   └── models/                  # GNN model architectures
│       └── gnn.py              # GCN, GAT, GIN models
├── data/
│   ├── splits/                  # Train/val/test splits (generated)
│   └── unzipped/                # Raw .st structure files
├── rfam/
│   └── rfam_types_full.pkl     # RFAM meta-type mappings
├── results/
│   └── runs/                    # Training outputs (generated)
├── train.py                     # Main training script
├── visualize_results.py         # Generate result visualizations
├── quick_test.py                # Quick pipeline test
└── test_data_pipeline.py        # Test data loading
```

## 🎨 Features

### Data Pipeline
- ✅ Parses .st RNA structure files
- ✅ Converts RNA to graph representation
- ✅ 14-dimensional node features (nucleotide + structure + position)
- ✅ Typed edges (backbone + 4 types of base pairs)
- ✅ Handles class imbalance with weighting
- ✅ Stratified train/val/test splits

### Models
- ✅ **GCN** - Graph Convolutional Network
- ✅ **GAT** - Graph Attention Network
- ✅ **GIN** - Graph Isomorphism Network
- ✅ Configurable depth, width, dropout
- ✅ Automatic GPU detection

### Training
- ✅ Class-weighted loss for imbalance
- ✅ Learning rate scheduling
- ✅ Early stopping via best model saving
- ✅ Comprehensive metrics (accuracy, F1, precision, recall)
- ✅ Automatic checkpointing

### Visualization
- ✅ Training curves (loss, accuracy, F1)
- ✅ Confusion matrices
- ✅ Per-class performance metrics
- ✅ Prediction confidence analysis

## 📈 Example Results

After training, you'll get:

**Training History:**
![Training curves showing loss, accuracy, and F1 score over epochs]

**Confusion Matrix:**
![Heatmap showing prediction accuracy per class]

**Per-Class Metrics:**
![Bar charts of F1, precision, recall per RNA type]

## 🔬 Graph Representation

### Nodes (Nucleotides)
Each nucleotide becomes a node with 14 features:
- **Nucleotide type** (A/U/G/C/N) - 5 dims (one-hot)
- **Structure** (E/S/H/I/M/B/X) - 7 dims (one-hot)
- **Pseudoknot** (binary) - 1 dim
- **Position** (normalized) - 1 dim

### Edges (Bonds)
- **Type 0**: Backbone (sequential)
- **Type 1**: Base pair `()`
- **Types 2-4**: Pseudoknots `[]` `{}` `<>`

### Labels
23 hierarchical RFAM meta-types:
```
Cis-reg; riboswitch;
Gene; miRNA;
Gene; ribozyme;
Gene; tRNA;
...
```

## 💻 Requirements

```bash
pip install torch torch-geometric scikit-learn matplotlib seaborn numpy
```

## 🎓 Usage Examples

### Basic Training
```bash
python train.py
```

### Advanced Training
```bash
python train.py \
  --model gin \
  --hidden-dim 256 \
  --num-layers 4 \
  --epochs 100 \
  --batch-size 32 \
  --dropout 0.3 \
  --use-class-weights
```

### Load Trained Model
```python
from src.models import get_model
import torch

model = get_model('gcn', num_node_features=14, num_classes=23)
checkpoint = torch.load('results/runs/gcn_20231209_143022/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Make Predictions
```python
from src.data import RNADataset, LabelEncoder

dataset = RNADataset(...)
label_encoder = LabelEncoder()

with torch.no_grad():
    data = dataset[0]
    output = model(data)
    prediction = output.argmax().item()
    rna_type = label_encoder.decode(prediction)

print(f"Predicted RNA type: {rna_type}")
```

## 🔧 Customization

### Add New Model
```python
# src/models/gnn.py
class MyCustomGNN(nn.Module):
    def __init__(self, num_node_features=14, num_classes=23):
        super().__init__()
        # Your architecture here

    def forward(self, data):
        # Your forward pass
        return output
```

### Modify Node Features
Edit `src/data/graph_builder.py` to add/remove features.

### Change Label Hierarchy
Edit `src/data/label_encoder.py` to use different class granularity.

## 📊 Performance Expectations

**Baseline (GCN, default settings):**
- Accuracy: ~55-65%
- F1 Score: ~50-60%

**Optimized (GIN, tuned hyperparameters):**
- Accuracy: ~65-75%
- F1 Score: ~60-70%

Note: Performance varies by class. Common classes (Gene; miRNA) have higher F1, rare classes (Gene; lncRNA) have lower F1.

## 🛠️ Development

### Run Tests
```bash
# Test data pipeline
python test_data_pipeline.py

# Quick training test
python quick_test.py
```

### Create New Splits
```bash
python src/data/split_dataset.py
```

## 📝 Citation

If you use this code or the bpRNA dataset, please cite:

```
# Add your citation here
```

## 📄 License

[Your License]

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

[Your Contact Information]

---

**Ready to classify some RNA? Start with `python quick_test.py`!** 🧬
