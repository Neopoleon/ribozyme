# RNA GNN Training Guide

## Quick Start

### 1. Train a Model (Default: GCN)

```bash
python train.py
```

This will:
- Train a GCN model with default hyperparameters
- Use class weights to handle imbalance
- Save results to `results/runs/gcn_<timestamp>/`
- Evaluate on test set and generate metrics

### 2. Visualize Results

```bash
python visualize_results.py results/runs/gcn_<timestamp>/
```

This generates:
- Training history plots (loss, accuracy, F1)
- Confusion matrices
- Per-class performance metrics
- Confidence analysis

## Training Options

### Model Architectures

**GCN (Graph Convolutional Network)** - Default, fast and effective
```bash
python train.py --model gcn
```

**GAT (Graph Attention Network)** - Uses attention mechanism
```bash
python train.py --model gat
```

**GIN (Graph Isomorphism Network)** - Most expressive
```bash
python train.py --model gin
```

### Hyperparameters

```bash
python train.py \
  --model gcn \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.001 \
  --use-class-weights
```

**Key Parameters:**
- `--hidden-dim`: Size of hidden layers (default: 128)
- `--num-layers`: Number of GNN layers (default: 3)
- `--dropout`: Dropout rate for regularization (default: 0.3)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--use-class-weights`: Use inverse frequency weighting for class imbalance
- `--pooling`: Graph pooling method: 'mean' or 'max' (default: mean)

### Example Configurations

**Fast Training (Lower quality)**
```bash
python train.py \
  --model gcn \
  --hidden-dim 64 \
  --num-layers 2 \
  --epochs 50 \
  --batch-size 64
```

**High Quality (Slower)**
```bash
python train.py \
  --model gin \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.4 \
  --epochs 200 \
  --batch-size 32 \
  --use-class-weights
```

**GPU Training (if available)**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 128
```

## Understanding the Output

### During Training

```
Epoch   1/100 | Train Loss: 3.0234 Acc: 0.2145 F1: 0.1823 | Val Loss: 2.8912 Acc: 0.2567 F1: 0.2234
Epoch   2/100 | Train Loss: 2.7456 Acc: 0.3012 F1: 0.2789 | Val Loss: 2.6234 Acc: 0.3234 F1: 0.3012
  → New best model saved (Val F1: 0.3012)
```

### After Training

```
Test Results:
  Loss: 2.4123
  Accuracy: 0.5678
  F1 Score: 0.5234

Classification Report:
                                      precision    recall  f1-score   support
Cis-reg;                                  0.45      0.52      0.48       219
Gene; miRNA;                              0.78      0.85      0.81       364
...
```

### Files Generated

```
results/runs/gcn_20231209_143022/
├── config.json                  # Training configuration
├── best_model.pt                # Best model checkpoint
├── history.json                 # Training metrics per epoch
├── test_results.txt             # Detailed test results
├── test_predictions.npy         # Test predictions
├── test_labels.npy              # Test true labels
├── test_probabilities.npy       # Test class probabilities
├── confusion_matrix.npy         # Confusion matrix
└── visualizations/
    ├── training_history.png     # Loss/accuracy/F1 curves
    ├── confusion_matrix.png     # Confusion matrix heatmap
    ├── confusion_matrix_normalized.png
    ├── per_class_metrics.png    # Per-class precision/recall/F1
    ├── confidence_analysis.png  # Prediction confidence
    └── summary.txt              # Results summary
```

## Interpreting Results

### Metrics Explained

**Accuracy**: Overall correctness (# correct / # total)
- Good for balanced datasets
- Can be misleading with class imbalance

**F1 Score (Weighted)**: Harmonic mean of precision and recall, weighted by class support
- Better for imbalanced datasets
- Main metric to optimize

**Precision**: Of predicted class X, how many are actually class X?
- High precision = few false positives

**Recall**: Of all true class X samples, how many did we predict correctly?
- High recall = few false negatives

### Confusion Matrix

- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- Look for systematic errors (e.g., Gene; miRNA confused with Gene; sRNA)

### Per-Class Metrics

Classes with low F1 typically have:
1. **Low support** (few training samples)
2. **Similar features** to other classes
3. **Noisy labels** or ambiguous examples

## Tips for Better Performance

### 1. Handle Class Imbalance

Always use `--use-class-weights` for this dataset:
```bash
python train.py --use-class-weights
```

### 2. Regularization

If overfitting (train >> val performance):
- Increase dropout: `--dropout 0.5`
- Add weight decay: `--weight-decay 1e-4`
- Use fewer layers: `--num-layers 2`

### 3. Model Capacity

If underfitting (poor train performance):
- Increase hidden dim: `--hidden-dim 256`
- Add more layers: `--num-layers 4`
- Try different architecture: `--model gin`

### 4. Training Time

If training is too slow:
- Reduce batch size: `--batch-size 16` (uses less memory but slower)
- Increase batch size: `--batch-size 128` (faster but needs more memory)
- Use fewer layers: `--num-layers 2`

### 5. Hyperparameter Search

Run multiple experiments with different settings:
```bash
# Try different models
for model in gcn gat gin; do
  python train.py --model $model --use-class-weights
done

# Try different hidden dimensions
for dim in 64 128 256; do
  python train.py --hidden-dim $dim --use-class-weights
done
```

## Advanced: Using the Trained Model

### Load and Use Model for Inference

```python
import torch
from src.models import get_model
from src.data import RNADataset, LabelEncoder

# Load model
model = get_model('gcn', num_node_features=14, num_classes=23)
checkpoint = torch.load('results/runs/gcn_20231209_143022/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
label_encoder = LabelEncoder()
dataset = RNADataset(...)

# Predict
with torch.no_grad():
    data = dataset[0]
    output = model(data)
    predicted_class = output.argmax().item()
    predicted_label = label_encoder.decode(predicted_class)

print(f"Predicted: {predicted_label}")
```

## Troubleshooting

**Out of memory error**
- Reduce batch size: `--batch-size 16`
- Reduce hidden dim: `--hidden-dim 64`
- Use CPU only: `CUDA_VISIBLE_DEVICES='' python train.py`

**Training too slow**
- Increase batch size: `--batch-size 128`
- Use fewer layers: `--num-layers 2`
- Reduce epochs: `--epochs 50`

**Poor performance**
- Enable class weights: `--use-class-weights`
- Try different model: `--model gin`
- Increase model capacity: `--hidden-dim 256 --num-layers 4`

**Model not improving**
- Check learning rate: try `--lr 0.0001` or `--lr 0.01`
- Reduce dropout: `--dropout 0.1`
- Check data loading: run `test_data_pipeline.py`

## Next Steps

1. **Experiment with architectures**: Try GCN, GAT, GIN
2. **Tune hyperparameters**: Grid search over hidden_dim, num_layers, dropout
3. **Ensemble models**: Combine predictions from multiple models
4. **Feature engineering**: Add more node features or edge attributes
5. **Data augmentation**: Augment RNA structures during training

## Example Workflow

```bash
# 1. Test data pipeline
python test_data_pipeline.py

# 2. Quick training run to verify everything works
python train.py --epochs 10 --batch-size 32

# 3. Full training with best settings
python train.py \
  --model gin \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 32 \
  --use-class-weights

# 4. Visualize results
python visualize_results.py results/runs/<your_run_name>

# 5. Compare with other architectures
python train.py --model gat --use-class-weights
python train.py --model gcn --use-class-weights
```

Good luck with your RNA classification! 🧬
