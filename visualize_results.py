"""Visualize training results and model performance"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.data import LabelEncoder


def plot_training_history(history_path, output_path):
    """Plot training and validation metrics over epochs"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(history['train_f1'], label='Train', linewidth=2)
    axes[2].plot(history['val_f1'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Training and Validation F1 Score', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {output_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path, normalize=False):
    """Plot confusion matrix"""
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        square=True,
    )
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")
    plt.close()


def plot_per_class_metrics(labels, predictions, class_names, output_path):
    """Plot per-class precision, recall, and F1 score"""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    # Sort by F1 score
    indices = np.argsort(f1)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # F1 Score
    axes[0, 0].barh(range(len(f1)), f1[indices], color='steelblue')
    axes[0, 0].set_yticks(range(len(f1)))
    axes[0, 0].set_yticklabels([class_names[i] for i in indices], fontsize=8)
    axes[0, 0].set_xlabel('F1 Score', fontsize=12)
    axes[0, 0].set_title('Per-Class F1 Score', fontsize=14)
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].invert_yaxis()

    # Precision
    axes[0, 1].barh(range(len(precision)), precision[indices], color='green', alpha=0.7)
    axes[0, 1].set_yticks(range(len(precision)))
    axes[0, 1].set_yticklabels([class_names[i] for i in indices], fontsize=8)
    axes[0, 1].set_xlabel('Precision', fontsize=12)
    axes[0, 1].set_title('Per-Class Precision', fontsize=14)
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].invert_yaxis()

    # Recall
    axes[1, 0].barh(range(len(recall)), recall[indices], color='orange', alpha=0.7)
    axes[1, 0].set_yticks(range(len(recall)))
    axes[1, 0].set_yticklabels([class_names[i] for i in indices], fontsize=8)
    axes[1, 0].set_xlabel('Recall', fontsize=12)
    axes[1, 0].set_title('Per-Class Recall', fontsize=14)
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].invert_yaxis()

    # Support (number of samples)
    axes[1, 1].barh(range(len(support)), support[indices], color='purple', alpha=0.7)
    axes[1, 1].set_yticks(range(len(support)))
    axes[1, 1].set_yticklabels([class_names[i] for i in indices], fontsize=8)
    axes[1, 1].set_xlabel('Number of Samples', fontsize=12)
    axes[1, 1].set_title('Per-Class Support', fontsize=14)
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-class metrics to {output_path}")
    plt.close()


def plot_top_predictions(probabilities, labels, class_names, output_path, top_k=10):
    """Plot confidence distribution for top-k most confident predictions"""
    # Get max probability for each prediction
    max_probs = np.max(probabilities, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    correct = predicted_classes == labels

    # Sort by confidence
    sorted_indices = np.argsort(max_probs)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top confident correct predictions
    correct_indices = sorted_indices[correct[sorted_indices]][:top_k]
    axes[0].barh(range(len(correct_indices)), max_probs[correct_indices], color='green', alpha=0.7)
    axes[0].set_yticks(range(len(correct_indices)))
    axes[0].set_yticklabels([
        f"{class_names[predicted_classes[i]]} (True: {class_names[labels[i]]})"
        for i in correct_indices
    ], fontsize=9)
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_title(f'Top {top_k} Most Confident Correct Predictions', fontsize=14)
    axes[0].set_xlim([0, 1])
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()

    # Top confident incorrect predictions
    incorrect_indices = sorted_indices[~correct[sorted_indices]][:top_k]
    if len(incorrect_indices) > 0:
        axes[1].barh(range(len(incorrect_indices)), max_probs[incorrect_indices], color='red', alpha=0.7)
        axes[1].set_yticks(range(len(incorrect_indices)))
        axes[1].set_yticklabels([
            f"{class_names[predicted_classes[i]]} (True: {class_names[labels[i]]})"
            for i in incorrect_indices
        ], fontsize=9)
        axes[1].set_xlabel('Confidence', fontsize=12)
        axes[1].set_title(f'Top {top_k} Most Confident Incorrect Predictions', fontsize=14)
        axes[1].set_xlim([0, 1])
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confidence analysis to {output_path}")
    plt.close()


def main(args):
    results_dir = Path(args.results_dir)

    # Load label encoder
    label_encoder = LabelEncoder()
    class_names = [label_encoder.decode(i) for i in range(label_encoder.num_classes)]

    # Create visualization output directory
    vis_dir = results_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    print(f"Generating visualizations for: {results_dir}")

    # 1. Training history
    if (results_dir / 'history.json').exists():
        plot_training_history(
            results_dir / 'history.json',
            vis_dir / 'training_history.png'
        )

    # Load test results
    test_labels = np.load(results_dir / 'test_labels.npy')
    test_preds = np.load(results_dir / 'test_predictions.npy')
    test_probs = np.load(results_dir / 'test_probabilities.npy')

    # 2. Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, class_names, vis_dir / 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, vis_dir / 'confusion_matrix_normalized.png', normalize=True)

    # 3. Per-class metrics
    plot_per_class_metrics(test_labels, test_preds, class_names, vis_dir / 'per_class_metrics.png')

    # 4. Confidence analysis
    plot_top_predictions(test_probs, test_labels, class_names, vis_dir / 'confidence_analysis.png')

    # 5. Generate summary statistics
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

For detailed per-class metrics, see: {results_dir}/test_results.txt
"""

    print(summary)

    # Save summary
    with open(vis_dir / 'summary.txt', 'w') as f:
        f.write(summary)

    print(f"\n✓ Visualizations complete! Saved to: {vis_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize RNA GNN training results')
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    args = parser.parse_args()
    main(args)
