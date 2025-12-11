"""Analyze and visualize scaling law experiment results

This script aggregates results from multiple scaling law runs and generates
comprehensive visualizations showing how model performance scales with data size.

Usage:
    python analyze_scaling_law.py [results_dir]

    # Default: analyzes results/scaling_law/
    python analyze_scaling_law.py

    # Custom directory:
    python analyze_scaling_law.py results/my_scaling_experiment/
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_scaling_results(results_dir: Path) -> pd.DataFrame:
    """Load all scaling law results into a DataFrame"""
    results = []

    # Find all summary.json files
    for summary_file in results_dir.rglob('summary.json'):
        with open(summary_file) as f:
            data = json.load(f)
            results.append(data)

    if not results:
        raise ValueError(f"No summary.json files found in {results_dir}")

    df = pd.DataFrame(results)
    return df


def plot_scaling_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance metrics vs. data fraction"""
    # Aggregate across seeds
    agg_df = df.groupby('data_fraction').agg({
        'num_train_samples': 'mean',
        'test_acc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std'],
        'training_time_seconds': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                      for col in agg_df.columns]

    # Sort by data fraction
    agg_df = agg_df.sort_values('data_fraction')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scaling Law: Model Performance vs. Training Data Size',
                 fontsize=16, fontweight='bold')

    # Plot 1: Accuracy vs. Data Fraction
    ax = axes[0, 0]
    ax.errorbar(agg_df['data_fraction'] * 100,
                agg_df['test_acc_mean'],
                yerr=agg_df['test_acc_std'],
                marker='o', capsize=5, capthick=2, linewidth=2,
                label='Test Accuracy')
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs. Data Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])

    # Plot 2: F1 Score vs. Data Fraction
    ax = axes[0, 1]
    ax.errorbar(agg_df['data_fraction'] * 100,
                agg_df['test_f1_mean'],
                yerr=agg_df['test_f1_std'],
                marker='o', capsize=5, capthick=2, linewidth=2,
                color='green', label='Test F1')
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score vs. Data Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])

    # Plot 3: Precision & Recall vs. Data Fraction
    ax = axes[1, 0]
    ax.errorbar(agg_df['data_fraction'] * 100,
                agg_df['test_precision_mean'],
                yerr=agg_df['test_precision_std'],
                marker='s', capsize=5, capthick=2, linewidth=2,
                label='Precision')
    ax.errorbar(agg_df['data_fraction'] * 100,
                agg_df['test_recall_mean'],
                yerr=agg_df['test_recall_std'],
                marker='^', capsize=5, capthick=2, linewidth=2,
                label='Recall')
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision & Recall vs. Data Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])

    # Plot 4: Training Time vs. Data Fraction
    ax = axes[1, 1]
    ax.errorbar(agg_df['data_fraction'] * 100,
                agg_df['training_time_seconds_mean'] / 60,
                yerr=agg_df['training_time_seconds_std'] / 60,
                marker='o', capsize=5, capthick=2, linewidth=2,
                color='red', label='Training Time')
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time vs. Data Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'scaling_curves.png'}")
    plt.close()


def plot_scaling_curves_logscale(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance metrics vs. number of samples (log scale)"""
    agg_df = df.groupby('data_fraction').agg({
        'num_train_samples': 'mean',
        'test_acc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
    }).reset_index()

    agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                      for col in agg_df.columns]
    agg_df = agg_df.sort_values('num_train_samples')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Scaling Law: Performance vs. Number of Training Samples (Log Scale)',
                 fontsize=16, fontweight='bold')

    # Accuracy
    ax = axes[0]
    ax.errorbar(agg_df['num_train_samples'],
                agg_df['test_acc_mean'],
                yerr=agg_df['test_acc_std'],
                marker='o', capsize=5, capthick=2, linewidth=2,
                label='Test Accuracy')
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs. Sample Count', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_ylim([0, 1])

    # F1 Score
    ax = axes[1]
    ax.errorbar(agg_df['num_train_samples'],
                agg_df['test_f1_mean'],
                yerr=agg_df['test_f1_std'],
                marker='o', capsize=5, capthick=2, linewidth=2,
                color='green', label='Test F1')
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 vs. Sample Count', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_curves_logscale.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'scaling_curves_logscale.png'}")
    plt.close()


def plot_overfitting_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot train-val gap (overfitting indicators)"""
    agg_df = df.groupby('data_fraction').agg({
        'train_val_acc_gap': ['mean', 'std'],
        'train_val_f1_gap': ['mean', 'std'],
    }).reset_index()

    agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                      for col in agg_df.columns]
    agg_df = agg_df.sort_values('data_fraction')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Overfitting Analysis: Train-Val Gap vs. Data Size',
                 fontsize=16, fontweight='bold')

    # Accuracy gap
    ax = axes[0]
    ax.errorbar(agg_df['data_fraction'] * 100,
                agg_df['train_val_acc_gap_mean'],
                yerr=agg_df['train_val_acc_gap_std'],
                marker='o', capsize=5, capthick=2, linewidth=2,
                color='orange', label='Train-Val Acc Gap')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Gap')
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Train - Val Accuracy', fontsize=12)
    ax.set_title('Accuracy Gap (Higher = More Overfitting)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # F1 gap
    ax = axes[1]
    ax.errorbar(agg_df['data_fraction'] * 100,
                agg_df['train_val_f1_gap_mean'],
                yerr=agg_df['train_val_f1_gap_std'],
                marker='o', capsize=5, capthick=2, linewidth=2,
                color='purple', label='Train-Val F1 Gap')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Gap')
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Train - Val F1', fontsize=12)
    ax.set_title('F1 Gap (Higher = More Overfitting)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'overfitting_analysis.png'}")
    plt.close()


def plot_variance_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot variance across seeds for each data fraction"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Variance Analysis Across Seeds', fontsize=16, fontweight='bold')

    # Box plot for accuracy
    ax = axes[0]
    df_sorted = df.sort_values('data_fraction')
    df_sorted['data_fraction_pct'] = df_sorted['data_fraction'] * 100
    sns.boxplot(data=df_sorted, x='data_fraction_pct', y='test_acc', ax=ax)
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Accuracy Distribution Across Seeds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Box plot for F1
    ax = axes[1]
    sns.boxplot(data=df_sorted, x='data_fraction_pct', y='test_f1', ax=ax)
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Test F1 Score', fontsize=12)
    ax.set_title('F1 Distribution Across Seeds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'variance_analysis.png'}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a summary table of results"""
    summary = df.groupby('data_fraction').agg({
        'num_train_samples': 'mean',
        'test_acc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std'],
        'train_val_acc_gap': ['mean', 'std'],
        'training_time_seconds': ['mean', 'std'],
        'best_epoch': ['mean', 'std'],
    }).reset_index()

    # Flatten and rename
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns]
    summary = summary.sort_values('data_fraction')

    # Format for display
    summary['num_train_samples'] = summary['num_train_samples'].round(0).astype(int)
    summary['data_fraction_pct'] = (summary['data_fraction'] * 100).round(1)

    # Reorder columns
    display_cols = [
        'data_fraction_pct',
        'num_train_samples',
        'test_acc_mean', 'test_acc_std',
        'test_f1_mean', 'test_f1_std',
        'test_precision_mean', 'test_precision_std',
        'test_recall_mean', 'test_recall_std',
        'train_val_acc_gap_mean', 'train_val_acc_gap_std',
        'training_time_seconds_mean', 'training_time_seconds_std',
        'best_epoch_mean', 'best_epoch_std',
    ]
    summary_display = summary[display_cols]

    # Save to CSV
    summary_display.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"Saved: {output_dir / 'summary_table.csv'}")

    # Create formatted text table
    with open(output_dir / 'summary_table.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("SCALING LAW SUMMARY TABLE\n")
        f.write("="*100 + "\n\n")

        for _, row in summary.iterrows():
            f.write(f"Training Data: {row['data_fraction']*100:.1f}% ({row['num_train_samples']:.0f} samples)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Test Accuracy:    {row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}\n")
            f.write(f"  Test F1:          {row['test_f1_mean']:.4f} ± {row['test_f1_std']:.4f}\n")
            f.write(f"  Test Precision:   {row['test_precision_mean']:.4f} ± {row['test_precision_std']:.4f}\n")
            f.write(f"  Test Recall:      {row['test_recall_mean']:.4f} ± {row['test_recall_std']:.4f}\n")
            f.write(f"  Train-Val Gap:    {row['train_val_acc_gap_mean']:+.4f} ± {row['train_val_acc_gap_std']:.4f}\n")
            f.write(f"  Training Time:    {row['training_time_seconds_mean']/60:.2f} ± {row['training_time_seconds_std']/60:.2f} min\n")
            f.write(f"  Best Epoch:       {row['best_epoch_mean']:.1f} ± {row['best_epoch_std']:.1f}\n")
            f.write("\n")

    print(f"Saved: {output_dir / 'summary_table.txt'}")

    # Print to console
    print("\n" + "="*100)
    print("SCALING LAW SUMMARY")
    print("="*100)
    print(summary_display.to_string(index=False))
    print("="*100 + "\n")


def main():
    """Main analysis function"""
    # Get results directory from command line or use default
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path('results/scaling_law')

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print(f"Usage: python {sys.argv[0]} [results_dir]")
        sys.exit(1)

    print(f"\nAnalyzing scaling law results from: {results_dir}")
    print("="*80)

    # Load all results
    df = load_scaling_results(results_dir)
    print(f"\nLoaded {len(df)} experiment runs")
    print(f"Data fractions: {sorted(df['data_fraction'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    # Create analysis output directory
    analysis_dir = results_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    print(f"\nSaving analysis to: {analysis_dir}")

    # Generate all visualizations and summaries
    print("\nGenerating visualizations...")
    plot_scaling_curves(df, analysis_dir)
    plot_scaling_curves_logscale(df, analysis_dir)
    plot_overfitting_analysis(df, analysis_dir)
    plot_variance_analysis(df, analysis_dir)

    print("\nGenerating summary tables...")
    generate_summary_table(df, analysis_dir)

    # Save raw aggregated data
    df.to_csv(analysis_dir / 'all_results.csv', index=False)
    print(f"Saved: {analysis_dir / 'all_results.csv'}")

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"All results saved to: {analysis_dir}")
    print("="*80 + "\n")

    # Key insights
    print("KEY INSIGHTS:")
    print("-" * 80)

    # Find minimum data fraction with >90% of max performance
    agg_df = df.groupby('data_fraction')['test_f1'].mean().sort_values(ascending=False)
    max_f1 = agg_df.iloc[0]
    threshold_90 = max_f1 * 0.9

    for frac in sorted(df['data_fraction'].unique()):
        mean_f1 = df[df['data_fraction'] == frac]['test_f1'].mean()
        if mean_f1 >= threshold_90:
            num_samples = df[df['data_fraction'] == frac]['num_train_samples'].mean()
            print(f"Minimum data for 90% of max F1: {frac*100:.1f}% ({num_samples:.0f} samples)")
            print(f"  F1 at {frac*100:.1f}%: {mean_f1:.4f} (max: {max_f1:.4f})")
            break

    print("-" * 80 + "\n")


if __name__ == '__main__':
    main()