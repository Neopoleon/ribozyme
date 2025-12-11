"""Compare scaling law results across GIN, GAT, and GCN models

This script loads results from all three model architectures and creates
side-by-side comparisons to understand which model is most data-efficient.

Usage:
    python compare_scaling_models.py

    # Custom directories:
    python compare_scaling_models.py --gin results/scaling_law_gin/ \
                                      --gat results/scaling_law_gat/ \
                                      --gcn results/scaling_law_gcn/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_model_results(results_dir: Path, model_name: str) -> pd.DataFrame:
    """Load all scaling law results for a given model"""
    results = []

    for summary_file in results_dir.rglob('summary.json'):
        with open(summary_file) as f:
            data = json.load(f)
            data['model'] = model_name
            results.append(data)

    if not results:
        print(f"Warning: No results found for {model_name} in {results_dir}")
        return pd.DataFrame()

    return pd.DataFrame(results)


def plot_model_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive comparison plots across models"""

    # Aggregate across seeds
    agg_df = df.groupby(['model', 'data_fraction']).agg({
        'num_train_samples': 'mean',
        'test_acc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std'],
        'train_val_acc_gap': ['mean', 'std'],
        'train_val_f1_gap': ['mean', 'std'],
        'training_time_seconds': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                      for col in agg_df.columns]

    # Main comparison plot: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scaling Law Comparison: GIN vs GAT vs GCN',
                 fontsize=18, fontweight='bold')

    models = agg_df['model'].unique()
    colors = {'GIN': '#2E86AB', 'GAT': '#A23B72', 'GCN': '#F18F01'}
    markers = {'GIN': 'o', 'GAT': 's', 'GCN': '^'}

    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    for model in models:
        model_data = agg_df[agg_df['model'] == model].sort_values('data_fraction')
        ax.errorbar(
            model_data['data_fraction'] * 100,
            model_data['test_acc_mean'],
            yerr=model_data['test_acc_std'],
            marker=markers[model], label=model, capsize=5,
            capthick=2, linewidth=2, markersize=8,
            color=colors[model]
        )
    ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs. Data Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])

    # Plot 2: F1 Score comparison
    ax = axes[0, 1]
    for model in models:
        model_data = agg_df[agg_df['model'] == model].sort_values('data_fraction')
        ax.errorbar(
            model_data['data_fraction'] * 100,
            model_data['test_f1_mean'],
            yerr=model_data['test_f1_std'],
            marker=markers[model], label=model, capsize=5,
            capthick=2, linewidth=2, markersize=8,
            color=colors[model]
        )
    ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score vs. Data Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])

    # Plot 3: Overfitting comparison (train-val gap)
    ax = axes[1, 0]
    for model in models:
        model_data = agg_df[agg_df['model'] == model].sort_values('data_fraction')
        ax.errorbar(
            model_data['data_fraction'] * 100,
            model_data['train_val_f1_gap_mean'],
            yerr=model_data['train_val_f1_gap_std'],
            marker=markers[model], label=model, capsize=5,
            capthick=2, linewidth=2, markersize=8,
            color=colors[model]
        )
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Gap')
    ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train-Val F1 Gap', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Analysis (Higher = More Overfitting)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Plot 4: Training time comparison
    ax = axes[1, 1]
    for model in models:
        model_data = agg_df[agg_df['model'] == model].sort_values('data_fraction')
        ax.errorbar(
            model_data['data_fraction'] * 100,
            model_data['training_time_seconds_mean'] / 60,
            yerr=model_data['training_time_seconds_std'] / 60,
            marker=markers[model], label=model, capsize=5,
            capthick=2, linewidth=2, markersize=8,
            color=colors[model]
        )
    ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time vs. Data Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison.png'}")
    plt.close()


def plot_performance_gap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance degradation relative to 100% data"""

    # Calculate performance relative to 100% baseline for each model
    results = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        # Get 100% performance for this model
        baseline = model_df[model_df['data_fraction'] == 1.0].groupby('model').agg({
            'test_acc': 'mean',
            'test_f1': 'mean',
        })

        if baseline.empty:
            continue

        baseline_acc = baseline['test_acc'].iloc[0]
        baseline_f1 = baseline['test_f1'].iloc[0]

        # Calculate degradation for each fraction
        for frac in sorted(model_df['data_fraction'].unique()):
            frac_data = model_df[model_df['data_fraction'] == frac]

            acc_mean = frac_data['test_acc'].mean()
            acc_std = frac_data['test_acc'].std()
            f1_mean = frac_data['test_f1'].mean()
            f1_std = frac_data['test_f1'].std()

            results.append({
                'model': model,
                'data_fraction': frac,
                'acc_retention': (acc_mean / baseline_acc) * 100,
                'acc_retention_std': (acc_std / baseline_acc) * 100,
                'f1_retention': (f1_mean / baseline_f1) * 100,
                'f1_retention_std': (f1_std / baseline_f1) * 100,
            })

    results_df = pd.DataFrame(results)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Retention Relative to 100% Data',
                 fontsize=18, fontweight='bold')

    colors = {'GIN': '#2E86AB', 'GAT': '#A23B72', 'GCN': '#F18F01'}
    markers = {'GIN': 'o', 'GAT': 's', 'GCN': '^'}

    # Accuracy retention
    ax = axes[0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model].sort_values('data_fraction')
        ax.errorbar(
            model_data['data_fraction'] * 100,
            model_data['acc_retention'],
            yerr=model_data['acc_retention_std'],
            marker=markers[model], label=model, capsize=5,
            capthick=2, linewidth=2, markersize=8,
            color=colors[model]
        )
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% Baseline')
    ax.axhline(y=90, color='red', linestyle=':', alpha=0.5, label='90% Threshold')
    ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Retention (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Retention', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # F1 retention
    ax = axes[1]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model].sort_values('data_fraction')
        ax.errorbar(
            model_data['data_fraction'] * 100,
            model_data['f1_retention'],
            yerr=model_data['f1_retention_std'],
            marker=markers[model], label=model, capsize=5,
            capthick=2, linewidth=2, markersize=8,
            color=colors[model]
        )
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% Baseline')
    ax.axhline(y=90, color='red', linestyle=':', alpha=0.5, label='90% Threshold')
    ax.set_xlabel('Training Data (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Retention (%)', fontsize=12, fontweight='bold')
    ax.set_title('F1 Retention', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_retention.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_retention.png'}")
    plt.close()


def generate_comparison_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comparison table across models"""

    summary = df.groupby(['model', 'data_fraction']).agg({
        'num_train_samples': 'mean',
        'test_acc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'training_time_seconds': ['mean', 'std'],
    }).reset_index()

    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns]

    # Pivot for easier comparison
    pivot_acc = summary.pivot(index='data_fraction', columns='model', values='test_acc_mean')
    pivot_f1 = summary.pivot(index='data_fraction', columns='model', values='test_f1_mean')
    pivot_time = summary.pivot(index='data_fraction', columns='model', values='training_time_seconds_mean')

    # Save to CSV
    summary.to_csv(output_dir / 'comparison_table.csv', index=False)
    print(f"Saved: {output_dir / 'comparison_table.csv'}")

    # Generate text report
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("MODEL COMPARISON: GIN vs GAT vs GCN\n")
        f.write("="*100 + "\n\n")

        f.write("ACCURACY COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(pivot_acc.to_string() + "\n\n")

        f.write("F1 SCORE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(pivot_f1.to_string() + "\n\n")

        f.write("TRAINING TIME COMPARISON (seconds)\n")
        f.write("-"*80 + "\n")
        f.write(pivot_time.to_string() + "\n\n")

        # Winner analysis
        f.write("="*100 + "\n")
        f.write("WINNER ANALYSIS\n")
        f.write("="*100 + "\n\n")

        for frac in sorted(df['data_fraction'].unique()):
            frac_data = summary[summary['data_fraction'] == frac]
            best_acc_model = frac_data.loc[frac_data['test_acc_mean'].idxmax(), 'model']
            best_f1_model = frac_data.loc[frac_data['test_f1_mean'].idxmax(), 'model']
            fastest_model = frac_data.loc[frac_data['training_time_seconds_mean'].idxmin(), 'model']

            f.write(f"Data Fraction: {frac*100:.1f}%\n")
            f.write(f"  Best Accuracy:  {best_acc_model}\n")
            f.write(f"  Best F1:        {best_f1_model}\n")
            f.write(f"  Fastest:        {fastest_model}\n\n")

    print(f"Saved: {output_dir / 'comparison_report.txt'}")

    # Print to console
    print("\n" + "="*100)
    print("ACCURACY COMPARISON")
    print("="*100)
    print(pivot_acc)
    print("\n" + "="*100)
    print("F1 SCORE COMPARISON")
    print("="*100)
    print(pivot_f1)
    print("\n")


def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description='Compare scaling law results across models')
    parser.add_argument('--gin', type=str, default='results/scaling_law_gin/',
                        help='Path to GIN results directory')
    parser.add_argument('--gat', type=str, default='results/scaling_law_gat/',
                        help='Path to GAT results directory')
    parser.add_argument('--gcn', type=str, default='results/scaling_law_gcn/',
                        help='Path to GCN results directory')
    parser.add_argument('--output', type=str, default='results/scaling_law_comparison/',
                        help='Output directory for comparison results')
    args = parser.parse_args()

    print("\n" + "="*100)
    print("SCALING LAW MODEL COMPARISON")
    print("="*100)

    # Load results from all models
    print("\nLoading results...")
    gin_df = load_model_results(Path(args.gin), 'GIN')
    gat_df = load_model_results(Path(args.gat), 'GAT')
    gcn_df = load_model_results(Path(args.gcn), 'GCN')

    # Combine all results
    all_df = pd.concat([gin_df, gat_df, gcn_df], ignore_index=True)

    if all_df.empty:
        print("Error: No results found. Make sure experiments have been run.")
        return

    print(f"Loaded {len(all_df)} total runs:")
    print(f"  GIN: {len(gin_df)} runs")
    print(f"  GAT: {len(gat_df)} runs")
    print(f"  GCN: {len(gcn_df)} runs")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving comparison to: {output_dir}")

    # Generate visualizations
    print("\nGenerating comparison plots...")
    plot_model_comparison(all_df, output_dir)
    plot_performance_gap(all_df, output_dir)

    # Generate tables
    print("\nGenerating comparison tables...")
    generate_comparison_table(all_df, output_dir)

    # Save combined data
    all_df.to_csv(output_dir / 'all_models_combined.csv', index=False)
    print(f"Saved: {output_dir / 'all_models_combined.csv'}")

    print("\n" + "="*100)
    print("✓ Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
