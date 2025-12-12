#!/usr/bin/env python3
"""
Create publication-ready 4x17 ablation heatmap from results_df.csv
Shows F1 scores for GCN, GAT, GIN, and Transformer across feature configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data():
    """Load CSV and return DataFrame"""
    df = pd.read_csv('/home/jeff/ribozyme/notebooks/results_df.csv')
    return df


def create_feature_config_mapping():
    """Define the 15 feature combinations in order (excluding None)"""
    configs = [
        # (N, S, P, PE, Label)
        (True, True, True, True, 'All'),
        (True, True, True, False, 'No PE'),
        (True, True, False, True, 'No P'),
        (True, False, True, True, 'No S'),
        (False, True, True, True, 'No N'),
        (True, True, False, False, 'N+S'),
        (True, False, True, False, 'N+P'),
        (True, False, False, True, 'N+PE'),
        (False, True, True, False, 'S+P'),
        (False, True, False, True, 'S+PE'),
        (False, False, True, True, 'P+PE'),
        (True, False, False, False, 'N only'),
        (False, True, False, False, 'S only'),
        (False, False, True, False, 'P only'),
        (False, False, False, True, 'PE only'),
    ]
    return configs


def get_config_label(row):
    """Map feature flags to config label"""
    n = row['use_nucleotide']
    s = row['use_structure_annotation']
    p = row['use_pseudoknot']
    pe = row['use_position_encoding']

    # Map to one of the 16 configs
    configs = create_feature_config_mapping()
    for config_n, config_s, config_p, config_pe, label in configs:
        if (n == config_n and s == config_s and
            p == config_p and pe == config_pe):
            return label
    return None


def prepare_heatmap_data(df, metric='f1_score'):
    """
    Convert DataFrame to 4x16 matrix
    Returns: data_matrix (4x16 numpy array), row_labels, col_labels
    """
    # Separate GNN runs from transformer and seq-only baselines
    gnn_runs = df[
        (df['model_type'].isin(['gcn', 'gat', 'gin'])) &
        (~df['folder'].str.contains('seq_baseline', na=False))
    ].copy()

    seq_baseline = df[df['folder'].str.contains('gnn_seq_baseline', na=False)].copy()
    transformer = df[df['model_type'] == 'seq_transformer'].copy()

    # Add config labels to GNN runs
    gnn_runs['config_label'] = gnn_runs.apply(get_config_label, axis=1)

    # Average duplicate runs
    gnn_avg = gnn_runs.groupby(['model_type', 'config_label'])[metric].mean().reset_index()

    # Pivot to get architectures as rows, configs as columns
    pivot = gnn_avg.pivot(index='model_type', columns='config_label', values=metric)

    # Ensure all 15 configs are present (fill missing with NaN)
    config_labels = [label for _, _, _, _, label in create_feature_config_mapping()]
    for label in config_labels:
        if label not in pivot.columns:
            pivot[label] = np.nan

    # Reorder columns with seq-only baseline first
    # Create seq-only baseline column
    seq_baseline_col = pd.Series(np.nan, index=['gcn', 'gat', 'gin'], name='Seq-only\nBaseline')
    for arch in ['gcn', 'gat', 'gin']:
        baseline_row = seq_baseline[seq_baseline['model_type'] == arch]
        if not baseline_row.empty:
            seq_baseline_col.loc[arch] = baseline_row[metric].values[0]

    # Reorder: seq-only baseline first, then feature configs
    pivot = pivot[config_labels]
    pivot.insert(0, 'Seq-only\nBaseline', seq_baseline_col)

    # Ensure row order: GCN, GAT, GIN
    pivot = pivot.reindex(['gcn', 'gat', 'gin'])

    # Add transformer as 4th row - only populate "Seq-only Baseline" column, rest are NaN
    transformer_row = pd.Series(
        [np.nan] * len(pivot.columns),
        index=pivot.columns,
        name='seq_transformer'
    )
    # Set only the "Seq-only Baseline" column (column 0)
    transformer_row['Seq-only\nBaseline'] = transformer[metric].values[0]

    pivot = pd.concat([pivot, transformer_row.to_frame().T])

    # Convert to numpy array
    data_matrix = pivot.values.astype(float)

    # Labels
    row_labels = ['GCN', 'GAT', 'GIN', 'Transformer']
    col_labels = list(pivot.columns)

    return data_matrix, row_labels, col_labels


def create_desaturated_colormap():
    """Create desaturated red-yellow-green colormap"""
    import matplotlib.colors as mcolors

    cmap = plt.cm.RdYlGn
    colors = cmap(np.linspace(0, 1, 256))

    # Reduce saturation by 25% (was 50%, now more vibrant)
    for i in range(len(colors)):
        rgb = colors[i, :3]
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[1] *= 0.75
        colors[i, :3] = mcolors.hsv_to_rgb(hsv)

    return mcolors.LinearSegmentedColormap.from_list('desaturated_RdYlGn', colors)


def plot_single_heatmap(data_matrix, row_labels, col_labels, ax, metric='f1_score', desaturated_cmap=None):
    """Plot a single heatmap on given axes"""

    # Metric-specific settings
    if metric == 'accuracy':
        metric_label = 'Accuracy'
        title = 'Accuracy by Architecture and Feature Configuration'
    else:
        metric_label = 'F1 Score'
        title = 'F1 Score by Architecture and Feature Configuration'

    sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.3f',
        cmap=desaturated_cmap,
        vmin=0.3, vmax=1.0,
        square=False,
        linewidths=1.5,
        linecolor='white',
        cbar_kws={
            'label': metric_label,
            'shrink': 0.8,
            'aspect': 20
        },
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        annot_kws={'fontsize': 11, 'fontweight': 'normal'}
    )

    # Title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Feature Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Architecture', fontsize=14, fontweight='bold')

    # Rotate column labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    # Add vertical separators
    ax.axvline(x=1, color='black', linewidth=3, linestyle='-', alpha=0.9)
    ax.axvline(x=2, color='gray', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=6, color='gray', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=11, color='gray', linewidth=2, linestyle='--', alpha=0.7)

    # Remove grid
    ax.grid(False)


def main():
    """Main execution"""
    print('Creating ablation heatmaps...\n')

    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.3)
    sns.set_palette("husl")

    # Enhanced font settings for electronic publication
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 800  # 800 DPI as requested
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['savefig.format'] = 'png'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

    # Create output directory
    output_dir = Path('/home/jeff/ribozyme/notebooks')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print('1. Loading data...')
    df = load_data()
    print(f'   Loaded {len(df)} rows')

    # Prepare data for both metrics
    print('\n2. Preparing heatmap data...')
    f1_matrix, row_labels, col_labels = prepare_heatmap_data(df, metric='f1_score')
    acc_matrix, _, _ = prepare_heatmap_data(df, metric='accuracy')

    print(f'   Matrix shape: {f1_matrix.shape}')
    print(f'   F1 Score range: [{np.nanmin(f1_matrix):.3f}, {np.nanmax(f1_matrix):.3f}]')
    print(f'   Accuracy range: [{np.nanmin(acc_matrix):.3f}, {np.nanmax(acc_matrix):.3f}]')

    # Create desaturated colormap
    desaturated_cmap = create_desaturated_colormap()

    # 1. Create F1 score heatmap
    print('\n3. Creating F1 score heatmap...')
    fig_f1, ax_f1 = plt.subplots(1, 1, figsize=(20, 6))
    plot_single_heatmap(f1_matrix, row_labels, col_labels, ax_f1, metric='f1_score', desaturated_cmap=desaturated_cmap)
    fig_f1.patch.set_facecolor('white')
    plt.tight_layout()

    output_path_f1 = output_dir / 'ablation_heatmap_f1.png'
    plt.savefig(output_path_f1, dpi=800, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'   ✓ Saved F1 heatmap to: {output_path_f1}')
    plt.close()

    # 2. Create Accuracy heatmap
    print('\n4. Creating Accuracy heatmap...')
    fig_acc, ax_acc = plt.subplots(1, 1, figsize=(20, 6))
    plot_single_heatmap(acc_matrix, row_labels, col_labels, ax_acc, metric='accuracy', desaturated_cmap=desaturated_cmap)
    fig_acc.patch.set_facecolor('white')
    plt.tight_layout()

    output_path_acc = output_dir / 'ablation_heatmap_accuracy.png'
    plt.savefig(output_path_acc, dpi=800, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'   ✓ Saved Accuracy heatmap to: {output_path_acc}')
    plt.close()

    # 3. Create combined stacked figure (2 rows, 1 column)
    print('\n5. Creating combined stacked heatmap...')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    # Plot F1 heatmap (top)
    plot_single_heatmap(f1_matrix, row_labels, col_labels, ax1, metric='f1_score', desaturated_cmap=desaturated_cmap)

    # Plot Accuracy heatmap (bottom)
    plot_single_heatmap(acc_matrix, row_labels, col_labels, ax2, metric='accuracy', desaturated_cmap=desaturated_cmap)

    # Set white background
    fig.patch.set_facecolor('white')

    # Tight layout
    plt.tight_layout()

    # Save PNG only at 800 DPI
    output_path = output_dir / 'ablation_heatmap.png'
    plt.savefig(output_path, dpi=800, bbox_inches='tight', facecolor='white', edgecolor='none')

    print(f'   ✓ Saved combined heatmap to: {output_path}')

    plt.close()

    print('\n✓ All heatmaps created successfully!')


if __name__ == '__main__':
    main()
