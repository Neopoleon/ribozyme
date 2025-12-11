"""Create stratified train/val/test splits for RNA dataset"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_all_fold_labels(results_dir: str = 'results') -> list[dict]:
    """Load and combine all fold label files."""
    all_labels = []
    for fold_num in range(1, 5):
        fold_path = Path(results_dir) / f'fold{fold_num}_labels.json'
        with open(fold_path, 'r') as f:
            fold_data = json.load(f)
            all_labels.extend(fold_data)
    return all_labels


def extract_rfid(reference_name: str) -> str:
    """Extract RFAM ID from reference name."""
    return reference_name.split('_')[0]


def create_stratified_splits(
    all_labels: list[dict],
    rfam_types_path: str = 'rfam/rfam_types_full.pkl',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Create stratified train/val/test splits ensuring balanced class distribution.

    Args:
        all_labels: Combined list of all label entries
        rfam_types_path: Path to rfam_types_full.pkl
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_labels, val_labels, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Load RFAM types
    with open(rfam_types_path, 'rb') as f:
        rfam_types = pickle.load(f)

    # Group entries by meta-type
    type_to_entries = defaultdict(list)
    entries_without_type = []

    for entry in all_labels:
        rfid = extract_rfid(entry['reference_name'])
        meta_type = rfam_types.get(rfid)

        if meta_type is not None:
            type_to_entries[meta_type].append(entry)
        else:
            entries_without_type.append(entry)

    if entries_without_type:
        print(f"Warning: {len(entries_without_type)} entries without meta-type mapping")

    # Split each class independently to maintain stratification
    train_data = []
    val_data = []
    test_data = []

    np.random.seed(random_seed)

    for meta_type, entries in type_to_entries.items():
        n_samples = len(entries)

        if n_samples < 3:
            # For very small classes, put everything in train
            print(f"Warning: Class '{meta_type}' has only {n_samples} samples. "
                  f"Adding all to train set.")
            train_data.extend(entries)
            continue

        # Calculate split sizes
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val  # Remaining goes to test

        # Ensure at least 1 sample in each split if possible
        if n_samples >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)

        # Shuffle entries
        indices = np.random.permutation(n_samples)

        # Split indices
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        # Add to splits
        train_data.extend([entries[i] for i in train_idx])
        val_data.extend([entries[i] for i in val_idx])
        test_data.extend([entries[i] for i in test_idx])

    # Shuffle the final splits
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/len(all_labels)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} samples ({len(val_data)/len(all_labels)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/len(all_labels)*100:.1f}%)")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} samples")

    return train_data, val_data, test_data


def save_splits(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
    output_dir: str = 'data/splits',
) -> None:
    """Save train/val/test splits to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
    }

    for split_name, split_data in splits.items():
        output_file = output_path / f'{split_name}_labels.json'
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_name} split to {output_file}")


def print_split_statistics(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
    rfam_types_path: str = 'rfam/rfam_types_full.pkl',
) -> None:
    """Print detailed statistics about the splits."""
    with open(rfam_types_path, 'rb') as f:
        rfam_types = pickle.load(f)

    def get_class_distribution(data):
        dist = defaultdict(int)
        for entry in data:
            rfid = extract_rfid(entry['reference_name'])
            meta_type = rfam_types.get(rfid, 'UNKNOWN')
            dist[meta_type] += 1
        return dist

    print("\nClass distribution per split:")
    print("-" * 80)

    train_dist = get_class_distribution(train_data)
    val_dist = get_class_distribution(val_data)
    test_dist = get_class_distribution(test_data)

    # Get all unique classes
    all_classes = sorted(set(list(train_dist.keys()) + list(val_dist.keys()) + list(test_dist.keys())))

    print(f"{'Class':<45} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 80)

    for cls in all_classes:
        if cls == 'UNKNOWN':
            continue
        train_count = train_dist.get(cls, 0)
        val_count = val_dist.get(cls, 0)
        test_count = test_dist.get(cls, 0)
        total = train_count + val_count + test_count

        print(f"{cls:<45} {train_count:8d} {val_count:8d} {test_count:8d} {total:8d}")


if __name__ == '__main__':
    # Load all fold labels
    print("Loading all fold labels...")
    all_labels = load_all_fold_labels()
    print(f"Loaded {len(all_labels)} total samples")

    # Create stratified splits
    print("\nCreating stratified splits (80/10/10)...")
    train_data, val_data, test_data = create_stratified_splits(
        all_labels,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    )

    # Save splits
    print("\nSaving splits...")
    save_splits(train_data, val_data, test_data)

    # Print detailed statistics
    print_split_statistics(train_data, val_data, test_data)

    print("\n✓ Split creation complete!")
