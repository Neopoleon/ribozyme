"""Test script for RNA GNN data pipeline"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import RNADataset, LabelEncoder
from torch_geometric.loader import DataLoader


def test_dataset_creation():
    """Test creating dataset instances for each split"""
    print("="*80)
    print("Testing Dataset Creation")
    print("="*80)

    label_encoder = LabelEncoder()
    splits = ['train', 'val', 'test']

    for split_name in splits:
        print(f"\n{split_name.upper()} Split:")
        print("-" * 40)

        dataset = RNADataset(
            root=f'data/processed/{split_name}',
            fold_labels_path=f'data/splits/{split_name}_labels.json',
            rfam_types_path='rfam/rfam_types_full.pkl',
            st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
            label_encoder=label_encoder,
        )

        print(f"  Total samples: {len(dataset)}")

        # Get statistics
        stats = dataset.get_statistics()
        print(f"  Number of classes: {stats['num_classes']}")

        # Test loading first sample
        print(f"\n  Testing first sample...")
        data = dataset[0]
        print(f"    Nodes: {data.x.shape}")
        print(f"    Node features: {data.x.shape[1]} dims")
        print(f"    Edges: {data.edge_index.shape}")
        print(f"    Edge types: {data.edge_attr.shape}")
        print(f"    Label: {data.y.item()} ({label_encoder.decode(data.y.item())})")
        print(f"    Sequence length: {data.sequence_length}")
        print(f"    BPRNA ID: {data.bprna_id}")
        print(f"    RFID: {data.rfid}")

    print("\n" + "="*80)
    print("✓ Dataset creation test passed!")
    print("="*80)


def test_dataloader():
    """Test PyG DataLoader with batching"""
    print("\n" + "="*80)
    print("Testing DataLoader with Batching")
    print("="*80)

    label_encoder = LabelEncoder()

    # Create dataset
    dataset = RNADataset(
        root='data/processed/train',
        fold_labels_path='data/splits/train_labels.json',
        rfam_types_path='rfam/rfam_types_full.pkl',
        st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
        label_encoder=label_encoder,
    )

    # Create dataloader
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nDataLoader config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(loader)}")

    # Test first batch
    print(f"\nTesting first batch...")
    batch = next(iter(loader))

    print(f"  Batch object type: {type(batch)}")
    print(f"  Total nodes in batch: {batch.x.shape[0]}")
    print(f"  Node features shape: {batch.x.shape}")
    print(f"  Total edges in batch: {batch.edge_index.shape[1]}")
    print(f"  Batch vector shape: {batch.batch.shape}")
    print(f"  Labels shape: {batch.y.shape}")
    print(f"  Num graphs in batch: {batch.num_graphs}")

    print("\n" + "="*80)
    print("✓ DataLoader test passed!")
    print("="*80)


def test_label_distribution():
    """Test label distribution across splits"""
    print("\n" + "="*80)
    print("Testing Label Distribution")
    print("="*80)

    label_encoder = LabelEncoder()
    splits = ['train', 'val', 'test']

    all_stats = {}

    for split_name in splits:
        dataset = RNADataset(
            root=f'data/processed/{split_name}',
            fold_labels_path=f'data/splits/{split_name}_labels.json',
            rfam_types_path='rfam/rfam_types_full.pkl',
            st_files_dir='data/unzipped/bpRNA_1m_90_STAFILES',
            label_encoder=label_encoder,
        )
        all_stats[split_name] = dataset.get_statistics()

    # Print class distribution table
    print("\nClass distribution across splits:")
    print("-" * 80)
    print(f"{'Class':<45} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 80)

    for i, meta_type in enumerate(label_encoder.meta_types):
        train_count = all_stats['train']['label_distribution'].get(meta_type, 0)
        val_count = all_stats['val']['label_distribution'].get(meta_type, 0)
        test_count = all_stats['test']['label_distribution'].get(meta_type, 0)

        print(f"{meta_type:<45} {train_count:8d} {val_count:8d} {test_count:8d}")

    print("\n" + "="*80)
    print("✓ Label distribution test passed!")
    print("="*80)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("RNA GNN Data Pipeline Test Suite")
    print("="*80)

    try:
        # Run tests
        test_dataset_creation()
        test_dataloader()
        test_label_distribution()

        print("\n" + "="*80)
        print("🎉 All tests passed successfully!")
        print("="*80)
        print("\nThe data pipeline is ready for GNN training!")
        print("\nQuick start:")
        print("  1. Import: from src.data import RNADataset, LabelEncoder")
        print("  2. Create dataset: dataset = RNADataset(...)")
        print("  3. Create loader: loader = DataLoader(dataset, batch_size=32)")
        print("  4. Train your GNN model!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
