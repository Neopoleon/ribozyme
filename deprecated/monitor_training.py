"""Monitor training progress"""

import json
import sys
from pathlib import Path
import argparse


def monitor_training(run_dir):
    """Display current training progress"""
    run_path = Path(run_dir)

    if not run_path.exists():
        print(f"❌ Directory not found: {run_dir}")
        print("\nAvailable runs:")
        runs_dir = Path('results/runs')
        if runs_dir.exists():
            runs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for run in runs[:5]:
                print(f"  {run.name}")
        return

    print("="*80)
    print(f"Training Monitor: {run_path.name}")
    print("="*80)

    # Check config
    config_file = run_path / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\nConfiguration:")
        print(f"  Model: {config.get('model', 'N/A')}")
        print(f"  Hidden dim: {config.get('hidden_dim', 'N/A')}")
        print(f"  Num layers: {config.get('num_layers', 'N/A')}")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  Total epochs: {config.get('epochs', 'N/A')}")
        print(f"  Class weights: {config.get('use_class_weights', False)}")

    # Check history
    history_file = run_path / 'history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)

        n_epochs = len(history['train_loss'])
        total_epochs = config.get('epochs', '?')

        print(f"\nProgress: {n_epochs}/{total_epochs} epochs completed ({n_epochs/total_epochs*100:.1f}%)")

        if n_epochs > 0:
            print(f"\n{'Epoch':<10} {'Train Loss':<12} {'Train F1':<10} {'Val Loss':<12} {'Val F1':<10} {'Status'}")
            print("-"*80)

            # Show first epoch
            print(f"{'1':<10} {history['train_loss'][0]:<12.4f} {history['train_f1'][0]:<10.4f} "
                  f"{history['val_loss'][0]:<12.4f} {history['val_f1'][0]:<10.4f}")

            # Show last few epochs
            start_idx = max(1, n_epochs - 5)
            for i in range(start_idx, n_epochs):
                epoch = i + 1
                status = ""
                if i > 0 and history['val_f1'][i] > max(history['val_f1'][:i]):
                    status = "⭐ Best"

                print(f"{epoch:<10} {history['train_loss'][i]:<12.4f} {history['train_f1'][i]:<10.4f} "
                      f"{history['val_loss'][i]:<12.4f} {history['val_f1'][i]:<10.4f} {status}")

            # Summary statistics
            best_val_f1 = max(history['val_f1'])
            best_epoch = history['val_f1'].index(best_val_f1) + 1
            current_val_f1 = history['val_f1'][-1]

            print("\n" + "="*80)
            print("Summary:")
            print(f"  Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
            print(f"  Current Val F1: {current_val_f1:.4f}")

            # Check if improving
            if n_epochs >= 10:
                recent_f1 = history['val_f1'][-5:]
                if max(recent_f1) == max(history['val_f1']):
                    print(f"  Status: ✅ Still improving")
                elif current_val_f1 < best_val_f1 - 0.05:
                    print(f"  Status: ⚠️  May have plateaued (best was {best_val_f1:.4f})")
                else:
                    print(f"  Status: 📊 Converging")

            # Estimate time
            if n_epochs >= 2:
                # Rough estimate: assume constant time per epoch
                print(f"\n  Estimated completion: {total_epochs - n_epochs} epochs remaining")

    else:
        print("\n⏳ Training not started yet or no history file found")
        print(f"   Looking for: {history_file}")

    # Check if training is complete
    test_results = run_path / 'test_results.txt'
    if test_results.exists():
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\nTest results available at: {test_results}")
        print("\nView full results:")
        print(f"  python visualize_results.py {run_dir}")

    print("\n" + "="*80)


def list_runs():
    """List all available training runs"""
    runs_dir = Path('results/runs')
    if not runs_dir.exists():
        print("No training runs found in results/runs/")
        return

    runs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)

    print("="*80)
    print("Available Training Runs (most recent first)")
    print("="*80)

    for i, run in enumerate(runs[:10], 1):
        history_file = run / 'history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            n_epochs = len(history['train_loss'])
            best_f1 = max(history['val_f1'])
            status = "✅ Complete" if (run / 'test_results.txt').exists() else f"🏃 {n_epochs} epochs"
        else:
            status = "⏳ Starting..."
            best_f1 = 0

        print(f"{i:2d}. {run.name:<40} | Best F1: {best_f1:.4f} | {status}")

    print("\nTo monitor a specific run:")
    print("  python monitor_training.py results/runs/<run_name>")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor RNA GNN training progress')
    parser.add_argument('run_dir', type=str, nargs='?', help='Path to training run directory')
    parser.add_argument('--list', action='store_true', help='List all training runs')

    args = parser.parse_args()

    if args.list:
        list_runs()
    elif args.run_dir:
        monitor_training(args.run_dir)
    else:
        # Show latest run
        runs_dir = Path('results/runs')
        if runs_dir.exists():
            runs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if runs:
                print("Monitoring latest run...\n")
                monitor_training(str(runs[0]))
            else:
                print("No training runs found")
                print("\nUsage: python monitor_training.py <run_dir>")
                print("   or: python monitor_training.py --list")
        else:
            print("No results/runs directory found")
            print("Start training first: python train.py")
