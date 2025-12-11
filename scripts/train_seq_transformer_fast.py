"""Optimized sequence Transformer training with DDP, cached data, and performance improvements."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb  # type: ignore
except Exception:  # noqa: BLE001
    wandb = None

from src.data import (
    LabelEncoder,
    PAD_TOKEN_ID,
    CachedRNASequenceDataset,
    sequence_collate_fn,
    VOCAB_SIZE,
)
from src.models import RNASequenceTransformer


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training. Returns (rank, local_rank, world_size)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        # Use GLOO backend if NCCL has issues (e.g., in tmux)
        backend = os.getenv('DDP_BACKEND', 'nccl')
        if backend == 'nccl':
            # Set NCCL env vars for better stability
            os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
            os.environ.setdefault('NCCL_DEBUG', 'WARN')

        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def compute_class_weights(dataset: CachedRNASequenceDataset, device: torch.device) -> torch.Tensor:
    counts = dataset.label_counts().float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(counts)
    return weights.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Gather metrics across all processes
    if world_size > 1:
        metrics = torch.tensor([total_loss, total_samples], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss = metrics[0].item()
        total_samples = int(metrics[1].item())

        # Gather predictions and labels from all ranks
        preds_tensor = torch.tensor(all_preds, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)

        # Create buffers for gathering
        if rank == 0:
            preds_list = [torch.zeros_like(preds_tensor) for _ in range(world_size)]
            labels_list = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
        else:
            preds_list = None
            labels_list = None

        dist.gather(preds_tensor, preds_list if rank == 0 else None, dst=0)
        dist.gather(labels_tensor, labels_list if rank == 0 else None, dst=0)

        if rank == 0:
            all_preds = torch.cat(preds_list).cpu().tolist()
            all_labels = torch.cat(labels_list).cpu().tolist()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    if rank == 0:
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    else:
        acc = 0.0
        f1 = 0.0

    return avg_loss, acc, f1


@hydra.main(config_path="../conf", config_name="seq_baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    set_seed(cfg.seed, rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"World size: {world_size} | Device: {device}")

    project_root = Path(get_original_cwd())
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"seq_transformer_fast_{run_timestamp}"
    output_dir = project_root / cfg.output_dir / run_name

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print("Configuration")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=False), f, indent=2)
        with open(output_dir / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # Optional Weights & Biases logging (only on rank 0)
    wandb_enabled = os.getenv("HYDRA_WANDB", "0").lower() in {"1", "true", "yes"}
    wandb_run = None
    if wandb_enabled and rank == 0:
        if wandb is None:
            print("HYDRA_WANDB set but wandb is not installed; skipping wandb logging.")
        else:
            try:
                wandb_config = OmegaConf.to_container(cfg, resolve=False)
                wandb_config['world_size'] = world_size  # Add DDP info
                wandb_config['distributed'] = world_size > 1
                wandb_run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", "ribozyme-seq-transformer"),
                    entity=os.getenv("WANDB_ENTITY"),
                    mode=os.getenv("WANDB_MODE", "online"),
                    name=run_name,
                    dir=str(output_dir),
                    config=wandb_config,
                )
                print(f"Weights & Biases logging enabled: {wandb_run.url}")
            except Exception as exc:  # noqa: BLE001
                print(f"HYDRA_WANDB set but wandb init failed: {exc}. Skipping wandb logging.")

    label_encoder = LabelEncoder()

    if rank == 0:
        print("\n=== Loading datasets with memory caching ===")

    # Use cached dataset for fast training
    train_dataset = CachedRNASequenceDataset(
        fold_labels_path=to_absolute_path(cfg.data.train_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
    )
    val_dataset = CachedRNASequenceDataset(
        fold_labels_path=to_absolute_path(cfg.data.val_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
    )
    test_dataset = CachedRNASequenceDataset(
        fold_labels_path=to_absolute_path(cfg.data.test_path),
        rfam_types_path=to_absolute_path(cfg.data.rfam_types_path),
        st_files_dir=to_absolute_path(cfg.data.st_files_dir),
        label_encoder=label_encoder,
    )

    if rank == 0:
        print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Can now use multiple workers since data is in memory
    num_workers = cfg.training.num_workers
    if rank == 0:
        print(f"Using {num_workers} DataLoader workers")

    # Create distributed samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed,
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = RNASequenceTransformer(
        vocab_size=VOCAB_SIZE,
        num_classes=label_encoder.num_classes,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        pad_token_id=PAD_TOKEN_ID,
        max_seq_len=cfg.model.max_seq_len,
    ).to(device)

    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Trainable parameters: {n_params:,}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=cfg.training.lr_patience, factor=0.5
    )

    class_weights = None
    if cfg.training.use_class_weights:
        class_weights = compute_class_weights(train_dataset, device)
        if rank == 0:
            print("Class weights computed.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    best_state = None
    history: dict[str, list[float]] = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
    }

    if rank == 0:
        print("Starting training loop...\n")

    for epoch in range(1, cfg.training.epochs + 1):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, optimizer, device, criterion, rank, world_size
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, None, device, criterion, rank, world_size
        )

        scheduler.step(val_f1)

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if rank == 0:
            # Log to wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "train/f1": train_f1,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "val/f1": val_f1,
                        "lr": scheduler.optimizer.param_groups[0]["lr"],
                        "epoch_time": epoch_time,
                    },
                    step=epoch,
                )

            print(
                f"Epoch {epoch:03d} ({epoch_time:.1f}s) | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if rank == 0:
                # Unwrap DDP model for saving
                model_to_save = model.module if hasattr(model, 'module') else model
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                }
                torch.save(best_state, output_dir / 'best_model.pt')

    if rank == 0:
        if best_state is None:
            raise RuntimeError("Training did not produce a valid checkpoint.")

        # Load best model
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(best_state['model_state_dict'])

        print("\nEvaluating on test set...")

    test_loss, test_acc, test_f1 = run_epoch(
        model, test_loader, None, device, criterion, rank, world_size
    )

    if rank == 0:
        print(f"Test Loss {test_loss:.4f} | Acc {test_acc:.4f} | F1 {test_f1:.4f}")

        # Log test metrics to wandb
        if wandb_run is not None:
            wandb.log(
                {
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/f1": test_f1,
                    "best_val_f1": best_val_f1,
                }
            )

        history_path = output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        results = {
            'val_best_f1': best_val_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
        }
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_dir}")

        # Finish wandb run
        if wandb_run is not None:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
