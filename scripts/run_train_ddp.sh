#!/bin/bash
# Launch script for full training with distributed training on 2 GPUs

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure we're in the project root
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA may not be installed."
    exit 1
fi

# Check number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "Warning: Less than 2 GPUs detected. Falling back to single GPU training."
    python scripts/train_seq_transformer_fast.py "$@"
else
    echo "Launching distributed training on 2 GPUs..."
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=2 \
        scripts/train_seq_transformer_fast.py "$@"
fi
