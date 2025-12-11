#!/bin/bash
# TensorBoard viewer script for server usage
# Usage: ./view_tensorboard.sh [experiment_name] [port]

# Activate conda environment - ensure it's properly sourced
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    # Initialize conda if not already initialized
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate usps
fi

# Verify conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "usps" ]; then
    echo "Warning: Conda environment 'usps' not activated. Current: $CONDA_DEFAULT_ENV"
    echo "Attempting to activate usps..."
    conda activate usps
fi

# Verify tensorboard is available
if ! command -v tensorboard &> /dev/null; then
    echo "Error: tensorboard command not found. Is it installed in the usps conda environment?"
    exit 1
fi

# Resolve repo-relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/outputs"
PORT=${2:-6006}

# Build list of TensorBoard run directories (directories that contain event files)
if [ -n "$1" ]; then
    SEARCH_DIR="${DATA_DIR}/$1"
    if [ ! -d "$SEARCH_DIR" ]; then
        echo "Error: Directory $SEARCH_DIR not found"
        echo "Available experiments:"
        ls -1 "$DATA_DIR"
        exit 1
    fi
else
    SEARCH_DIR="$DATA_DIR"
fi

mapfile -t TB_DIRS < <(find "$SEARCH_DIR" -type f -name "events.out.tfevents.*" -printf '%h\n' | sort -u)

if [ ${#TB_DIRS[@]} -eq 0 ]; then
    echo "Error: No TensorBoard event files found under $SEARCH_DIR"
    exit 1
fi

echo "Starting TensorBoard on port $PORT..."
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "TensorBoard version: $(tensorboard --version 2>/dev/null || echo 'unknown')"
echo "Search base: $SEARCH_DIR"
echo "TensorBoard runs found:"
for dir in "${TB_DIRS[@]}"; do
    relative="${dir#$DATA_DIR/}"
    echo "  - ${relative%/tb}"
done
echo ""
echo "To access TensorBoard from your local machine:"
echo "1. Run this command on your LOCAL machine (not on server):"
echo "   ssh -L ${PORT}:localhost:${PORT} your_username@server_address"
echo ""
echo "2. Then open in your browser:"
echo "   http://localhost:${PORT}"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""
# Use --logdir pointing to parent directory - TensorBoard will discover all runs
# This is more reliable than --logdir_spec for multiple runs
exec tensorboard --logdir="$SEARCH_DIR" --port="$PORT" --host=0.0.0.0