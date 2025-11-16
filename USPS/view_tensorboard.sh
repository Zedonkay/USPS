#!/bin/bash
# TensorBoard viewer script for server usage
# Usage: ./view_tensorboard.sh [experiment_name] [port]
# Default values
DATA_DIR="outputs/quadruped_run-adv"
PORT=${2:-6006}
# If experiment name is provided, use that directory
if [ -n "$1" ]; then
    LOG_DIR="${DATA_DIR}/$1"
    if [ ! -d "$LOG_DIR" ]; then
        echo "Error: Directory $LOG_DIR not found"
        echo "Available experiments:"
        ls -1 "$DATA_DIR"
        exit 1
    fi
else
    # Use parent directory to view all experiments
    LOG_DIR="$DATA_DIR"
fi
echo "Starting TensorBoard on port $PORT..."
echo "Log directory: $LOG_DIR"
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
# Start TensorBoard
tensorboard --logdir="$LOG_DIR" --port="$PORT" --host=0.0.0.0