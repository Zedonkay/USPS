#!/bin/bash
# Visualization script for test results
# Modify the variables at the top to change the run directory and visualization mode

# ==============================================================================
# CONFIGURATION - Modify these variables as needed
# ==============================================================================

# Run directory (relative to USPS/ or absolute path)
# Examples:
#   "outputs/cube_in_hand-adv_adaptive/10-0351-12345"
#   "/home/ubuntu/USPS/USPS/outputs/cube_in_hand-adv_adaptive/10-0351-12345"
RUN_DIR="outputs/cube_in_hand-adv_adaptive/10-0351-12345"

# Visualization mode/plots to generate
# Options: 'all', 'mean', 'distribution', 'min', 'summary', 'heatmap'
# You can specify multiple: 'mean' 'distribution' 'summary'
MODE="mean"

# ==============================================================================
# SCRIPT - Do not modify below unless you know what you're doing
# ==============================================================================

# Get script directory and resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment if available
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate usps 2>/dev/null || true
fi

# Get Python command
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_CMD="${CONDA_PREFIX}/bin/python"
else
    PYTHON_CMD="python3"
fi

# Resolve test directory path
if [[ "$RUN_DIR" == /* ]]; then
    # Absolute path
    TEST_DIR="${RUN_DIR}/test"
else
    # Relative path - assume relative to USPS directory
    TEST_DIR="${SCRIPT_DIR}/${RUN_DIR}/test"
fi

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory does not exist: $TEST_DIR"
    echo ""
    echo "Please check your RUN_DIR setting: $RUN_DIR"
    exit 1
fi

# Check if test directory has JSON files
JSON_COUNT=$(find "$TEST_DIR" -maxdepth 1 -name "*.json" | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "Warning: No JSON files found in test directory: $TEST_DIR"
    echo "The visualization may not produce any results."
    echo ""
fi

# Print configuration
echo "=========================================="
echo "Test Results Visualization"
echo "=========================================="
echo "Run directory: $RUN_DIR"
echo "Test directory: $TEST_DIR"
echo "Mode: $MODE"
echo "Python: $PYTHON_CMD"
echo "JSON files found: $JSON_COUNT"
echo "=========================================="
echo ""

# Run visualization script
"$PYTHON_CMD" python_scripts/visualize_test_results.py \
    --test_dir "$TEST_DIR" \
    --plots $MODE

echo ""
echo "Visualization complete!"
echo "Plots saved to: $TEST_DIR/images/"
