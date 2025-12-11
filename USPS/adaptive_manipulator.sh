
#!/bin/bash
# Train cube_in_hand with adaptive robust_coef
# Runs all experiments in background with nohup and logs to files
# Safe to close SSH tunnel after starting

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate usps

# Get full path to python in conda environment (must be after conda activate)
PYTHON_CMD="${CONDA_PREFIX}/bin/python"

# mujoco
# Use glfw for macOS compatibility (egl is for Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    export MUJOCO_GL=glfw
    # On macOS, prioritize conda environment's mujoco library (211) over system mujoco210
    # Set DYLD_LIBRARY_PATH to point to conda's mujoco library first
    CONDA_ENV_LIB=$(conda info --base)/envs/usps-py38/lib/python3.8/site-packages/mujoco
    if [ -d "$CONDA_ENV_LIB" ]; then
        export DYLD_LIBRARY_PATH=$CONDA_ENV_LIB:${DYLD_LIBRARY_PATH:-}
    fi
else
    export MUJOCO_GL=egl
    # Only set LD_LIBRARY_PATH for Linux if using mujoco210
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
fi
# Note: dm_control should use its bundled MuJoCo version (211)

# set up cuda 
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# select exp setting and regularizer setting
overrides=cube_in_hand
robust_method=l2_adv_param
# l2 adv with ADAPTIVE robust_coef
robust_coef=5e-4  # initial value, will be adapted
adaptive_robust_coef=true
robust_coef_min=1e-4
robust_coef_max=1e-3
robust_buffer_size=250
exp_base_name=manipulation_adaptive_fixed_target_entropy

# Set device based on OS (macOS doesn't have CUDA)
if [[ "$OSTYPE" == "darwin"* ]]; then
    DEVICE_OVERRIDE="device=cpu"
else
    DEVICE_OVERRIDE="device=cuda:0"
fi

# Create log directory
LOG_DIR="logs/train_cube_in_hand-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "Starting training with adaptive robust_coef..."
echo "Log directory: ${LOG_DIR}"
echo "All output will be logged to: ${LOG_DIR}/"
echo ""

cuda_id=0
for seed in 12345; do
    # set up cuda (only for Linux)
    if [[ "$OSTYPE" != "darwin"* ]]; then
        export CUDA_VISIBLE_DEVICES=${cuda_id}
    fi
    
    exp_name="${exp_base_name}"
    
    # Log file for this experiment
    log_file="${LOG_DIR}/train_adaptive_seed${seed}.log"
    
    echo "Starting training with adaptive robust_coef (initial=${robust_coef}, min=${robust_coef_min}, max=${robust_coef_max}), experiment=${exp_name}"
    echo "  Logging to: ${log_file}"
    
    # train in background, redirecting stdout and stderr to log file
    # Use nohup and explicit redirection to ensure logs are captured
    # Use -u flag for unbuffered output so logs appear immediately
    nohup ${PYTHON_CMD} -u python_scripts/train.py \
        overrides=${overrides} \
        seed=${seed} \
        device=${DEVICE_OVERRIDE#device=} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        agent.params.adaptive_robust_coef=${adaptive_robust_coef} \
        agent.params.robust_coef_min=${robust_coef_min} \
        agent.params.robust_coef_max=${robust_coef_max} \
        agent.params.robust_buffer_size=${robust_buffer_size} \
        experiment=${exp_name} \
        > "${log_file}" 2>&1 &
    
    # Store PID
    pid=$!
    echo "  Started with PID: ${pid}"
    echo "${pid}:${exp_name}:adaptive" >> "${LOG_DIR}/pids.txt"
    echo ""
done

echo "All training jobs started in background!"
echo "PID list saved to: ${LOG_DIR}/pids.txt"
echo "Monitor progress with: tail -f ${LOG_DIR}/train_*.log"
echo "Check running jobs with: ps aux | grep train.py"


