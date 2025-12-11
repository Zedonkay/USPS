#!/bin/bash
# Test cube_in_hand robustness
# Runs all tests in background with nohup and logs to files
# Safe to close SSH tunnel after starting

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate usps

# Get full path to python in conda environment (must be after conda activate)
PYTHON_CMD="${CONDA_PREFIX}/bin/python"

# set up mujoco_py
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# cube_in_hand
base_dir="outputs/cube_in_hand-adv_adaptive/10-0351-12345"
perturb_param_list="object_mass object_inertia cube_friction fingertip_friction actuator_kp actuator_kv contact_damping gravity_scale"
perturb_min_list="0.7 0.7 0.8 0.6 0.8 0.8 0.7 0.95"
perturb_max_list="1.3 1.3 1.2 1.4 1.2 1.2 1.3 1.05"

perturb_param_list=($perturb_param_list)
perturb_min_list=($perturb_min_list)
perturb_max_list=($perturb_max_list)
length=${#perturb_param_list[@]} 

# Create log directory
LOG_DIR="logs/test_robust_cube_in_hand-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "Starting robustness testing..."
echo "Log directory: ${LOG_DIR}"
echo "All output will be logged to: ${LOG_DIR}/"
echo ""

cuda_id=0
for seed in 12345; do
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    cuda_id=$(($cuda_id+1))
    exp_dir=$base_dir 
    for ((i=0; i<${length}; i++));do
        # Log file for this test
        log_file="${LOG_DIR}/test_${perturb_param_list[$i]}_seed${seed}.log"
        
        echo "Starting test: ${perturb_param_list[$i]} (min=${perturb_min_list[$i]}, max=${perturb_max_list[$i]})"
        echo "  Logging to: ${log_file}"
        
        # Run test in background with nohup, redirecting stdout and stderr to log file
        # Use -u flag for unbuffered output so logs appear immediately
        nohup ${PYTHON_CMD} -u python_scripts/test.py \
            --experiments_dir ${exp_dir} \
            --agent_dir ${exp_dir} \
            --num_steps 1000 \
            --perturb_param ${perturb_param_list[$i]} \
            --perturb_min ${perturb_min_list[$i]} \
            --perturb_max ${perturb_max_list[$i]} \
            > "${log_file}" 2>&1 &
        
        # Store PID
        pid=$!
        echo "  Started with PID: ${pid}"
        echo "${pid}:${perturb_param_list[$i]}:seed${seed}" >> "${LOG_DIR}/pids.txt"
        echo ""
    done
done

echo "All test jobs started in background!"
echo "PID list saved to: ${LOG_DIR}/pids.txt"
echo "Monitor progress with: tail -f ${LOG_DIR}/test_*.log"
echo "Check running jobs with: ps aux | grep test.py"
echo ""
echo "To check if all jobs completed, run:"
echo "  ps aux | grep test.py | grep -v grep"
echo "If no output, all jobs have finished."

