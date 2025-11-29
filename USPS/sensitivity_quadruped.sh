

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
overrides=quadruped_run
robust_method=l2_adv_param
exp_name=sensitivity_adv

# Set device based on OS (macOS doesn't have CUDA)
if [[ "$OSTYPE" == "darwin"* ]]; then
    DEVICE_OVERRIDE="device=cpu"
else
    DEVICE_OVERRIDE="device=cuda:0"
fi

cuda_id=0
for seed in 12345; do
    # set up cuda (only for Linux)
    if [[ "$OSTYPE" != "darwin"* ]]; then
        export CUDA_VISIBLE_DEVICES=${cuda_id}
        cuda_id=$(($cuda_id+1))
    fi
    # train
    robust_coef=7e-5
    python python_scripts/train.py \
        overrides=${overrides} \
        seed=${seed} \
        device=${DEVICE_OVERRIDE#device=} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        experiment=${exp_name}-${robust_coef} &
    
    robust_coef=5e-6
    python python_scripts/train.py \
        overrides=${overrides} \
        seed=${seed} \
        device=${DEVICE_OVERRIDE#device=} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        experiment=${exp_name}-${robust_coef} &
    
    robust_coef=1e-6
    python python_scripts/train.py \
        overrides=${overrides} \
        seed=${seed} \
        device=${DEVICE_OVERRIDE#device=} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        experiment=${exp_name}-${robust_coef} &
    
    robust_coef=5e-4
    python python_scripts/train.py \
        overrides=${overrides} \
        seed=${seed} \
        device=${DEVICE_OVERRIDE#device=} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        experiment=${exp_name}-${robust_coef} &
done



