#!/bin/bash

# =========================
# Env setup
# =========================
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0   # single GPU

# =========================
# Cartpole swingup, Adv-USR (l2_adv_param)
# =========================
base_dir="outputs/cartpole_swingup-cartpole_swingup_l2_adv"

# parameter names and ranges
perturb_param_list="pole_length pole_mass joint_damping slider_damping"
perturb_min_list="0.3 0.1 2e-6 5e-4"
perturb_max_list="3.0 10.0 2e-1 3.0"

perturb_param_list=($perturb_param_list)
perturb_min_list=($perturb_min_list)
perturb_max_list=($perturb_max_list)
length=${#perturb_param_list[@]}

# the seeds you trained with
seed_list="12345 23451 34512"

echo "Running robustness tests for Cartpole Swingup (l2_adv)"
echo "Base dir: $base_dir"
echo "Seeds: $seed_list"
echo ""

for seed in $seed_list; do
    # find the run directory that ends with this seed, e.g. 29-2041-12345
    exp_dir=$(ls -d ${base_dir}/*${seed} 2>/dev/null | head -n 1)

    if [ -z "$exp_dir" ]; then
        echo "WARNING: no experiment dir found for seed $seed under $base_dir"
        continue
    fi

    echo "----------------------------------------"
    echo " Seed: $seed"
    echo " Using exp_dir: $exp_dir"
    echo "----------------------------------------"

    for ((i=0; i<${length}; i++)); do
        param=${perturb_param_list[$i]}
        pmin=${perturb_min_list[$i]}
        pmax=${perturb_max_list[$i]}

        echo "  -> Testing param: $param ([$pmin, $pmax])"

        python python_scripts/test.py \
            --experiments_dir "$exp_dir" \
            --agent_dir "$exp_dir" \
            --num_steps 1000 \
            --perturb_param "$param" \
            --perturb_min "$pmin" \
            --perturb_max "$pmax"
    done

    echo ""
done

echo "All robustness tests for Cartpole Swingup (l2_adv) completed."