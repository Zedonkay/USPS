#!/bin/bash

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

base_dir="outputs/cartpole_swingup-cartpole_swingup_non_reg"

perturb_param_list="pole_length pole_mass joint_damping slider_damping"
perturb_min_list="0.3 0.1 2e-6 5e-4"
perturb_max_list="3.0 10.0 2e-1 3.0"

perturb_param_list=($perturb_param_list)
perturb_min_list=($perturb_min_list)
perturb_max_list=($perturb_max_list)
length=${#perturb_param_list[@]}

seed_list="12345 23451 34512"

for seed in $seed_list; do
    exp_dir=$(ls -d ${base_dir}/*${seed} | head -n1)

    echo "Testing seed = $seed -> $exp_dir"

    for ((i=0; i<$length; i++)); do
        python python_scripts/test.py \
            --experiments_dir "$exp_dir" \
            --agent_dir "$exp_dir" \
            --num_steps 1000 \
            --perturb_param "${perturb_param_list[$i]}" \
            --perturb_min "${perturb_min_list[$i]}" \
            --perturb_max "${perturb_max_list[$i]}"
    done
done