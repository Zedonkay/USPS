#!/bin/bash

# set up mujoco_py
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# cube_in_hand
base_dir="outputs/cube_in_hand-adv"
perturb_param_list="object_mass object_inertia cube_friction fingertip_friction actuator_kp actuator_kv contact_damping gravity_scale"
perturb_min_list="0.7 0.7 0.8 0.6 0.8 0.8 0.7 0.95"
perturb_max_list="1.3 1.3 1.2 1.4 1.2 1.2 1.3 1.05"


perturb_param_list=($perturb_param_list)
perturb_min_list=($perturb_min_list)
perturb_max_list=($perturb_max_list)
length=${#perturb_param_list[@]} 


cuda_id=0
for seed in 12345 23451 34512 45123 51234; do
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    cuda_id=$(($cuda_id+1))
    exp_dir=$base_dir/$seed 
    for ((i=0; i<${length}; i++));do
         python python_scripts/test.py \
             --experiments_dir ${exp_dir} \
             --agent_dir ${exp_dir} \
             --num_steps 1000 \
             --perturb_param ${perturb_param_list[$i]} \
             --perturb_min ${perturb_min_list[$i]} \
             --perturb_max ${perturb_max_list[$i]} &
    done
done

wait  # Wait for all background jobs to complete

