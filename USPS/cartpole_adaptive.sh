#!/bin/bash

# mujoco 210
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda 
export CUDA_DEVICE_ORDER=PCI_BUS_ID

overrides=cartpole_balance
# overrides=cartpole_swingup 

# # non robust
# robust_method=none
# robust_coef=0
# adaptive_robust_coef=false
 
# # l2 reg (static)
# robust_method=l2_reg
# robust_coef=1e-4
# adaptive_robust_coef=false
 
# # l1 reg (static)
# robust_method=l1_reg
# robust_coef=1e-5
# adaptive_robust_coef=false
 
# # l2 param (static)
# robust_method=l2_param
# robust_coef=1e-4
# adaptive_robust_coef=false
 
# # l1 param (static)
# robust_method=l1_param
# robust_coef=1e-4
# adaptive_robust_coef=false
#adaptive

# l2 adv with ADAPTIVE robust_coef
robust_method=l2_adv_param
robust_coef=1e-4  # initial value, will be adapted
adaptive_robust_coef=true
robust_coef_min=1e-5
robust_coef_max=1e-3
robust_buffer_size=250

cuda_id=0
for seed in 12345 23451 34512 45123 51234; do
    # set up cuda
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    cuda_id=$(($cuda_id+1))
    # train
    python python_scripts/train.py \
        overrides=${overrides} \
        seed=${seed} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        agent.params.adaptive_robust_coef=${adaptive_robust_coef} \
        agent.params.robust_coef_min=${robust_coef_min} \
        agent.params.robust_coef_max=${robust_coef_max} \
        agent.params.robust_buffer_size=${robust_buffer_size} \
        experiment=adv_adaptive1 &
done