
# set up mujoco_py
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# cartpole_blance/cartpole_swingup
base_dir="outputs/cartpole_swingup-cartpole_swingup_l2_adv"
perturb_param_list="pole_length"
perturb_min_list="0.3"
perturb_max_list="3.0"
 
# # walker_stand/walker_walk
# base_dir="outputs/walker_stand-adv"
# perturb_param_list="thigh_length"
# perturb_min_list="0.1"
# perturb_max_list="0.7"

# # quadruped_walk/quadruped_run
# base_dir="outputs/quadruped_walk-adv"
# perturb_param_list="contact_friction"
# perturb_min_list="0.1"
# perturb_max_list="4.5"

perturb_param_list=($perturb_param_list)
perturb_min_list=($perturb_min_list)
perturb_max_list=($perturb_max_list)
length=${#perturb_param_list[@]} 


seed_list="12345"

cuda_id=0
for seed in $seed_list; do
    exp_dir=$(find "${base_dir}" -maxdepth 1 -type d -name "*${seed}" | head -n 1)

    if [ -z "$exp_dir" ]; then
        echo "WARNING: no experiment dir found for seed $seed under $base_dir"
        continue
    fi

    if [ ! -f "$exp_dir/.hydra/config.yaml" ]; then
        echo "WARNING: experiment dir \"$exp_dir\" is missing .hydra/config.yaml; skipping"
        continue
    fi

    echo "Rendering videos for seed $seed using run dir: $exp_dir"

    export CUDA_VISIBLE_DEVICES=${cuda_id}
    cuda_id=$(($cuda_id+1))

    for ((i=0; i<${length}; i++));do
         python python_scripts/video.py \
             --experiments_dir "${exp_dir}" \
             --agent_dir "${exp_dir}" \
             --num_steps 100 \
             --perturb_param "${perturb_param_list[$i]}" \
             --perturb_min "${perturb_min_list[$i]}" \
             --perturb_max "${perturb_max_list[$i]}" \
             --save_video &
    done
done


