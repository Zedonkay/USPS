#!/bin/bash

# ------------------------------
# (1) Environment setup
# ------------------------------
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID


# ============================================================
#  Choose the task + parameter ranges
# ============================================================

## --- cartpole_swingup example ---
base_dir="outputs/cartpole_swingup-sensitivity_adv-1e-4"
perturb_param_list="pole_length pole_mass joint_damping slider_damping"
perturb_min_list="0.3 0.1 2e-6 5e-4"
perturb_max_list="3.0 10.0 2e-1 3.0"

## --- walker_stand example ---
# base_dir="outputs/walker_stand-adv"
# perturb_param_list="thigh_length torso_length joint_damping contact_friction"
# perturb_min_list="0.1 0.1 0.1 0.01"
# perturb_max_list="0.7 0.7 10.0 2.0"

## --- quadruped_walk example ---
# base_dir="outputs/quadruped_walk-adv"
# perturb_param_list="shin_length torso_density joint_damping contact_friction"
# perturb_min_list="0.25 500.0 10.0 0.1"
# perturb_max_list="2.0 10000.0 150.0 4.5"


# Convert to bash arrays
perturb_param_list=($perturb_param_list)
perturb_min_list=($perturb_min_list)
perturb_max_list=($perturb_max_list)
length=${#perturb_param_list[@]}

# ============================================================
#  Seeds to aggregate over (modify if needed)
# ============================================================
seed_list="12345 23451 34512"    # <--- Add more seeds here if needed: "12345 2022 2023 ..."


# ============================================================
#  (2) Run aggregation for each perturbation parameter
# ============================================================

echo "Running Robust-AUC aggregation..."
echo "Base dir = $base_dir"
echo ""

for ((i=0; i<${length}; i++)); do
    param=${perturb_param_list[$i]}
    echo "----------------------------------------"
    echo " Aggregating AUC for param: $param"
    echo "----------------------------------------"

    # Collect COMMA-SEPARATED experiment dirs
    exp_dirs=""
    for seed in $seed_list; do
        exp_dirs="${exp_dirs}${base_dir}/${seed},"
    done
    exp_dirs="${exp_dirs::-1}"   # remove last comma

    echo "exp_dirs = $exp_dirs"

    # Output file
    out_file="${base_dir}/auc_${param}.txt"

    # Call the Python aggregation script
    python python_scripts/aggregate_robust_auc.py \
        --exp_dirs "$exp_dirs" \
        --perturb_param "$param" \
        | tee "$out_file"

    echo "Saved to: $out_file"
    echo ""
done

echo "All AUC aggregations completed!"
