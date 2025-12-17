# USPS: Uncertainty-Sensitive Policy Search for Robust Reinforcement Learning

USPS is a reinforcement learning framework that implements robust RL algorithms with adaptive regularization for training policies that are resilient to dynamics perturbations and distributional shifts. The framework is built on Soft Actor-Critic (SAC) and extends it with various robust regularization methods, including a novel **adaptive robust coefficient** mechanism that automatically adjusts regularization strength based on training performance.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Cube in Hand Manipulation Environment](#cube-in-hand-manipulation-environment)
- [Adaptive Robust Coefficient Implementation](#adaptive-robust-coefficient-implementation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

USPS provides a comprehensive framework for training robust RL agents that can handle:
- **Dynamics perturbations**: Variations in physical parameters (mass, friction, damping, etc.)
- **Distributional shifts**: Mismatches between training and test conditions
- **Adversarial robustness**: Resilience to worst-case perturbations

The framework implements multiple robust regularization methods:
- **L1/L2 parameter regularization**: Penalizes large parameter values
- **L1/L2/L∞ adversarial regularization**: Defends against worst-case state perturbations
- **Adaptive robust coefficient**: Automatically tunes regularization strength during training

---

## Key Features

### 1. Robust RL Algorithms
- **Soft Actor-Critic (SAC)** with robust extensions
- Multiple regularization methods: `l1_reg`, `l2_reg`, `l1_adv_param`, `l2_adv_param`, `linf_adv_param`
- Adversarial training with gradient-based perturbations

### 2. Adaptive Robust Coefficient
- Automatically adjusts regularization strength based on training performance
- Uses rolling performance buffer to track policy quality
- Maps performance trends to coefficient values using tanh scaling

### 3. Diverse Environments
- **Cube in Hand**: Dextrous manipulation with Torobo hand
- **Locomotion**: Quadruped (A1), humanoid, walker tasks
- **Control**: Cartpole swingup/balance
- **RealWorldRL Suite**: Standardized real-world RL benchmarks

### 4. Dynamics Perturbations
- Configurable perturbation specifications
- Support for multiple physical parameters (mass, friction, damping, gravity, etc.)
- Range-based and fixed-value perturbation modes

---

## Installation

### Prerequisites

- Python 3.8
- CUDA 11.8 (for GPU training on Linux)
- Conda (recommended)

### Step 1: Create Conda Environment

```bash
conda create -n usps python=3.8 -y
conda activate usps
conda install -c conda-forge glew
```

### Step 2: Install USPS

```bash
# Install USPS package
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Install specific PyTorch version (CUDA 11.8)
pip install setuptools==59.5.0
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install RealWorldRL Suite
cd USPS/envs/realworldrl_suite
pip install -e . 
cd ../..
```

### Step 3: Environment-Specific Setup

#### macOS
```bash
export MUJOCO_GL=glfw
```

#### Linux
```bash
export MUJOCO_GL=egl
```

---

## Project Structure

```
USPS/
├── USPS/
│   ├── agent/              # RL agent implementations
│   │   ├── agent.py       # SAC agent with robust extensions
│   │   ├── actor.py       # Policy network
│   │   └── critic.py      # Q-function networks
│   ├── envs/              # Environment implementations
│   │   ├── cube_in_hand_env.py    # Cube manipulation environment
│   │   ├── inhand_env.py          # ShadowHand wrapper
│   │   ├── a1_env_mujoco.py      # A1 quadruped (MuJoCo)
│   │   ├── a1_env_bullet.py      # A1 quadruped (PyBullet)
│   │   └── ...
│   ├── infra/             # Infrastructure code
│   │   ├── logger.py      # TensorBoard logging
│   │   ├── replay_buffer.py  # Experience replay with adversarial sampling
│   │   └── utils.py       # Utility functions
│   ├── python_scripts/    # Training and testing scripts
│   │   ├── train.py       # Main training script
│   │   ├── test.py        # Evaluation script
│   │   └── ...
│   └── configs/           # Hydra configuration files
│       ├── train.yaml     # Base training config
│       ├── agent/         # Agent configurations
│       └── overrides/     # Environment-specific overrides
├── requirements.txt
├── setup.py
└── README.md
```

---

## Cube in Hand Manipulation Environment

The **Cube in Hand** environment is a challenging dexterous manipulation task where a multi-fingered robotic hand (Torobo hand) must reorient a cube to match a target orientation. This environment is particularly well-suited for testing robust RL algorithms due to its sensitivity to physical parameters.

### Environment Details

**Task**: Reorient a cube held in a multi-fingered hand to match a randomly generated target quaternion orientation.

**Robot**: Torobo hand with 16 degrees of freedom (DOF), 12 actuated joints
- 4 fingers (thumb + 3 fingers)
- Each finger has multiple joints for dexterous manipulation

**Observation Space**: 92-dimensional (46 dimensions × 2 frame stack)
- 16 joint positions (scaled to [-1, 1])
- 3 object position (x, y, z)
- 4 object quaternion (w, x, y, z)
- 3 target position (constant)
- 4 target quaternion (w, x, y, z)
- 4 quaternion difference (object relative to target)
- 12 last processed action

**Action Space**: 12-dimensional continuous actions in [-1, 1]
- Actions are scaled by `action_scale=0.15` and applied as position deltas
- Joint positions are clipped to soft limits to prevent damage

**Reward**: Negative rotation distance between object and target quaternions
- `reward = -2 * arcsin(min(||quat_diff||, 1.0))`
- Success threshold: rotation distance < 0.3 radians

**Episode Termination**:
- Success: Cube orientation matches target (within tolerance)
- Failure: Cube drops below 0.30m height or moves outside ±0.3m bounds
- Time limit: 1000 steps (10 seconds at 100Hz control)

### Dynamics Perturbations

The environment supports configurable dynamics perturbations to test robustness:

**Perturbable Parameters**:
1. **`object_mass`**: Cube mass multiplier
2. **`object_inertia`**: Cube inertia tensor multiplier
3. **`cube_friction`**: Friction coefficient for cube surfaces
4. **`fingertip_friction`**: Friction coefficient for fingertip contacts
5. **`actuator_kp`**: Position gain for actuators
6. **`actuator_kv`**: Velocity gain for actuators
7. **`contact_damping`**: Contact solver damping parameters
8. **`gravity_scale`**: Gravity magnitude multiplier

**Perturbation Modes**:
- **`range`**: Sample uniformly from [min, max] each episode
- **`fixed`**: Use fixed multiplier value (for evaluation sweeps)

**Example Configuration**:
```yaml
task_kwargs:
  perturb_spec:
    mode: "range"
    params:
      object_mass:
        min: 0.5
        max: 2.0
      fingertip_friction:
        min: 0.5
        max: 1.5
```

### Implementation Details

**Control Architecture**:
- Control frequency: 100Hz (`control_dt=0.01`)
- Simulation decimation: 4 substeps per control step
- Position control: Actions specify target joint positions
- Soft joint limits: 90% of hard limits to prevent damage

**Frame Stacking**:
- 2-frame history for temporal information
- Helps capture velocity and acceleration implicitly

**MuJoCo Integration**:
- Uses MuJoCo 2.1+ (mujoco Python package)
- Direct model manipulation for perturbations
- Efficient forward kinematics and contact computation

### Training Considerations

**Initialization**:
- Hand starts in neutral position (all joints at zero)
- Cube starts in hand with random target orientation
- Small random actions during exploration to maintain grasp

**Challenges**:
- High-dimensional action space (12 DOF)
- Complex contact dynamics
- Sensitive to physical parameters
- Requires precise coordination between fingers

**Recommended Settings**:
- Learning rates: `1e-4` for actor, critic, and temperature
- Batch size: 1024
- Replay buffer: 1M transitions
- Exploration: Epsilon-greedy with decay (0.4 → 0.05 over 500k steps)

---

## Adaptive Robust Coefficient Implementation

The **adaptive robust coefficient** is a novel mechanism that automatically adjusts the regularization strength (`robust_coef`) during training based on the agent's performance. This eliminates the need for manual hyperparameter tuning and adapts to the learning dynamics.

### Motivation

Traditional robust RL methods require careful tuning of the regularization coefficient `λ` (robust_coef):
- Too small: Insufficient robustness, poor generalization
- Too large: Over-regularization, slow learning, reduced performance

The adaptive mechanism addresses this by:
1. **Tracking performance**: Maintains a rolling buffer of episode rewards
2. **Detecting trends**: Compares recent performance to historical average
3. **Adjusting coefficient**: Maps performance trends to regularization strength

### Algorithm

The adaptive robust coefficient updates at the end of each training episode:

```python
def update_adaptive_robust_coef(self):
    # 1. Calculate statistics from rolling buffer
    buffer_array = np.array(self.performance_buffer)
    mean_performance = np.mean(buffer_array)
    std_performance = np.std(buffer_array)
    
    # 2. Get recent performance (last 10% of buffer)
    recent_window = max(1, len(self.performance_buffer) // 5)
    recent_performance = np.mean(list(self.performance_buffer)[-recent_window:])
    
    # 3. Calculate normalized improvement
    if std_performance > 0:
        normalized_improvement = (recent_performance - mean_performance) / std_performance
    else:
        normalized_improvement = 0
    
    # 4. Map improvement to coefficient using tanh
    scale_factor = 0.5
    adjustment = np.tanh(normalized_improvement * scale_factor)
    
    # 5. Map [-1, 1] to [min_coef, max_coef]
    coef_range = self.robust_coef_max - self.robust_coef_min
    self.robust_coef = self.robust_coef_min + (adjustment + 1) / 2 * coef_range
```

### Key Components

#### 1. Performance Buffer

A rolling buffer (default size: 250 episodes) stores mean episode rewards:
- **Purpose**: Tracks long-term performance trends
- **Size**: Configurable via `robust_buffer_size`
- **Update**: Appended at end of each episode

```python
self.performance_buffer = deque(maxlen=robust_buffer_size)
```

#### 2. Normalized Improvement Metric

Compares recent performance to historical average:
- **Recent window**: Last 20% of buffer (configurable)
- **Normalization**: Uses standard deviation for scale-invariance
- **Interpretation**: 
  - Positive: Performance improving → increase regularization
  - Negative: Performance degrading → decrease regularization

#### 3. Tanh Mapping

Maps normalized improvement to adjustment factor:
- **Function**: `adjustment = tanh(normalized_improvement * 0.5)`
- **Range**: [-1, 1]
- **Properties**: Smooth, bounded, saturates for extreme values

#### 4. Coefficient Range

Maps adjustment to actual coefficient value:
- **Range**: [`robust_coef_min`, `robust_coef_max`]
- **Default**: `[1e-5, 1e-3]` for adversarial methods
- **Linear mapping**: `coef = min + (adjustment + 1) / 2 * (max - min)`

### Integration with Training

The adaptive mechanism integrates seamlessly with the training loop:

```python
# In train.py
if done:
    # Log episode reward
    self.agent.log_epoch_reward(episode_reward)
    
    # Update adaptive robust_coef at end of episode
    epoch_mean_reward = self.agent.finalize_epoch()
    
    # Log statistics
    stats = self.agent.get_robust_stats()
    logger.log('train_robust/coef', stats['robust_coef'], step)
```

**Update Frequency**: Once per episode (not per step)

**Initialization**: Starts with `robust_coef` initial value, adapts after buffer fills

### Usage

#### Configuration

Enable adaptive robust coefficient in agent config:

```yaml
agent:
  params:
    adaptive_robust_coef: true
    robust_coef: 5e-4          # Initial value
    robust_coef_min: 1e-5      # Minimum bound
    robust_coef_max: 1e-3      # Maximum bound
    robust_buffer_size: 250    # Buffer size (episodes)
```

#### Command Line

```bash
python python_scripts/train.py \
    overrides=cube_in_hand \
    agent.params.adaptive_robust_coef=true \
    agent.params.robust_coef=5e-4 \
    agent.params.robust_coef_min=1e-5 \
    agent.params.robust_coef_max=1e-3 \
    agent.params.robust_buffer_size=250
```

#### Monitoring

The coefficient is logged to TensorBoard:
- `train_robust/coef`: Current robust coefficient value
- `train_robust/mean_performance`: Historical mean reward
- `train_robust/recent_performance`: Recent mean reward
- `train_robust/buffer_size`: Current buffer size

### Design Rationale

**Why rolling buffer?**
- Captures long-term trends, not just recent noise
- Provides stable statistics for comparison
- Adapts to non-stationary learning dynamics

**Why normalized improvement?**
- Scale-invariant: Works across different reward scales
- Relative comparison: Focuses on trends, not absolute values
- Robust to outliers: Standard deviation provides natural scaling

**Why tanh mapping?**
- Smooth transitions: Avoids abrupt coefficient changes
- Bounded: Prevents extreme values
- Saturates: Limits sensitivity to very large improvements

**Why episode-based updates?**
- Aligns with natural learning cycles
- Reduces computational overhead
- Provides stable performance estimates

### Expected Behavior

**Early Training**:
- Performance improving → Coefficient increases
- More regularization as policy becomes more capable

**Mid Training**:
- Performance plateaus → Coefficient stabilizes
- Balanced regularization for fine-tuning

**Late Training**:
- Performance may degrade → Coefficient decreases
- Less regularization to avoid over-constraining

### Advantages

1. **Automatic Tuning**: No manual hyperparameter search
2. **Adaptive**: Responds to learning dynamics
3. **Robust**: Handles non-stationary performance
4. **Interpretable**: Clear mapping from performance to regularization
5. **Efficient**: Minimal computational overhead

### Limitations

1. **Buffer Size**: Requires sufficient episodes to fill buffer
2. **Initialization**: Starts with fixed initial value
3. **Sensitivity**: May be sensitive to reward scale
4. **Non-stationarity**: Assumes performance trends are meaningful

---

## Usage

### Training

#### Basic Training

```bash
python python_scripts/train.py \
    overrides=cube_in_hand \
    seed=12345 \
    experiment=my_experiment
```

#### With Robust Regularization

```bash
python python_scripts/train.py \
    overrides=cube_in_hand \
    agent.params.robust_method=l2_adv_param \
    agent.params.robust_coef=5e-4 \
    experiment=robust_training
```

#### With Adaptive Robust Coefficient

```bash
python python_scripts/train.py \
    overrides=cube_in_hand \
    agent.params.robust_method=l2_adv_param \
    agent.params.adaptive_robust_coef=true \
    agent.params.robust_coef=5e-4 \
    agent.params.robust_coef_min=1e-5 \
    agent.params.robust_coef_max=1e-3 \
    experiment=adaptive_robust
```

### Using Shell Scripts

The repository includes convenient shell scripts for common experiments:

```bash
# Train cube in hand with adaptive robust coefficient
bash USPS/adaptive_manipulator.sh

# Train quadruped with adaptive robust coefficient
bash USPS/quadruped_adaptive.sh

# Train cartpole with adaptive robust coefficient
bash USPS/cartpole_adaptive.sh
```

### Evaluation

```bash
python python_scripts/test.py \
    overrides=cube_in_hand \
    agent_dir=./outputs/cube_in_hand-my_experiment/.../checkpoints \
    num_eval_episodes=100
```

### Monitoring

View TensorBoard logs:

```bash
tensorboard --logdir=./outputs
```

Or use the provided script:

```bash
bash USPS/view_tensorboard.sh
```

---

## Configuration

### Agent Configuration

Key parameters in `configs/agent/agent.yaml`:

```yaml
agent:
  params:
    # Robust method: "no", "l1_reg", "l2_reg", "l1_adv_param", "l2_adv_param", "linf_adv_param"
    robust_method: "l2_adv_param"
    
    # Robust coefficient (fixed or adaptive)
    robust_coef: 5e-4
    
    # Adaptive robust coefficient settings
    adaptive_robust_coef: true
    robust_coef_min: 1e-5
    robust_coef_max: 1e-3
    robust_buffer_size: 250
    
    # Learning rates
    actor_lr: 1e-4
    critic_lr: 1e-4
    alpha_lr: 1e-4
    
    # Network architecture
    hidden_dim: 1024
    hidden_depth: 2
    
    # Training hyperparameters
    batch_size: 1024
    discount: 0.99
    critic_tau: 0.005
```

### Environment Configuration

Example for cube in hand (`configs/overrides/cube_in_hand.yaml`):

```yaml
env:
  name: "cube_in_hand"
  class: "envs.cube_in_hand_env.CubeInHandEnv"
  params:
    control_dt: 0.01
    sim_decimation: 4
    action_scale: 0.15
    max_episode_steps: 1000
    task_kwargs:
      perturb_spec:
        mode: "range"
        params:
          object_mass:
            min: 0.5
            max: 2.0
```

### Training Configuration

Key parameters in `configs/train.yaml`:

```yaml
seed: 1
device: "cuda:0"

num_train_steps: 1e6
num_random_steps: 5e3
eval_frequency: 1e4
num_eval_episodes: 10

replay_buffer_capacity: 1e6
```

---

## Troubleshooting

### Common Issues

#### 1. `AttributeError: module 'setuptools._distutils' has no attribute 'version'`

**Solution**: Ensure you're using setuptools 59.5.0:
```bash
pip install setuptools==59.5.0
```

#### 2. `ModuleNotFoundError: No module named 'envs'`

**Solution**: The training scripts automatically add the repository root to `sys.path`. If you encounter this error:
- Ensure you're running scripts from the repository root
- Check that `USPS/` directory is in your Python path

#### 3. MuJoCo Rendering Issues

**macOS**:
```bash
export MUJOCO_GL=glfw
```

**Linux**:
```bash
export MUJOCO_GL=egl
```

#### 4. CUDA Out of Memory

**Solutions**:
- Reduce `batch_size` in agent config
- Reduce `replay_buffer_capacity`
- Use CPU: `device=cpu` (slower but works)

#### 5. Training Instability

**For cube in hand**:
- Start with smaller `robust_coef` (1e-5 to 1e-4)
- Use adaptive robust coefficient
- Increase `num_random_steps` for better exploration
- Check that action scaling is appropriate

---

## Acknowledgments
# Citation

This repository uses code from the following paper. If you use this code in your research, please cite:

```bibtex
@inproceedings{
    zhang2023robust,
    title={Robust Reinforcement Learning in Continuous Control Tasks with Uncertainty Set Regularization},
    author={Yuan Zhang and Jianhong Wang and Joschka Boedecker},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=keAPCON4jHC}
}
```

- MuJoCo physics simulator
- RealWorldRL Suite for standardized benchmarks
- Soft Actor-Critic (SAC) algorithm
- Hydra for configuration management
