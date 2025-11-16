# Environment Construction Guide

This guide explains how environments are constructed in this codebase and how to add your own MuJoCo environment.

## Overview

The training and testing scripts use **Hydra** for configuration management. Environments are instantiated via `hydra.utils.instantiate()` in the `infra/utils.py` module. The key function is `make_env(cfg)`, which creates an environment from the Hydra config.

## How Environments Are Created

### 1. Environment Factory (`infra/utils.py`)

The `make_env(cfg)` function creates environments:

```python
def make_env(cfg):
    """Helper function to create dm_control environment"""
    if "dm-" in cfg.env.name:
        # For dm_control environments (not fully implemented)
        raise NotADirectoryError
    else:
        # Uses Hydra to instantiate the environment class with params
        env = hydra.utils.instantiate(cfg.env)
    return env
```

### 2. Configuration Structure

Environments are configured in YAML files under `configs/overrides/`. The structure is:

```yaml
env: 
  name: "environment_name"           # Identifier for the environment
  class: "module.path.ClassName"     # Full Python path to env class
  params:                            # Keyword arguments for __init__
    param1: value1
    param2: value2
```

**Example** (`configs/overrides/a1_mujoco.yaml`):
```yaml
env: 
  name: "a1"
  class: "envs.a1_env_mujoco.A1EnvMujoco"
  params: 
    random: ${seed}
    real_robot: False
    task_name: locomotion
```

### 3. Environment Interface Requirements

All environments must:
1. **Inherit from `gym.Env`**
2. **Implement required methods:**
   - `__init__(**params)`: Constructor that accepts config params
   - `reset()`: Returns initial observation
   - `step(action)`: Returns `(observation, reward, done, info)`
   - `close()`: Cleanup
   - `render(mode)` (optional): Rendering support
3. **Define spaces:**
   - `self.observation_space`: Gym space (e.g., `gym.spaces.Box`)
   - `self.action_space`: Gym space (e.g., `gym.spaces.Box`)

## Existing Environment Examples

### Example 1: MuJoCo Environment (`envs/a1_env_mujoco.py`)

This wraps MuJoCo environments from the `walk_in_the_park` package:

```python
class A1EnvMujoco(gym.Env):
    def __init__(self, task_name="locomotion", random=2022, real_robot=False):
        if real_robot:
            from real.envs.a1_env import A1Real
            env = A1Real(zero_action=np.asarray([0.05, 0.9, -1.8] * 4))
        else:
            env_name = 'A1Run-v0'
            control_frequency = 20
            env = make_mujoco_env(
                env_name, 
                control_frequency=control_frequency,
                action_filter_high_cut=None,
                action_history=1)
        
        env = wrap_gym(env, rescale_actions=True)
        env.seed(random)
        self._env = env
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
```

### Example 2: PyBullet Environment (`envs/a1_env_bullet.py`)

Wraps PyBullet environments from `fine-tuning-locomotion`:

```python
class A1EnvBullet(gym.Env):
    def __init__(self, task_name="reset", random=2022, real_robot=False):
        self._env = env_builder.build_env(
            task_name,
            mode="train",
            enable_randomizer=enable_randomizer,
            enable_rendering=False,
            reset_at_current_position=False,
            use_real_robot=real_robot,
            realistic_sim=False)
        
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
```

### Example 3: Simple Custom Environment (`envs/toy_env.py`)

Minimal example showing the required interface:

```python
class ToyEnv(gym.Env):
    def __init__(self, perturb_method="l2", perturb_val=0.1, random=2022):
        self.observation_space = Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # ... initialization code ...
    
    def reset(self):
        # Return initial observation
        return self.curr_state
    
    def step(self, action):
        next_state = self.transition_fn(action)
        reward = self.reward_fn(next_state)
        done = self.done_fn(next_state)
        return next_state, reward, done, {}
```

## How to Add Your Own MuJoCo Environment

### Step 1: Create Your Environment Class

Create a new file `envs/my_mujoco_env.py`:

```python
import os, sys
import numpy as np
import gym
from gym.spaces import Box

class MyMujocoEnv(gym.Env):
    """
    Custom MuJoCo environment wrapper.
    
    This example assumes you have a MuJoCo XML file and want to use it
    with the dm_control or mujoco_py libraries.
    """
    
    def __init__(self, 
                 model_path="path/to/your/model.xml",
                 task_name="default_task",
                 random=2022,
                 control_frequency=20,
                 **kwargs):
        """
        Initialize the environment.
        
        Args:
            model_path: Path to MuJoCo XML model file
            task_name: Task identifier
            random: Random seed
            control_frequency: Control frequency in Hz
            **kwargs: Additional arguments
        """
        # TODO: Load your MuJoCo model here
        # Example with mujoco_py:
        # import mujoco_py
        # self.model = mujoco_py.load_model_from_path(model_path)
        # self.sim = mujoco_py.MjSim(self.model)
        # self.viewer = None
        
        # Or with dm_control:
        # from dm_control import mjcf
        # self.model = mjcf.from_path(model_path)
        # ...
        
        # Define observation and action spaces
        # Adjust dimensions based on your model
        obs_dim = 10  # Example: 10-dimensional observations
        action_dim = 6  # Example: 6-dimensional actions
        
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.action_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        self.seed(random)
        self.reset()
    
    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        if hasattr(self, 'sim'):
            self.sim.set_state(self.sim.get_state())  # For mujoco_py
        return [seed]
    
    def reset(self):
        """Reset environment to initial state."""
        # TODO: Reset your MuJoCo simulation
        # Example with mujoco_py:
        # self.sim.reset()
        # obs = self._get_obs()
        
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs
    
    def step(self, action):
        """
        Execute one timestep.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Clip action to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # TODO: Apply action to MuJoCo simulation
        # Example with mujoco_py:
        # self.sim.data.ctrl[:] = action
        # self.sim.step()
        
        # TODO: Compute reward, done, and get observation
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._is_done(obs)
        info = {}
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Extract observation from simulation."""
        # TODO: Extract observation from MuJoCo sim
        # Example:
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # obs = np.concatenate([qpos, qvel])
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs
    
    def _compute_reward(self, obs, action):
        """Compute reward based on observation and action."""
        # TODO: Implement your reward function
        return 0.0
    
    def _is_done(self, obs):
        """Check if episode is done."""
        # TODO: Implement termination conditions
        return False
    
    def render(self, mode='human'):
        """Render the environment."""
        # TODO: Implement rendering
        # Example with mujoco_py:
        # if self.viewer is None:
        #     self.viewer = mujoco_py.MjViewer(self.sim)
        # self.viewer.render()
        pass
    
    def close(self):
        """Clean up resources."""
        # TODO: Close viewers, etc.
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer = None
```

### Step 2: Create Configuration File

Create `configs/overrides/my_mujoco_env.yaml`:

```yaml
# @package _group_
env: 
  name: "my_mujoco"
  class: "envs.my_mujoco_env.MyMujocoEnv"
  params: 
    model_path: "path/to/your/model.xml"
    task_name: "default_task"
    random: ${seed}
    control_frequency: 20

max_episode_steps: 1000

# training params 
num_random_steps: 5e3
num_train_steps: 1e6
eval_frequency: 1e4
num_eval_episodes: 10

# algorithm params
sac_critic_lr: 1e-4
sac_actor_lr: 1e-4
sac_alpha_lr: 1e-4

# replay buffer
replay_buffer_capacity: ${num_train_steps}
```

### Step 3: Update `__init__.py` (if needed)

If your environment is in a subdirectory, ensure `envs/__init__.py` is set up correctly or import it directly in your class path.

### Step 4: Run Training/Testing

**Training:**
```bash
python python_scripts/train.py -cn train.yaml env=my_mujoco_env
```

**Testing:**
```bash
python python_scripts/test.py --experiments_dir /path/to/experiment --agent_dir /path/to/agent
```

## Key Points for Integration

1. **Gym Interface**: Your environment must follow the standard Gym interface
2. **Observation/Action Spaces**: Define `observation_space` and `action_space` as Gym spaces
3. **Config Integration**: Use Hydra configs with `class` and `params` structure
4. **Return Types**: 
   - `reset()` returns `observation` (numpy array)
   - `step(action)` returns `(observation, reward, done, info)` where:
     - `observation`: numpy array
     - `reward`: float
     - `done`: boolean
     - `info`: dict (can be empty)
5. **Action Space**: Training scripts expect actions in `[-1, 1]` range. If your actions are in a different range, either:
   - Set `action_space` to `[-1, 1]` and scale internally, OR
   - Adjust the action space bounds (the scripts will use `action_space.low` and `action_space.high`)

## Using dm_control for MuJoCo

If you want to use `dm_control` (which provides a higher-level interface to MuJoCo), you can follow the pattern from `A1EnvMujoco`:

1. Create a composer environment with your MuJoCo model
2. Convert to Gym using `DMCGYM` wrapper
3. Add any additional wrappers (FlattenObservation, TimeLimit, etc.)

See `envs/walk_in_the_park/env_utils.py` for examples of dm_control integration.

## Troubleshooting

- **Import errors**: Ensure your environment class path in the config matches the actual Python path
- **Action space issues**: Check that actions are properly clipped and scaled
- **Observation shape**: Ensure observation dimensions match what your agent expects
- **Seed/reproducibility**: Implement proper seeding in your environment

## New Environments: CubeInHand and ShadowHand

### CubeInHand Environment (`envs/cube_in_hand_env.py`)

The CubeInHand environment is a faithful port of the `torobo_mujoco` cube reorientation task.

**Setup:**
- Assets are located in `envs/torobo_assets/hand_v4/` (automatically copied from torobo_mujoco)
- Requires MuJoCo Python bindings (`pip install mujoco`)
- Requires `scipy` for quaternion math

**Observation Space:**
- Shape: `(92,)` - two stacked 46-dimensional frames
- Each 46-d frame contains:
  - Scaled joint positions (16): Joint positions normalized to [-1, 1] range
  - Object position (3): Cube center-of-mass position (x, y, z)
  - Object quaternion (4): Cube orientation in MuJoCo format (w, x, y, z)
  - Target position (3): Constant target position [0.13, -0.02, 0.49]
  - Target quaternion (4): Target orientation in MuJoCo format (w, x, y, z)
  - Quaternion difference (4): Orientation error between cube and target
  - Last processed action (12): Previous actuator target positions

**Action Space:**
- Shape: `(12,)` - Box space with bounds [-1, 1]
- Actions are relative position commands (delta positions)
- Internally scaled by `action_scale=0.15` and added to previous targets
- Final targets are clipped to soft joint limits (0.9 factor of hard limits)

**Reward:**
- `reward = -rotation_distance` where rotation distance is the quaternion error angle
- Dense, continuous reward signal
- Success when rotation distance < 0.3 radians (target updates but episode continues)

**Dynamics Perturbations (Always Enabled):**
The environment always applies dynamics perturbations at reset. Supported parameters:
- `object_mass`: ±30% variation (default: 0.7-1.3)
- `object_inertia`: ±30% variation (default: 0.7-1.3)
- `cube_friction`: ±20% variation (default: 0.8-1.2)
- `fingertip_friction`: ±40% variation (default: 0.6-1.4)
- `actuator_kp`: ±20% variation (default: 0.8-1.2) - position gain
- `actuator_kv`: ±20% variation (default: 0.8-1.2) - velocity gain
- `contact_damping`: ±30% variation (default: 0.7-1.3) - contact solver damping
- `gravity_scale`: ±5% variation (default: 0.95-1.05)

**Configuration:**
Use `configs/overrides/cube_in_hand.yaml`:
```bash
python python_scripts/train.py -cn train.yaml env=cube_in_hand
```

**Testing with Perturbation Sweeps:**
```bash
python python_scripts/test.py --experiments_dir <dir> --perturb_param object_mass --perturb_min 0.7 --perturb_max 1.3
```

### ShadowHand InHand Environments (`envs/inhand_env.py`)

Wrapper for OpenAI Gym ShadowHand manipulation tasks.

**Setup:**
- Requires `gym[robotics]` package: `pip install gym[robotics]`
- Requires MuJoCo Python bindings

**Supported Tasks:**
- `HandManipulateBlockRotateXYZ-v0`: Block rotation task
- `HandManipulatePen-v0`: Pen manipulation task
- `HandManipulateEggRotate-v0`: Egg rotation task

**Observation Space:**
- Flattened observation from Gym environment (dict observations are automatically flattened)
- Dtype: `np.float32` for compatibility with replay buffer

**Action Space:**
- Actions are automatically rescaled from [-1, 1] to environment's native action bounds
- Compatible with SAC agent which outputs [-1, 1] actions

**Reward:**
- Uses native Gym environment reward function (no modification)

**Dynamics Perturbations (Always Enabled):**
- `torque_scale`: ±30% variation (default: 0.7-1.3) - multiplies actions before applying
- `joint_damping`: ±20% variation (default: 0.8-1.2) - scales joint damping coefficients
- `joint_friction`: ±20% variation (default: 0.8-1.2) - scales joint friction losses
- `gravity_scale`: ±5% variation (default: 0.95-1.05) - scales gravity vector

**Configuration:**
Use task-specific configs:
- `configs/overrides/inhand_block.yaml`
- `configs/overrides/inhand_pen.yaml`
- `configs/overrides/inhand_egg.yaml`

Example usage:
```bash
python python_scripts/train.py -cn train.yaml env=inhand_block
```

**Testing with Perturbation Sweeps:**
```bash
python python_scripts/test.py --experiments_dir <dir> --perturb_param torque_scale --perturb_min 0.7 --perturb_max 1.3
```

## Perturbations and Reparameterizations

This section explains how different environments handle dynamics perturbations and reparameterizations for robustness evaluation and domain randomization.

### Perturbation Specification Format

All environments that support perturbations use a common `perturb_spec` structure passed via `task_kwargs`:

```yaml
task_kwargs:
  perturb_spec:
    mode: "range"  # or "fixed" for testing sweeps
    params:
      param_name:
        min: 0.7    # minimum value (for range mode)
        max: 1.3    # maximum value (for range mode)
        start: 1.0  # fixed value (for fixed mode, used by test.py)
```

**Modes:**
- `"range"`: Sample uniformly from `[min, max]` at each reset (used during training)
- `"fixed"`: Use a fixed value from `start` or `value` (used during testing sweeps)

### How Environments Handle Perturbations

#### 1. CubeInHand Environment (`envs/cube_in_hand_env.py`)

**Implementation:** Direct MuJoCo model parameter modification in `_apply_dynamics_perturb()`

**When Applied:** At every `reset()` call - perturbations are resampled each episode

**How It Works:**
- Perturbations are applied by directly modifying the MuJoCo model parameters
- The method `_apply_dynamics_perturb()` is called during `reset()`
- Model parameters are multiplied by the sampled perturbation value
- Original model is loaded fresh each time, so perturbations accumulate multiplicatively

**Supported Parameters:**
- `object_mass`: Multiplies `model.body_mass[object_body_id]`
- `object_inertia`: Multiplies `model.body_inertia[object_body_id, :]`
- `cube_friction`: Multiplies `model.geom_friction[cube_geom_ids, :]`
- `fingertip_friction`: Multiplies `model.geom_friction[fingertip_geom_ids, :]`
- `actuator_kp`: Multiplies `model.actuator_gainprm[actuator_ids, 0]` (position gain)
- `actuator_kv`: Multiplies `model.actuator_biasprm[actuator_ids, 1]` (velocity gain)
- `contact_damping`: Modifies `model.geom_solref` and `model.geom_solimp` (contact solver parameters)
- `gravity_scale`: Multiplies `model.opt.gravity[:]`

**Code Pattern:**
```python
def _apply_dynamics_perturb(self, perturb_spec):
    params = perturb_spec.get('params', {})
    mode = perturb_spec.get('mode', 'range')
    
    for param_name, param_config in params.items():
        if mode == 'range':
            value = self.rng.uniform(min_val, max_val)
        elif mode == 'fixed':
            value = param_config.get('start', 1.0)
        
        # Apply to MuJoCo model
        if param_name == 'object_mass':
            self.model.body_mass[self.object_body_id] *= value
        # ... other parameters
```

**Info Dict:** Current perturbation values are stored in `self.current_perturb` and returned in `info['perturb_spec']`

#### 2. ShadowHand InHand Environments (`envs/inhand_env.py`)

**Implementation:** Wrapper-based approach using `ShadowHandPerturbWrapper`

**When Applied:** At every `reset()` call - wrapper resamples perturbations

**How It Works:**
- Uses a Gym wrapper (`ShadowHandPerturbWrapper`) that wraps the base Gym environment
- Wrapper intercepts `reset()` and `step()` calls
- Perturbations are applied either to the MuJoCo model (if accessible) or to actions
- Accesses MuJoCo model via `env.unwrapped.sim.model` (Gym's MuJoCo environments expose this)

**Supported Parameters:**
- `torque_scale`: Applied in `step()` by multiplying actions before passing to environment
- `joint_damping`: Multiplies `sim_model.dof_damping[:]`
- `joint_friction`: Multiplies `sim_model.dof_frictionloss[:]`
- `gravity_scale`: Multiplies `sim_model.opt.gravity[:]`

**Code Pattern:**
```python
class ShadowHandPerturbWrapper(gym.Wrapper):
    def _sample_perturbations(self):
        # Sample values based on mode
        if mode == 'range':
            value = rng.uniform(min_val, max_val)
        elif mode == 'fixed':
            value = param_config.get('start', 1.0)
        
        # Apply to model or store for step()
        if param_name == 'torque_scale':
            # Store for step()
            pass
        elif param_name == 'joint_damping':
            self.sim_model.dof_damping[:] *= value
    
    def step(self, action):
        if 'torque_scale' in self.current_perturb:
            action = action * self.current_perturb['torque_scale']
        return self.env.step(action)
```

**Info Dict:** Perturbation values are injected into `info['perturb_spec']` in the wrapper

#### 3. RealWorldRL Suite Environments (`envs/rwrl_env.py`)

**Implementation:** Delegates to `realworldrl_suite` library's built-in perturbation system

**When Applied:** Handled internally by the `realworldrl_suite` library

**How It Works:**
- Passes `perturb_spec` directly to `rwrl.load()` function
- The `realworldrl_suite` library handles all perturbation logic internally
- Supports more complex perturbation schedules (e.g., `scheduler: "uniform"`, `period: 1`)

**Configuration Format:**
```yaml
task_kwargs:
  perturb_spec:
    enable: True/False
    period: 1
    scheduler: "uniform"  # or other schedulers
    # Additional params depend on realworldrl_suite API
```

**Note:** The exact parameters depend on the `realworldrl_suite` API and may vary by domain/task

#### 4. Toy Environment (`envs/toy_env.py`)

**Implementation:** Custom reparameterization via `set_perturb()` method

**When Applied:** At every `reset()` call via `set_perturb()`

**How It Works:**
- Uses a different approach: reparameterizes the transition function rather than physics parameters
- Modifies `self.cur_inertia` which affects the transition: `next_state = state + action * cur_inertia`
- Supports multiple sampling methods: `l2`, `ellipsoid`, `l+`

**Supported Methods:**
- `l2`: Samples from unit sphere, scales uniformly
- `ellipsoid`: Samples from unit sphere, applies transformation matrix
- `l+`: Independent uniform sampling per dimension

**Code Pattern:**
```python
def set_perturb(self):
    if self.perturb_method == "l2":
        z = np.random.randn(2)
        z_on_unit_sphere = z / np.linalg.norm(z, ord=2)
        scale = np.random.uniform(0, 1) ** 0.5
        z_scaled = z_on_unit_sphere * scale
        self.cur_inertia = self.init_inertia + z_scaled * self.perturb_val
```

**Configuration:** Uses `perturb_method` and `perturb_val` constructor parameters (not `perturb_spec`)

#### 5. A1 Environments (Mujoco/Bullet)

**A1EnvMujoco (`envs/a1_env_mujoco.py`):**
- No built-in perturbation support
- Wraps `walk_in_the_park` environments which may have their own perturbation mechanisms
- Uses `wrap_gym(env, rescale_actions=True)` wrapper

**A1EnvBullet (`envs/a1_env_bullet.py`):**
- Supports domain randomization via task name suffix: `task_name="locomotion-dr"`
- When `-dr` suffix is present, sets `enable_randomizer=True` in `env_builder.build_env()`
- Domain randomization is handled by the `fine-tuning-locomotion` package internally

### Training with Perturbations (Domain Randomization)

Training with perturbations enables **domain randomization** - a technique where the agent learns to be robust by training across a distribution of environment dynamics.

#### How Training Works with Perturbations

**Automatic Application:**
- Perturbations are automatically applied during training when configured in the environment's config file
- The training script (`train.py`) doesn't need any special handling - it just calls `env.reset()` normally
- Each episode reset triggers perturbation resampling (when `mode: "range"`)

**Training Loop:**
```python
# In train.py - simplified
while step < num_train_steps:
    if done:
        obs = self.env.reset()  # Perturbations resampled here!
        # ... episode setup ...
    
    action = agent.act(obs)
    next_obs, reward, done, _ = self.env.step(action)
    # ... store in replay buffer ...
```

**Key Behavior:**
1. **Each Episode = New Dynamics**: Every `reset()` call samples new perturbation values from the specified ranges
2. **Continuous Variation**: The agent experiences different dynamics throughout training
3. **Robustness Learning**: By training across the perturbation distribution, the agent learns policies that work across a range of conditions

#### Configuration for Training

**Example: CubeInHand with Perturbations**
```yaml
# configs/overrides/cube_in_hand.yaml
env: 
  name: "cube_in_hand"
  class: "envs.cube_in_hand_env.CubeInHandEnv"
  params: 
    task_kwargs:
      perturb_spec:
        mode: "range"  # Use "range" for training
        params:
          object_mass: {min: 0.7, max: 1.3}
          object_inertia: {min: 0.7, max: 1.3}
          cube_friction: {min: 0.8, max: 1.2}
          # ... other parameters
```

**Training Command:**
```bash
python python_scripts/train.py -cn train.yaml env=cube_in_hand
```

#### Training vs. Testing Perturbations

| Aspect | Training (`mode: "range"`) | Testing (`mode: "fixed"`) |
|--------|---------------------------|--------------------------|
| **Sampling** | Uniform random from `[min, max]` | Fixed value from `start` |
| **Frequency** | Resampled every episode | Fixed for all episodes |
| **Purpose** | Domain randomization for robustness | Evaluate specific conditions |
| **Variation** | High - each episode different | None - consistent across episodes |

#### Evaluation During Training

**Important Note:** Evaluation during training (via `evaluate()` in `train.py`) uses the **same perturbed environment**:
- Evaluation episodes also resample perturbations at each reset
- This means evaluation performance varies due to different perturbation values
- For consistent evaluation, you may want to:
  1. Use a separate evaluation environment with `mode: "fixed"` and `start: 1.0` (no perturbations)
  2. Or average over multiple evaluation episodes to account for perturbation variance

**Current Implementation:**
```python
# In train.py evaluate()
for episode in range(self.cfg.num_eval_episodes):
    obs = self.env.reset()  # Resamples perturbations!
    # ... evaluation ...
```

#### Benefits of Training with Perturbations

1. **Sim-to-Real Transfer**: Training with varied dynamics helps policies transfer to real robots
2. **Robustness**: Policies learn to handle uncertainty and variations
3. **Generalization**: Better performance across different conditions
4. **Safety**: Policies that work under perturbations are more reliable

#### Choosing Perturbation Ranges

**Guidelines:**
- **Start Conservative**: Begin with small ranges (e.g., ±10%) and increase gradually
- **Match Real Variations**: If possible, base ranges on real-world measurements
- **Balance**: Too small = limited robustness, too large = training instability
- **Per-Parameter**: Different parameters may need different ranges

**Example Ranges (from existing configs):**
- Mass/Inertia: ±30% (0.7-1.3) - common for object properties
- Friction: ±20% (0.8-1.2) - moderate variation
- Actuator gains: ±20% (0.8-1.2) - controller parameters
- Gravity: ±5% (0.95-1.05) - small, realistic variation

#### Disabling Perturbations During Training

To train without perturbations, either:
1. **Remove `perturb_spec`** from config:
```yaml
env: 
  params: 
    task_kwargs: {}  # No perturb_spec
```

2. **Use empty `params`**:
```yaml
env: 
  params: 
    task_kwargs:
      perturb_spec:
        mode: "range"
        params: {}  # No parameters to perturb
```

3. **Set all ranges to [1.0, 1.0]** (no variation):
```yaml
perturb_spec:
  mode: "range"
  params:
    object_mass: {min: 1.0, max: 1.0}
```

### Testing with Perturbations

The `test.py` script supports perturbation sweeps for robustness evaluation:

**Single Parameter Sweep:**
```bash
python python_scripts/test.py \
    --experiments_dir <dir> \
    --perturb_param object_mass \
    --perturb_min 0.7 \
    --perturb_max 1.3
```

**Multiple Parameters (All at Once):**
```bash
python python_scripts/test.py \
    --experiments_dir <dir> \
    --perturb_min_list "0.7|0.8|0.7" \
    --perturb_max_list "1.3|1.2|1.3"
```

**How Test Script Works:**
1. Creates `perturb_spec` with `mode: "fixed"` and `start: <sweep_value>`
2. Passes via `task_kwargs.perturb_spec` to environment
3. Environment applies fixed perturbation value for all episodes in that test
4. Results saved as JSON files with perturbation values in filename

**Test Script Pattern:**
```python
# In test.py - Current format (for RealWorldRL environments)
test_perturb_spec = {
    "enable": True,
    "period": 1,
    "param": self.args.perturb_param,
    "scheduler": "constant",
    "start": float(perturb_val)  # Fixed value for sweep
}
self.cfg.env.params.task_kwargs.perturb_spec = test_perturb_spec
self.env = utils.make_env(self.cfg)
```

**Note on Format Compatibility:**
- The test script currently uses a format designed for RealWorldRL environments
- Newer environments (CubeInHand, InHand) expect `perturb_spec` with `mode` and `params` structure
- For these environments, the test script would need to convert the format:
```python
# Converted format for CubeInHand/InHand environments
test_perturb_spec = {
    "mode": "fixed",
    "params": {
        self.args.perturb_param: {
            "start": float(perturb_val)
        }
    }
}
```

### Adversarial Training Implementation

This codebase implements **adversarial training** for robust reinforcement learning. Unlike standard domain randomization (which samples perturbations uniformly), adversarial training actively finds the worst-case perturbations that minimize the value function, then trains the agent to be robust against these adversarial perturbations.

#### Overview

Adversarial training in this codebase works by:
1. **Computing Adversarial Directions**: Finding observation-space directions that decrease the value function
2. **Robust Regularization**: Penalizing the value function in these adversarial directions during training
3. **Online Computation**: Computing adversarial directions on-the-fly during training updates

#### How Adversarial Training Works

**Step 1: Adversarial Direction Computation**

The replay buffer computes adversarial directions by taking the **negative gradient** of the value function with respect to next observations:

```python
# In replay_buffer.py - sample_and_adv()
x = next_obses.clone()
x.requires_grad = True
value = model._compute_target_value(x)  # V(s')
value.backward()  # Compute gradient
neg_grad = -x.grad.data  # Negative gradient = direction that decreases V
# Normalize to unit L2 norm
next_obs_dir = neg_grad / ||neg_grad||_2 * sqrt(obs_dim)
```

**Mathematical Formulation:**
- Adversarial direction: `d = -∇_s V(s') / ||∇_s V(s')||_2`
- This direction points in the direction of **steepest decrease** of the value function
- The normalization ensures the direction has unit L2 norm scaled by observation dimension

**Step 2: Robust Regularizer Computation**

The agent computes a robust regularizer that penalizes the value function in adversarial directions:

```python
# In agent.py - _compute_reg()
# Sample perturbations around next_obs
delta_obs_sample ~ N(0, sigma^2)  # Gaussian noise
next_obs_sample = next_obs + delta_obs_sample

# Compute value at perturbed state
target_V_sample = V(next_obs_sample)

# Gradient of perturbation probability
next_obs_prob_grad = delta_obs_sample / sigma^2

# Adversarial regularization term
if "adv" in robust_method:
    target_V_reg = V(s') * (delta_obs / sigma^2) * d  # [bs, obs_dim]
    # Project onto norm (L2, L1, or Linf)
    target_V_reg = ||target_V_reg||_p  # [bs, 1]
```

**Mathematical Formulation:**
- Regularizer: `R(s') = E_δ[V(s' + δ) * (δ/σ²) · d]` where `d` is the adversarial direction
- For adversarial training: `R(s') = E_δ[V(s' + δ) * (δ/σ²) · (-∇V(s')/||∇V(s')||)]`
- This measures how much the value function decreases in the adversarial direction

**Step 3: Value Function Update**

The target value function is modified to include the robust regularizer:

```python
# In agent.py - update_critic()
reg_V = _compute_reg(next_obs, next_obs_dir)
target_V = (1 - robust_coef) * target_V - robust_coef * reg_V
```

**Mathematical Formulation:**
- Modified target: `V_target = (1-λ)V(s') - λR(s')`
- Where `λ = robust_coef` controls the strength of adversarial training
- The regularizer is **subtracted** (negative sign), which penalizes low values in adversarial directions

#### Robust Method Options

The `robust_method` parameter controls the type of robust training:

| Method | Description | Norm Type |
|--------|-------------|-----------|
| `"no"` | No robust training (standard SAC) | N/A |
| `"l2_adv_param"` | **Adversarial training** with L2 norm | L2 (Euclidean) |
| `"l1_adv_param"` | **Adversarial training** with L1 norm | L1 (Manhattan) |
| `"linf_adv_param"` | **Adversarial training** with Linf norm | Linf (Chebyshev) |
| `"l2_param"` | Non-adversarial L2 regularization | L2 |
| `"l1_param"` | Non-adversarial L1 regularization | L1 |
| `"l2_reg"` | L2 weight regularization (on network parameters) | L2 |
| `"l1_reg"` | L1 weight regularization (on network parameters) | L1 |

**Key Distinction:**
- Methods with `"adv"` use adversarial directions (negative gradient of value function)
- Methods without `"adv"` use isotropic/non-directional regularization

#### Configuration

**Agent Configuration:**
```yaml
# configs/agent/sac.yaml
agent:
  params:
    robust_method: "l2_adv_param"  # Adversarial training method
    robust_coef: 5e-4              # Regularization coefficient (λ)
```

**Training Command:**
```bash
python python_scripts/train.py \
    -cn train.yaml \
    env=quadruped_walk \
    agent.params.robust_method=l2_adv_param \
    agent.params.robust_coef=5e-4
```

#### Implementation Details

**1. Online Adversarial Direction Computation**

Adversarial directions are computed **on-the-fly** during each training update:

```python
# In agent.py - update()
if "adv" in self.robust_method:
    # Compute adversarial directions for current batch
    obs, ..., next_obs_dir = replay_buffer.sample_and_adv(batch_size, self)
else:
    # Use default directions (all ones)
    obs, ..., next_obs_dir = replay_buffer.sample(batch_size)
```

**Why Online?**
- Value function changes during training, so adversarial directions must be recomputed
- Computing directions on-the-fly ensures they're always aligned with current value function
- More computationally expensive but more accurate than pre-computed directions

**2. Adversarial Direction Storage (Optional)**

The replay buffer also supports pre-computing adversarial directions:

```python
# In replay_buffer.py - compute_adv_dir()
def compute_adv_dir(self, model, ratio=0.1):
    # Pre-compute adversarial directions for 10% of buffer
    for i in range(update_len):
        x = next_obses[i]
        x.requires_grad = True
        value = model._compute_target_value(x.unsqueeze(0))
        value.backward()
        neg_grad = -x.grad.data
        next_obs_dir = neg_grad / ||neg_grad||_2 * sqrt(obs_dim)
        self.next_obses_dir[i] = next_obs_dir
```

**Note:** This method is available but the current implementation uses `sample_and_adv()` for online computation.

**3. Regularization Computation**

The regularizer uses Monte Carlo sampling:

```python
# Sample M perturbations
for m in range(M):
    delta_obs_sample ~ N(0, sigma^2)  # sigma = 1.0
    next_obs_sample = next_obs + delta_obs_sample
    target_V_sample = V(next_obs_sample)
    
    # Compute regularization term
    prob_grad = delta_obs_sample / sigma^2
    if "adv" in robust_method:
        reg_term = V(s') * prob_grad * adversarial_dir
    else:
        reg_term = V(s') * prob_grad
    
    # Apply norm
    reg_term = ||reg_term||_p
    
# Average over samples
reg_V = mean(reg_terms)
```

**4. Value Function Modification**

The target value is modified to penalize adversarial directions:

```python
# Standard target value
target_V = min(Q1, Q2) - alpha * log_prob

# Add robust regularizer
reg_V = _compute_reg(next_obs, next_obs_dir)
target_V = (1 - robust_coef) * target_V - robust_coef * reg_V

# Use modified target for Q-learning
target_Q = reward + discount * target_V
```

#### Theoretical Foundation

**Adversarial Training Objective:**

The adversarial training objective can be viewed as solving a **min-max optimization**:

```
min_θ max_δ E[Q(s,a) - (1-λ)V(s') - λR(s', δ)]
```

Where:
- `θ`: Policy/critic parameters
- `δ`: Adversarial perturbation in observation space
- `R(s', δ)`: Robust regularizer measuring value decrease in adversarial direction

**Connection to Distributional Robustness:**

Adversarial training is related to **distributionally robust optimization**:
- Instead of training on the nominal distribution, train on worst-case distributions
- The adversarial direction finds the worst-case perturbation
- The regularizer measures sensitivity to these perturbations

#### Comparison: Adversarial vs. Domain Randomization

| Aspect | Domain Randomization | Adversarial Training |
|--------|---------------------|---------------------|
| **Perturbation Selection** | Uniform random sampling | Gradient-based worst-case |
| **Computation** | Environment-level (physics params) | Agent-level (observation space) |
| **When Applied** | During environment reset | During value function update |
| **Objective** | Robust to random variations | Robust to worst-case variations |
| **Computational Cost** | Low (just sampling) | Higher (gradient computation) |

**Key Insight:** Adversarial training finds the **hardest** perturbations (those that decrease value the most), while domain randomization uses **random** perturbations. They can be combined for even stronger robustness.

#### Best Practices

1. **Start with Small `robust_coef`**: Begin with `1e-4` to `5e-4` and adjust
2. **Use L2 Norm**: `l2_adv_param` is most commonly used and stable
3. **Combine with Domain Randomization**: Use both for maximum robustness
4. **Monitor Regularization Loss**: Check `train_critic/value_reg` in logs
5. **Tune Based on Task**: Different tasks may need different `robust_coef` values

#### Example: Full Training Pipeline

```python
# 1. Environment with domain randomization
env = make_env(cfg)  # Has perturb_spec with mode="range"

# 2. Agent with adversarial training
agent = SACAgent(
    robust_method="l2_adv_param",
    robust_coef=5e-4
)

# 3. Training loop
for step in range(num_steps):
    # Collect experience (with random dynamics)
    obs = env.reset()  # Resamples physics perturbations
    action = agent.act(obs)
    next_obs, reward, done, _ = env.step(action)
    
    # Update agent (with adversarial training)
    agent.update(replay_buffer, logger, step)
    # Inside update():
    #   - Computes adversarial directions from value function
    #   - Computes robust regularizer
    #   - Modifies target value to penalize adversarial directions
```

### Best Practices for Adding Perturbations

1. **Use `task_kwargs` for Configuration:**
   - Always accept `task_kwargs` parameter in `__init__()`
   - Extract `perturb_spec` from `task_kwargs.get('perturb_spec', {...})`

2. **Support Both Modes:**
   - `"range"` mode: Sample uniformly for training/domain randomization
   - `"fixed"` mode: Use fixed value for testing sweeps

3. **Apply at Reset:**
   - Resample perturbations at every `reset()` call
   - This ensures each episode has different dynamics (for range mode)

4. **Store Current Values:**
   - Keep `self.current_perturb` dict with current values
   - Return in `info['perturb_spec']` for logging/debugging

5. **Handle Model Access:**
   - For MuJoCo: Access via `self.model` (direct) or `env.unwrapped.sim.model` (wrapped)
   - For other simulators: Use appropriate API

6. **Multiplicative Scaling:**
   - Most perturbations multiply existing values (e.g., `value *= scale`)
   - This preserves relative relationships and is more intuitive

7. **Info Dict:**
   - Always include perturbation values in `info` dict for analysis
   - Helps track which perturbations were active during episodes

## Summary

1. Create a Gym-compatible environment class in `envs/`
2. Create a Hydra config file in `configs/overrides/`
3. Specify `class` as full Python path and `params` as constructor arguments
4. Run training/testing with the override: `python train.py -cn train.yaml env=my_env`
5. For environments with perturbations, use `test.py` with `--perturb_param` for robustness evaluation
