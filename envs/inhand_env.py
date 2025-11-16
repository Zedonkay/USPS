"""
ShadowHand InHand Manipulation Environment Wrapper
Provides Gym-compatible interface for OpenAI ShadowHand tasks with dynamics perturbations
"""

import os
import numpy as np
import gym
from gym.spaces import Box, Dict
from gym.wrappers import TimeLimit, FlattenObservation
from scipy.spatial.transform import Rotation as R
import mujoco


class RescaleAction(gym.ActionWrapper):
    """Rescale actions from [-1, 1] to environment's action bounds."""
    
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, Box), "RescaleAction only works with Box action spaces"
        
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype
        )
    
    def action(self, action):
        """Rescale action from [-1, 1] to [low, high]."""
        # Linear interpolation
        return self.low + (action + 1.0) * 0.5 * (self.high - self.low)


class ShadowHandPerturbWrapper(gym.Wrapper):
    """Wrapper that applies dynamics perturbations to ShadowHand environments."""
    
    def __init__(self, env, perturb_spec):
        super().__init__(env)
        self.perturb_spec = perturb_spec
        self.current_perturb = {}
        
        # Check if environment exposes MuJoCo model
        self.has_mujoco_model = False
        try:
            self.sim_model = env.unwrapped.sim.model
            self.has_mujoco_model = True
        except (AttributeError, NameError):
            # Some environments might not expose MuJoCo directly
            pass
        
        # Sample initial perturbations
        self._sample_perturbations()
    
    def _sample_perturbations(self):
        """Sample perturbation values based on spec."""
        params = self.perturb_spec.get('params', {})
        mode = self.perturb_spec.get('mode', 'range')
        self.current_perturb = {}
        
        # Use environment's RNG if available, otherwise use numpy random
        if hasattr(self.env.unwrapped, 'np_random'):
            rng = self.env.unwrapped.np_random
        else:
            rng = np.random
        
        for param_name, param_config in params.items():
            if mode == 'range':
                min_val = param_config.get('min', 1.0)
                max_val = param_config.get('max', 1.0)
                value = rng.uniform(min_val, max_val)
            elif mode == 'fixed':
                # For tester sweeps
                value = param_config.get('start', param_config.get('value', 1.0))
            else:
                value = 1.0
            
            self.current_perturb[param_name] = value
            
            # Apply perturbations to MuJoCo model if available
            if self.has_mujoco_model and param_name == 'torque_scale':
                # Torque scaling is applied in step(), store value
                pass
            elif self.has_mujoco_model and param_name == 'joint_damping':
                # Scale joint damping
                self.sim_model.dof_damping[:] *= value
            elif self.has_mujoco_model and param_name == 'joint_friction':
                # Scale joint friction
                self.sim_model.dof_frictionloss[:] *= value
            elif self.has_mujoco_model and param_name == 'gravity_scale':
                # Scale gravity
                self.sim_model.opt.gravity[:] *= value
    
    def reset(self, **kwargs):
        """Reset environment and resample perturbations."""
        self._sample_perturbations()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Apply torque scaling to action if specified."""
        # Apply torque scaling if specified
        if 'torque_scale' in self.current_perturb:
            action = action * self.current_perturb['torque_scale']
        
        obs, reward, done, info = self.env.step(action)
        
        # Inject perturbation info
        if not isinstance(info, dict):
            info = {}
        info['perturb_spec'] = self.current_perturb.copy()
        
        return obs, reward, done, info


class InHandEnv(gym.Env):
    """Wrapper for OpenAI ShadowHand manipulation tasks."""
    
    def __init__(self,
                 env_id,
                 action_rescale=True,
                 obs_flatten=True,
                 max_episode_steps=100,
                 task_kwargs=None,
                 random=2022,
                 gym_kwargs=None):
        """Initialize the InHand environment wrapper.
        
        Args:
            env_id: Gym environment ID (e.g., "HandManipulateBlockRotateXYZ-v0")
            action_rescale: Whether to rescale actions from [-1,1] to env bounds
            obs_flatten: Whether to flatten dict observations
            max_episode_steps: Maximum episode length
            task_kwargs: Dict containing perturb_spec
            random: Random seed
            gym_kwargs: Optional kwargs passed to gym.make
        """
        super().__init__()
        
        self.env_id = env_id
        self.task_kwargs = task_kwargs or {}
        
        # Create base environment
        gym_kwargs = gym_kwargs or {}
        self.base_env = gym.make(env_id, **gym_kwargs)
        
        # Apply wrappers
        env = self.base_env
        
        # Flatten observations if needed
        if obs_flatten and isinstance(env.observation_space, Dict):
            env = FlattenObservation(env)
        
        # Rescale actions if needed
        if action_rescale and isinstance(env.action_space, Box):
            env = RescaleAction(env)
        
        # Apply time limit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        
        # Apply perturbation wrapper
        perturb_spec = self.task_kwargs.get('perturb_spec', {
            'mode': 'range',
            'params': {}
        })
        env = ShadowHandPerturbWrapper(env, perturb_spec)
        
        self._env = env
        
        # Expose spaces
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
        # Ensure observation dtype is float32
        if isinstance(self.observation_space, Box):
            self.observation_space = Box(
                low=self.observation_space.low.astype(np.float32),
                high=self.observation_space.high.astype(np.float32),
                shape=self.observation_space.shape,
                dtype=np.float32
            )
        
        # Seed environment
        self.seed(random)
        
        print(f'\nObservation Space: {self.observation_space.shape} dtype: {self.observation_space.dtype}')
        print(f'Action Space: {self.action_space.shape} dtype: {self.action_space.dtype}')
    
    def seed(self, seed=None):
        """Set random seed."""
        return self._env.seed(seed)
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs = self._env.reset(**kwargs)
        # Ensure float32 dtype
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        elif isinstance(obs, dict):
            obs = {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        return obs
    
    def step(self, action):
        """Execute one step."""
        obs, reward, done, info = self._env.step(action)
        # Ensure float32 dtype
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        elif isinstance(obs, dict):
            obs = {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render environment."""
        return self._env.render(mode=mode)
    
    def close(self):
        """Close environment."""
        return self._env.close()
    
    def __getattr__(self, attr):
        """Forward attribute access to wrapped environment."""
        return getattr(self._env, attr)

