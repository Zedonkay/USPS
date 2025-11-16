"""
Cube in Hand Environment - Torobo MuJoCo Reorientation Task
Converted to Gym-compatible environment with dynamics perturbations
"""

import os
import numpy as np
import gym
from gym.spaces import Box
from collections import deque
from scipy.spatial.transform import Rotation as R
import mujoco


class CubeInHandEnv(gym.Env):
    """Cube reorientation environment matching original torobo_mujoco implementation."""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 model_path=None,
                 control_dt=0.01,
                 sim_decimation=4,
                 action_scale=0.15,
                 soft_joint_limit_factor=0.9,
                 success_tolerance=0.3,
                 frame_stack=2,
                 max_episode_steps=1000,
                 task_kwargs=None,
                 random=2022):
        """Initialize the environment.
        
        Args:
            model_path: Path to MuJoCo XML file
            control_dt: Simulation timestep
            sim_decimation: Number of substeps per control step
            action_scale: Scaling factor for actions
            soft_joint_limit_factor: Safety factor for joint limits
            success_tolerance: Success threshold in radians
            frame_stack: Number of frames to stack (fixed at 2)
            max_episode_steps: Maximum episode length
            task_kwargs: Dict containing perturb_spec
            random: Random seed
        """
        super().__init__()
        
        # Set default model path if not provided
        if model_path is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(cur_dir, "torobo_assets/hand_v4/hand_v4_with_cube.xml")
        
        # Store parameters
        self.control_dt = control_dt
        self.sim_decimation = sim_decimation
        self.action_scale = action_scale
        self.soft_joint_limit_factor = soft_joint_limit_factor
        self.success_tolerance = success_tolerance
        self.frame_stack = frame_stack
        self.max_episode_steps = max_episode_steps
        self.task_kwargs = task_kwargs or {}
        
        # Parse perturbation spec (always present)
        self.perturb_spec = self.task_kwargs.get('perturb_spec', {
            'mode': 'range',
            'params': {}
        })
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = control_dt
        self.data = mujoco.MjData(self.model)
        
        # Initialize RNG
        self.rng = np.random.RandomState(seed=random)
        
        # Cache MuJoCo IDs
        self._cache_mujoco_ids()
        
        # Precompute joint limits
        self._precompute_joint_limits()
        
        # Initialize buffers
        self._init_buffers()
        
        # Initialize viewer (lazy)
        self.viewer = None
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(92,),  # 46 * 2 frames
            dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )
        
        # Episode counters
        self.episode_step = 0
        self.count_lowlevel = 0
        
        # Current perturbation values (for info dict)
        self.current_perturb = {}
        
        print(f'\nObservation Space: {self.observation_space.shape}')
        print(f'Action Space: {self.action_space.shape}')
    
    def _cache_mujoco_ids(self):
        """Cache MuJoCo IDs for fast lookup."""
        # Body IDs
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        
        # Geom IDs - cube
        self.cube_geom_ids = []
        geom_names = ["object", "object_hidden"]
        for name in geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                self.cube_geom_ids.append(geom_id)
        
        # Geom IDs - fingertips (we'll identify these by name patterns)
        self.fingertip_geom_ids = []
        # Fingertips are typically the last link of each finger
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and ('link_3' in geom_name or 'link4' in geom_name):
                self.fingertip_geom_ids.append(i)
        
        # Joint IDs - all 16 joints
        self.joint_ids = []
        self.isaac_joint_names = [
            'hand_first_finger_base_joint', 'hand_second_finger_base_joint', 'hand_third_finger_base_joint',
            'hand_thumb_joint_1', 'hand_first_finger_joint_1', 'hand_second_finger_joint_1',
            'hand_thumb_joint_2', 'hand_first_finger_joint_2', 'hand_third_finger_joint_1',
            'hand_second_finger_joint_2', 'hand_thumb_joint_3', 'hand_first_finger_joint_3',
            'hand_third_finger_joint_2', 'hand_second_finger_joint_3', 'hand_thumb_joint_4', 'hand_third_finger_joint_3'
        ]
        self.mujoco_joint_names = [
            'hand_first_finger_base_joint', 'hand_first_finger_joint_1', 'hand_first_finger_joint_2', 'hand_first_finger_joint_3',
            'hand_second_finger_base_joint', 'hand_second_finger_joint_1', 'hand_second_finger_joint_2', 'hand_second_finger_joint_3',
            'hand_third_finger_base_joint', 'hand_third_finger_joint_1', 'hand_third_finger_joint_2', 'hand_third_finger_joint_3',
            'hand_thumb_joint_1', 'hand_thumb_joint_2', 'hand_thumb_joint_3', 'hand_thumb_joint_4'
        ]
        self.actuated_joint_names = [
            'hand_first_finger_base_joint', 'hand_second_finger_base_joint',
            'hand_thumb_joint_1', 'hand_first_finger_joint_1', 'hand_second_finger_joint_1',
            'hand_thumb_joint_2', 'hand_first_finger_joint_2', 'hand_third_finger_joint_1',
            'hand_second_finger_joint_2', 'hand_thumb_joint_3', 'hand_third_finger_joint_2', 'hand_thumb_joint_4'
        ]
        
        # Map joint names to IDs and create reordering indices
        mujoco_joint_ids = []
        for name in self.mujoco_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                mujoco_joint_ids.append(joint_id)
        
        # Create mapping from MuJoCo order to Isaac order
        self.mujoco2isaac_indices = []
        for isaac_name in self.isaac_joint_names:
            try:
                idx = self.mujoco_joint_names.index(isaac_name)
                self.mujoco2isaac_indices.append(idx)
            except ValueError:
                # Joint might not exist, skip
                pass
        
        # Actuator IDs - 12 actuators
        self.actuator_ids = []
        for name in self.actuated_joint_names:
            # MuJoCo actuators use joint names
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id >= 0:
                self.actuator_ids.append(act_id)
        
        # Ensure we have exactly 12 actuators
        if len(self.actuator_ids) != 12:
            # Try alternative: actuators are indexed sequentially
            self.actuator_ids = list(range(12))
    
    def _precompute_joint_limits(self):
        """Precompute joint limit arrays."""
        # Lower and upper limits (16 DOFs)
        self.lower_pos_limit = np.array([
            -0.10472, -0.10472, -0.10472,
            0, 0, 0,
            -0.13962, -0.13962, 0,
            -0.13962, -0.314159, -0.13962,
            -0.13962, -0.13962, -0.314159, -0.13962
        ], dtype=np.float64)
        
        self.upper_pos_limit = np.array([
            0.10472, 0.10472, 0.10472,
            1.5009, 1.5009, 1.5009,
            0.9250, 1.8849, 1.5009,
            1.8849, 1.53588, 1.8849,
            1.8849, 1.8849, 1.53588, 1.8849
        ], dtype=np.float64)
        
        # Soft limits with safety factor
        mean_limit = 0.5 * (self.lower_pos_limit + self.upper_pos_limit)
        self.lower_pos_limit_soft = mean_limit - (mean_limit - self.lower_pos_limit) * self.soft_joint_limit_factor
        self.upper_pos_limit_soft = mean_limit + (self.upper_pos_limit - mean_limit) * self.soft_joint_limit_factor
        
        # Actuated joint limits (12 DOFs) - extract for actuated joints
        actuated_lower = np.array([self.lower_pos_limit_soft[self.mujoco2isaac_indices[i]] 
                                    for i, name in enumerate(self.isaac_joint_names) 
                                    if name in self.actuated_joint_names])
        actuated_upper = np.array([self.upper_pos_limit_soft[self.mujoco2isaac_indices[i]] 
                                    for i, name in enumerate(self.isaac_joint_names) 
                                    if name in self.actuated_joint_names])
        
        # Apply soft limit factor again to actuated limits
        actuated_mean = 0.5 * (actuated_lower + actuated_upper)
        self.actuated_lower_pos_limit = actuated_mean - (actuated_mean - actuated_lower) * self.soft_joint_limit_factor
        self.actuated_upper_pos_limit = actuated_mean + (actuated_upper - actuated_mean) * self.soft_joint_limit_factor
        
        # Target position (constant)
        self.target_pos = np.array([0.13, -0.02, 0.49], dtype=np.float64)
    
    def _init_buffers(self):
        """Initialize observation buffers."""
        self.prev_target_q = np.zeros(12, dtype=np.float64)
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.prev_joint_pos = np.zeros(16, dtype=np.float64)
        self.prev_object_pos = np.zeros(3, dtype=np.float64)
        self.prev_object_quat = np.zeros(4, dtype=np.float64)
        self.prev_target_pos = np.zeros(7, dtype=np.float64)
        self.prev_target_quat_diff = np.zeros(4, dtype=np.float64)
        self.prev_last_processed_action = np.zeros(12, dtype=np.float64)
        
        # Frame deque for stacking
        self.frame_deque = deque(maxlen=self.frame_stack)
    
    def _scale(self, x, lower, upper):
        """Scale values from [lower, upper] to [-1, 1]."""
        return (2.0 * x - upper - lower) / (upper - lower)
    
    def _build_frame(self):
        """Build a single 46-d observation frame."""
        qpos = self.data.qpos.astype(np.float64)
        
        # Extract joint positions (last 16 are hand joints)
        q = qpos[-16:]
        
        # Reorder to Isaac order
        q_isaac = q[self.mujoco2isaac_indices]
        
        # Scale joint positions
        joint_pos = self._scale(q_isaac, self.lower_pos_limit, self.upper_pos_limit)
        
        # Extract object position
        object_pos = qpos[4:7]
        
        # Extract object quaternion (MuJoCo order: w,x,y,z)
        object_quat = qpos[7:11]
        
        # Extract target quaternion (MuJoCo order: w,x,y,z)
        target_quat = qpos[0:4]
        
        # Compute quaternion difference
        # Convert to scipy format (x,y,z,w)
        obj_quat_scipy = object_quat[[1, 2, 3, 0]]
        target_quat_scipy = target_quat[[1, 2, 3, 0]]
        
        R_obj = R.from_quat(obj_quat_scipy)
        R_target = R.from_quat(target_quat_scipy)
        quat_diff_scipy = (R_obj * R_target.inv()).as_quat()
        
        # Convert back to MuJoCo order (w,x,y,z)
        target_quat_diff = quat_diff_scipy[[3, 0, 1, 2]]
        
        # Last processed action
        last_processed_action = self.prev_target_q.copy()
        
        # Concatenate frame: 16 + 3 + 4 + 3 + 4 + 4 + 12 = 46 dimensions
        frame = np.concatenate([
            joint_pos,          # 16
            object_pos,         # 3
            object_quat,        # 4
            self.target_pos,    # 3
            target_quat,        # 4
            target_quat_diff,   # 4
            last_processed_action  # 12
        ])
        
        return frame.astype(np.float32)
    
    def _compute_reward_and_success(self):
        """Compute reward and check for success."""
        qpos = self.data.qpos.astype(np.float64)
        
        # Extract quaternions
        obj_quat_mj = qpos[7:11]  # MuJoCo order: w,x,y,z
        target_quat_mj = qpos[0:4]  # MuJoCo order: w,x,y,z
        
        # Convert to scipy format (x,y,z,w)
        obj_quat_scipy = obj_quat_mj[[1, 2, 3, 0]]
        target_quat_scipy = target_quat_mj[[1, 2, 3, 0]]
        
        # Create Rotation objects
        R_obj = R.from_quat(obj_quat_scipy)
        R_target = R.from_quat(target_quat_scipy)
        
        # Compute quaternion difference
        quat_diff = (R_obj * R_target.inv()).as_quat()  # Returns scipy format (x,y,z,w)
        
        # Compute rotation distance
        rot_dist = 2 * np.arcsin(min(np.linalg.norm(quat_diff[0:3]), 1.0))
        
        # Reward is negative rotation distance
        reward = -rot_dist
        
        # Check for success
        success = (rot_dist < self.success_tolerance)
        
        # Update target if successful
        if success:
            # Generate new random target
            delta_R = R.random(random_state=self.rng)
            new_target_R = delta_R * R_target
            new_target_quat_scipy = new_target_R.as_quat()  # (x,y,z,w)
            
            # Convert to MuJoCo order (w,x,y,z)
            new_target_quat_mj = new_target_quat_scipy[[3, 0, 1, 2]]
            
            # Update MuJoCo state
            self.data.qpos[0:4] = new_target_quat_mj
        
        return float(reward), bool(success), float(rot_dist)
    
    def _apply_dynamics_perturb(self, perturb_spec):
        """Apply dynamics perturbations to MuJoCo model."""
        params = perturb_spec.get('params', {})
        mode = perturb_spec.get('mode', 'range')
        self.current_perturb = {}
        
        # Sample or use fixed values
        for param_name, param_config in params.items():
            if mode == 'range':
                min_val = param_config.get('min', 1.0)
                max_val = param_config.get('max', 1.0)
                value = self.rng.uniform(min_val, max_val)
            elif mode == 'fixed':
                # For tester sweeps
                value = param_config.get('start', param_config.get('value', 1.0))
            else:
                value = 1.0
            
            self.current_perturb[param_name] = value
            
            # Apply perturbations
            if param_name == 'object_mass':
                if self.object_body_id >= 0:
                    self.model.body_mass[self.object_body_id] *= value
            
            elif param_name == 'object_inertia':
                if self.object_body_id >= 0:
                    self.model.body_inertia[self.object_body_id, :] *= value
            
            elif param_name == 'cube_friction':
                for geom_id in self.cube_geom_ids:
                    if geom_id >= 0:
                        self.model.geom_friction[geom_id, :] *= value
            
            elif param_name == 'fingertip_friction':
                for geom_id in self.fingertip_geom_ids:
                    if geom_id >= 0 and geom_id < self.model.ngeom:
                        self.model.geom_friction[geom_id, :] *= value
            
            elif param_name == 'actuator_kp':
                # Position gain (gainprm[:, 0])
                for act_id in self.actuator_ids:
                    if act_id < self.model.nu:
                        self.model.actuator_gainprm[act_id, 0] *= value
            
            elif param_name == 'actuator_kv':
                # Velocity gain (biasprm[:, 1])
                for act_id in self.actuator_ids:
                    if act_id < self.model.nu:
                        self.model.actuator_biasprm[act_id, 1] *= value
            
            elif param_name == 'contact_damping':
                # Adjust contact solver parameters
                # solref[0] is time constant, solimp[0] is damping
                for geom_id in self.cube_geom_ids:
                    if geom_id >= 0 and geom_id < self.model.ngeom:
                        # Scale damping-related parameters
                        # solref[0] is time constant (inverse affects damping)
                        if self.model.geom_solref[geom_id, 0] > 0:
                            self.model.geom_solref[geom_id, 0] /= value
                        # solimp[0] is damping parameter
                        self.model.geom_solimp[geom_id, 0] *= value
            
            elif param_name == 'gravity_scale':
                self.model.opt.gravity[:] *= value
    
    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            self.rng.seed(seed)
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set hand joints to zeros
        self.data.qpos[-16:] = 0.0
        self.data.qvel[-16:] = 0.0
        
        # Initialize target quaternion to identity (MuJoCo order: w,x,y,z)
        self.data.qpos[0:4] = [1.0, 0.0, 0.0, 0.0]
        
        # Optionally randomize target once to avoid zero reward
        random_target = R.random(random_state=self.rng)
        target_quat_scipy = random_target.as_quat()  # (x,y,z,w)
        target_quat_mj = target_quat_scipy[[3, 0, 1, 2]]  # (w,x,y,z)
        self.data.qpos[0:4] = target_quat_mj
        
        # Zero all buffers
        self._init_buffers()
        
        # Apply perturbations
        self._apply_dynamics_perturb(self.perturb_spec)
        
        # Forward kinematics (needed for observation)
        mujoco.mj_forward(self.model, self.data)
        
        # Build initial frame
        initial_frame = self._build_frame()
        
        # Populate deque with duplicate frames
        for _ in range(self.frame_stack):
            self.frame_deque.append(initial_frame.copy())
        
        # Reset counters
        self.episode_step = 0
        self.count_lowlevel = 0
        
        # Return stacked observation
        obs = np.concatenate(list(self.frame_deque))
        return obs.astype(np.float32)
    
    def step(self, action):
        """Execute one environment step."""
        # Clip action
        action = np.clip(action, -1.0, 1.0)
        action_clipped = not np.allclose(action, np.clip(action, -1.0, 1.0))
        
        # Process action at control timestep
        is_control_step = (self.count_lowlevel % self.sim_decimation == 0)
        
        if is_control_step:
            # Compute delta
            delta = self.action_scale * action
            
            # Update target positions
            target_q = self.prev_target_q + delta
            
            # Clip to soft limits
            target_q = np.clip(target_q, self.actuated_lower_pos_limit, self.actuated_upper_pos_limit)
            
            # Save previous processed action before updating
            self.prev_last_processed_action = self.prev_target_q.copy()
            
            # Update buffers
            self.prev_target_q = target_q.copy()
            self.prev_action = action.copy()
        
        # Use current target_q for actuators
        target_q = self.prev_target_q
        
        # Apply to actuators (must be done every step, but target_q only updates at control timestep)
        # Note: MuJoCo actuators should be in the same order as actuated joints
        for i, act_id in enumerate(self.actuator_ids):
            if act_id < self.model.nu:
                self.data.ctrl[act_id] = target_q[i]
        
        # Perform simulation substeps
        done = False
        failure_reason = None
        
        for _ in range(self.sim_decimation):
            mujoco.mj_step(self.model, self.data)
            
            # Check failure conditions
            object_height = self.data.qpos[4 + 2]  # z-coordinate
            object_x = self.data.qpos[4]
            object_y = self.data.qpos[5]
            
            if object_height < 0.30:
                done = True
                failure_reason = "dropped"
                break
            
            if abs(object_x) > 0.3 or abs(object_y) > 0.3:
                done = True
                failure_reason = "out_of_bounds"
                break
        
        # Increment counters
        self.count_lowlevel += 1
        self.episode_step += 1
        
        # Check time limit
        if self.episode_step >= self.max_episode_steps:
            done = True
        
        # Build observation at control timestep
        if is_control_step:
            # Update previous buffers (for next frame)
            qpos = self.data.qpos.astype(np.float64)
            q = qpos[-16:]
            q_isaac = q[self.mujoco2isaac_indices]
            joint_pos = self._scale(q_isaac, self.lower_pos_limit, self.upper_pos_limit)
            
            object_pos = qpos[4:7]
            object_quat = qpos[7:11]
            target_quat = qpos[0:4]
            
            # Compute quaternion difference for previous frame
            obj_quat_scipy = object_quat[[1, 2, 3, 0]]
            target_quat_scipy = target_quat[[1, 2, 3, 0]]
            R_obj = R.from_quat(obj_quat_scipy)
            R_target = R.from_quat(target_quat_scipy)
            quat_diff_scipy = (R_obj * R_target.inv()).as_quat()
            target_quat_diff = quat_diff_scipy[[3, 0, 1, 2]]
            
            target_pose = np.concatenate([self.target_pos, target_quat[[3, 0, 1, 2]]])
            
            # Update previous buffers
            self.prev_joint_pos = joint_pos.copy()
            self.prev_object_pos = object_pos.copy()
            self.prev_object_quat = object_quat.copy()
            self.prev_target_pos = target_pose.copy()
            self.prev_target_quat_diff = target_quat_diff.copy()
            
            # Build current frame and update deque
            current_frame = self._build_frame()
            self.frame_deque.append(current_frame)
        
        # Get stacked observation (always use latest deque state)
        obs = np.concatenate(list(self.frame_deque))
        
        # Compute reward and success
        reward, success, rot_dist = self._compute_reward_and_success()
        
        # Build info dict
        info = {
            'rotation_distance': rot_dist,
            'success': success,
            'object_height': float(self.data.qpos[4 + 2]),
            'action_clipped': action_clipped,
            'perturb_spec': self.current_perturb.copy()
        }
        
        if failure_reason:
            info['failure_reason'] = failure_reason
        
        return obs.astype(np.float32), reward, done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        elif mode == 'rgb_array':
            # Offscreen rendering would require additional setup
            return None
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

