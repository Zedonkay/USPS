import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
import math
from typing import Dict, Set

import hydra


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if "dm-" in cfg.env.name:
        domain_name, task_name = cfg.env.split('-')[1].split('_')
        raise NotADirectoryError

    else:
        env = hydra.utils.instantiate(cfg.env)

    # assert env.action_space.low.min() >= -1
    # assert env.action_space.high.max() <= 1

    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

import omegaconf

def load_hydra_cfg(results_dir) -> omegaconf.DictConfig:
    """Loads a Hydra configuration from the given directory path.

    Tries to load the configuration from "results_dir/.hydra/config.yaml".

    Args:
        results_dir (str or pathlib.Path): the path to the directory containing the config.

    Returns:
        (omegaconf.DictConfig): the loaded configuration.

    """
    cfg_file = os.path.join(results_dir, ".hydra/config.yaml")
    cfg = omegaconf.OmegaConf.load(cfg_file)
    if not isinstance(cfg, omegaconf.DictConfig):
        raise RuntimeError("Configuration format not a omegaconf.DictConf")
    return cfg


class GaussianPolicyPosterior:
    """Diagonal Gaussian posterior over module parameters.

    This helper tracks a Gaussian prior/posterior over a subset of parameters as
    a lightweight approximation of the posterior sampling strategy from
    "Posterior Sampling for Continuing Environments" (Xu et al., 2025). We use
    the running squared gradients as a proxy for the Fisher information so that
    sampling can be performed without storing the full covariance.
    """

    def __init__(self,
                 module: nn.Module,
                 prior_variance: float = 1.0,
                 likelihood_variance: float = 1.0,
                 min_precision: float = 1e-6,
                 max_variance: float = 5.0,
                 allowed_param_names: Set[str] = None):
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        if likelihood_variance <= 0:
            raise ValueError("likelihood_variance must be positive")
        self.prior_precision = 1.0 / prior_variance
        self.likelihood_precision = 1.0 / likelihood_variance
        self.min_precision = min_precision
        self.max_variance = max_variance
        self.allowed_param_names = set(allowed_param_names or [])
        self.precisions: Dict[str, torch.Tensor] = {}
        self.reset(module)

    def reset(self, module: nn.Module):
        """Initializes precisions so the posterior equals the prior."""
        self.precisions = {}
        for name, param in module.named_parameters():
            if self.allowed_param_names and name not in self.allowed_param_names:
                continue
            self.precisions[name] = torch.full_like(
                param.data, self.prior_precision, device=param.device)

    def update(self, module: nn.Module):
        """Updates posterior precisions using the current parameter gradients."""
        for name, param in module.named_parameters():
            if self.allowed_param_names and name not in self.allowed_param_names:
                continue
            if not param.requires_grad or param.grad is None:
                continue
            # Use squared gradients as a Fisher-information proxy
            grad_sq = param.grad.detach() ** 2
            self.precisions[name] = torch.clamp(
                self.precisions[name] + self.likelihood_precision * grad_sq,
                min=self.min_precision)

    def sample(self, source_module: nn.Module, target_module: nn.Module):
        """Samples parameters from the posterior into ``target_module``."""
        target_params = dict(target_module.named_parameters())
        for name, src_param in source_module.named_parameters():
            if self.allowed_param_names and name not in self.allowed_param_names:
                continue
            if name not in self.precisions or name not in target_params:
                continue
            precision = torch.clamp(self.precisions[name],
                                    min=self.min_precision)
            # Variance derived from precision, optionally capped to avoid blow-up
            variance = torch.clamp(precision.reciprocal(),
                                   max=self.max_variance)
            std = torch.sqrt(variance)
            noise = torch.randn_like(src_param) * std
            sampled_param = src_param.detach() + noise
            target_params[name].data.copy_(sampled_param)

    def state_dict(self):
        return {name: tensor.detach().clone()
                for name, tensor in self.precisions.items()}

    def load_state_dict(self, state_dict):
        for name, tensor in state_dict.items():
            if name in self.precisions:
                self.precisions[name].data.copy_(
                    tensor.to(self.precisions[name].device))
