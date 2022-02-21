from base64 import encode
from dis import dis
from operator import imod
from typing import Dict, List, Tuple
from aem import con
from matplotlib.pyplot import cla
import numpy as np
import yaml
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard.writer import SummaryWriter


class GaussianPolicy(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, log_std_min: float, log_std_max: float) -> None:
        super(GaussianPolicy, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.encoder = nn.Sequential(
            nn.Linear(o_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(nn.Linear(128, a_dim))
        self.log_std_head = nn.Sequential(nn.Linear(128, a_dim))

    def act(self, obs: torch.tensor) -> np.array:
        z = self.encoder(obs)
        mean, log_std = self.mean_head(z), self.log_std_head(z)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = torch.tanh(dist.sample())
        action = action.cpu().detach().numpy()
        return action

    def __call__(self, obs: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        z = self.encoder(obs)
        mean, log_std = self.mean_head(z), self.log_std_head(z)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        arctah_action = dist.rsample()  # reparametric trick for gradient flow
        action = torch.tanh(arctah_action)
        log_prob = dist.log_prob(arctah_action) - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keep_dim=True)

        return action, log_prob


class TwinQFunction(nn.Module):
    def __init__(self, o_dim: int, a_dim: int) -> None:
        super(TwinQFunction, self).__init__()
        self.o_dim, self.a_dim = o_dim, a_dim
        self.Q1_model = nn.Sequential(
            nn.Linear(o_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.Q2_model = nn.Sequential(
            nn.Linear(o_dim + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def __call__(self, obs: torch.tensor, a: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x = torch.cat((obs, a), dim=-1)
        q1_value, q2_value = self.Q1_model(x), self.Q2_model(x)
        return q1_value, q2_value

    def call_q1(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.Q1_model(torch.cat((obs, a), dim=-1))

    def call_q2(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.Q2_model(torch.cat((obs, a), dim=-1))


class SACAgent(object):
    def __init__(self, config: Dict, env_params: Dict) -> None:
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.action_boundary = env_params['action_boundary']

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.initial_alpha = config['initial_alpha']
        self.target_entropy = - torch.tensor(self.a_dim, dtype=torch.float32)
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']