from collections import deque
from copy import copy
import random
from typing import Dict, List, Tuple
from matplotlib.pyplot import cla
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from RobustRLBenchmarks.mujoco.domainrandom.dr_halfcheetah import DRHalfcheetah


class DeterministicPolicy(nn.Module):
    def __init__(self, o_dim: int, a_dim: int) -> None:
        super(DeterministicPolicy, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.model = nn.Sequential(
            nn.Linear(o_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim),
            nn.Tanh()
        )

    def __call__(self, obs: torch.tensor) -> torch.tensor:
        return self.model(obs)


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
        return self.Q1_model(x), self.Q2_model(x)

    def call_Q1(self, obs: torch.tensor, a:  torch.tensor) -> torch.tensor:
        x = torch.cat((obs, a), dim=-1)
        return self.Q1_model(x)

    def call_Q2(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        x = torch.cat((obs, a), dim=-1)
        return self.Q2_model(x)


class Buffer(object):
    def __init__(self, buffer_size: int) -> None:
        super(Buffer, self).__init__()
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def store(self, trans: Tuple) -> None:
        self.buffer.append(trans)

    def sample(self, batch_size: int) -> Tuple:
        assert batch_size <= len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        obs, a, r, done, obs_ = zip(*batch)
        obs, a, r, done, obs_ = np.stack(obs, 0), np.stack(a, 0), np.array(r), np.array(done), np.stack(obs_, 0)
        return obs, a, r, done, obs_


def hard_update(source_model: nn.Module, target_model: nn.Module) -> None:
    target_model.load_state_dict(source_model.state_dict())


def soft_update(model: nn.Module, target_model: nn.Module, tau: float) -> None:
    for param, param_target in zip(model.parameters(), target_model.parameters()):
        param_target.data.copy_(tau * param.data + (1 - tau) * param_target.data)


class TD3Agent(object):
    def __init__(self, config: Dict, env_params: Dict) -> None:
        super(TD3Agent, self).__init__()
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.action_boundary = env_params['action_boundary']

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.buffer_size = int(config['buffer_size'])
        self.batch_size = config['batch_size']
        self.noise_std = config['noise_std']
        self.noise_clip = config['noise_clip']
        self.train_delay = config['train_delay']
        self.device = torch.device(config['device'])

        self.buffer = Buffer(self.buffer_size)

        self.policy = DeterministicPolicy(self.o_dim, self.a_dim).to(self.device)
        self.policy_tar = DeterministicPolicy(self.o_dim, self.a_dim).to(self.device)
        self.critic = TwinQFunction(self.o_dim, self.a_dim).to(self.device)
        self.critic_tar = TwinQFunction(self.o_dim, self.a_dim).to(self.device)

        self.optimizer_q = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.lr)

        hard_update(self.policy, self.policy_tar)
        hard_update(self.critic, self.critic_tar)
        
        self.update_count = 0
        self.log_loss_policy = 0.
        self.log_loss_q = 0.

    def selection_action(self, obs: np.array, w_noise: bool = True) -> np.array:
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action = self.policy(obs)
        action = action.cpu().detach().numpy()
        if w_noise:
            noise = np.random.randn(*action.shape) * self.noise_std
            action += noise
            return np.clip(action, -self.action_boundary, self.action_boundary)
        else:
            return np.clip(action, -self.action_boundary, self.action_boundary)

    def update(self) -> Tuple[float, float]:
        obs, a, r, done, obs_ = self.buffer.sample(self.batch_size)
        obs = torch.from_numpy(obs).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        r = torch.from_numpy(r).float().unsqueeze(dim=-1).to(self.device)
        done = torch.from_numpy(done).int().unsqueeze(dim=-1).to(self.device)
        obs_ = torch.from_numpy(obs_).float().to(self.device)

        with torch.no_grad():
            next_action_tar = self.policy_tar(obs_)
            next_action_noise = torch.rand_like(next_action_tar).float().to(self.device) * self.noise_std
            next_action_tar = next_action_tar + torch.clamp(next_action_noise, -self.noise_clip, self.noise_clip)
            q1_target, q2_target = self.critic_tar(obs_, next_action_tar)
            next_q_target = torch.min(q1_target, q2_target)
        q_update_target = r + (1 - done) * self.gamma * next_q_target
        q1_pred, q2_pred = self.critic(obs, a)
        loss_q = F.mse_loss(q1_pred, q_update_target) + F.mse_loss(q2_pred, q_update_target)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        if self.update_count % self.train_delay == 0:
            loss_policy = - (self.critic.call_Q1(obs, self.policy(obs))).mean()
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()

            self.log_loss_policy = loss_policy.cpu().detach().item()

        self.log_loss_q = loss_q.cpu().detach().item()
        self.update_count += 1

        return copy(self.log_loss_policy), copy(self.log_loss_q)

    def evaluate(self, env: DRHalfcheetah, episode_num: int) -> float:
        average_r = 0.
        for _ in range(episode_num):
            done = False
            obs = env.reset()
            while not done:
                a = self.selection_action(obs, False)
                obs, r, done, info = env.step(a)
                average_r += r
        return average_r / episode_num



config = {
    'seed': 10,
    'train_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 4).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 0, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'test_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 8).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 0, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'lr': 5e-4,
    'gamma': 0.998,
    'tau': 0.001,
    'train_delay': 2,
    'buffer_size': 1e6,
    'batch_size': 256,
    'noise_std': 0.4,
    'noise_clip': 0.2,
    'device': 'cuda',
    'result_path': '/home/xukang/GitRepo/RobustRLBenchmarks/test/results/TD3_mujoco_halfcheetah_DR/',
    'max_episode': 10000,
    'evaluation_interval': 50,
    'evaluation_episode': 10,
    'train_begin_episode': 10
}

config_oral = copy(config)
config_oral.update({
    'train_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 8).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 0, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'test_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 8).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 0, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
})

config_baseline = copy(config)
config_baseline.update({
    'train_env_config': {
        'fix_system': True,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 8).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 0, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'test_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 8).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 0, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
})



def train(config: Dict, exp_name: str) -> None:
    train_env = DRHalfcheetah(config['train_env_config'])
    test_env = DRHalfcheetah(config['test_env_config'])

    np.random.seed(config['seed'])
    train_env.seed(config['seed'])
    test_env.seed(config['seed'])
    torch.manual_seed(config['seed'])

    env_params = {
        'o_dim': train_env.observation_space.shape[0],
        'a_dim': train_env.action_space.shape[0],
        'action_boundary': train_env.action_space.high[0],
        'max_episode_steps': train_env.episode_length,
    }
    agent = TD3Agent(config, env_params)
    config.update({
        'result_path': config['result_path'] + '{}_{}/'.format(exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    })
    logger = SummaryWriter(log_dir=config['result_path'])
    with open(config['result_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)

    total_steps = 0
    total_episodes = 0
    evaluation_score = 0.
    loss_pi, loss_q = 0., 0.
    for i in range(config['max_episode']):
        train_score = 0.
    
        obs = train_env.reset()
        done = False
        while not done:
            a = agent.selection_action(obs)
            obs_, r, done, info = train_env.step(a)
            agent.buffer.store((obs, a, r, done, obs_))
            total_steps += 1
            train_score += r
            obs = obs_

            if i > config['train_begin_episode']:
                loss_pi, loss_q = agent.update()

        total_episodes += 1

        if i % config['evaluation_interval'] == 0:
            evaluation_score = agent.evaluate(test_env, config['evaluation_episode'])

            logger.add_scalar('Indicator/train_score', train_score, total_steps)
            logger.add_scalar('Indicator/train_score_episode', train_score, total_episodes)
            logger.add_scalar('Indicator/evaluation_score', evaluation_score, total_steps)
            logger.add_scalar('Indicator/evaluation_score_episode', evaluation_score, total_episodes)
            logger.add_scalar('Loss/loss_pi', loss_pi, total_steps)
            logger.add_scalar('Loss/loss_pi_episode', loss_pi, total_episodes)
            logger.add_scalar('Loss/loss_q', loss_q, total_steps)
            logger.add_scalar('Loss/loss_q_episode', loss_q, total_episodes)

            print(f'----Episode: {total_episodes}  train_score: {train_score}, test_score: {evaluation_score}-----')


if __name__ == '__main__':
    #train(config, 'main')
    #train(config_oral, 'oral')
    train(config_baseline, 'baseline')
    #Process(target=train, args=(config, 'main')).start()
    #Process(target=train, args=(config_oral, 'oral')).start()
    #Process(target=train, args=(config_baseline, 'baseline')).start()