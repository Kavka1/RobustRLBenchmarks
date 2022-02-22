from base64 import encode
from copy import copy
from dis import dis
from multiprocessing.spawn import import_main_path
from operator import imod
from time import sleep
from typing import Dict, List, Tuple
from matplotlib.pyplot import cla
import numpy as np
import yaml
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard.writer import SummaryWriter
from RobustRLBenchmarks.test.TD3_mujoco_halfcheetah_dr import Buffer, hard_update, soft_update
from RobustRLBenchmarks.bullet.basic.minitaur import MinitaurBulletEnv
from pybullet_envs.bullet.minitaur_env_randomizer import MinitaurEnvRandomizer


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
        log_prob = log_prob.sum(dim=-1, keepdim=True)

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
        self.noise_clip = config['noise_clip']
        self.train_delay = config['train_delay']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.logstd_min, self.logstd_max = config['logstd_min'], config['logstd_max']
        self.device = torch.device(config['device'])
        self.target_entropy = - torch.tensor(self.a_dim, dtype=torch.float32)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha)

        self.policy = GaussianPolicy(self.o_dim, self.a_dim, self.logstd_min, self.logstd_max).to(self.device)
        self.policy_target = GaussianPolicy(self.o_dim, self.a_dim, self.logstd_min, self.logstd_max).to(self.device)
        self.critic = TwinQFunction(self.o_dim, self.a_dim).to(self.device)
        self.critic_target = TwinQFunction(self.o_dim, self.a_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), self.lr)
        self.optimizer_q = optim.Adam(self.critic.parameters(), self.lr)
        self.optimizer_alpha = optim.Adam([self.log_alpha], self.lr)

        self.buffer = Buffer(self.buffer_size)

        hard_update(self.policy, self.policy_target)
        hard_update(self.critic, self.critic_target)

        self.update_count = 0
        self.logger_loss_pi = 0.
        self.logger_loss_q = 0.
        self.logger_loss_alpha = 0.
        self.logger_alpha = 0.

    def selection_action(self, obs: np.array) -> np.array:
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action = self.policy.act(obs)
            action = action.cpu().detach().numpy()
            action = np.clip(action, -self.action_boundary, self.action_boundary)
        return action

    def update(self) -> Tuple[float, float, float, float]:
        batch = self.buffer.sample(self.batch_size)
        obs, a, r, done, obs_ = batch
        obs = torch.from_numpy(obs).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        r = torch.from_numpy(r).float().to(self.device).unsqueeze(dim=-1)
        done = torch.from_numpy(done).int().to(self.device).unsqueeze(dim=-1)
        obs_ = torch.from_numpy(obs_).float().to(self.device)

        with torch.no_grad():
            next_a_target, next_a_logprob = self.policy_target(obs_)
            noise = torch.randn_like(next_a_target).float().to(self.device)
            next_a_target = next_a_target + torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_q1_target, next_q2_target = self.critic_target(obs_, next_a_target)
            next_q = torch.min(next_q1_target, next_q2_target)
            update_target_q = r + (1 - done) * self.gamma * (next_q - self.alpha * next_a_logprob)
        pred_q1, pred_q2 = self.critic(obs, a)
        loss_q = F.mse_loss(pred_q1, update_target_q) + F.mse_loss(pred_q2, update_target_q)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        if self.update_count % self.train_delay == 0:
            action, log_prob = self.policy(obs)
            loss_policy = (self.alpha * log_prob - self.critic.call_q1(obs, action)).mean()
            self.optimizer_pi.zero_grad()
            loss_policy.backward()
            self.optimizer_pi.step()

            log_prob = torch.tensor(log_prob.tolist(), requires_grad=False, device=self.device)
            loss_alpha = (- torch.exp(self.log_alpha) * (log_prob + self.target_entropy)).mean()
            self.optimizer_alpha.zero_grad()
            loss_alpha.backward()
            self.optimizer_alpha.step()

            self.alpha = torch.exp(self.log_alpha)

            soft_update(self.policy, self.policy_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)

            self.logger_loss_pi = loss_policy.item()
            self.logger_loss_alpha = loss_alpha.item()
            self.logger_alpha = self.alpha.item()

        self.logger_loss_q = loss_q.item()
        self.update_count += 1

        return self.logger_loss_pi, self.logger_loss_q, self.logger_loss_alpha, self.logger_alpha

    def evaluation(self, env, episode_num: int) -> Tuple[float, int]:
        cumulative_r = 0
        evaluation_steps = 0
        for episode in range(episode_num):
            done = False
            obs = env.reset()
            while not done:
                a = self.selection_action(obs)
                obs_, r, done, info = env.step(a)
                cumulative_r += r
                obs = obs_
                evaluation_steps += 1
        return cumulative_r / episode_num, evaluation_steps


CONFIG = {
    'seed': 10,
    'train_env_randomizer': MinitaurEnvRandomizer(
        minitaur_base_mass_err_range = (-0.2, 0.2),
        minitaur_leg_mass_err_range = (-0.2, 0.2),
        battery_voltage_range = (14.8, 16.8),
        motor_viscous_damping_range=(0, 0.01)
    ),
    'test_env_randomizer': MinitaurEnvRandomizer(
        minitaur_base_mass_err_range = (-0.2, 0.5),
        minitaur_leg_mass_err_range = (-0.2, 0.2),
        battery_voltage_range = (12.8, 18.8),
        motor_viscous_damping_range=(0, 0.01)
    ),
    'lr': 7e-4,
    'gamma': 0.99,
    'tau': 0.001,
    'noise_clip': 0.2,
    'train_delay': 2,
    'logstd_min': -2,
    'logstd_max': 20,
    'buffer_size': 1000000,
    'batch_size': 256,
    'device': 'cuda',
    'result_path': '/home/xukang/GitRepo/RobustRLBenchmarks/test/results/SAC_Bullet_Minitaur_DR/',
    'max_steps': 1500000,
    'train_score_log_interval': 10,
    'evaluation_interval': 50,
    'evaluation_episode': 15,
    'start_train_steps': 1000
}


CONFIG_BASELINE = copy(CONFIG)
CONFIG_BASELINE.update({
    'train_env_randomizer': None
})


def train(config: Dict, exp_name: str) -> None:
    train_env = MinitaurBulletEnv(env_randomizer=config['train_env_randomizer'])
    test_env = MinitaurBulletEnv(env_randomizer=config['test_env_randomizer'])

    np.random.seed(config['seed'])
    train_env.seed(config['seed'])
    test_env.seed(config['seed'])
    torch.manual_seed(config['seed'])

    env_params = {
        'o_dim': train_env.observation_space.shape[0],
        'a_dim': train_env.action_space.shape[0],
        'action_boundary': train_env.action_space.high[0],
        'max_episode_steps': train_env.max_episode_length
    }
    agent = SACAgent(config, env_params)
    config.update({
        'exp_path': config['result_path'] + '{}_{}_{}/'.format(exp_name, config['seed'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    })
    logger = SummaryWriter(log_dir=config['exp_path'])
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)

    total_steps, episode_steps, total_episodes = 0, 0, 0
    train_score, evaluation_score = 0, 0
    cumulative_r = 0
    loss_pi, loss_q, loss_alpha, alpha = 0, 0, 0, 0
    while total_steps < config['max_steps']:
        episode_steps = 0

        done = False
        obs = train_env.reset()
        while not done:
            a = agent.selection_action(obs)
            obs_, r, done, info = train_env.step(a)
            agent.buffer.store((obs, a, r, done, obs_))
            
            obs = obs_
            total_steps += 1
            episode_steps += 1
            cumulative_r += r

            if total_steps > config['start_train_steps']:
                loss_pi, loss_q, loss_alpha, alpha = agent.update()

        total_episodes += 1

        if total_episodes % config['train_score_log_interval'] == 0:
            train_score = cumulative_r / config['train_score_log_interval']
            logger.add_scalar('Indicator/train_score', train_score, total_steps)
            logger.add_scalar('Indicator/train_score_episode', train_score, total_episodes)
            logger.add_scalar('Loss/loss_pi', loss_pi, total_steps)
            logger.add_scalar('Loss/loss_q', loss_q, total_steps)
            logger.add_scalar('Loss/loss_alpha', loss_alpha, total_steps)
            logger.add_scalar('Loss/alpha', alpha, total_steps)
            cumulative_r = 0

        if total_episodes % config['evaluation_interval'] == 0:
            evaluation_score, evaluation_steps = agent.evaluation(test_env, config['evaluation_episode'])
            logger.add_scalar('Indicator/evaluation_score', evaluation_score, total_steps)

        print(f'----Episode: {total_episodes} Episode_steps: {episode_steps} Total_steps: {total_steps} train_score: {train_score}, test_score: {evaluation_score}-----')



train_setting = {
    'dr': (CONFIG, 'dr'),
    'baseline': (CONFIG_BASELINE, 'baseline')
}


if __name__ == '__main__':
    config, exp_name = train_setting['baseline']
    for seed in [10, 20, 30, 40, 50]:
        config.update({'seed': seed})
        train(config, exp_name)