from copy import copy
from typing import Dict, List, Tuple
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard.writer import SummaryWriter
from RobustRLBenchmarks.mujoco.domainrandom.dr_halfcheetah import DRHalfcheetah
from multiprocessing import Process


class GaussianPolicy(nn.Module):
    def __init__(self, o_dim, a_dim, action_var, device):
        super(GaussianPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(o_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim),
            nn.Tanh()
        )
        self.action_var = torch.full(size=[a_dim, ], fill_value=action_var).to(device)
        self.cov_mat = torch.diag(self.action_var)

    def __call__(self, o):
        a_mean = self.model(o)
        dist = MultivariateNormal(loc=a_mean, covariance_matrix=self.cov_mat)
        return dist


class VFunction(nn.Module):
    def __init__(self, o_dim):
        super(VFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(o_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def __call__(self, o):
        return self.model(o)


class TrajectoryBuffer():
    def __init__(self, capacity, o_dim, a_dim):
        self.o = np.zeros(shape=[capacity, o_dim], dtype=np.float32)
        self.a = np.zeros(shape=[capacity, a_dim], dtype=np.float32)
        self.r = np.zeros(shape=[capacity], dtype=np.float32)
        self.o_ = np.zeros(shape=[capacity, o_dim], dtype=np.float32)
        self.log_prob = np.zeros(shape=[capacity], dtype=np.float32)

        self.write = 0
        self.size = 0
        self.capacity = capacity

    def store(self, o, a, r, o_, lp):
        idx = self.write
        self.o[idx] = o
        self.a[idx] = a
        self.r[idx] = r
        self.o_[idx] = o_
        self.log_prob[idx] = lp
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def get_trajectory(self):
        return self.o[:self.size], self.a[:self.size], self.r[:self.size], self.o_[:self.size], self.log_prob[:self.size]

    def clear(self):
        self.size = 0
        self.write = 0


class PPOAgent(object):
    def __init__(self, config: Dict, env_params: Dict) -> None:
        super().__init__()
        self.o_dim = env_params['o_dim']
        self.a_dim = env_params['a_dim']
        self.action_boundary = env_params['action_boundary']
        self.max_episode_steps = env_params['max_episode_steps']

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.lamda = config['lamda']
        self.action_var = config['action_var']
        self.ratio_clip = config['ratio_clip']
        self.temperature_coef = config['temperature_coef']
        self.num_update = config['num_update']
        self.device = torch.device(config['device'])

        self.buffer = TrajectoryBuffer(self.max_episode_steps, self.o_dim, self.a_dim)
        self.policy = GaussianPolicy(self.o_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.policy_old = GaussianPolicy(self.o_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.V = VFunction(self.o_dim).to(self.device)
        self.V_old = VFunction(self.o_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_v = optim.Adam(self.V.parameters(), lr=self.lr)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.V_old.load_state_dict(self.V.state_dict())

    def act(self, obs: np.array) -> Tuple[np.array, np.float32]:
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            dist = self.policy_old(obs)
            a = dist.sample()
            log_prob = dist.log_prob(a)
        a = a.cpu().detach().numpy()
        log_prob = log_prob.cpu().detach().numpy()
        a = np.clip(a, - self.action_boundary, self.action_boundary)
        return a, log_prob

    def gae_estimator(self, rewards: np.array, values: np.array) -> Tuple[np.array, np.array, np.array]:
        length = len(rewards)
        returns = np.zeros(shape=(length,), dtype=np.float32)
        deltas = np.zeros(shape=(length,), dtype=np.float32)
        advantages = np.zeros(shape=(length,), dtype=np.float32)

        prev_return = 0.
        prev_value = 0.
        prev_advantage = 0.
        for i in reversed(range(length)):
            returns[i] = rewards[i] + self.gamma * prev_return
            deltas[i] = rewards[i] + self.gamma * prev_value - values[i]
            advantages[i] = deltas[i] + self.gamma * self.lamda * prev_advantage
            prev_advantage = advantages[i]
            prev_value = values[i]
            prev_return = returns[i]
        
        return returns, deltas, advantages

    def update(self) -> Tuple[float, float]:
        o, a, r, o_, log_prob = self.buffer.get_trajectory()
        o = torch.from_numpy(o).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        o_ = torch.from_numpy(o_).to(self.device)
        log_prob = torch.from_numpy(log_prob).to(self.device)

        values = self.V_old(o).squeeze(dim=-1).cpu().detach().numpy()
        returns, deltas, advantages = self.gae_estimator(r, values)

        returns = torch.from_numpy(returns).to(self.device)
        returns = (returns - returns.mean())/(returns.std() + 1e-7)
        advantages = torch.from_numpy(advantages).to(self.device)

        for i in range(self.num_update):
            dist = self.policy(o)
            new_log_prob = dist.log_prob(a)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_prob - log_prob.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.ratio_clip, 1+self.ratio_clip) * advantages
            
            loss_pi = (- torch.min(surr1, surr2)).mean() - (self.temperature_coef * entropy).mean()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

            values = self.V(o).squeeze(dim=-1)
            loss_v = F.mse_loss(values, returns)
            self.optimizer_v.zero_grad()
            loss_v.backward()
            self.optimizer_v.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.V_old.load_state_dict(self.V.state_dict())

        return loss_pi.cpu().detach().item(), loss_v.cpu().detach().item()

    def rollout(self, env: DRHalfcheetah, is_train: bool = True) -> Tuple[float, float, float, int]:
        cumulative_r = 0.
        total_step = 0.
        
        obs = env.reset()
        done = False
        while not done:
            a, log_prob = self.act(obs)
            obs_, r, done, info = env.step(a)
            if is_train:
                self.buffer.store(obs, a, r, obs_, log_prob)
    
            total_step += 1
            cumulative_r += r
            obs = obs_

        if is_train:
            loss_pi, loss_v = self.update()
            return cumulative_r, loss_pi, loss_v, total_step
        else:
            return cumulative_r, total_step


config = {
    'seed': 10,
    'train_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 5).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 1, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'test_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 1, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'lr': 5e-4,
    'gamma': 0.998,
    'lamda': 0.995,
    'action_var': 0.2,
    'ratio_clip': 0.2,
    'temperature_coef': 0.01,
    'num_update': 50,
    'device': 'cuda',
    'result_path': '/home/xukang/GitRepo/RobustRLBenchmarks/test/results/PPO_Mujoco_Halfcheetah_DR/',
    'max_episode': 10000,
    'evaluation_interval': 50,
    'evaluation_episode': 10
}

config_oral = copy(config)
config_oral.update({
    'train_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 1, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'test_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 1, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
})

config_baseline = copy(config)
config_baseline.update({
    'train_env_config': {
        'fix_system': True,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.5, 10).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 1, 0, 0, 1],
        'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    'test_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
        'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
        'mass_change_body': [0, 0, 0, 0, 1, 0, 0, 1],
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
    agent = PPOAgent(config, env_params)
    logger = SummaryWriter(log_dir=config['result_path'] + '{}_{}'.format(exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    total_steps = 0
    total_episodes = 0
    evaluation_score = 0.
    for i in range(config['max_episode']):
        train_return, loss_pi, loss_v, steps = agent.rollout(train_env, True)

        total_steps += steps
        total_episodes += 1

        if i % config['evaluation_interval'] == 0:
            evaluation_score = 0
            for j in range(config['evaluation_episode']):
                score, evaluation_step = agent.rollout(test_env, False)
                evaluation_score  += score
            evaluation_score /= config['evaluation_episode']

            logger.add_scalar('Indicator/train_score', train_return, total_steps)
            logger.add_scalar('Indicator/train_score_episode', train_return, total_episodes)
            logger.add_scalar('Indicator/evaluation_score', evaluation_score, total_steps)
            logger.add_scalar('Indicator/evaluation_score_episode', evaluation_score, total_episodes)
            logger.add_scalar('Loss/loss_pi', loss_pi, total_steps)
            logger.add_scalar('Loss/loss_pi_episode', loss_pi, total_episodes)
            logger.add_scalar('Loss/loss_v', loss_v, total_steps)
            logger.add_scalar('Loss/loss_v_episode', loss_v, total_episodes)

            print(f'----Episode: {total_episodes}  train_score: {train_return}, test_score: {evaluation_score}-----')


if __name__ == '__main__':
    train(config, 'main')
    #train(config_oral, 'oral')
    #train(config_baseline, 'baseline')
    #Process(target=train, args=(config, 'main')).start()
    #Process(target=train, args=(config_oral, 'oral')).start()
    #Process(target=train, args=(config_baseline, 'baseline')).start()