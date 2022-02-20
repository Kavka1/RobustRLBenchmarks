from copy import copy
from typing import Dict, List, Tuple
import gym
import numpy as np
import yaml
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard.writer import SummaryWriter
from RobustRLBenchmarks.mujoco.basic.inverted_pendulum import InvertedPendulumEnv
from RobustRLBenchmarks.mujoco.domainrandom.dr_halfcheetah import DRHalfcheetahEnv


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
        trajectory = self.o[:self.size], self.a[:self.size], self.r[:self.size], self.o_[:self.size], self.log_prob[:self.size]
        return trajectory

    def clear(self):
        self.size = 0
        self.write = 0


class Dataset(object):
    def __init__(self, data: Dict) -> None:
        super(Dataset, self).__init__()
        self.data = data
        self.n = len(data['ret'])
        self._next_id = 0
        self.shuffle()

    def shuffle(self) -> None:
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        for key in list(self.data.keys()):
            self.data[key] = self.data[key][perm]

    def next_batch(self, batch_size: int) -> Tuple:
        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - cur_id)
        self._next_id += cur_batch_size

        batch = dict()
        for key in list(self.data.keys()):
            batch[key] = self.data[key][cur_id: cur_id + cur_batch_size]
        return batch

    def iterate_once(self, batch_size: int) -> Tuple:
        self.shuffle()
        while self._next_id < self.n:
            yield self.next_batch(batch_size)
        self._next_id = 0


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
        self.batch_size = config['batch_size']
        self.device = torch.device(config['device'])

        self.buffer = TrajectoryBuffer(self.max_episode_steps, self.o_dim, self.a_dim)
        self.policy = GaussianPolicy(self.o_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.policy_old = GaussianPolicy(self.o_dim, self.a_dim, self.action_var, self.device).to(self.device)
        self.V = VFunction(self.o_dim).to(self.device)

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_v = optim.Adam(self.V.parameters(), lr=self.lr)

        self.policy_old.load_state_dict(self.policy.state_dict())

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

        values = self.V(o).squeeze(dim=-1).cpu().detach().numpy()
        returns, deltas, advantages = self.gae_estimator(r, values)

        returns = torch.from_numpy(returns).to(self.device)      
        advantages = torch.from_numpy(advantages).to(self.device)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-7)

        all_batch = Dataset({
            'obs': o,
            'action': a,
            'ret': returns,
            'logp': log_prob,
            'adv': advantages
        })

        log_loss_pi, log_loss_v, update_count = 0, 0, 0

        for i in range(self.num_update):
            for batch in all_batch.iterate_once(self.batch_size):
                o_batch, a_batch, logp_batch, adv_batch, ret_batch = batch['obs'], batch['action'], batch['logp'], batch['adv'], batch['ret']

                dist = self.policy(o_batch)
                new_log_prob = dist.log_prob(a_batch)
                entropy = dist.entropy()
                values = self.V(o_batch).squeeze(dim=-1)

                ratio = torch.exp(new_log_prob - logp_batch.detach())
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1-self.ratio_clip, 1+self.ratio_clip) * adv_batch
                
                loss_pi = (- torch.min(surr1, surr2) - self.temperature_coef * entropy).mean()
                loss_v = 0.5 * F.mse_loss(values, ret_batch)
                loss = loss_pi + loss_v
                self.optimizer_pi.zero_grad()
                self.optimizer_v.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
                self.optimizer_pi.step()
                self.optimizer_v.step()

                log_loss_pi += loss_pi.cpu().detach().item()
                log_loss_v += loss_v.cpu().detach().item()
                update_count += 1

        self.policy_old.load_state_dict(self.policy.state_dict())

        return log_loss_pi / update_count, log_loss_v / update_count

    def rollout(self, env: DRHalfcheetahEnv, is_train: bool = True) -> Tuple[float, float, float, int]:
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
            self.buffer.clear()
            return cumulative_r, loss_pi, loss_v, total_step
        else:
            return cumulative_r, total_step


CONFIG = {
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
    'lr': 7e-4,
    'gamma': 0.99,
    'lamda': 0.95,
    'action_var': 0.16,
    'ratio_clip': 0.2,
    'temperature_coef': 0.01,
    'num_update': 50,
    'batch_size': 256,
    'device': 'cuda',
    'result_path': '/home/xukang/GitRepo/RobustRLBenchmarks/test/results/PPO_Mujoco_Halfcheetah_DR/',
    'max_episode': 10000,
    'evaluation_interval': 50,
    'evaluation_episode': 10
}

CONFIG_ORAL = copy(CONFIG)
CONFIG_ORAL.update({
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

CONFIG_BASELINE = copy(CONFIG)
CONFIG_BASELINE.update({
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


def train(config: Dict, exp_name: str, TrainEnv, TestEnv) -> None:
    train_env = TrainEnv
    test_env = TestEnv

    np.random.seed(config['seed'])
    train_env.seed(config['seed'])
    test_env.seed(config['seed'])
    torch.manual_seed(config['seed'])

    env_params = {
        'o_dim': train_env.observation_space.shape[0],
        'a_dim': train_env.action_space.shape[0],
        'action_boundary': train_env.action_space.high[0],
        'max_episode_steps': train_env._max_episode_steps #train_env.episode_length,
    }
    agent = PPOAgent(config, env_params)
    config.update({
        'exp_path': config['result_path'] + '{}_{}/'.format(exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    })
    logger = SummaryWriter(log_dir=config['exp_path'])
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)

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


TRAIN_SETTING = {
    'dr': (CONFIG, 'dr'),
    'oral': (CONFIG_ORAL, 'oral'),
    'baseline': (CONFIG_BASELINE, 'baseline')
}


if __name__ == '__main__':
    '''
    config, exp_name = TRAIN_SETTING['baseline']
    for seed in [10, 20, 30]:
        config.update({'seed': seed})
        train(config, exp_name)
    '''
    env = gym.make('HalfCheetah-v2')
    train(CONFIG, 'Invertpendulum', env, env)
