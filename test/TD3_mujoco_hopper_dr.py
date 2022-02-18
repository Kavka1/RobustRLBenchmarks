from copy import copy
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import yaml
import datetime
from typing import Dict
from RobustRLBenchmarks.test.TD3_mujoco_halfcheetah_dr import TD3Agent
from RobustRLBenchmarks.mujoco.domainrandom.dr_hopper import DRHopperEnv


CONFIG = {
    'seed': 10,
    'train_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1],
        'fix_fric_coeff': [1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.8, 1.2, 4).tolist(),
        'fric_coeff_sweep': np.linspace(0.8, 1.2, 4).tolist(),
        'mass_change_body': [0, 1, 1, 1, 1],
        'fric_change_geom': [0, 1, 1, 1, 1]
    },
    'test_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1],
        'fix_fric_coeff': [1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.5, 1.5, 6).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 6).tolist(),
        'mass_change_body': [0, 1, 1, 1, 1],
        'fric_change_geom': [0, 1, 1, 1, 1]
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
    'result_path': '/home/xukang/GitRepo/RobustRLBenchmarks/test/results/TD3_mujoco_hopper_DR/',
    'max_steps': 1000000,
    'evaluation_interval': 50,
    'evaluation_episode': 10,
    'train_begin_episode': 10
}


CONFIG_ORAL = copy(CONFIG)
CONFIG_ORAL.update({
    'train_env_config': {
        'fix_system': False,
        'fix_mass_coeff': [1, 1, 1, 1, 1],
        'fix_fric_coeff': [1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.5, 1.5, 6).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 6).tolist(),
        'mass_change_body': [0, 1, 1, 1, 1],
        'fric_change_geom': [0, 1, 1, 1, 1]
    },
})


CONFIG_BASELINE = copy(CONFIG)
CONFIG_BASELINE.update({
    'train_env_config': {
        'fix_system': True,
        'fix_mass_coeff': [1, 1, 1, 1, 1],
        'fix_fric_coeff': [1, 1, 1, 1, 1],
        'mass_coeff_sweep': np.linspace(0.5, 1.5, 6).tolist(),
        'fric_coeff_sweep': np.linspace(0.5, 1.5, 6).tolist(),
        'mass_change_body': [0, 1, 1, 1, 1],
        'fric_change_geom': [0, 1, 1, 1, 1]
    },
})


def train(config: Dict, exp_name: str) -> None:
    train_env = DRHopperEnv(config['train_env_config'])
    test_env = DRHopperEnv(config['test_env_config'])

    np.random.seed(config['seed'])
    train_env.seed(config['seed'])
    test_env.seed(config['seed'])
    torch.manual_seed(config['seed'])

    env_params = {
        'o_dim': train_env.observation_space.shape[0],
        'a_dim': train_env.action_space.shape[0],
        'action_boundary': train_env.action_space.high[0],
    }
    agent = TD3Agent(config, env_params)
    config.update({
        'result_path': config['result_path'] + '{}_{}_{}/'.format(exp_name, config['seed'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    })
    logger = SummaryWriter(log_dir=config['result_path'])
    with open(config['result_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)

    total_steps = 0
    total_episodes = 0
    episode_step = 0.
    evaluation_score = 0.
    loss_pi, loss_q = 0., 0.
    while total_steps <= config['max_steps']:
        train_score = 0.
        episode_step = 0.
    
        obs = train_env.reset()
        done = False
        while not done:
            a = agent.selection_action(obs)
            obs_, r, done, info = train_env.step(a)
            agent.buffer.store((obs, a, r, done, obs_))
            total_steps += 1
            episode_step += 1
            train_score += r
            obs = obs_

            if total_episodes > config['train_begin_episode']:
                loss_pi, loss_q = agent.update()

        total_episodes += 1

        if total_episodes % config['evaluation_interval'] == 0:
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

        print(f'-----Episode {total_episodes} Step {episode_step} TotalStep: {total_steps}-----')


TRAIN_SETTING = {
    'dr': (CONFIG, 'dr'),
    'oral': (CONFIG_ORAL, 'oral'),
    'baseline': (CONFIG_BASELINE, 'baseline')
}


if __name__ == '__main__':
    config, exp_name = TRAIN_SETTING['baseline']
    for seed in [10, 20, 30]:
        config.update({'seed': seed})
        train(config, exp_name)