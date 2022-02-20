from typing import Dict, List, Tuple
import datetime
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from copy import copy
import yaml
from RobustRLBenchmarks.test.PPO_mujoco_halfcheetah_dr import PPOAgent
from RobustRLBenchmarks.bullet.basic.minitaur import MinitaurBulletEnv
from pybullet_envs.bullet.minitaur_env_randomizer import MinitaurEnvRandomizer


# Relative range.
MINITAUR_BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
MINITAUR_LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
# Absolute range.
BATTERY_VOLTAGE_RANGE = (14.8, 16.8)  # Unit: Volt
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.01)  # Unit: N*m*s/rad (torque/angular vel)
MINITAUR_LEG_FRICTION = (0.8, 1.5)  # Unit: dimensionless



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
    'lamda': 0.95,
    'action_var': 0.16,
    'ratio_clip': 0.2,
    'temperature_coef': 0.01,
    'num_update': 50,
    'batch_size': 256,
    'device': 'cuda',
    'result_path': '/home/xukang/GitRepo/RobustRLBenchmarks/test/results/PPO_Bullet_Minitaur_DR/',
    'max_steps': 1000000,
    'train_score_log_interval': 10,
    'evaluation_interval': 50,
    'evaluation_episode': 15,
}


CONFIG_BASELINE = copy(CONFIG)
CONFIG_BASELINE.update({
    'train_env_randomizer': None
})


def train(config: Dict, exp_name: str) -> None:
    train_env = MinitaurBulletEnv(env_randomizer=config['train_env_randomizer'], distance_limit=20.)
    test_env = MinitaurBulletEnv(env_randomizer=config['test_env_randomizer'], distance_limit=20.)

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
    agent = PPOAgent(config, env_params)
    config.update({
        'exp_path': config['result_path'] + '{}_{}_{}/'.format(exp_name, config['seed'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    })
    logger = SummaryWriter(log_dir=config['exp_path'])
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, indent=2)

    total_steps = 0
    total_episodes = 0
    cumulative_r = 0.
    train_score, evaluation_score = 0, 0
    loss_pi, loss_v = 0., 0.
    while total_steps <= config['max_steps']:
        train_return, loss_pi, loss_v, steps = agent.rollout(train_env, True)
        
        total_steps += steps
        total_episodes += 1
        cumulative_r += train_return

        if total_episodes % config['train_score_log_interval'] == 0:
            train_score = cumulative_r / config['train_score_log_interval']
            logger.add_scalar('Indicator/train_score', train_score, total_steps)
            logger.add_scalar('Indicator/train_score_episode', train_score, total_episodes)
            logger.add_scalar('Loss/loss_pi', loss_pi, total_steps)
            logger.add_scalar('Loss/loss_v', loss_v, total_steps)
            cumulative_r = 0

        if total_episodes % config['evaluation_interval'] == 0:
            for j in range(config['evaluation_episode']):
                score, evaluation_step = agent.rollout(test_env, False)
                evaluation_score  += score
            evaluation_score /= config['evaluation_episode']
            logger.add_scalar('Indicator/evaluation_score', evaluation_score, total_steps)

        print(f'----Episode: {total_episodes} Episode_steps: {steps} Total_steps: {total_steps} train_score: {train_score}, test_score: {evaluation_score}-----')



TRAIN_SETTING = {
    'dr': (CONFIG, 'dr'),
    'baseline': (CONFIG_BASELINE, 'baseline')
}


if __name__ == '__main__':
    config, exp_name = TRAIN_SETTING['baseline']
    for seed in [10, 20, 30]:
        config.update({'seed': seed})
        train(config, exp_name)