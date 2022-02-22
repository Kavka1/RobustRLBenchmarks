import argparse
from typing import List, Dict, Tuple
from copy import copy
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import yaml
import datetime
import pandas as pd
from RobustRLBenchmarks.bullet.basic.minitaur import MinitaurBulletEnv
from RobustRLBenchmarks.test.PPO_mujoco_halfcheetah_dr import PPOAgent
from RobustRLBenchmarks.test.TD3_mujoco_halfcheetah_dr import TD3Agent
from RobustRLBenchmarks.test.SAC_bullet_minitaur_dr import SACAgent
from pybullet_envs.bullet.minitaur_env_randomizer import MinitaurEnvRandomizer



train_env_randomizer_seq = [
    MinitaurEnvRandomizer(
        minitaur_base_mass_err_range = (-0.2, -0.1),
        minitaur_leg_mass_err_range = (-0.2, 0.2),
        battery_voltage_range = (12.8, 18.8),
        motor_viscous_damping_range=(0, 0.01)
    ),
    MinitaurEnvRandomizer(
        minitaur_base_mass_err_range = (-0.1, 0.),
        minitaur_leg_mass_err_range = (-0.2, 0.2),
        battery_voltage_range = (12.8, 18.8),
        motor_viscous_damping_range=(0, 0.01)
    ),
    MinitaurEnvRandomizer(
        minitaur_base_mass_err_range = (-0., 0.1),
        minitaur_leg_mass_err_range = (-0.2, 0.2),
        battery_voltage_range = (12.8, 18.8),
        motor_viscous_damping_range=(0, 0.01)
    ),
    MinitaurEnvRandomizer(
        minitaur_base_mass_err_range = (-0.1, 0.2),
        minitaur_leg_mass_err_range = (-0.2, 0.2),
        battery_voltage_range = (12.8, 18.8),
        motor_viscous_damping_range=(0, 0.01)
    ),
    MinitaurEnvRandomizer(
        minitaur_base_mass_err_range = (-0.2, 0.3),
        minitaur_leg_mass_err_range = (-0.2, 0.2),
        battery_voltage_range = (12.8, 18.8),
        motor_viscous_damping_range=(0, 0.01)
    )
]

test_env_randomizer = MinitaurEnvRandomizer(
    minitaur_base_mass_err_range = (-0.2, 0.5),
    minitaur_leg_mass_err_range = (-0.2, 0.2),
    battery_voltage_range = (12.8, 18.8),
    motor_viscous_damping_range=(0, 0.01)
)


config_TD3 = {
    'seed': 10,
    'lr': 5e-4,
    'gamma': 0.998,
    'tau': 0.001,
    'train_delay': 2,
    'buffer_size': 1e6,
    'batch_size': 256,
    'noise_std': 0.4,
    'noise_clip': 0.2,
    'device': 'cuda',
    'result_path': '/home/xukang/GitRepo/RobustRLBenchmarks/test/results/TD3_Minitaur_DR_Range/',
    'max_steps': 1500000,
    'train_score_log_interval': 10,
    'evaluation_interval': 50,
    'evaluation_episode': 15,
    'train_begin_episode': 20
}


def train() -> None:
    def single_run(config: Dict, train_env: MinitaurBulletEnv, test_env: MinitaurBulletEnv, train_set_index: int) -> None:
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
            'exp_path': config['result_path'] + 'Index_{}_Seed_{}_T_{}/'.format(train_set_index, config['seed'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        })
        logger = SummaryWriter(log_dir=config['exp_path'])
        with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, indent=2)

        data_log = {'Episode': [], 'Total_Steps': [], 'Train_Score': [], 'Evaluation_Score': []}

        total_steps = 0
        total_episodes = 0
        cumulative_r = 0.
        train_score, evaluation_score = 0, 0
        loss_pi, loss_q = 0., 0.
        while total_steps <= config['max_steps']:
            episode_step = 0
        
            obs = train_env.reset()
            done = False
            while not done:
                a = agent.selection_action(obs)
                obs_, r, done, info = train_env.step(a)
                agent.buffer.store((obs, a, r, done, obs_))
                
                episode_step += 1
                total_steps += 1
                cumulative_r += r
                obs = obs_

                if total_episodes > config['train_begin_episode']:
                    loss_pi, loss_q = agent.update()

            total_episodes += 1

            if total_episodes % config['train_score_log_interval'] == 0:
                train_score = cumulative_r / config['train_score_log_interval']
                logger.add_scalar('Indicator/train_score', train_score, total_steps)
                logger.add_scalar('Indicator/train_score_episode', train_score, total_episodes)
                logger.add_scalar('Loss/loss_pi', loss_pi, total_steps)
                logger.add_scalar('Loss/loss_q', loss_q, total_steps)
                cumulative_r = 0

            if total_episodes % config['evaluation_interval'] == 0:
                evaluation_score = agent.evaluate(test_env, config['evaluation_episode'])
                logger.add_scalar('Indicator/evaluation_score', evaluation_score, total_steps)
                
                data_log['Episode'].append(total_episodes)
                data_log['Total_Steps'].append(total_steps)
                data_log['Train_Score'].append(train_score)
                data_log['Evaluation_Score'].append(evaluation_score)
                dataframe = pd.DataFrame(data_log)
                dataframe.to_csv(config['exp_path'] + 'data_log.csv', index=False, sep=',')


            print(f'----Episode: {total_episodes} Episode_steps: {episode_step} Total_steps: {total_steps} train_score: {train_score}, test_score: {evaluation_score}-----')


    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_index', default=None, type=int)
    args = parser.parse_args()

    train_set_index = args.train_set_index
    assert train_set_index < len(train_env_randomizer_seq), f"Train set index must be less than {len(train_env_randomizer_seq)}"

    config = copy(config_TD3)
    for seed in [10, 20, 30]:
        train_env = MinitaurBulletEnv(env_randomizer=train_env_randomizer_seq[train_set_index])
        test_env = MinitaurBulletEnv(env_randomizer=test_env_randomizer)
        config.update({'seed': seed})
        single_run(config, train_env, test_env, train_set_index)


if __name__ == '__main__':
    train()