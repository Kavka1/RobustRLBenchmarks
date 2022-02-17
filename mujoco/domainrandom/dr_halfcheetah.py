from typing import Dict, List, Tuple
import numpy as np
import random
import gym
from copy import copy
from gym import utils
from gym.envs.mujoco import mujoco_env
from torch import rand


# Example
# body_names: 'world', 'torso', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'
# geom_names: 'floor', 'torso', 'head', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'
CONFIG = {
    'fix_system': False,
    'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
    'fix_fric_coeff': [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1],
    'mass_coeff_sweep': np.linspace(0.2, 1.5, 10).tolist(),
    'fric_coeff_sweep': np.linspace(0.1, 1.2, 10).tolist(),
    'mass_change_body': [0, 0, 0, 0, 1, 1, 1, 1],
    'fric_change_geom': [0, 0, 0, 0, 0, 0, 0, 0, 0]
}


class DRHalfcheetah(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, config: Dict, episode_length: int = 1000):

        self.fix_system = config['fix_system']  # whether fix the mass and friction coefficients
        self.fix_mass_coeff = config['fix_mass_coeff']  # [1. ] * 8 for fix system
        self.fix_fric_coeff = config['fix_fric_coeff']  # [1. ] * 9 for fix system
        # For domain randomization environments
        self.mass_coeff_sweep = config['mass_coeff_sweep']  # [0.1, 0.3, 0.5, ..., 1.5]
        self.fric_coeff_sweep = config['fric_coeff_sweep']  # [0.2, 0.4, 0.5, ..., 1.2]
        self.mass_change_body = config['mass_change_body']    # [0, 1, ..., 1, 1] (8,) whether apply the sweep to the body
        self.fric_change_geom = config['fric_change_geom']    # (9,)

        self.episode_length = episode_length
        self.episode_step = 0
        self.episode_count = 0

        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

        self.initial_body_mass = copy(self.model.body_mass)
        self.initial_geom_fraction = copy(self.model.geom_friction)
        
    def step(self, action):
        self.episode_step += 1

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = self.episode_step >= self.episode_length
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def resample_model_coefficients(self) -> None:
        if self.fix_system:
            self.model.body_mass[:] = self.initial_body_mass * self.fix_mass_coeff
            self.model.geom_friction[:] = [self.initial_geom_fraction[i] * self.fix_fric_coeff[i] for i in range(len(self.fix_fric_coeff))]
        else:
            new_mass_coefficients = [random.choice(self.mass_coeff_sweep) for _ in range(len(self.mass_change_body))]
            new_fric_coefficients = [random.choice(self.fric_coeff_sweep) for _ in range(len(self.fric_change_geom))]
            
            mass_change_idx = [i for i in range(len(self.mass_change_body)) if self.mass_change_body[i] == 1]
            fric_change_idx = [i for i in range(len(self.fric_change_geom)) if self.fric_change_geom[i] == 1]
            self.model.body_mass[mass_change_idx] = (self.initial_body_mass * new_mass_coefficients)[mass_change_idx]
            self.model.geom_friction[fric_change_idx] = np.array(
                [self.initial_geom_fraction[i] * new_fric_coefficients[i] for i in range(len(self.fric_change_geom))]
            )[fric_change_idx]

    def reset_model(self):
        self.episode_step = 0
        self.episode_count += 1

        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

        self.resample_model_coefficients()  # random generate new combinations of coefficients

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == '__main__':
    env = DRHalfcheetah(CONFIG)
    action_high = env.action_space.high
    action_low = env.action_space.low

    for _ in range(1000):
        obs = env.reset()
        done = False
        print(f'Episode: {_}\n mass: {env.model.body_mass}\n friction: {env.model.geom_friction}')
        while not done:
            env.render()
            a = np.random.randn(env.action_space.shape[0], )
            a = np.clip(a, action_low, action_high)
            obs, r, done, info = env.step(a)
