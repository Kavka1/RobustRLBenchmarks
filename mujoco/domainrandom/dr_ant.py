from typing import Dict
from copy import copy
import random
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


# body_names: 'world', 'torso', 'front_left_leg', 'aux_1', 'front_right_leg', 'aux_2', 'back_leg', 'aux_3', 'right_back_leg', 'aux_4', '', '', '', ''
# geom_names: 'floor', 'torso_geom', 'aux_1_geom', 'left_leg_geom, 'left_ankle_geom', 'aux_2_geom', 'right_leg_geom, 'right_ankle_geom', 'aux_3_geom', 'back_leg_geom, 'third_ankle_geom', 'aux_4_geom', 'rightback_leg_geom, 'fourth_ankle_geom'
CONFIG = {
    'fix_system': True,
    'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'fix_fric_coeff': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'mass_coeff_sweep': np.linspace(0.2, 1.5, 10).tolist(),
    'fric_coeff_sweep': np.linspace(0.1, 1.2, 10).tolist(),
    'mass_change_body': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'fric_change_geom': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}


class DRAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
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

        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

        self.initial_body_mass = copy(self.model.body_mass)
        self.initial_geom_fraction = copy(self.model.geom_friction)

    def step(self, a):
        self.episode_step += 1

        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = (not notdone) or self.episode_step >= self.episode_length
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
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
        self.episode_count += 1
        self.episode_step = 0

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

        self.resample_model_coefficients()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


if __name__ == '__main__':
    env = DRAntEnv(CONFIG)
    action_high = env.action_space.high
    action_low = env.action_space.low

    model = env.model

    for _ in range(1000):
        obs = env.reset()
        done = False
        print(f'Episode: {_}\n mass: {env.model.body_mass}\n friction: {env.model.geom_friction}')
        while not done:
            env.render()
            a = np.random.randn(env.action_space.shape[0], )
            a = np.clip(a, action_low, action_high)
            obs, r, done, info = env.step(a)