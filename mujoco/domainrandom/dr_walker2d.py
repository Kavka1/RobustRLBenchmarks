from typing import Dict
import random
from copy import copy
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


# Example
# body_names: 'world', 'torso', 'thigh', 'leg', 'foot', 'thigh_left', 'leg_left', 'foot_left'
# geom_names: 'floor', 'torso_geom', 'thigh_geom', 'leg_geom', 'foot_geom', 'thigh_left_geom', 'leg_left_geom', 'foot_left_geom'
CONFIG = {
    'fix_system': True,
    'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
    'fix_fric_coeff': [1, 1, 1, 1, 1, 1, 1, 1],
    'mass_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
    'fric_coeff_sweep': np.linspace(0.5, 1.5, 10).tolist(),
    'mass_change_body': [0, 1, 1, 1, 1, 1, 1, 1],
    'fric_change_geom': [0, 1, 1, 1, 1, 1, 1, 1]
}


class DRWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, config: Dict, episode_length: int = 1000):
        self.fix_system = config['fix_system']
        self.fix_mass_coeff = config['fix_mass_coeff']
        self.fix_fric_coeff = config['fix_fric_coeff']
        self.mass_coeff_sweep = config['mass_coeff_sweep']
        self.fric_coeff_sweep = config['fric_coeff_sweep']
        self.mass_change_body = config['mass_change_body']
        self.fric_change_geom = config['fric_change_geom']

        self.episode_count = 0
        self.episode_length = episode_length
        self.episode_step = 0

        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        
        self.initial_body_mass = copy(self.model.body_mass)
        self.initial_geom_fraction = copy(self.model.geom_friction)

    def step(self, a):
        self.episode_step += 1

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = (not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)) or (self.episode_step >= self.episode_length)
        ob = self._get_obs()
        return ob, reward, done, {}

    def resample_system_coefficients(self) -> None:
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

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.episode_count += 1
        self.episode_step = 0

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )

        self.resample_system_coefficients()

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



if __name__ == '__main__':
    env = DRWalker2dEnv(CONFIG)
    action_high = env.action_space.high
    action_low = env.action_space.low

    for _ in range(1000):
        obs = env.reset()
        model = env.model
        done = False
        print(f'Episode: {_}\n mass: {env.model.body_mass}\n friction: {env.model.geom_friction}')
        while not done:
            #env.render()
            a = np.random.randn(env.action_space.shape[0], )
            a = np.clip(a, action_low, action_high)
            obs, r, done, info = env.step(a)
