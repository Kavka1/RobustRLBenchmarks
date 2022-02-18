from typing import Dict
import random
from copy import copy
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils


# body_names: 'world', 'torso', 'lwaist', 'pelvis', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm'
# geom_names: 'floor', 'torso1', 'head', 'uwaist', 'lwaist', 'butt', 'right_thigh1', 'right_shin1', 'right_foot', 'left_thigh1', 'left_shin1', 'left_foot', 'right_uarm1', 'right_larm', 'right_hand', 'left_uarm1', 'left_larm', 'left_hand'
CONFIG = {
    'fix_system': True,
    'fix_mass_coeff': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'fix_fric_coeff': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'mass_coeff_sweep': np.linspace(0.2, 1.5, 10).tolist(),
    'fric_coeff_sweep': np.linspace(0.1, 1.2, 10).tolist(),
    'mass_change_body': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'fric_change_geom': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class DRHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
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

        mujoco_env.MujocoEnv.__init__(self, "humanoid.xml", 5)
        utils.EzPickle.__init__(self)

        self.initial_body_mass = copy(self.model.body_mass)
        self.initial_geom_fraction = copy(self.model.geom_friction)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def step(self, a):
        self.episode_step += 1

        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0)) or (self.episode_step >= self.episode_length)
        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        self.episode_count += 1
        self.episode_step = 0

        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )

        self.resample_model_coefficients()

        return self._get_obs()

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

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


if __name__ == '__main__':
    env = DRHumanoidEnv(CONFIG)
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