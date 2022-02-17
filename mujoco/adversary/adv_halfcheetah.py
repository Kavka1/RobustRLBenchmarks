from typing import Dict, List, Tuple
import numpy as np
import gym
from gym.spaces import Box
from gym import utils
from gym.envs.mujoco import mujoco_env


class AdvHalfcheetah(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, config: Dict, episode_length: int = 1000):
        self.episode_length = episode_length
        self.episode_step = 0                           # step in each episode
        self.episode_count = 0                          # count for total episodes

        self.adv_amplitude = config['adv_amplitude']    # The adversary action coefficient: a = a_agent + coefficient * a_adv
        self.episode_length = config['episode_length']  # maximum episode length

        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    @property
    def adv_action_space(self) -> Box:
        low = np.array(self.action_space.low.tolist())
        high = np.array(self.action_space.high.tolist())
        box = Box(low = low * self.adv_amplitude, high = high * self.adv_amplitude, shape=None, dtype=np.float32)
        return box

    @property
    def adv_observation_space(self) -> Box:
        return self.observation_space

    def step(self, action_agent: np.array, action_adv: np.array) -> Tuple(np.array, Dict, bool, Dict):
        self.episode_step += 1

        action_agent = np.clip(action_agent, self.action_space.low, self.action_space.high)
        action_adv = np.clip(action_adv, self.adv_action_space.low, self.adv_action_space.high)
        action = np.clip(action_agent + action_adv, self.action_space.low, self.action_space.high)

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        reward_adv = - reward  # reward for adversary
        reward_dict = {'reward_agent': reward, 'reward_adv': reward_adv}
        
        done = self.episode_step > self.episode_length

        info = {
            'step': self.episode_step,
            'reward_agent': reward,
            'episode_count': self.episode_count
        }

        return ob, reward_dict, done, info

    def _get_obs(self) -> np.array:
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self) -> np.array:
        self.episode_step = 0
        self.episode_count += 1

        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5