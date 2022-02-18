import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, episode_length: int = 1000):
        self.episode_length = episode_length
        self.episode_count = 0
        self.episode_step = 0

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "inverted_pendulum.xml", 2)

    def step(self, a):
        self.episode_step += 1

        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
        done = not notdone or self.episode_step >= self.episode_length
        return ob, reward, done, {}

    def reset_model(self):
        self.episode_step = 0
        self.episode_count += 1

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent



if __name__ == '__main__':
    env = InvertedPendulumEnv()
    model = env.model
    
    body_names = model.body_names
    for name in body_names:
        idx = model.body_names.index(name)
        model.body_mass[idx] = float(idx)

    print('---------debuging---------')