import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environments.taggame.taggame import TagGame
from environments.taggame.config import WIDTH, HEIGHT, MAX_VELOCITY
from environments.taggame.train import feature_extractor, N_FEATURES
from environments.taggame import config


class TagGameGymEnv(gym.Env):

    def __init__(self, render=False, max_episode_steps=None, chaser_policy_idx=None):
        super().__init__()

        self.chaser_policy_idx = chaser_policy_idx if chaser_policy_idx is not None else config.CURRENT_CHASER_POLICY_IDX
        
        self.env = TagGame(render=render)
        self.env.initialize()
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else config.MAX_EPISODE_STEPS
        self.step_count = 0

        self.action_space = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(N_FEATURES,),
            dtype=np.float32
        )

        self.current_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.env.reset()
        self.step_count = 0
        obs = feature_extractor(self.current_state, self.chaser_policy_idx)
        return obs, {}

    def step(self, action):
        angle = float(action[0])
        vx = np.cos(angle) * MAX_VELOCITY
        vy = np.sin(angle) * MAX_VELOCITY

        game_action = (vx, vy)

        next_state, reward = self.env.step(self.current_state, game_action, chaser_policy_idx=self.chaser_policy_idx)

        self.step_count += 1
        terminated = self.env.is_terminal(next_state)
        truncated = self.step_count >= self.max_episode_steps

        obs = feature_extractor(next_state, self.chaser_policy_idx)

        self.current_state = next_state

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        self.env.close()
