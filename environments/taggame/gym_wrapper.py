import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environments.taggame.taggame import TagGame
from environments.taggame.config import WIDTH, HEIGHT, MAX_VELOCITY
from environments.taggame.run import feature_extractor, N_FEATURES


class TagGameGymEnv(gym.Env):
    """Gym wrapper for TagGame environment with continuous action space."""

    def __init__(self, render=False):
        super().__init__()

        self.env = TagGame(render=render)
        self.env.initialize()

        # Continuous action space: angle in radians [-pi, pi]
        # Agent only controls DIRECTION, magnitude is always MAX_VELOCITY
        self.action_space = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space: feature vector
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
        obs = feature_extractor(self.current_state)
        return obs, {}

    def step(self, action):
        """
        Args:
            action: numpy array [angle] in range [-pi, pi]
            Agent only controls direction, magnitude is always MAX_VELOCITY

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert angle to velocity components
        angle = float(action[0])
        vx = np.cos(angle) * MAX_VELOCITY
        vy = np.sin(angle) * MAX_VELOCITY

        # Use continuous action (no rounding)
        game_action = (vx, vy)

        # Take step in environment
        next_state, reward = self.env.step(self.current_state, game_action)

        # Check if done
        terminated = self.env.is_terminal(next_state)
        truncated = False

        # Get observation
        obs = feature_extractor(next_state)

        self.current_state = next_state

        return obs, reward, terminated, truncated, {}

    def render(self):
        # Rendering is handled by TagGame internally if render=True
        pass

    def close(self):
        self.env.close()
