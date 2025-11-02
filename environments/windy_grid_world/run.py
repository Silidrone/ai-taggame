import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from environments.windy_grid_world.windy_grid_world import WindyGridWorld
from environments.windy_grid_world.config import GRID_WIDTH, GRID_HEIGHT
from rl import DQNHyperParameters, DQN, QNetwork
from util import standard_saver, plot_training_progress

class GridWorldQNetwork(QNetwork):
    def __init__(self, n_features: int, n_actions: int):
        super().__init__(n_features, n_actions)
        self.layer1 = nn.Linear(n_features, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

N_FEATURES = GRID_WIDTH * GRID_HEIGHT

gridworld_hyperparams = DQNHyperParameters(
    batch_size=128,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=2000,
    tau=0.005,
    lr=1e-3,
    memory_size=10000
)

def feature_extractor(state):
    row, col = state
    features = np.zeros(GRID_WIDTH * GRID_HEIGHT, dtype=np.float32)
    idx = row * GRID_WIDTH + col
    features[idx] = 1.0
    return features


def run(mode, n_episodes, save_freq, render, logger, log_dir):
    env = WindyGridWorld()
    env.initialize()

    agent = DQN(env, feature_extractor, N_FEATURES, gridworld_hyperparams, GridWorldQNetwork, logger)

    checkpoint_path = os.path.join(log_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)

    if mode == 'train':
        logger.info(f"Starting training for {n_episodes} episodes")

        saver = standard_saver(agent, save_freq, log_dir, logger)
        episode_rewards, episode_durations = agent.train(n_episodes, saver)

        plot_training_progress(episode_rewards, episode_durations, log_dir)
        logger.info(f"Saved training plots to {log_dir}")

    elif mode == 'evaluate':
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"No checkpoint found at {checkpoint_path}")
        episode_rewards, episode_durations = agent.evaluate(n_episodes)

    env.close()
