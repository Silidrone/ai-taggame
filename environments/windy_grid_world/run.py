import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from environments.windy_grid_world.windy_grid_world import WindyGridWorld
from environments.windy_grid_world.config import (
    GRID_WIDTH, GRID_HEIGHT,
    BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LEARNING_RATE, MEMORY_SIZE
)
from rl import DQNHyperParameters, DQN, QNetwork
from util import standard_saver, plot_training_progress

N_FEATURES = GRID_WIDTH * GRID_HEIGHT

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

gridworld_hyperparams = DQNHyperParameters(
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    eps_start=EPS_START,
    eps_end=EPS_END,
    eps_decay=EPS_DECAY,
    tau=TAU,
    lr=LEARNING_RATE,
    memory_size=MEMORY_SIZE
)

def feature_extractor(state):
    row, col = state
    features = np.zeros(GRID_WIDTH * GRID_HEIGHT, dtype=np.float32)
    idx = row * GRID_WIDTH + col
    features[idx] = 1.0
    return features


def run(mode, n_episodes, save_freq, render, logger, log_dir, fps_limit=None, curriculum_phase=False):
    env = WindyGridWorld()
    env.initialize()

    agent = DQN(env, feature_extractor, N_FEATURES, gridworld_hyperparams, GridWorldQNetwork, logger)

    checkpoint_path = os.path.join(log_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path, curriculum_phase=curriculum_phase)

    if mode == 'train':
        logger.info(f"Starting training for {n_episodes} episodes")

        saver = standard_saver(agent, save_freq, log_dir, logger)
        episode_rewards, episode_durations = agent.train(n_episodes, saver)

    elif mode == 'evaluate':
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"No checkpoint found at {checkpoint_path}")
        episode_rewards, episode_durations = agent.evaluate(n_episodes, fps_limit=fps_limit)

    env.close()
