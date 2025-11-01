import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from environments.windy_grid_world.windy_grid_world import WindyGridWorld
from environments.windy_grid_world.config import GRID_WIDTH, GRID_HEIGHT
from rl import DQNHyperParameters, DQN, QNetwork

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
    batch_size=128,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=2000,
    tau=0.005,
    lr=1e-3,
    memory_size=10000
)

N_EPISODES = 1000
N_FEATURES = GRID_WIDTH * GRID_HEIGHT

def feature_extractor(state):
    """Convert grid state (row, col) to one-hot encoded features"""
    row, col = state
    features = np.zeros(GRID_WIDTH * GRID_HEIGHT, dtype=np.float32)
    idx = row * GRID_WIDTH + col
    features[idx] = 1.0
    return features


def main():
    env = WindyGridWorld()
    env.initialize()

    agent = DQN(env, feature_extractor, N_FEATURES, gridworld_hyperparams, GridWorldQNetwork)
    agent.train(N_EPISODES)

    env.close()


if __name__ == '__main__':
    main()
