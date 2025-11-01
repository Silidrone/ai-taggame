import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from environments.taggame.taggame import TagGame
from environments.taggame.config import (
    WIDTH, HEIGHT, MAX_VELOCITY, ENABLE_RENDERING
)
from rl import DQNHyperParameters, DQN, QNetwork

class TagGameQNetwork(QNetwork):
    def __init__(self, n_features: int, n_actions: int):
        super().__init__(n_features, n_actions)
        self.layer1 = nn.Linear(n_features, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

N_EPISODES = 20000
N_FEATURES = 11

taggame_hyperparams = DQNHyperParameters(
    batch_size=128,
    gamma=0.995,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=5000,
    tau=0.005,
    lr=1e-4,
    memory_size=100000
)


def feature_extractor(state):
    my_pos, my_vel, tag_pos, tag_vel, is_tagged = state

    # Wall distances
    dist_left = my_pos[0] / WIDTH
    dist_right = (WIDTH - my_pos[0]) / WIDTH
    dist_top = my_pos[1] / HEIGHT
    dist_bottom = (HEIGHT - my_pos[1]) / HEIGHT

    # Agent velocity
    norm_vel_x = my_vel[0] / MAX_VELOCITY
    norm_vel_y = my_vel[1] / MAX_VELOCITY

    # Tagger velocity
    norm_tagger_vel_x = tag_vel[0] / MAX_VELOCITY
    norm_tagger_vel_y = tag_vel[1] / MAX_VELOCITY

    # Relative position to tagger
    dx = my_pos[0] - tag_pos[0]
    dy = my_pos[1] - tag_pos[1]
    distance = math.sqrt(dx * dx + dy * dy) / math.sqrt(WIDTH**2 + HEIGHT**2)

    angle = math.atan2(dy, dx)
    normalized_angle = (angle + math.pi) / (2 * math.pi)

    # Bias term
    bias = 1.0

    return np.array([
        dist_left, dist_right, dist_top, dist_bottom,
        norm_vel_x, norm_vel_y,
        norm_tagger_vel_x, norm_tagger_vel_y,
        distance, normalized_angle,
        bias
    ], dtype=np.float32)


def main():
    env = TagGame(render=ENABLE_RENDERING)
    env.initialize()

    agent = DQN(env, feature_extractor, N_FEATURES, taggame_hyperparams, TagGameQNetwork)
    agent.train(N_EPISODES)

    env.close()    


if __name__ == '__main__':
    main()
