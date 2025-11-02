import math
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from environments.taggame.taggame import TagGame
from environments.taggame.config import (
    WIDTH, HEIGHT, MAX_VELOCITY
)
from rl import DQNHyperParameters, DQN, QNetwork
from util import standard_saver, plot_training_progress

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

    dist_left = my_pos[0] / WIDTH
    dist_right = (WIDTH - my_pos[0]) / WIDTH
    dist_top = my_pos[1] / HEIGHT
    dist_bottom = (HEIGHT - my_pos[1]) / HEIGHT

    norm_vel_x = my_vel[0] / MAX_VELOCITY
    norm_vel_y = my_vel[1] / MAX_VELOCITY

    norm_tagger_vel_x = tag_vel[0] / MAX_VELOCITY
    norm_tagger_vel_y = tag_vel[1] / MAX_VELOCITY

    dx = my_pos[0] - tag_pos[0]
    dy = my_pos[1] - tag_pos[1]
    distance = math.sqrt(dx * dx + dy * dy) / math.sqrt(WIDTH**2 + HEIGHT**2)

    angle = math.atan2(dy, dx)
    normalized_angle = (angle + math.pi) / (2 * math.pi)

    bias = 1.0

    return np.array([
        dist_left, dist_right, dist_top, dist_bottom,
        norm_vel_x, norm_vel_y,
        norm_tagger_vel_x, norm_tagger_vel_y,
        distance, normalized_angle,
        bias
    ], dtype=np.float32)


def run(mode, n_episodes, save_freq, render, logger, log_dir):
    env = TagGame(render=render)
    env.initialize()

    agent = DQN(env, feature_extractor, N_FEATURES, taggame_hyperparams, TagGameQNetwork, logger)

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
