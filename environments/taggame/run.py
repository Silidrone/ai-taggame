import math
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from environments.taggame.taggame import TagGame
from environments.taggame.config import (
    WIDTH, HEIGHT, MAX_VELOCITY,
    BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LEARNING_RATE, MEMORY_SIZE,
    HIDDEN_SIZE
)
from rl import DQNHyperParameters, DQN, QNetwork
from util import standard_saver

N_FEATURES = 21

class TagGameQNetwork(QNetwork):
    def __init__(self, n_features: int, n_actions: int):
        super().__init__(n_features, n_actions)
        self.layer1 = nn.Linear(n_features, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.layer4 = nn.Linear(HIDDEN_SIZE // 2, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

taggame_hyperparams = DQNHyperParameters(
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

    # Normalized distance components to tagger
    norm_dx = dx / WIDTH
    norm_dy = dy / HEIGHT

    angle = math.atan2(dy, dx)
    normalized_angle = (angle + math.pi) / (2 * math.pi)

    # Corner positions
    corners = [(0, 0), (0, HEIGHT), (WIDTH, 0), (WIDTH, HEIGHT)]
    max_corner_dist = math.sqrt(WIDTH**2 + HEIGHT**2)

    # Distance from my position to each corner
    my_corner_dists = [
        math.sqrt((my_pos[0] - cx)**2 + (my_pos[1] - cy)**2) / max_corner_dist
        for cx, cy in corners
    ]

    # Distance from tagger position to each corner
    tagger_corner_dists = [
        math.sqrt((tag_pos[0] - cx)**2 + (tag_pos[1] - cy)**2) / max_corner_dist
        for cx, cy in corners
    ]

    bias = 1.0

    return np.array([
        dist_left, dist_right, dist_top, dist_bottom,
        norm_vel_x, norm_vel_y,
        norm_tagger_vel_x, norm_tagger_vel_y,
        distance, normalized_angle,
        norm_dx, norm_dy,
        my_corner_dists[0], my_corner_dists[1], my_corner_dists[2], my_corner_dists[3],
        tagger_corner_dists[0], tagger_corner_dists[1], tagger_corner_dists[2], tagger_corner_dists[3],
        bias
    ], dtype=np.float32)


def run(mode, n_episodes, save_freq, render, logger, log_dir, fps_limit=None, curriculum_phase=False):
    env = TagGame(render=render)
    env.initialize()

    agent = DQN(env, feature_extractor, N_FEATURES, taggame_hyperparams, TagGameQNetwork, logger)

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
