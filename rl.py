import math
import random
import logging
import numpy as np
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from itertools import count
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

class TrainingInterrupted(Exception):
    pass

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

@dataclass
class DQNHyperParameters:
    batch_size: int
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    tau: float
    lr: float
    memory_size: int

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class QNetwork(nn.Module, ABC):
    @abstractmethod
    def __init__(self, n_features: int, n_actions: int):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

class DQN:
    def __init__(self, env, feature_extractor, n_features, hyperparams: DQNHyperParameters, qnetwork_class: type[QNetwork], logger):
        self.env = env
        self.n_features = n_features
        self.all_actions = self.env.all_possible_actions()
        self.n_actions = len(self.all_actions)
        self.feature_extractor = feature_extractor
        self.hyperparams = hyperparams
        self.qnetwork_class = qnetwork_class
        self.device = device
        self.logger = logger

        self.policy_net = self.qnetwork_class(self.n_features, self.n_actions).to(self.device)
        self.target_net = self.qnetwork_class(self.n_features, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.hyperparams.lr, amsgrad=True)

        self.steps_done = 0
        self.current_episode = 0

    def save(self, path):
        torch.save({
            'episode': self.current_episode,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.current_episode = checkpoint['episode']
        self.logger.info(f"Loaded checkpoint from episode {self.current_episode}")

    def epsilon_greedy_sample(self, state, policy_net, n_actions, steps_done):
        sample = random.random()
        eps_threshold = self.hyperparams.eps_end + (self.hyperparams.eps_start - self.hyperparams.eps_end) * math.exp(-1. * steps_done / self.hyperparams.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


    def update(self, memory, policy_net, target_net, optimizer):
        """Perform one step of optimization"""
        if len(memory) < self.hyperparams.batch_size:
            return

        transitions = memory.sample(self.hyperparams.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                    device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using target network
        next_state_values = torch.zeros(self.hyperparams.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.hyperparams.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        
    def train(self, n_episodes, saver=None):
        self.logger.info(f"State features: {self.n_features}")
        self.logger.info(f"Actions: {self.n_actions}")

        memory = ReplayMemory(self.hyperparams.memory_size)

        episode_durations = []
        episode_rewards = []

        self.logger.info(f"Training on {device}")
        self.logger.info(f"Episodes: {n_episodes}")
        self.logger.info("-" * 60)

        for i_episode in range(n_episodes):
            # Reset using TagGame
            game_state = self.env.reset()
            state = self.feature_extractor(game_state)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            total_reward = 0
            for t in count():
                # Select action
                action_idx = self.epsilon_greedy_sample(state, self.policy_net, self.n_actions, self.steps_done)

                # Execute action in TagGame
                game_action = self.all_actions[action_idx.item()]
                next_game_state, reward = self.env.step(game_state, game_action)

                total_reward += reward
                reward_tensor = torch.tensor([reward], device=device)
                self.steps_done += 1

                # Check if terminal
                done = self.env.is_terminal(next_game_state)

                if done:
                    next_state = None
                else:
                    next_state_features = self.feature_extractor(next_game_state)
                    next_state = torch.tensor(next_state_features, dtype=torch.float32, device=device).unsqueeze(0)

                # Store transition
                memory.push(state, action_idx, next_state, reward_tensor)

                # Move to next state
                state = next_state
                game_state = next_game_state

                self.update(memory, self.policy_net, self.target_net, self.optimizer)

                # Soft update of target network
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.hyperparams.tau + target_net_state_dict[key] * (1 - self.hyperparams.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    episode_rewards.append(total_reward)
                    break

            self.current_episode += 1

            # Print progress
            if self.current_episode % 100 == 0:
                avg_duration = np.mean(episode_durations[-100:])
                avg_reward = np.mean(episode_rewards[-100:])
                eps = self.hyperparams.eps_end + (self.hyperparams.eps_start - self.hyperparams.eps_end) * math.exp(-1. * self.steps_done / self.hyperparams.eps_decay)
                self.logger.info(f"Episode {self.current_episode}/{self.current_episode + n_episodes - i_episode - 1} | "
                    f"Avg Steps: {avg_duration:.1f} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Epsilon: {eps:.3f}")

            if saver is not None:
                saver(self.current_episode)

        self.logger.info("Training complete!")

        return episode_rewards, episode_durations

    def evaluate(self, n_episodes=100):
        self.logger.info(f"Evaluating agent for {n_episodes} episodes...")

        self.policy_net.eval()

        episode_rewards = []
        episode_durations = []

        for i_episode in range(n_episodes):
            game_state = self.env.reset()
            state_features = self.feature_extractor(game_state)
            state = torch.tensor(state_features, dtype=torch.float32, device=self.device).unsqueeze(0)

            total_reward = 0
            steps = 0

            while True:
                steps += 1

                with torch.no_grad():
                    q_values = self.policy_net(state)
                    action_idx = q_values.max(1).indices.view(1, 1).item()

                game_action = self.all_actions[action_idx]
                next_game_state, reward = self.env.step(game_state, game_action)
                total_reward += reward

                done = self.env.is_terminal(next_game_state)

                if done:
                    episode_rewards.append(total_reward)
                    episode_durations.append(steps)
                    break

                next_state_features = self.feature_extractor(next_game_state)
                state = torch.tensor(next_state_features, dtype=torch.float32, device=self.device).unsqueeze(0)
                game_state = next_game_state

                if steps > 10000:
                    episode_rewards.append(total_reward)
                    episode_durations.append(steps)
                    break

            if (i_episode + 1) % 10 == 0:
                self.logger.info(f"Episode {i_episode + 1}/{n_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")

        self.policy_net.train()

        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"Episodes: {n_episodes}")
        self.logger.info(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        self.logger.info(f"Average Steps: {np.mean(episode_durations):.1f} ± {np.std(episode_durations):.1f}")
        self.logger.info(f"Min Reward: {np.min(episode_rewards):.2f}")
        self.logger.info(f"Max Reward: {np.max(episode_rewards):.2f}")

        return episode_rewards, episode_durations