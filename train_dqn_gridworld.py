import math
import random
import numpy as np
from collections import deque, namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environments.windy_grid_world.windy_grid_world import WindyGridWorld

# DQN Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 2000
TAU = 0.005  # Soft update parameter
LR = 1e-3
MEMORY_SIZE = 10000
N_EPISODES = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def state_to_features(state, grid_width, grid_height):
    """Convert grid state (row, col) to one-hot encoded features"""
    row, col = state
    # One-hot encoding for position
    features = np.zeros(grid_width * grid_height, dtype=np.float32)
    idx = row * grid_width + col
    features[idx] = 1.0
    return features


def select_action(state, policy_net, n_actions, steps_done):
    """Epsilon-greedy action selection"""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer):
    """Perform one step of optimization"""
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute mask of non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states using target network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def main():
    # Initialize WindyGridWorld environment
    env = WindyGridWorld()
    env.initialize()

    # Get action space
    all_actions = env.all_possible_actions()
    n_actions = len(all_actions)

    # Grid dimensions
    grid_width = env._grid_width
    grid_height = env._grid_height
    n_observations = grid_width * grid_height

    print(f"Environment: WindyGridWorld")
    print(f"Grid size: {grid_width}x{grid_height}")
    print(f"State features: {n_observations} (one-hot)")
    print(f"Actions: {n_actions}")
    print(f"Start: {env._start_state}, Goal: {env._goal_state}")

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    episode_durations = []
    episode_rewards = []

    print(f"Training on {device}")
    print(f"Episodes: {N_EPISODES}")
    print("-" * 60)

    for i_episode in range(N_EPISODES):
        # Reset
        game_state = env.reset()
        state_features = state_to_features(game_state, grid_width, grid_height)
        state = torch.tensor(state_features, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        for t in count():
            # Select action
            action_idx = select_action(state, policy_net, n_actions, steps_done)

            # Execute action
            game_action = all_actions[action_idx.item()]
            next_game_state, reward = env.step(game_state, game_action)

            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device)
            steps_done += 1

            # Check if terminal
            done = env.is_terminal(next_game_state)

            if done:
                next_state = None
            else:
                next_state_features = state_to_features(next_game_state, grid_width, grid_height)
                next_state = torch.tensor(next_state_features, dtype=torch.float32, device=device).unsqueeze(0)

            # Store transition
            memory.push(state, action_idx, next_state, reward_tensor)

            # Move to next state
            state = next_state
            game_state = next_game_state

            # Optimize model
            optimize_model(memory, policy_net, target_net, optimizer)

            # Soft update of target network
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.mul_(1 - TAU).add_(policy_param.data, alpha=TAU)

            if done or t > 1000:
                episode_durations.append(t + 1)
                episode_rewards.append(total_reward)
                break

        # Print progress
        if (i_episode + 1) % 100 == 0:
            avg_duration = np.mean(episode_durations[-100:])
            avg_reward = np.mean(episode_rewards[-100:])
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            print(f"Episode {i_episode + 1}/{N_EPISODES} | "
                  f"Avg Steps: {avg_duration:.1f} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {eps:.3f}")

        # Save checkpoint
        if (i_episode + 1) % 1000 == 0:
            torch.save({
                'episode': i_episode + 1,
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps_done': steps_done,
            }, f'gridworld_checkpoint_ep{i_episode + 1}.pt')
            print(f"Saved checkpoint at episode {i_episode + 1}")

    print("\nTraining complete!")

    # Test final policy
    print("\nTesting final policy (greedy):")
    test_episodes = 10
    test_steps = []

    for _ in range(test_episodes):
        game_state = env.reset()
        state_features = state_to_features(game_state, grid_width, grid_height)
        state = torch.tensor(state_features, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            with torch.no_grad():
                action_idx = policy_net(state).max(1).indices.view(1, 1)

            game_action = all_actions[action_idx.item()]
            next_game_state, _ = env.step(game_state, game_action)

            if env.is_terminal(next_game_state):
                test_steps.append(t + 1)
                break

            next_state_features = state_to_features(next_game_state, grid_width, grid_height)
            state = torch.tensor(next_state_features, dtype=torch.float32, device=device).unsqueeze(0)
            game_state = next_game_state

            if t > 1000:
                test_steps.append(1001)
                break

    print(f"Test episodes: {test_episodes}")
    print(f"Average steps to goal: {np.mean(test_steps):.1f}")
    print(f"Min steps: {min(test_steps)}, Max steps: {max(test_steps)}")

    torch.save(policy_net.state_dict(), 'gridworld_policy_net_final.pt')


if __name__ == '__main__':
    main()
