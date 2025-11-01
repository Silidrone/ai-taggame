import math
import random
import numpy as np
from collections import deque, namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environments.taggame.taggame import TagGame
from environments.taggame.constants import (
    WIDTH, HEIGHT, MAX_VELOCITY, ENABLE_RENDERING
)

# DQN Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.995
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 5000
TAU = 0.005  # Soft update parameter
LR = 1e-4
MEMORY_SIZE = 100000
N_EPISODES = 20000

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
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


def extract_state_features(state):
    """Extract state features (11 features) from TagGameState"""
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

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select columns of actions taken
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
    env = TagGame(render=ENABLE_RENDERING)
    env.initialize()

    # Get action space
    all_actions = env.all_possible_actions()
    n_actions = len(all_actions)
    n_observations = 11

    print(f"Environment: TagGame")
    print(f"State features: {n_observations}")
    print(f"Actions: {n_actions}")

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
        # Reset using TagGame
        game_state = env.reset()
        state = extract_state_features(game_state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        for t in count():
            # Select action
            action_idx = select_action(state, policy_net, n_actions, steps_done)

            # Execute action in TagGame
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
                next_state_features = extract_state_features(next_game_state)
                next_state = torch.tensor(next_state_features, dtype=torch.float32, device=device).unsqueeze(0)

            # Store transition
            memory.push(state, action_idx, next_state, reward_tensor)

            # Move to next state
            state = next_state
            game_state = next_game_state

            # Optimize model
            optimize_model(memory, policy_net, target_net, optimizer)

            # Soft update of target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
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
            }, f'checkpoint_ep{i_episode + 1}.pt')
            print(f"Saved checkpoint at episode {i_episode + 1}")

    print("Training complete!")
    torch.save(policy_net.state_dict(), 'policy_net_final.pt')

    env.close()


if __name__ == '__main__':
    main()
