#!/usr/bin/env python3

import os
import torch
from typing import List, Tuple

from sarsa import SARSA
from environments.windy_grid_world.windy_grid_constants import (
    DECAY_RATE, DISCOUNT_RATE, LEARNING_RATE, MIN_EPSILON,
    N_OF_EPISODES, MODEL_DIR, POLICY_EPSILON, VALUE_FILE
)
from environments.windy_grid_world.windy_grid_world import WindyGridWorld
from environments.windy_grid_world.windy_grid_models import WindyGridWorldQNet, feature_extractor, set_device
from policy import EpsilonGreedyPolicy, GreedyPolicy
from value_strategy import TorchValueStrategy


def visualize_trajectory(environment: WindyGridWorld, trajectory: List[Tuple[int, int]]) -> None:
    grid_width = environment._grid_width
    grid_height = environment._grid_height
    
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
    
    start_r, start_c = environment._start_state
    goal_r, goal_c = environment._goal_state
    grid[start_r][start_c] = 'S'
    grid[goal_r][goal_c] = 'G'
    
    steps = len(trajectory)
    for i in range(1, steps):
        prev_state = trajectory[i-1]
        curr_state = trajectory[i]
        r, c = curr_state
        
        if (r, c) == environment._start_state or (r, c) == environment._goal_state:
            continue
        
        dr = curr_state[0] - prev_state[0]
        dc = curr_state[1] - prev_state[1]
        
        if dr < 0 and dc == 0:
            arrow = '↑'
        elif dr > 0 and dc == 0:
            arrow = '↓'
        elif dr == 0 and dc > 0:
            arrow = '→'
        elif dr == 0 and dc < 0:
            arrow = '←'
        elif dr < 0 and dc > 0:
            arrow = '↗'
        elif dr < 0 and dc < 0:
            arrow = '↖'
        elif dr > 0 and dc > 0:
            arrow = '↘'
        elif dr > 0 and dc < 0:
            arrow = '↙'
        else:
            arrow = 'o'
        
        grid[r][c] = arrow
    
    print("\nOptimal Trajectory:")
    print('+' + '-' * (grid_width * 2 - 1) + '+')
    for row in grid:
        print('|' + '|'.join(cell for cell in row) + '|')
    print('+' + '-' * (grid_width * 2 - 1) + '+')
    print(f"Steps: {steps - 1}")


def main() -> None:
     
    print("Starting Windy Grid World training with PyTorch and SARSA algorithm...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    environment = WindyGridWorld()
    environment.initialize()
    
    print("Environment layout:")
    environment.print_grid()
    
    plots_dir = os.path.join(MODEL_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_device(device)
    
    input_size = environment._grid_height * environment._grid_width * len(environment.all_possible_actions())
    hidden_size = 32
    model = WindyGridWorldQNet(input_size, hidden_size)
    model.to(device)
    
    value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE)
    value_strategy.initialize(environment)
    
    model_path = os.path.join(MODEL_DIR, VALUE_FILE)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print("Successfully loaded model from file.")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with a new model.")
    else:
        print("No existing model found. Starting with a new model.")
    
    policy = EpsilonGreedyPolicy(value_strategy, POLICY_EPSILON, MIN_EPSILON, DECAY_RATE)
    mdp_solver = SARSA(environment, policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, True)
    
    print(f"Starting policy iteration with neural network on {device}...")
    
    try:
        mdp_solver.policy_iteration()
        
        print("Training completed. Evaluating final policy...")
        
        optimal_policy = GreedyPolicy(value_strategy)
        
        n_eval_episodes = 10
        lengths = []
        
        for i in range(n_eval_episodes):
            steps, _ = run_episode(environment, optimal_policy)
            lengths.append(steps)
        
        avg_length = sum(lengths) / len(lengths)
        print(f"Average episode length after training: {avg_length:.1f} steps")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An exception occurred during policy iteration: {e}")
    
    try:
        model.to(torch.device("cpu"))
        torch.save(model.state_dict(), model_path)
        print(f"Successfully saved the model to {model_path}.")
    except Exception as e:
        print(f"Failed to save the model: {e}")
    
    steps, trajectory = run_episode(environment, policy)
    visualize_trajectory(environment, trajectory)

def run_episode(environment: WindyGridWorld, policy: EpsilonGreedyPolicy, 
                max_steps: int = 1000, debug: bool = False) -> Tuple[int, List[Tuple[int, int]]]:
    state = environment.reset()
    steps = 0
    trajectory = [state]
    
    if hasattr(policy.value_strategy, 'q_network'):
        policy.value_strategy.q_network.eval()
    
    if debug:
        print("\nStarting episode from state:", state)
    
    while not environment.is_terminal(state) and steps < max_steps:
        try:
            action, value = policy.greedy_action(state)
            
            if debug:
                print(f"Step {steps+1}: At state {state}, taking action {action} (Q-value: {value:.2f})")
            
            next_state, reward = environment.step(state, action)
            
            if debug:
                print(f"  Result: New state {next_state}, reward: {reward}")
            
            state = next_state
            trajectory.append(state)
            steps += 1
        except Exception as e:
            if debug:
                print(f"Error during episode: {e}")
            break
    
    return steps, trajectory


if __name__ == "__main__":
    main()