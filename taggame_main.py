import os
import torch
import time
import argparse
import string
import random
import warnings

import sys
sys.path.append('/home/silidrone/silidev/aiplane_py')

# Suppress pygame warnings and hello message
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from td import SARSA, QLearning
from environments.taggame.constants import (
    DECAY_RATE, DISCOUNT_RATE, ENABLE_RENDERING, LEARNING_RATE, MIN_EPSILON,
    N_OF_EPISODES, DATA_DIR, POLICY_EPSILON, MODEL_FILE, HIDDEN_SIZE,
    LEARNING_RATE_DECAY, MIN_LEARNING_RATE, OUTPUT_FREQ, MODEL_SAVE_FREQ
)
from environments.taggame.taggame import TagGame
from environments.taggame.models import TagGameQNet, feature_extractor, set_device, state_to_readable
from policy import EpsilonGreedyPolicy, GreedyPolicy
from value_strategy import TorchValueStrategy

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

import logging

def setup_training(mode='train', run_id=None, model_path_arg=None):
    if mode == 'train':
        if run_id is None:
            run_id = generate_run_id()
        
        run_data_dir = os.path.join(DATA_DIR, run_id)
        os.makedirs(run_data_dir, exist_ok=True)
        
        model_path = os.path.join(run_data_dir, MODEL_FILE)
        plot_dir = run_data_dir
        plot_path = f"{run_id}/taggame_training_final.png"
        log_path = os.path.join(run_data_dir, "training.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
        print(f"Training run ID: {run_id}")
        print(f"Data will be saved to: {run_data_dir}/")
        print(f"Model: {model_path}")
        print(f"Plots: {run_data_dir}/")
        print(f"Logs: {log_path}")
    else:
        if model_path_arg:
            model_path = os.path.join(DATA_DIR, model_path_arg)
        elif run_id:
            model_path = os.path.join(DATA_DIR, run_id, MODEL_FILE)
        else:
            model_path = os.path.join(DATA_DIR, MODEL_FILE)
        
        plot_dir = None
        plot_path = None
        run_id = None
    
    os.makedirs(DATA_DIR, exist_ok=True)

    environment = TagGame(render=ENABLE_RENDERING)
    environment.initialize()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_device(device)
    
    input_size = 14
    
    model = TagGameQNet(input_size, HIDDEN_SIZE)
    model.to(device)
    
    value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE, LEARNING_RATE_DECAY, MIN_LEARNING_RATE)
    value_strategy.initialize(environment)
    
    print(f"Using device: {device}")
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with a new model.")
    else:
        if mode == 'evaluate':
            raise FileNotFoundError(f"No existing model found at {model_path}. Cannot evaluate without a trained model.")
        print("No existing model found. Starting with a new model.")
    
    policy = EpsilonGreedyPolicy(value_strategy, POLICY_EPSILON, MIN_EPSILON, DECAY_RATE)
    mdp_solver = QLearning(environment, policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, True)
    
    return environment, model, value_strategy, policy, mdp_solver, model_path, plot_dir, plot_path, run_id

def save_intermediate(mdp_solver, model, episode, model_path, plot_dir):
    if episode % MODEL_SAVE_FREQ == 0:
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
    
    if episode % OUTPUT_FREQ == 0:
        plot_filename = f"{os.path.basename(plot_dir)}/taggame_training_ep{episode:06d}.png"
        mdp_solver._logger.plot_training_progress(plot_filename)
        print(f"Generated plot in {plot_dir}/taggame_training_ep{episode:06d}.png")

def onexit(mdp_solver, model, training_time, model_path, plot_path, exc=None):
    torch.save(model.state_dict(), model_path)
    if exc:
        print(f"Exit with an exception. Saved model. Training completed in {training_time:.2f} seconds. Exit reason: {exc}")
    else:
        print(f"Saved model. Training completed in {training_time:.2f} seconds.")
                
    mdp_solver._logger.plot_training_progress(plot_path)

def train(mdp_solver, model, model_path, plot_dir, plot_path):
    print("Starting Q-learning training...")
    start_time = time.time()
    try:
        mdp_solver.set_episode_end_callback(lambda episode: save_intermediate(mdp_solver, model, episode, model_path, plot_dir))
        mdp_solver.policy_iteration()
        onexit(mdp_solver, model, time.time() - start_time, model_path, plot_path)
    except Exception as e:
        onexit(mdp_solver, model, time.time() - start_time, model_path, plot_path, e)

def evaluate(environment, value_strategy):
    if hasattr(value_strategy, 'q_network'):
        value_strategy.q_network.eval()
    
    greedy_policy = GreedyPolicy(value_strategy)
    
    for i in range(N_OF_EPISODES):
        state = environment.reset()
        done = False
        
        while not done:
            action, _ = greedy_policy.greedy_action(state)
            next_state, reward = environment.step(state, action)
            state = next_state
            done = environment.is_terminal(state)

def main():
    parser = argparse.ArgumentParser(description='TagGame RL Training')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate'],
                        help='Mode to run: train or evaluate')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID for training or evaluation (e.g., a3k7x2)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model path relative to models/ directory for evaluation (e.g., a3k7x2/taggame_model.pt)')
                        
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        if args.run_id and args.model_path:
            parser.error("Cannot specify both --run_id and --model_path for evaluation")
        if not args.run_id and not args.model_path:
            raise ValueError("Must specify either --run_id or --model_path for evaluation mode")
    
    environment, model, value_strategy, policy, mdp_solver, model_path, plot_dir, plot_path, run_id = setup_training(args.mode, args.run_id, args.model_path)
    try:
        if args.mode == 'train':
            train(mdp_solver, model, model_path, plot_dir, plot_path)
        elif args.mode == 'evaluate':
            evaluate(environment, value_strategy)
        
        if environment.render_enabled:
            environment.close()
    except Exception as e:
        print(f"Stopped: {e}")


if __name__ == "__main__":
    main()