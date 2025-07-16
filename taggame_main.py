import os
import torch
import time
import argparse

import sys
sys.path.append('/home/silidrone/silidev/aiplane_py')

from sarsa import SARSA
from environments.taggame.constants import (
    DECAY_RATE, DISCOUNT_RATE, ENABLE_RENDERING, LEARNING_RATE, MIN_EPSILON,
    N_OF_EPISODES, MODEL_DIR, POLICY_EPSILON, MODEL_FILE, HIDDEN_SIZE
)
from environments.taggame.taggame import TagGame
from environments.taggame.models import TagGameQNet, feature_extractor, set_device, state_to_readable
from policy import EpsilonGreedyPolicy, GreedyPolicy
from value_strategy import TorchValueStrategy

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

def setup_training(mode='train'):
    os.makedirs(MODEL_DIR, exist_ok=True)

    environment = TagGame(render=ENABLE_RENDERING)
    environment.initialize()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_device(device)
    
    input_size = 14
    
    model = TagGameQNet(input_size, HIDDEN_SIZE)
    model.to(device)
    
    value_strategy = TorchValueStrategy(model, feature_extractor, LEARNING_RATE)
    value_strategy.initialize(environment)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with a new model.")
    else:
        if mode == 'evaluate':
            raise FileNotFoundError(f"No existing model found at {MODEL_PATH}. Cannot evaluate without a trained model.")
        print("No existing model found. Starting with a new model.")
    
    policy = EpsilonGreedyPolicy(value_strategy, POLICY_EPSILON, MIN_EPSILON, DECAY_RATE)
    mdp_solver = SARSA(environment, policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, True)
    
    return environment, model, value_strategy, policy, mdp_solver

def onexit(mdp_solver, model, training_time, exc=None):
    torch.save(model.state_dict(), MODEL_PATH)
    if exc:
        print(f"Exit with an exception. Saved model. Training completed in {training_time:.2f} seconds. Exit reason: {exc}")
    else:
        print(f"Saved model. Training completed in {training_time:.2f} seconds.")
                
    mdp_solver._logger.plot_training_progress("taggame_training_progress.png")

def train(mdp_solver, model):
    print(f"Starting training with neural network...")
    
    start_time = time.time()
    try:
        mdp_solver.policy_iteration()
        onexit(mdp_solver, model, time.time() - start_time)
    except Exception as e:
        onexit(mdp_solver, model, time.time() - start_time, e)

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
                        
    args = parser.parse_args()
    
    environment, model, value_strategy, policy, mdp_solver = setup_training(args.mode)
    try:
        if args.mode == 'train':
            train(mdp_solver, model)
        elif args.mode == 'evaluate':
            evaluate(environment, value_strategy)
        
        if environment.render_enabled:
            environment.close()
    except Exception as e:
        print(f"Stopped: {e}")


if __name__ == "__main__":
    main()