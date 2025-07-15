# ai-taggame: Reinforcement Learning Agent for a 2D Tag Game

An AI agent that learns to play an evader in a 2D tag game environment using reinforcement learning. The agent uses a neural network to learn optimal evasion strategies, successfully avoiding capture through intelligent movement and spatial awareness.

Built on top of an extensible reinforcement learning framework designed to work with any MDP (Markov Decision Process) environment.

## Architecture

The agent uses the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm with neural network function approximation. The RL framework models MDP environments and is designed to be extensible, any environment that implements the MDP interface can work with the RL algorithms. The tag game environment is implemented in Python using Pygame for rendering and physics.

## Development History

This framework was originally built as part of the [ai-from-scratch](https://github.com/Silidrone/ai-from-scratch) repository in C++, where the neural network was implemented from scratch with custom backpropagation. The original C++ version worked with an environment that the agent communicated with over sockets, a 2D tag game written in Java that handled the game physics and rendering.

The RL framework code was ported from C++ to Python, while the neural network component now uses PyTorch instead of the custom C++ implementation. The motivation for this Python port was to compare performance between custom neural network implementations and a high-end library like PyTorch.

## Model Persistence

The trained neural network model is saved as `models/taggame_model.pt` (or `windygridworld_model.pt` for windygridworld). When running training mode, the model is automatically loaded if it exists to continue learning from the previous state. At the end of training, the updated model is saved back to the same file. When running evaluation mode, the saved model is loaded to demonstrate the learned policy without any further training. In evaluation mode, the agent acts 100% greedily with respect to the model's Q-values and the neural network is set to evaluation mode, which disables dropout and batch normalization updates.

## Neural Network Architecture

- **Type**: Fully connected feedforward network
- **Input**: 14 features (agent position, velocity, tagger position, velocity, action, distance, relative angle, tagged status, bias)
- **Hidden layers**: 2 layers with 64 neurons each, ReLU activation
- **Output**: Single Q-value
- **Optimizer**: Adam with learning rate 0.001
- **Weight initialization**: Xavier uniform

## Hyperparameters

- **Algorithm**: SARSA
- **Discount factor**: 0.99
- **Exploration**: ε-greedy (start: 0.3, min: 0.01, decay: 0.999)
- **Training episodes**: 5,000
- **Convergence**: Optimal policy learned in approximately 5,000 episodes

## Unit Testing

The windygridworld environment from the Sutton-Barto RL book was also ported from C++ and serves as a unit test. Whenever I modify the RL logic or AI components, I run windygridworld and compare the results with the book's proposed optimal policy (Barto Sutton Reinforcement Learning: An Introduction, page 130, Example 6.5) to verify everything still works correctly.

## Usage

Train the tag game agent:
```bash
python taggame_main.py --mode train
```

Evaluate trained agent:
```bash
python taggame_main.py --mode evaluate
```

Run unit test:
```bash
python windy_grid_main.py
```
