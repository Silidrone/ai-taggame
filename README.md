# ai-taggame: Reinforcement Learning Agent for a 2D Tag Game

An AI agent that learns to play an evader in a 2D tag game environment using reinforcement learning. The agent uses a neural network to learn optimal evasion strategies, successfully avoiding capture through intelligent movement and spatial awareness.

Built on top of an extensible reinforcement learning framework designed to work with any MDP (Markov Decision Process) environment.

## Architecture

The agent uses Q-learning reinforcement learning algorithm with neural network function approximation. The RL framework models MDP environments and is designed to be extensible, any environment that implements the MDP interface can work with the RL algorithms. The tag game environment is implemented in Python using Pygame for rendering and physics.

## Environment and Episode Structure

The tag game environment operates on a simple episode structure where each episode represents a complete round of the game. An episode begins when both the tagger and the evader (our RL agent) are placed at random positions on the game field. During the episode, the RL agent attempts to avoid being tagged by the tagger while the tagger pursues the agent.

When the RL agent gets tagged, the episode terminates and the environment automatically resets both players to new random positions to begin the next episode. This reset mechanism ensures that each episode starts from a fresh state, preventing the agent from becoming overly specialized to specific starting positions and encouraging it to learn general evasion strategies that work from any initial configuration.

## Development History

This framework was originally built as part of the [ai-from-scratch](https://github.com/Silidrone/ai-from-scratch) repository in C++, where the neural network was implemented from scratch with custom backpropagation. The original C++ version worked with an environment that the agent communicated with over sockets, a 2D tag game written in Java that handled the game physics and rendering.

The RL framework code was ported from C++ to Python, while the neural network component now uses PyTorch instead of the custom C++ implementation. The motivation for this Python port was to compare performance between custom neural network implementations and a high-end library like PyTorch.

## Data Management

All training data is organized in the `data/` directory with each training run getting its own subdirectory identified by a unique 6-character alphanumeric ID (e.g., `a3k7x2`). Each run directory contains:

- **Model**: `taggame_model.pt` - The trained neural network
- **Plots**: Training progress plots (e.g., `taggame_training_ep000500.png`)
- **Logs**: `training.log` - Training statistics and progress

When running training mode, the model is automatically loaded if it exists to continue learning from the previous state. At the end of training, the updated model is saved back to the same file. When running evaluation mode, the saved model is loaded to demonstrate the learned policy without any further training. In evaluation mode, the agent acts 100% greedily with respect to the model's Q-values and the neural network is set to evaluation mode, which disables dropout and batch normalization updates.

## Neural Network Architecture

- **Type**: Fully connected feedforward network
- **Input**: 14 features (agent position, velocity, tagger position, velocity, action, distance, relative angle, tagged status, bias)
- **Hidden layers**: 2 layers with 64 neurons each, ReLU activation
- **Output**: Single Q-value
- **Optimizer**: Adam with learning rate 0.001
- **Weight initialization**: Xavier uniform

## Hyperparameters

- **Algorithm**: Q-learning
- **Discount factor**: 0.99
- **Exploration**: ε-greedy (start: 0.3, min: 0.01, decay: 0.9975)
- **Learning rate**: 0.0001 (no decay)
- **Training episodes**: 5,000,000
- **Output frequency**: Every 500 episodes (model saves, plots, logs)

## Unit Testing

The windygridworld environment from the Sutton-Barto RL book was also ported from C++ and serves as a unit test. Whenever I modify the RL logic or AI components, I run windygridworld and compare the results with the book's proposed optimal policy (Barto Sutton Reinforcement Learning: An Introduction, page 130, Example 6.5) to verify everything still works correctly.

## Usage

### Training

Start new training (generates random run ID):
```bash
python taggame_main.py
```

Start/continue training with specific run ID:
```bash
python taggame_main.py --run_id my_model
```

### Evaluation

Evaluate trained agent by run ID:
```bash
python taggame_main.py --mode evaluate --run_id my_model
```

Evaluate trained agent by model path:
```bash
python taggame_main.py --mode evaluate --model_path my_model/taggame_model.pt
```

### Training Run Management

Each training run is assigned a unique 6-character alphanumeric ID (e.g., `a3k7x2`) displayed at the start of training. All data for a run is stored in `data/{run_id}/` containing the model, plots, and training logs.

You can specify your own run ID using `--run_id` to:
- **Continue training**: If the run ID exists, training resumes from the saved model
- **Name your runs**: Use meaningful names instead of random IDs
- **Organize experiments**: Group related training runs with consistent naming

Output is automatically saved every 500 episodes (configurable via `OUTPUT_FREQ` constant):
- **Model**: `data/{run_id}/taggame_model.pt`
- **Plots**: `data/{run_id}/taggame_training_ep000500.png`, etc.
- **Logs**: `data/{run_id}/training.log` (episode statistics)

Run unit test:
```bash
python windy_grid_main.py
```

## Configuration

The `environments/taggame/constants.py` file contains both hyperparameters and game parameters. The current configuration optimized for Q-learning:

```python
# Game Environment
WIDTH = 1000
HEIGHT = 1000
FRAME_RATE_CAP = 3000  # High performance training
ENABLE_RENDERING = False
TIME_COEFFICIENT = 1
MAX_VELOCITY = 50
PLAYER_RADIUS = 20
TAG_COOLDOWN_MS = 10

# RL Hyperparameters
DISCOUNT_RATE = 0.99
N_OF_EPISODES = 5000000
POLICY_EPSILON = 0.3
MIN_EPSILON = 0.01
DECAY_RATE = 0.9975  # Faster epsilon decay for Q-learning
LEARNING_RATE = 0.0001  # Lower learning rate for stability
LEARNING_RATE_DECAY = 1.0  # No decay
MIN_LEARNING_RATE = 0.0001
HIDDEN_SIZE = 64

# Data Management
DATA_DIR = "data/"
MODEL_FILE = "taggame_model.pt"
OUTPUT_FREQ = 500  # Save frequency for models/plots/logs

# Game Setup
PLAYER_COUNT = 2
RL_PLAYER_NAME = "Sili"
```

**Note**: For evaluation mode, you must set `ENABLE_RENDERING = True` and it is recommended to set `TIME_COEFFICIENT = 0.02` to slow down the simulation and better observe how the agent behaves.

## Performance Optimization

Rendering is disabled by default during training (`ENABLE_RENDERING = False`) for significant performance gains. Training runs 10-50x faster without graphics rendering, making the full 50,000 episodes practical. Enable rendering only for evaluation to observe the trained agent's behavior.

The agent automatically detects and uses GPU (CUDA) when available, providing additional performance improvements for neural network training. GPU acceleration is recommended for training, while CPU is sufficient for evaluation.
