# ai-taggame: Reinforcement Learning Agent for a 2D Tag Game

AI agent that learns an evasion behavior through Deep Q-Networks. The agent employs deep Q-learning with prioritized experience replay, target networks, n-step returns, and adaptive learning rate scheduling to develop optimal evasion strategies.

Built on top of a custom extensible reinforcement learning framework designed to work with any MDP (Markov Decision Process) environment.

## Architecture

The agent uses Deep Q-Network (DQN) with several advanced techniques:

- **Deep Q-Learning**: Neural network function approximation with target networks for stable training
- **Prioritized Experience Replay (PER)**: Efficient sampling of important experiences using TD-error priorities
- **N-Step Returns**: Multi-step bootstrapping for improved sample efficiency
- **Adaptive Learning Rate**: PyTorch schedulers that adjust learning rate based on performance
- **Enhanced Action Space**: Fine-grained directional control for precise movement
- **Enhanced Feature Engineering**: Spatial features including wall distances and opponent vector

The RL framework models MDP environments and is designed to be extensible. Any environment implementing the MDP interface can work with the RL algorithms. The tag game environment is implemented in Python using Pygame for rendering and physics.

## Environment and Episode Structure

The tag game environment operates on a simple episode structure where each episode represents a complete round of the game. An episode begins when both the tagger and the evader (our RL agent) are placed at random positions on the game field. During the episode, the RL agent attempts to avoid being tagged by the tagger, which follows a deterministic algorithm that chases our agent.

When the RL agent gets tagged, the episode terminates and the environment automatically resets both players to new random positions to begin the next episode. This reset mechanism ensures that each episode starts from a fresh state, preventing the agent from becoming overly specialized to specific starting positions and encouraging it to learn general evasion strategies that work from any initial configuration.

## Development History

This framework was originally built in C++ with a custom neural network implementation featuring hand-written backpropagation. The original C++ version communicated with a 2D tag game environment written in Java over sockets. The RL framework was later ported to Python with PyTorch replacing the custom neural network implementation to achieve more efficient environment integration, more efficient neural networks, and access to many different capabilities.

## Data Management

All training data is organized in the `data/` directory with each training run getting its own subdirectory identified by a unique 6-character alphanumeric ID (e.g., `a3k7x2`). Each run directory contains:

- **Model**: `taggame_model.pt` - The trained neural network
- **Plots**: Training progress plots (e.g., `taggame_training_ep000500.png`)
- **Logs**: `training.log` - Training statistics and progress

When running training mode, the model is automatically loaded if it exists to continue learning from the previous state. At the end of training, the updated model is saved back to the same file. When running evaluation mode, the saved model is loaded to demonstrate the learned policy without any further training. In evaluation mode, the agent acts 100% greedily with respect to the model's Q-values and the neural network is set to evaluation mode, which disables dropout and batch normalization updates.

## Neural Network Architecture

- **Type**: Fully connected feedforward network with target network
- **Input**: Spatial features including wall distances, opponent velocity, action vector, distance, relative angle, opponent difference vector
- **Hidden layers**: Multi-layer architecture with ReLU activation and He initialization
- **Output**: Single Q-value per state-action pair
- **Optimizer**: Adam with adaptive learning rate scheduling
- **Target Network**: Configurable update frequency for stable bootstrapping
- **Gradient Clipping**: Prevents gradient explosion during training

## Hyperparameters

### Core RL Parameters
- **Algorithm**: Deep Q-Learning with target networks
- **Exploration**: ε-greedy with configurable decay
- **Training episodes**: Configurable via `N_OF_EPISODES`

### Neural Network & Learning
- **Learning rate**: Adaptive scheduling with ReduceLROnPlateau
- **Architecture**: Multi-layer network with configurable hidden size
- **Batch training**: Configurable batch size for stable learning
- **Target network updates**: Periodic synchronization for stability

### Experience Replay
- **Buffer**: Configurable replay buffer with prioritized sampling
- **Prioritization**: TD-error based experience sampling
- **Importance sampling**: Bias correction with annealing
- **N-step returns**: Multi-step bootstrapping

### Output & Monitoring
- **Model saves**: Configurable via `MODEL_SAVE_FREQ`
- **Plot generation**: Configurable via `OUTPUT_FREQ`
- **Learning rate scheduling**: Configurable patience and decay factors

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
python taggame_main.py --run_id a3k7x2
```

### Evaluation

Evaluate trained agent by run ID:
```bash
python taggame_main.py --mode evaluate --run_id a3k7x2
```

Evaluate trained agent by model path (relative to `DATA_DIR`):
```bash
python taggame_main.py --mode evaluate --model_path a3k7x2/taggame_model.pt
```

### Training Run Management

Each training run is assigned a unique 6-character alphanumeric ID (e.g., `a3k7x2`) displayed at the start of training. All data for a run is stored in `data/{run_id}/` containing the model, plots, and training logs.

You can specify your own run ID using `--run_id` to:
- **Continue training**: If the run ID exists, training resumes from the saved model
- **Name your runs**: Use meaningful names instead of random IDs
- **Organize experiments**: Group related training runs with consistent naming

Output is automatically saved (configurable via `OUTPUT_FREQ` constant):
- **Model**: `data/{run_id}/taggame_model.pt`
- **Plots**: `data/{run_id}/taggame_training_ep000500.png`, etc.
- **Logs**: `data/{run_id}/training.log` (episode statistics)

Run unit test:
```bash
python windy_grid_main.py
```

## Configuration

The `environments/taggame/constants.py` file contains both hyperparameters and game parameters. The current configuration optimized for advanced DQN with PER:

```python
# Environment
WIDTH = 350
HEIGHT = 350
FRAME_RATE_CAP = 3000
ENABLE_RENDERING = True
TIME_COEFFICIENT = 0.2
PLAYER_RADIUS = 5
MAX_VELOCITY = 50
TAG_COOLDOWN_MS = 10

# Core RL
DISCOUNT_RATE = 0.99
N_STEP_RETURNS = 5
N_OF_EPISODES = 10000
HIDDEN_SIZE = 128

# Exploration
POLICY_EPSILON = 1.0       # Start with full exploration
MIN_EPSILON = 0.01         # Slower decay to minimum
DECAY_RATE = 0.995         # Gradual epsilon reduction

# Learning & Neural Network
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.0003
BATCH_SIZE = 64            # Stable gradient estimates
TARGET_NETWORK_UPDATE_FREQ = 500

# Prioritized Experience Replay (PER)
REPLAY_BUFFER_SIZE = 200000
PER_ALPHA = 0.7            # Prioritization strength
PER_BETA = 0.4             # Importance sampling correction
PER_BETA_ANNEAL_STEPS = 200000
PER_EPSILON = 1e-6

# Learning Rate Scheduler
LR_SCHEDULER_TYPE = "ReduceLROnPlateau"
LR_SCHEDULER_PATIENCE = 1000
LR_SCHEDULER_FACTOR = 0.5

# Output & Saving
OUTPUT_FREQ = 10           # Frequent monitoring
MODEL_SAVE_FREQ = 10
```

**Note**: For evaluation mode, set `ENABLE_RENDERING = True` and consider `TIME_COEFFICIENT = 0.02` to slow down visualization.

## Performance Optimization

**Rendering**: Disable during training (`ENABLE_RENDERING = False`) for performance gains. Enable only for evaluation to observe learned behaviors.

**Hardware**: Automatic GPU detection and CUDA acceleration when available. GPU recommended for training with experience replay; CPU sufficient for evaluation.

**Memory Efficiency**: Prioritized experience replay provides stable learning while managing memory usage effectively.
