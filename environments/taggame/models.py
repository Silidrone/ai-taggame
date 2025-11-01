import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from torch_model import TorchModel
from environments.taggame.constants import (
    WIDTH, HEIGHT, MAX_VELOCITY, HIDDEN_SIZE
)
from environments.taggame.taggame import TagGameState, TagGameAction

_DEVICE = None

def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE

def set_device(device: Optional[torch.device] = None) -> None:
    global _DEVICE
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DEVICE = device

class TagGameQNet(TorchModel):
    def __init__(self, input_size: int, n_actions: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, n_actions)

        # He initialization for ReLU
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.xavier_uniform_(self.output.weight)

        # Zero bias initialization
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions. Shape: (batch_size, n_actions)"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)

def feature_extractor(state: TagGameState, device=None) -> torch.Tensor:
    """Extract state features (no action). Returns 11 features."""
    if device is None:
        device = get_device()

    my_pos, my_vel, tag_pos, tag_vel, is_tagged = state

    norm_my_vel_x = my_vel[0] / MAX_VELOCITY
    norm_my_vel_y = my_vel[1] / MAX_VELOCITY

    norm_tag_vel_x = tag_vel[0] / MAX_VELOCITY
    norm_tag_vel_y = tag_vel[1] / MAX_VELOCITY

    dx = my_pos[0] - tag_pos[0]
    dy = my_pos[1] - tag_pos[1]
    distance = math.sqrt(dx * dx + dy * dy) / math.sqrt(WIDTH**2 + HEIGHT**2)

    angle_to_predator = math.atan2(dy, dx)
    normalized_angle = (angle_to_predator + math.pi) / (2 * math.pi)

    dist_to_left = my_pos[0] / WIDTH
    dist_to_right = (WIDTH - my_pos[0]) / WIDTH
    dist_to_top = my_pos[1] / HEIGHT
    dist_to_bottom = (HEIGHT - my_pos[1]) / HEIGHT

    features = [
        dist_to_left,
        dist_to_right,
        dist_to_top,
        dist_to_bottom,
        norm_my_vel_x,
        norm_my_vel_y,
        norm_tag_vel_x,
        norm_tag_vel_y,
        distance,
        normalized_angle,
        1.0,
    ]

    return torch.tensor([features], dtype=torch.float32, device=device)

def state_to_readable(state: TagGameState) -> str:
    my_pos, my_vel, tag_pos, tag_vel, is_tagged = state
    return (
        f"MyPos: ({my_pos[0]:.1f}, {my_pos[1]:.1f}), "
        f"MyVel: ({my_vel[0]:.1f}, {my_vel[1]:.1f}), "
        f"TagPos: ({tag_pos[0]:.1f}, {tag_pos[1]:.1f}), "
        f"TagVel: ({tag_vel[0]:.1f}, {tag_vel[1]:.1f}), "
        f"Tagged: {is_tagged}"
    )