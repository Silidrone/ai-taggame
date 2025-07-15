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
        print(f"Feature extractor using device: {_DEVICE}")
    return _DEVICE

def set_device(device: Optional[torch.device] = None) -> None:
    global _DEVICE
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DEVICE = device
    print(f"Feature extractor device set to: {_DEVICE}")

class TagGameQNet(TorchModel):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))            
        x = F.relu(self.fc2(x))
        return self.output(x)

def feature_extractor(state: TagGameState, action: TagGameAction, device=None) -> torch.Tensor:
    if device is None:
        device = get_device()
    
    my_pos, my_vel, tag_pos, tag_vel, is_tagged = state
    action_x, action_y = action
    
    norm_my_pos_x = my_pos[0] / WIDTH
    norm_my_pos_y = my_pos[1] / HEIGHT
    norm_my_vel_x = my_vel[0] / MAX_VELOCITY
    norm_my_vel_y = my_vel[1] / MAX_VELOCITY
    norm_tag_pos_x = tag_pos[0] / WIDTH
    norm_tag_pos_y = tag_pos[1] / HEIGHT
    norm_tag_vel_x = tag_vel[0] / MAX_VELOCITY
    norm_tag_vel_y = tag_vel[1] / MAX_VELOCITY
    
    norm_action_x = action_x / MAX_VELOCITY
    norm_action_y = action_y / MAX_VELOCITY
    
    dx = my_pos[0] - tag_pos[0]
    dy = my_pos[1] - tag_pos[1]
    distance = math.sqrt(dx * dx + dy * dy) / math.sqrt(WIDTH**2 + HEIGHT**2)
    
    if abs(norm_my_vel_x) > 0.001 or abs(norm_my_vel_y) > 0.001:
        my_dir = math.atan2(norm_my_vel_y, norm_my_vel_x)
        tag_dir = math.atan2(tag_pos[1] - my_pos[1], tag_pos[0] - my_pos[0])
        rel_angle = abs((my_dir - tag_dir + math.pi) % (2 * math.pi) - math.pi) / math.pi
    else:
        rel_angle = 0
    
    features = [
        norm_my_pos_x,
        norm_my_pos_y,
        norm_my_vel_x,
        norm_my_vel_y,
        norm_tag_pos_x,
        norm_tag_pos_y,
        norm_tag_vel_x,
        norm_tag_vel_y,
        norm_action_x,
        norm_action_y,
        distance,
        rel_angle,
        1.0 if is_tagged else 0.0,  # Is tagged flag
        1.0,  # Bias term
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