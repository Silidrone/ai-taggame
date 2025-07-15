import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any


class TorchModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass