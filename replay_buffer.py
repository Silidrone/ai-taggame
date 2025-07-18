import random
import torch
from collections import deque
from typing import List, Tuple, Any

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple[Any, Any, float, Any, bool]]:
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} experiences, cannot sample {batch_size}")
        
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size
    
    def clear(self) -> None:
        self.buffer.clear()