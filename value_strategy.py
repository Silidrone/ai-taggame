from abc import ABC, abstractmethod
from typing import Callable, Generic, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from mdp import MDP, Action, State
from torch_model import TorchModel

Return = float

class ValueStrategy(Generic[State, Action], ABC):
    @abstractmethod
    def initialize(self, mdp: MDP[State, Action]) -> None:
        pass
    
    @abstractmethod
    def get_best_action(self, state: State) -> Tuple[Action, Return]:
        pass
    
    @abstractmethod
    def Q(self, state: State, action: Action) -> float:
        pass
    
    @abstractmethod
    def update(self, state: State, action: Action, target_q: float) -> None:
        pass
    
    def save(self, path: str) -> None:
        raise NotImplementedError("Save not implemented for this ValueStrategy")
    
    def load(self, path: str) -> None:
        raise NotImplementedError("Load not implemented for this ValueStrategy")

class TorchValueStrategy(ValueStrategy[State, Action]):
    def __init__(self, network: TorchModel, 
                 feature_extractor: Callable[[State, Action], torch.Tensor],
                 step_size: float = 0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TorchValueStrategy initialized with device: {self.device}")
        
        self.q_network = network
        self.feature_extractor = feature_extractor
        self.q_network.to(self.device)
        self.optimizer = optim.Adam(network.parameters(), lr=step_size)
        self._mdp = None
        self._step_size = step_size
    
    def initialize(self, mdp: MDP[State, Action]) -> None:
        self._mdp = mdp
    
    def get_best_action(self, state: State) -> Tuple[Action, Return]:
        if self._mdp is None:
            raise RuntimeError("TorchValueStrategy not initialized with an MDP")
        
        best_action = None
        best_value = float('-inf')
        
        for action in self._mdp.actions(state):
            if self._mdp.is_valid(state, action):
                value = self.Q(state, action)
                if value > best_value:
                    best_value = value
                    best_action = action
        
        return best_action, best_value
    
    def Q(self, state: State, action: Action) -> float:
        with torch.no_grad():
            state_action = self.feature_extractor(state, action, device=self.device)
            if state_action.device != next(self.q_network.parameters()).device:
                state_action = state_action.to(next(self.q_network.parameters()).device)
            return self.q_network(state_action).item()
    
    def update(self, state: State, action: Action, target_q: float) -> None:
        state_action = self.feature_extractor(state, action, device=self.device)
        model_device = next(self.q_network.parameters()).device
        if state_action.device != model_device:
            state_action = state_action.to(model_device)
        
        current_q = self.q_network(state_action)
        target = torch.tensor([[target_q]], dtype=current_q.dtype, device=model_device)
        
        loss = nn.functional.mse_loss(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, path: str) -> None:
        try:
            self.q_network.to('cpu')
            torch.save(self.q_network.state_dict(), path)
            self.q_network.to(self.device)
        except Exception as e:
            self.q_network.to(self.device)
            raise IOError(f"Failed to save model: {e}")
    
    def load(self, path: str) -> None:
        try:
            self.q_network.load_state_dict(torch.load(path, map_location=self.device))
            self.q_network.to(self.device)
        except Exception as e:
            raise IOError(f"Failed to load model: {e}")
    
