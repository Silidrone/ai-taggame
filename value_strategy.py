from abc import ABC, abstractmethod
from typing import Callable, Generic, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from mdp import MDP, Action, State
from torch_model import TorchModel
from replay_buffer import PrioritizedReplayBuffer

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
    
    def target_q(self, state: State, action: Action) -> float:
        return self.Q(state, action)
    
    def save(self, path: str) -> None:
        raise NotImplementedError("Save not implemented for this ValueStrategy")
    
    def load(self, path: str) -> None:
        raise NotImplementedError("Load not implemented for this ValueStrategy")

class TorchValueStrategy(ValueStrategy[State, Action]):
    def __init__(self, network: TorchModel, 
                 feature_extractor: Callable[[State, Action], torch.Tensor],
                 step_size: float = 0.01, min_lr: float = 0.0001, 
                 target_update_freq: int = 500, scheduler_type: str = "None",
                 scheduler_patience: int = 500, scheduler_factor: float = 0.5,
                 use_per: bool = True, buffer_size: int = 50000, 
                 batch_size: int = 32, per_alpha: float = 0.6, per_beta: float = 0.4,
                 per_beta_anneal_steps: int = 100000, per_epsilon: float = 1e-6,
                 discount_rate: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = network
        self.target_network = type(network)(network.fc1.in_features, network.fc1.out_features)
        self.target_network.load_state_dict(network.state_dict())
        
        self.feature_extractor = feature_extractor
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.Adam(network.parameters(), lr=step_size)
        
        self.scheduler = None
        if scheduler_type == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=scheduler_factor, 
                patience=scheduler_patience, min_lr=min_lr
            )
        elif scheduler_type == "ExponentialLR":
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        elif scheduler_type == "StepLR":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=scheduler_patience, gamma=scheduler_factor)
        
        self._mdp = None
        self._step_size = step_size
        self.min_lr = min_lr
        self.update_target_freq = target_update_freq
        self.update_count = 0
        
        self.use_per = use_per
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size, per_alpha, per_beta, per_beta_anneal_steps, per_epsilon
            )
    
    def initialize(self, mdp: MDP[State, Action]) -> None:
        self._mdp = mdp
    
    def step_scheduler(self, metric: float = None) -> None:
        if self.scheduler is not None:
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
    
    def get_current_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]['lr']
    
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
    
    def _compute_q(self, state: State, action: Action, network) -> float:
        with torch.no_grad():
            state_action = self.feature_extractor(state, action, device=self.device)
            if state_action.device != next(network.parameters()).device:
                state_action = state_action.to(next(network.parameters()).device)
            return network(state_action).item()
    
    def Q(self, state: State, action: Action) -> float:
        return self._compute_q(state, action, self.q_network)
    
    def target_q(self, state: State, action: Action) -> float:
        return self._compute_q(state, action, self.target_network)
    
    def add_experience(self, state: State, action: Action, reward: float, 
                      next_state: State, done: bool, td_error: Optional[float] = None):
        if self.use_per:
            self.replay_buffer.add(state, action, reward, next_state, done, td_error)
    
    def update(self, state: State, action: Action, target_q: float) -> None:
        if not self.use_per:
            self._single_update(state, action, target_q)
        elif len(self.replay_buffer) >= self.batch_size:
            self._batch_update()
    
    def _single_update(self, state: State, action: Action, target_q: float) -> None:
        state_action = self.feature_extractor(state, action, device=self.device)
        model_device = next(self.q_network.parameters()).device
        if state_action.device != model_device:
            state_action = state_action.to(model_device)
        
        current_q = self.q_network(state_action)
        target = torch.tensor([[target_q]], dtype=current_q.dtype, device=model_device)
        
        loss = nn.functional.mse_loss(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _batch_update(self) -> None:
        experiences, idxs, is_weights = self.replay_buffer.sample(self.batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in experiences:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)
        
        batch_features = []
        targets = []
        
        for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            state_action = self.feature_extractor(state, action, device=self.device)
            batch_features.append(state_action.squeeze())
            
            if done:
                target = reward
            else:
                best_action, best_q = self.get_best_action(next_state)
                target = reward + self.discount_rate * best_q
            targets.append(target)
        
        batch_features = torch.stack(batch_features)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        
        current_q_values = self.q_network(batch_features).squeeze()
        
        td_errors = targets - current_q_values
        weighted_loss = (td_errors.pow(2) * is_weights).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy().tolist())
        
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict()) # update target network
    
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
    
