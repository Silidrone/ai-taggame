import random
from abc import ABC, abstractmethod
from typing import Dict, Generic, Tuple, TypeVar

from mdp import MDP, Action, State

class ValueStrategy(Generic[State, Action]):
    pass

Return = float

class Policy(Generic[State, Action], ABC):
    def __init__(self, value_strategy: 'ValueStrategy[State, Action]' = None):
        self._mdp = None
        self._value_strategy = value_strategy
    
    def initialize(self, mdp: MDP[State, Action], value_strategy: 'ValueStrategy[State, Action]') -> None:
        if mdp is None:
            raise ValueError("MDP cannot be None")
        
        if value_strategy is None:
            raise ValueError("Value strategy cannot be None")
        
        self._mdp = mdp
        self._value_strategy = value_strategy
    
    def sample(self, state: State) -> Action:
        return self.greedy_action(state)[0]
    
    def greedy_action(self, state: State) -> Tuple[Action, Return]:
        return self._value_strategy.get_best_action(state)
    
    def optimal(self) -> Dict[State, Action]:
        if self._mdp is None or self._value_strategy is None:
            raise RuntimeError("Policy not properly initialized with MDP and ValueStrategy")
        
        optimal_policy = {}
        
        for state in self._mdp.states():
            if not self._mdp.is_terminal(state):
                best_action, _ = self.greedy_action(state)
                optimal_policy[state] = best_action
        
        return optimal_policy
    
    @property
    def value_strategy(self) -> 'ValueStrategy[State, Action]':
        return self._value_strategy

class EpsilonGreedyPolicy(Policy[State, Action]):
    def __init__(self, value_strategy: 'ValueStrategy[State, Action]', epsilon: float,
                 min_epsilon: float = 0.01, epsilon_decay: float = 1.0):
        super().__init__(value_strategy)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1 (inclusive)")
        
        if not 0.0 <= min_epsilon <= epsilon:
            raise ValueError("Min epsilon must be between 0 and epsilon")
        
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("Epsilon decay must be between 0 and 1")
        
        self._epsilon = epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
    
    def sample(self, state: State) -> Action:
        if self._epsilon > 0 and random.random() < self._epsilon:
            actions = self._mdp.actions(state)
            if not actions:
                raise RuntimeError(f"No available actions for state {state}")
            return random.choice(actions)
        else:
            return self.greedy_action(state)[0]
    
    @property
    def epsilon(self) -> float:
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1!")
        self._epsilon = epsilon
    
    @property
    def min_epsilon(self) -> float:
        return self._min_epsilon
    
    @min_epsilon.setter
    def min_epsilon(self, min_epsilon: float) -> None:
        if not 0.0 < min_epsilon <= 1.0:
            raise ValueError("Min epsilon must be between 0 and 1")
        self._min_epsilon = min_epsilon
    
    @property
    def epsilon_decay(self) -> float:
        return self._epsilon_decay
    
    @epsilon_decay.setter
    def epsilon_decay(self, epsilon_decay: float) -> None:
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("Epsilon decay must be between 0 and 1")
        self._epsilon_decay = epsilon_decay
    
    def decay_epsilon(self) -> None:
        self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)
        
class GreedyPolicy(EpsilonGreedyPolicy[State, Action]):
    def __init__(self, value_strategy: 'ValueStrategy[State, Action]'):
        super().__init__(value_strategy, 0, 0, 1)
