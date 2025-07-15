from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from mdp import MDP, Action, State
from policy import Policy
from value_strategy import ValueStrategy

Threshold = float

class MDPSolver(Generic[State, Action], ABC):
    def __init__(self, mdp: MDP[State, Action], policy: Policy[State, Action],
                 discount_rate: float, policy_threshold: float):
        if not 0.0 <= discount_rate <= 1.0:
            raise ValueError("Discount rate must be between 0 and 1")
        
        self._mdp = mdp
        self._policy = policy
        self._discount_rate = discount_rate
        self._policy_threshold = policy_threshold
    
    @abstractmethod
    def policy_iteration(self) -> None:
        pass
    
    @property
    def mdp(self) -> MDP[State, Action]:
        return self._mdp
    
    @property
    def policy(self) -> Policy[State, Action]:
        return self._policy
    
    @property
    def discount_rate(self) -> float:
        return self._discount_rate
    
    @property
    def policy_threshold(self) -> float:
        return self._policy_threshold