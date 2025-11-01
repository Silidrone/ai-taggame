import random
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Tuple, TypeVar

State = TypeVar('State')
Action = TypeVar('Action')
Reward = float
Probability = float
Transition = Tuple[State, Reward, Probability]


class MDP(Generic[State, Action], ABC):
    def __init__(self, is_continuous: bool = False):
        self._states: List[State] = []
        self._terminal_states: List[State] = []
        self._actions: Dict[State, List[Action]] = {}
        self._dynamics: Dict[Tuple[State, Action], List[Transition]] = {}
        self._is_continuous = is_continuous
    
    @property
    def is_continuous(self) -> bool:
        return self._is_continuous
    
    @abstractmethod
    def initialize(self) -> None:
        pass
    
    def states(self) -> List[State]:
        return self._states
    
    def terminal_states(self) -> List[State]:
        return self._terminal_states
    
    def actions(self, state: State, fallback: bool = True) -> List[Action]:
        if state in self._actions:
            return self._actions[state]
        
        if fallback:
            return self.all_possible_actions()
        else:
            raise KeyError(f"State {state} not found in action map")
    
    def all_actions(self) -> Dict[State, List[Action]]:
        return self._actions
    
    def all_possible_actions(self) -> List[Action]:
        raise NotImplementedError("all_possible_actions is not implemented in this environment.")
    
    def is_valid(self, state: State, action: Action) -> bool:
        raise NotImplementedError("is_valid is not implemented in this environment.")
    
    def transitions(self, state: State, action: Action) -> List[Transition]:
        return self._dynamics[(state, action)]
    
    def dynamics(self) -> Dict[Tuple[State, Action], List[Transition]]:
        return self._dynamics
    
    def reset(self) -> State:
        raise NotImplementedError("The reset function is not available in this environment.")
    
    def step(self, state: State, action: Action) -> Tuple[State, Reward]:
        raise NotImplementedError("The step function is not available in this environment.")
    
    def is_terminal(self, state: State) -> bool:
        if self._is_continuous:
            return False
        
        return state in self._terminal_states
    
    def random_action(self, state: State) -> Action:
        actions = self.actions(state)
        if not actions:
            raise RuntimeError(f"No available actions for state {state}")
        
        return random.choice(actions)