from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Tuple, TypeVar

State = TypeVar('State')
Action = TypeVar('Action')
Reward = float


class MDP(Generic[State, Action], ABC):
    def __init__(self, is_continuous: bool = False):
        self._states: List[State] = []
        self._terminal_states: List[State] = []
        self._actions: Dict[State, List[Action]] = {}
        self._is_continuous = is_continuous

    @property
    def is_continuous(self) -> bool:
        return self._is_continuous

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def actions(self, state: State, fallback: bool = True) -> List[Action]:
        if state in self._actions:
            return self._actions[state]

        if fallback:
            return self.all_possible_actions()
        else:
            raise KeyError(f"State {state} not found in action map")

    def all_possible_actions(self) -> List[Action]:
        raise NotImplementedError("all_possible_actions is not implemented in this environment.")

    def reset(self) -> State:
        raise NotImplementedError("The reset function is not available in this environment.")

    def step(self, state: State, action: Action) -> Tuple[State, Reward]:
        raise NotImplementedError("The step function is not available in this environment.")

    def is_terminal(self, state: State) -> bool:
        if self._is_continuous:
            return False

        return state in self._terminal_states