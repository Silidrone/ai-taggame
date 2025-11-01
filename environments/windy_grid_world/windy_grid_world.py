from typing import List, Tuple

from mdp import MDP, Reward
from environments.windy_grid_world.config import (GRID_HEIGHT, GRID_WIDTH, START_STATE, GOAL_STATE, WIND_STRENGTH)

GridPosition = Tuple[int, int]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
ACTIONS = [UP, RIGHT, DOWN, LEFT]

class WindyGridWorld(MDP[GridPosition, Tuple[int, int]]):
    def __init__(self):
        super().__init__()
        
        self._grid_width = GRID_WIDTH
        self._grid_height = GRID_HEIGHT
        self._start_state = START_STATE
        self._goal_state = GOAL_STATE
        self._wind_strength = WIND_STRENGTH
        
        self._terminal_states = [self._goal_state]
    
    def initialize(self) -> None:
        for r in range(self._grid_height):
            for c in range(self._grid_width):
                state = (r, c)
                self._states.append(state)
                available_actions = []
                for action in ACTIONS:
                    if self.is_valid(state, action):
                        available_actions.append(action)
                
                self._actions[state] = available_actions
    
    def _walk(self, state: GridPosition, action: Tuple[int, int]) -> GridPosition:
        row, col = state
        dr, dc = action
        return (row + dr, col + dc)
    
    def _walk_with_wind(self, state: GridPosition, action: Tuple[int, int]) -> GridPosition:
        row, col = state
        next_state = self._walk(state, action)
        next_row, next_col = next_state
        next_row = max(0, min(self._grid_height - 1, next_row))
        next_col = max(0, min(self._grid_width - 1, next_col))
        if 0 <= col < len(self._wind_strength):
            wind = self._wind_strength[col]
            next_row = max(0, min(self._grid_height - 1, next_row - wind))
        
        return (next_row, next_col)
    
    def all_possible_actions(self) -> List[Tuple[int, int]]:
        return ACTIONS
    
    def is_valid(self, state: GridPosition, action: Tuple[int, int]) -> bool:
        next_state = self._walk(state, action)
        row, col = next_state
        return (0 <= row < self._grid_height and 0 <= col < self._grid_width)
    
    def step(self, state: GridPosition, action: Tuple[int, int]) -> Tuple[GridPosition, Reward]:
        next_state = self._walk_with_wind(state, action)

        if self.is_terminal(next_state):
            return next_state, 0.0
        
        return next_state, -1.0
    
    def reset(self) -> GridPosition:
        return self._start_state
    
    def is_terminal(self, state: GridPosition) -> bool:
        return state == self._goal_state
    
    def get_state_grid_representation(self) -> List[List[str]]:
        grid = [[' ' for _ in range(self._grid_width)] for _ in range(self._grid_height)]
        start_r, start_c = self._start_state
        goal_r, goal_c = self._goal_state
        
        grid[start_r][start_c] = 'S'
        grid[goal_r][goal_c] = 'G'
        
        return grid
    
    def print_grid(self) -> None:
        grid = self.get_state_grid_representation()

        print('    ', end='')
        for c in range(self._grid_width):
            print(f"{c} ", end='')
        print()
        
        print('  +' + '-' * (self._grid_width * 2 - 1) + '+')
        
        for r in range(self._grid_height):
            print(f"{r} |" + '|'.join(cell for cell in grid[r]) + '|')
        
        print('  +' + '-' * (self._grid_width * 2 - 1) + '+')
        
        print('    ', end='')
        for c, w in enumerate(self._wind_strength):
            print(f"{w} " if w >= 0 else "  ", end='')
        print("\n    Wind strength")
    
    def close():
        pass
