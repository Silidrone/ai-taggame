import math
from typing import List, Tuple
import time
import pygame
import random

from environments.taggame.dumb_tag_steering import DumbTagSteering
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer
from mdp import MDP, Reward
from environments.taggame.constants import (
    FRAME_RATE_CAP, WIDTH, HEIGHT, PLAYER_RADIUS, MAX_VELOCITY,
    TIME_COEFFICIENT, TAG_COOLDOWN_MS, RL_PLAYER_NAME, CORNER_EPISODES_PERCENTAGE
)

Position = Tuple[float, float]
Velocity = Tuple[float, float]
TagGameState = Tuple[Position, Velocity, Position, Velocity, bool]
TagGameAction = Tuple[int, int]

        
class TagGame(MDP[TagGameState, TagGameAction]):
    def __init__(self, render: bool = False):
        super().__init__()
        self.width = WIDTH
        self.height = HEIGHT
        self.player_radius = PLAYER_RADIUS
        self.max_velocity = MAX_VELOCITY
        self.time_coefficient = TIME_COEFFICIENT
        self.max_distance = math.sqrt(self.width**2 + self.height**2) - self.player_radius
        
        self.players: List[TagPlayer] = []
        self.tag_player = None
        self.tag_changed_time = 0
        self.render_enabled = render
        self.screen = None
        
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Tag Game RL")
            self.font = pygame.font.SysFont(None, 24)
            self.clock = pygame.time.Clock()
    
    def initialize(self) -> None:
        self._reset_game()    
        self._actions = {s: self.all_possible_actions() for s in self._states}
    
    def _reset_game(self) -> None:
        self.players.clear()
        self.tag_player = None
        self.tag_changed_time = 0
        
        player_count = 2
        
        for i in range(player_count):
            if random.random() < CORNER_EPISODES_PERCENTAGE:
                position = self._get_corner_position()
            else:
                position = self._get_random_position()

            player = TagPlayer(
                RL_PLAYER_NAME if i == 0 else f"P{i+1}",
                StaticInfo(position),
                self.max_velocity,
                self.player_radius,
                self.width,
                self.height
            )
            self.players.append(player)
        
        self.set_tag(self._get_random_non_rl_player())
        
        invalid_reset = False
        for player in self.players:
            if player != self.tag_player and self.tag_player.is_tagging(player):
                invalid_reset = True
                break
                
        if invalid_reset:
            self._reset_game()
    
    def reset(self) -> TagGameState:
        self._reset_game()
        return self._get_state()
    
    def _get_state(self) -> TagGameState:
        rl_player = self._get_rl_player()
        tag_player = self.tag_player
        
        if rl_player == tag_player:
            tag_player = next((p for p in self.players if p != rl_player), None)
        
        if not tag_player:
            return ((0, 0), (0, 0), (0, 0), (0, 0), False)
        
        return (
            (rl_player.static_info.pos.x, rl_player.static_info.pos.y),
            (rl_player.velocity.x, rl_player.velocity.y),
            (tag_player.static_info.pos.x, tag_player.static_info.pos.y),
            (tag_player.velocity.x, tag_player.velocity.y),
            rl_player.is_tagged
        )
    
    def step(self, state: TagGameState, action: TagGameAction) -> Tuple[TagGameState, Reward]:
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    
        rl_player = self._get_rl_player()
        
        x, y = action
        action_vector = Vector2D(x, y)
        if action_vector.length() > 0:
            action_vector = action_vector.normalize().times(self.max_velocity)
        
        rl_player.set_velocity(action_vector)
        
        current_time = pygame.time.get_ticks() if pygame.get_init() else int(time.time() * 1000)
        tagger_sleeping = (current_time - self.tag_changed_time < TAG_COOLDOWN_MS)
        
        if not tagger_sleeping:
            tagger = self.tag_player
            if tagger != rl_player:
                tagger.set_steering_behavior(
                    DumbTagSteering(tagger, self, self.width, self.height, self.max_velocity)
                )
                self._handle_tagging_logic()
        
        for player in self.players:
            if player != rl_player:
                player.update(1.0 * self.time_coefficient)
        
        rl_player.update(1.0 * self.time_coefficient)
        new_state = self._get_state()
        reward = self._calculate_reward(state, new_state)
        
        if self.render_enabled:
            self._render()
            self.clock.tick(FRAME_RATE_CAP)
            
        return new_state, reward
    
    def _calculate_reward(self, old_state: TagGameState, new_state: TagGameState) -> Reward:
        old_rl_pos, old_rl_vel, old_tag_pos, old_tag_vel, old_is_tagged = old_state
        rl_pos, rl_vel, tag_pos, tag_vel, new_is_tagged = new_state

        if not old_is_tagged and new_is_tagged:
            return -1.0

        # enemy is on cooldown
        if tag_pos == old_tag_pos:
            return 0.0

        return 0.1
        
    def is_terminal(self, state: TagGameState) -> bool:
        _, _, _, _, is_tagged = state
        return is_tagged
    
    def is_valid(self, state: TagGameState, action: TagGameAction) -> bool:
        return True
    
    def all_possible_actions(self) -> List[TagGameAction]:
        actions = []
        for i in range(16):
            angle = i * 22.5
            rad = math.radians(angle)
            x = int(round(math.cos(rad) * self.max_velocity))
            y = int(round(math.sin(rad) * self.max_velocity))
            if x != 0 or y != 0:
                actions.append((x, y))
        return actions
    
    def _get_rl_player(self) -> TagPlayer:
        return next((p for p in self.players if p.name == RL_PLAYER_NAME), None)
    
    def _get_random_non_rl_player(self) -> TagPlayer:
        non_rl_players = [p for p in self.players if p.name != RL_PLAYER_NAME]
        if not non_rl_players:
            return None
        return random.choice(non_rl_players)
    
    def _get_random_position(self) -> Point2D:
        return Point2D(
            random.uniform(self.player_radius, self.width - self.player_radius),
            random.uniform(self.player_radius, self.height - self.player_radius)
        )

    def _get_corner_position(self) -> Point2D:
        """Get a random position in the vicinity of a corner (within 70 pixels)."""
        corner_radius = 70
        corner = random.choice([
            'bottom_left', 'bottom_right', 'top_left', 'top_right'
        ])

        if corner == 'bottom_left':
            x = random.uniform(self.player_radius, corner_radius)
            y = random.uniform(self.height - corner_radius, self.height - self.player_radius)
        elif corner == 'bottom_right':
            x = random.uniform(self.width - corner_radius, self.width - self.player_radius)
            y = random.uniform(self.height - corner_radius, self.height - self.player_radius)
        elif corner == 'top_left':
            x = random.uniform(self.player_radius, corner_radius)
            y = random.uniform(self.player_radius, corner_radius)
        else:  # top_right
            x = random.uniform(self.width - corner_radius, self.width - self.player_radius)
            y = random.uniform(self.player_radius, corner_radius)

        return Point2D(x, y)

    def set_tag(self, new_player: TagPlayer) -> None:
        self.tag_player = new_player
        for player in self.players:
            player.set_is_tagged(player == new_player)
        self.tag_changed_time = pygame.time.get_ticks() if pygame.get_init() else int(time.time() * 1000)
    
    def _handle_tagging_logic(self) -> None:
        for player in self.players:
            if player != self.tag_player and self.tag_player.is_tagging(player):
                self.set_tag(player)
                self.tag_player.set_steering_behavior(lambda s, v: Vector2D(0, 0))
                return
    
    def _render(self) -> None:
        if not self.render_enabled or not self.screen:
            return
            
        self.screen.fill((0, 0, 0))

        # Draw border (visual only, doesn't affect physics)
        pygame.draw.rect(self.screen, (100, 100, 100), (0, 0, self.width, self.height), 3)

        for player in self.players:
            if player.name == RL_PLAYER_NAME:
                color = (0, 0, 255)  # Blue
            else:
                color = (0, 255, 0)  # Green
            
            pygame.draw.circle(
                self.screen,
                color,
                (int(player.static_info.pos.x), int(player.static_info.pos.y)),
                int(player.radius)
            )
            
            end_x = player.static_info.pos.x + player.radius * math.cos(player.static_info.orientation)
            end_y = player.static_info.pos.y + player.radius * math.sin(player.static_info.orientation)
            pygame.draw.line(
                self.screen,
                (255, 255, 255), # White
                (int(player.static_info.pos.x), int(player.static_info.pos.y)),
                (int(end_x), int(end_y)),
                2
            )
            
            if player.is_tagged:
                pygame.draw.circle(
                    self.screen,
                    (255, 0, 0),  # Red
                    (int(player.static_info.pos.x), int(player.static_info.pos.y)),
                    int(player.radius + 1),
                    2
                )
        
        pygame.display.flip()
    
    def close(self) -> None:
        if pygame.get_init():
            pygame.quit()