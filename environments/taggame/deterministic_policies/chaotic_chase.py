import math
import random
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class ChaoticChasePolicy:

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.width = width
        self.height = height
        self.noise_scale = 0.7
        self.direction_change_prob = 0.3
        self.last_direction = Vector2D(1, 0)

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))
        target_pos = target.static_info.pos

        base_direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)
        
        if base_direction.length() == 0:
            return Vector2D(0, 0)
            
        base_direction = base_direction.normalize()

        noise_x = random.gauss(0, 0.3)
        noise_y = random.gauss(0, 0.3)
        
        chaotic_direction = Vector2D(
            base_direction.x * 0.8 + noise_x * 0.2,
            base_direction.y * 0.8 + noise_y * 0.2
        )
        
        if random.random() < 0.15:
            jitter_angle = random.uniform(-0.5, 0.5)
            cos_j = math.cos(jitter_angle)
            sin_j = math.sin(jitter_angle)
            rotated_x = chaotic_direction.x * cos_j - chaotic_direction.y * sin_j
            rotated_y = chaotic_direction.x * sin_j + chaotic_direction.y * cos_j
            chaotic_direction = Vector2D(rotated_x, rotated_y)

        if chaotic_direction.length() > 0:
            chaotic_direction = chaotic_direction.normalize()
            self.last_direction = chaotic_direction
            return chaotic_direction.times(self.max_velocity)

        return Vector2D(0, 0)