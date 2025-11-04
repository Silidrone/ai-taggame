"""Random walk - moves in random directions with some bias towards target."""
import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class RandomWalkPolicy:
    """Moves somewhat randomly but with bias towards the target."""

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.direction_change_counter = 0
        self.current_random_angle = 0

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))
        target_pos = target.static_info.pos

        # Change random direction every 30 steps
        self.direction_change_counter += 1
        if self.direction_change_counter % 30 == 0:
            # Deterministic "random" based on position (ensures determinism)
            seed = int(my_position.x + my_position.y) % 360
            self.current_random_angle = math.radians(seed)

        # Direction towards target
        chase_direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)

        if chase_direction.length() > 0:
            chase_direction = chase_direction.normalize()

            # Random direction
            random_vx = math.cos(self.current_random_angle)
            random_vy = math.sin(self.current_random_angle)
            random_direction = Vector2D(random_vx, random_vy)

            # Blend 70% chase, 30% random
            final_direction = chase_direction.times(0.7).plus(random_direction.times(0.3))

            return final_direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
