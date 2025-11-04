"""Center control - maintains position near center to control space."""
import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class CenterControlPolicy:
    """Tries to stay near center while chasing, controlling the space."""

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.center = Point2D(width / 2, height / 2)

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))
        target_pos = target.static_info.pos

        # Direction towards target
        chase_direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)

        # Direction towards center
        center_direction = Vector2D(self.center.x - my_position.x, self.center.y - my_position.y)

        # Distance from center determines blending
        distance_from_center = my_position.distance(self.center)
        center_weight = min(1.0, distance_from_center / 200.0)

        if chase_direction.length() > 0 and center_direction.length() > 0:
            chase_direction = chase_direction.normalize()
            center_direction = center_direction.normalize()

            # Blend: chase more when close to center, return to center when far
            final_direction = chase_direction.times(1.0 - center_weight * 0.5).plus(
                center_direction.times(center_weight * 0.5)
            )

            return final_direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
