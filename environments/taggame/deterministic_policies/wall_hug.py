"""Wall hug - stays near walls and chases along the perimeter."""
import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class WallHugPolicy:
    """Tries to stay near walls while chasing, cornering the opponent."""

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.width = width
        self.height = height

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))
        target_pos = target.static_info.pos

        # Find nearest wall
        dist_to_walls = [
            my_position.x,  # left
            self.width - my_position.x,  # right
            my_position.y,  # top
            self.height - my_position.y  # bottom
        ]
        min_wall_dist = min(dist_to_walls)

        # Direction towards target
        direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)

        if direction.length() > 0:
            direction = direction.normalize()

            # If far from walls, move towards nearest wall
            if min_wall_dist > 100:
                wall_idx = dist_to_walls.index(min_wall_dist)
                wall_direction = Vector2D(0, 0)

                if wall_idx == 0:  # left
                    wall_direction = Vector2D(-1, 0)
                elif wall_idx == 1:  # right
                    wall_direction = Vector2D(1, 0)
                elif wall_idx == 2:  # top
                    wall_direction = Vector2D(0, -1)
                else:  # bottom
                    wall_direction = Vector2D(0, 1)

                # Blend wall seeking with chasing
                direction = direction.times(0.7).plus(wall_direction.times(0.3))

            return direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
