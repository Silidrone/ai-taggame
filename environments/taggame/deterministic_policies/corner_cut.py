import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class CornerCutPolicy:

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.width = width
        self.height = height
        self.corners = [
            Point2D(0, 0),
            Point2D(0, height),
            Point2D(width, 0),
            Point2D(width, height)
        ]

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))
        target_pos = target.static_info.pos

        nearest_corner = min(self.corners, key=lambda c: target_pos.distance(c))
        corner_distance = target_pos.distance(nearest_corner)

        if corner_distance < 200:
            cutoff_x = (target_pos.x + nearest_corner.x) / 2
            cutoff_y = (target_pos.y + nearest_corner.y) / 2
            cutoff_pos = Point2D(cutoff_x, cutoff_y)

            direction = Vector2D(cutoff_pos.x - my_position.x, cutoff_pos.y - my_position.y)
        else:
            direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)

        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
