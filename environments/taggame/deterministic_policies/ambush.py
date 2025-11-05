import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class AmbushPolicy:

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
        target_vel = target.velocity

        if target_vel.length() > 10:
            perp_direction = Vector2D(-target_vel.y, target_vel.x).normalize()

            prediction_time = 1.5
            predicted_x = target_pos.x + target_vel.x * prediction_time
            predicted_y = target_pos.y + target_vel.y * prediction_time

            ambush_x = predicted_x + perp_direction.x * 100
            ambush_y = predicted_y + perp_direction.y * 100

            ambush_x = max(0, min(self.width, ambush_x))
            ambush_y = max(0, min(self.height, ambush_y))

            ambush_pos = Point2D(ambush_x, ambush_y)
            direction = Vector2D(ambush_pos.x - my_position.x, ambush_pos.y - my_position.y)
        else:
            direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)

        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
