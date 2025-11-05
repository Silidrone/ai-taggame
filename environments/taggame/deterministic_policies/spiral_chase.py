import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class SpiralChasePolicy:

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.angle_offset = 0

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))
        target_pos = target.static_info.pos

        direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)

        if direction.length() > 0:
            direction = direction.normalize()

            distance = my_position.distance(target_pos)
            spiral_strength = min(1.0, distance / 200.0)
            self.angle_offset += 0.05

            spiral_angle = self.angle_offset * spiral_strength

            cos_a = math.cos(spiral_angle)
            sin_a = math.sin(spiral_angle)
            new_vx = direction.x * cos_a - direction.y * sin_a
            new_vy = direction.x * sin_a + direction.y * cos_a

            return Vector2D(new_vx, new_vy).normalize().times(self.max_velocity)

        return Vector2D(0, 0)
