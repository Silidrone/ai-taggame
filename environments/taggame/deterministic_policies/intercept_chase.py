"""Intercept chase - predicts opponent's future position and moves to intercept."""
import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class InterceptChasePolicy:
    """Predicts where the opponent will be and intercepts."""

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.prediction_time = 1.0  # Predict 1 second ahead

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        # Chase closest opponent
        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))

        # Predict future position
        predicted_x = target.static_info.pos.x + target.velocity.x * self.prediction_time
        predicted_y = target.static_info.pos.y + target.velocity.y * self.prediction_time
        predicted_pos = Point2D(predicted_x, predicted_y)

        direction = Vector2D(predicted_pos.x - my_position.x, predicted_pos.y - my_position.y)
        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
