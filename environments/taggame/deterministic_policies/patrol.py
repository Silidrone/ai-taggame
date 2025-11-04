"""Patrol - patrols in a pattern and chases when close to target."""
import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class PatrolPolicy:
    """Patrols the arena in a pattern, switching to chase when close."""

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.width = width
        self.height = height

        # Patrol waypoints (rectangle pattern)
        self.waypoints = [
            Point2D(width * 0.25, height * 0.25),
            Point2D(width * 0.75, height * 0.25),
            Point2D(width * 0.75, height * 0.75),
            Point2D(width * 0.25, height * 0.75),
        ]
        self.current_waypoint_idx = 0
        self.chase_distance = 200

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))
        target_pos = target.static_info.pos
        distance_to_target = my_position.distance(target_pos)

        # If close to target, chase directly
        if distance_to_target < self.chase_distance:
            direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)
        else:
            # Otherwise patrol waypoints
            current_waypoint = self.waypoints[self.current_waypoint_idx]

            # Move to current waypoint
            direction = Vector2D(current_waypoint.x - my_position.x, current_waypoint.y - my_position.y)

            # If close to waypoint, move to next
            if my_position.distance(current_waypoint) < 50:
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)

        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
