"""Evader policy - flees from tagger using corner-based evasion strategy."""
import math
from typing import List, Tuple
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class EvaderPolicy:
    """
    Evasion policy that flees from the tagger.
    Uses corner awareness and weighted flee behavior.
    """

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.corners = [
            Point2D(0, 0),
            Point2D(0, height),
            Point2D(width, 0),
            Point2D(width, height)
        ]

        self.max_distance = math.sqrt(width*width + height*height)
        self.safe_distance_threshold = 0.3 * self.max_distance

        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        tagged_opponent = next((p for p in other_players if p.is_tagged), None)

        if not tagged_opponent:
            return Vector2D(0, 0)

        opponent_position = tagged_opponent.static_info.pos
        ordered_corners = self.order_corners_by_distance(tagged_opponent)
        distance_to_opponent = my_position.distance(opponent_position)

        flee_weight = max(0, (self.safe_distance_threshold - distance_to_opponent) /
                         self.safe_distance_threshold)
        seek_weight = 1.0 - flee_weight

        furthest_corner = ordered_corners[-1][0]

        desired_direction = Vector2D(
            furthest_corner.x - my_position.x,
            furthest_corner.y - my_position.y
        ).normalize()

        from_me_to_opponent = Vector2D(
            opponent_position.x - my_position.x,
            opponent_position.y - my_position.y
        ).normalize()

        desired_velocity = desired_direction.times(seek_weight).plus(
            from_me_to_opponent.times(-flee_weight)
        )

        return desired_velocity.normalize().times(self.max_velocity)

    def order_corners_by_distance(self, player: TagPlayer) -> List[Tuple[Point2D, float]]:
        distance_entries = []

        for corner in self.corners:
            distance = player.static_info.pos.distance(corner)
            distance_entries.append((corner, distance))

        return sorted(distance_entries, key=lambda x: x[1])
