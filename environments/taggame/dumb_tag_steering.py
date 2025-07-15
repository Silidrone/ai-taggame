import math
from typing import List, Tuple
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class DumbTagSteering:
    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.corners = [
            Point2D(0, 0),
            Point2D(0, height),
            Point2D(width, 0),
            Point2D(width, height)
        ]
        
        self.max_distance = math.sqrt(width*width + height*height)
        self.safe_distance_threshold = 0.3 * self.max_distance
        self.chasing_corner_distance_threshold = 0.2 * self.max_distance
        self.corner_weight_multiplier = 0.3 * self.max_distance
        
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
    
    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        desired_velocity = Vector2D(0, 0)
        is_tagged = self.me.is_tagged
        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]
        
        if is_tagged:
            target_opponent = None
            closest_distance = float('inf')
            
            for opponent in other_players:
                opponent_position = opponent.static_info.pos
                distance = my_position.distance(opponent_position)
                
                nearest_corner = self.get_nearest_corner(opponent)
                corner_weight = max(0, 1.0 - (opponent_position.distance(nearest_corner) / 
                                             self.chasing_corner_distance_threshold))
                
                if distance - corner_weight * self.corner_weight_multiplier < closest_distance:
                    closest_distance = distance
                    target_opponent = opponent
            
            if target_opponent:
                target_pos = target_opponent.static_info.pos
                direction = Vector2D(target_pos.x - my_position.x, target_pos.y - my_position.y)
                desired_velocity = direction.normalize().times(self.max_velocity)
        else:
            tagged_opponent = next((p for p in other_players if p.is_tagged), None)
            
            if tagged_opponent:
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
                
                desired_velocity = desired_velocity.normalize().times(self.max_velocity)
        
        return desired_velocity
    
    def get_nearest_corner(self, player: TagPlayer) -> Point2D:
        return self.order_corners_by_distance(player)[0][0]
    
    def order_corners_by_distance(self, player: TagPlayer) -> List[Tuple[Point2D, float]]:
        distance_entries = []
        
        for corner in self.corners:
            distance = player.static_info.pos.distance(corner)
            distance_entries.append((corner, distance))
        
        return sorted(distance_entries, key=lambda x: x[1])
