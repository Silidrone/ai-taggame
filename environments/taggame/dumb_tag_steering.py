import math
import random
from typing import List, Tuple
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer
from environments.taggame.config import TAGGER_NOISE_LEVEL


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
                # Predict where opponent will be (intercept instead of chase)
                opponent_vel = target_opponent.velocity
                # Simple linear prediction: where will they be in 0.5 seconds?
                prediction_time = 0.5
                predicted_x = target_opponent.static_info.pos.x + opponent_vel.x * prediction_time
                predicted_y = target_opponent.static_info.pos.y + opponent_vel.y * prediction_time

                target_pos = Point2D(predicted_x, predicted_y)
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

        # Apply noise based on TAGGER_NOISE_LEVEL
        if TAGGER_NOISE_LEVEL > 0 and desired_velocity.length() > 0:
            # Chance to stop (0-10% based on noise level)
            stop_chance = 0.1 * TAGGER_NOISE_LEVEL
            if random.random() < stop_chance:
                return Vector2D(0, 0)

            # Add angle noise (0 to ±90 degrees based on noise level)
            max_angle_noise = math.pi / 2 * TAGGER_NOISE_LEVEL  # 0 to ±90 degrees
            noise_angle = random.uniform(-max_angle_noise, max_angle_noise)
            cos_noise = math.cos(noise_angle)
            sin_noise = math.sin(noise_angle)

            # Rotate velocity vector by noise angle
            new_vx = desired_velocity.x * cos_noise - desired_velocity.y * sin_noise
            new_vy = desired_velocity.x * sin_noise + desired_velocity.y * cos_noise

            # Add speed variation (100% ± 40% * noise_level)
            speed_range = 0.4 * TAGGER_NOISE_LEVEL
            speed_multiplier = random.uniform(1.0 - speed_range, 1.0 + speed_range)
            desired_velocity = Vector2D(new_vx, new_vy).normalize().times(self.max_velocity * speed_multiplier)

        return desired_velocity
    
    def get_nearest_corner(self, player: TagPlayer) -> Point2D:
        return self.order_corners_by_distance(player)[0][0]
    
    def order_corners_by_distance(self, player: TagPlayer) -> List[Tuple[Point2D, float]]:
        distance_entries = []
        
        for corner in self.corners:
            distance = player.static_info.pos.distance(corner)
            distance_entries.append((corner, distance))
        
        return sorted(distance_entries, key=lambda x: x[1])
