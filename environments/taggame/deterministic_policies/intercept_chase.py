import math
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class InterceptChasePolicy:

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.prediction_time = 1.0

    def __call__(self, static_info: StaticInfo, current_velocity: Vector2D) -> Vector2D:
        if not self.me.is_tagged:
            return Vector2D(0, 0)

        my_position = self.me.static_info.pos
        other_players = [p for p in self.arena.players if p != self.me]

        if not other_players:
            return Vector2D(0, 0)

        target = min(other_players, key=lambda p: my_position.distance(p.static_info.pos))

        intercept_point = self._calculate_intercept(my_position, target.static_info.pos, target.velocity)
        
        if intercept_point is None:
            direction = Vector2D(target.static_info.pos.x - my_position.x, target.static_info.pos.y - my_position.y)
        else:
            direction = Vector2D(intercept_point.x - my_position.x, intercept_point.y - my_position.y)
        
        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity)

        return Vector2D(0, 0)
    
    def _calculate_intercept(self, chaser_pos: Point2D, target_pos: Point2D, target_vel: Vector2D) -> Point2D:
        rel_x = target_pos.x - chaser_pos.x
        rel_y = target_pos.y - chaser_pos.y
        rel_vx = target_vel.x
        rel_vy = target_vel.y
        
        a = rel_vx * rel_vx + rel_vy * rel_vy - self.max_velocity * self.max_velocity
        b = 2 * (rel_x * rel_vx + rel_y * rel_vy)
        c = rel_x * rel_x + rel_y * rel_y
        
        if abs(a) < 1e-6:
            if abs(b) < 1e-6:
                return None
            t = -c / b
        else:
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return None
            
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)
            
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                return None
        
        if t <= 0:
            return None
            
        intercept_x = target_pos.x + target_vel.x * t
        intercept_y = target_pos.y + target_vel.y * t
        
        return Point2D(intercept_x, intercept_y)
