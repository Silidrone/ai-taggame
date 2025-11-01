import math
from typing import Tuple


class Vector2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self):
        length = self.length()
        if length < 0.0001:     
            return Vector2D(0, 0)
        return Vector2D(self.x / length, self.y / length)
    
    def times(self, scalar: float):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def plus(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def minus(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

class Point2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def distance(self, other) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def plus(self, vector):
        if isinstance(vector, Vector2D):
            return Point2D(self.x + vector.x, self.y + vector.y)
        return Point2D(self.x + vector[0], self.y + vector[1])
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

class StaticInfo:
    def __init__(self, pos: Point2D):
        self.pos = pos
        self.orientation = 0.0
    
    def update(self, velocity: Vector2D, time: float):
        self.pos = self.pos.plus(velocity.times(time))
        if velocity.length() > 0.001:
            self.orientation = math.atan2(velocity.y, velocity.x)



