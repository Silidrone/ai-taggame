from environments.taggame.static_info import StaticInfo, Vector2D


class TagPlayer:
    def __init__(self, name: str, static_info: StaticInfo, max_velocity: float, 
                 radius: float, game_width: int, game_height: int):
        self.name = name
        self.static_info = static_info
        self.velocity = Vector2D(0, 0)
        self.steering_behavior = None
        self.is_tagged = False
        self.max_velocity = max_velocity
        self.radius = radius
        self.game_width = game_width
        self.game_height = game_height
    
    def set_velocity(self, velocity: Vector2D):
        if velocity.length() > self.max_velocity:
            velocity = velocity.normalize().times(self.max_velocity)
        self.velocity = velocity
    
    def set_steering_behavior(self, behavior):
        self.steering_behavior = behavior
    
    def update(self, time: float):
        if self.steering_behavior:
            self.set_velocity(self.steering_behavior(self.static_info, self.velocity))
        
        velocity = Vector2D(self.velocity.x, self.velocity.y)
        
        if ((self.static_info.pos.x + self.radius >= self.game_width and velocity.x > 0) or
            (self.static_info.pos.x - self.radius <= 0 and velocity.x < 0)):
            velocity = Vector2D(0, velocity.y)
        
        if ((self.static_info.pos.y + self.radius >= self.game_height and velocity.y > 0) or
            (self.static_info.pos.y - self.radius <= 0 and velocity.y < 0)):
            velocity = Vector2D(velocity.x, 0)
        
        self.static_info.update(velocity, time)
    
    def is_tagging(self, other_player) -> bool:
        return (self.radius + other_player.radius >= 
                self.static_info.pos.distance(other_player.static_info.pos))
    
    def set_is_tagged(self, value: bool):
        self.is_tagged = value
