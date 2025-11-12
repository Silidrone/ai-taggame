"""Human-like chase - mimics human player behavior with imperfections and adaptive strategies."""
import math
import random
from environments.taggame.static_info import Point2D, StaticInfo, Vector2D
from environments.taggame.tag_player import TagPlayer


class HumanLikePolicy:
    """Mimics human chasing behavior with realistic imperfections and decision-making."""

    def __init__(self, me: TagPlayer, arena, width: int, height: int, max_velocity: float):
        self.me = me
        self.arena = arena
        self.max_velocity = max_velocity
        self.width = width
        self.height = height
        
        # Human-like state tracking
        self.last_target_pos = None
        self.reaction_delay = 0  # Frames of delay before reacting to target movement
        self.max_reaction_delay = 8  # Human reaction time ~250ms at 30fps
        self.momentum_factor = 0.7  # How much we stick to current direction
        self.prediction_confidence = 0.6  # How confident we are in predictions
        
        # Adaptive behavior
        self.chase_mode = "direct"  # "direct", "cutting", "anticipatory"
        self.mode_switch_cooldown = 0
        self.target_lost_frames = 0
        
        # Human errors and hesitation
        self.aim_error = 0.15  # Random aiming error
        self.hesitation_chance = 0.1  # Chance to hesitate per frame
        self.last_direction = Vector2D(1, 0)
        
        # Memory of target patterns
        self.target_velocity_history = []
        self.max_history = 30  # Remember last 30 positions

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
        
        # Update target tracking
        self._update_target_tracking(target_pos, target_vel)
        
        # Human-like decision making
        intended_direction = self._make_chase_decision(my_position, target_pos, target_vel)
        
        # Apply human imperfections
        final_direction = self._apply_human_imperfections(intended_direction, current_velocity)
        
        # Remember our decision
        if final_direction.length() > 0:
            self.last_direction = final_direction.normalize()
        
        return final_direction

    def _update_target_tracking(self, target_pos, target_vel):
        """Update our memory of target movement patterns."""
        # Track velocity history for pattern recognition
        if len(self.target_velocity_history) >= self.max_history:
            self.target_velocity_history.pop(0)
        self.target_velocity_history.append((target_vel.x, target_vel.y))
        
        # Update reaction delay
        if self.last_target_pos:
            movement = math.sqrt((target_pos.x - self.last_target_pos.x)**2 + 
                               (target_pos.y - self.last_target_pos.y)**2)
            if movement > 5:  # Target moved significantly
                self.reaction_delay = min(self.max_reaction_delay, self.reaction_delay + 2)
            else:
                self.reaction_delay = max(0, self.reaction_delay - 1)
        
        self.last_target_pos = target_pos
        
        # Update mode switching cooldown
        if self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1

    def _make_chase_decision(self, my_pos, target_pos, target_vel):
        """Make chase decision based on current mode and situation."""
        distance = my_pos.distance(target_pos)
        
        # Switch modes occasionally (human adaptability)
        if self.mode_switch_cooldown == 0 and random.random() < 0.05:
            old_mode = self.chase_mode
            if distance > 200:
                self.chase_mode = random.choice(["direct", "anticipatory"])
            elif distance > 100:
                self.chase_mode = random.choice(["cutting", "anticipatory"])
            else:
                self.chase_mode = random.choice(["direct", "cutting"])
            
            if self.chase_mode != old_mode:
                self.mode_switch_cooldown = 60  # Don't switch again for 2 seconds
        
        # Execute chosen strategy
        if self.chase_mode == "direct":
            return self._direct_chase(my_pos, target_pos)
        elif self.chase_mode == "cutting":
            return self._cutting_chase(my_pos, target_pos, target_vel)
        else:  # anticipatory
            return self._anticipatory_chase(my_pos, target_pos, target_vel)

    def _direct_chase(self, my_pos, target_pos):
        """Simple direct chase with slight overshoot tendency."""
        direction = Vector2D(target_pos.x - my_pos.x, target_pos.y - my_pos.y)
        
        # Humans tend to overshoot slightly when excited
        overshoot = 1.1 if random.random() < 0.3 else 1.0
        
        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity * overshoot)
        return Vector2D(0, 0)

    def _cutting_chase(self, my_pos, target_pos, target_vel):
        """Try to cut off the target's path (imperfect cutting)."""
        # Estimate where target is going
        if target_vel.length() > 10:
            # Imperfect prediction - humans aren't perfect at this
            prediction_time = random.uniform(0.5, 1.5)
            predicted_x = target_pos.x + target_vel.x * prediction_time
            predicted_y = target_pos.y + target_vel.y * prediction_time
            
            # Add some error to the prediction
            error_x = random.gauss(0, 30)
            error_y = random.gauss(0, 30)
            cut_point = Point2D(predicted_x + error_x, predicted_y + error_y)
        else:
            # Target not moving much, just go direct
            cut_point = target_pos
        
        direction = Vector2D(cut_point.x - my_pos.x, cut_point.y - my_pos.y)
        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity)
        return Vector2D(0, 0)

    def _anticipatory_chase(self, my_pos, target_pos, target_vel):
        """Advanced chase that tries to predict target patterns."""
        # Analyze target's recent movement pattern
        if len(self.target_velocity_history) < 5:
            # Not enough data, fall back to direct chase
            return self._direct_chase(my_pos, target_pos)
        
        # Look for patterns in velocity changes
        recent_vels = self.target_velocity_history[-10:]
        avg_vel_x = sum(v[0] for v in recent_vels) / len(recent_vels)
        avg_vel_y = sum(v[1] for v in recent_vels) / len(recent_vels)
        
        # Predict based on average recent movement (with confidence factor)
        confidence = self.prediction_confidence * random.uniform(0.7, 1.3)
        prediction_time = confidence * 1.0
        
        predicted_x = target_pos.x + avg_vel_x * prediction_time
        predicted_y = target_pos.y + avg_vel_y * prediction_time
        
        # Humans sometimes second-guess themselves
        if random.random() < 0.2:
            # "Wait, maybe they'll change direction..."
            predicted_x = target_pos.x + avg_vel_x * 0.3  # Shorter prediction
            predicted_y = target_pos.y + avg_vel_y * 0.3
        
        direction = Vector2D(predicted_x - my_pos.x, predicted_y - my_pos.y)
        if direction.length() > 0:
            return direction.normalize().times(self.max_velocity)
        return Vector2D(0, 0)

    def _apply_human_imperfections(self, intended_direction, current_velocity):
        """Apply human-like imperfections to the intended direction."""
        if intended_direction.length() == 0:
            return intended_direction
        
        # Reaction delay - humans don't react instantly
        if self.reaction_delay > 0:
            # Stick more to current direction when delayed
            momentum_weight = 0.8
            new_weight = 0.2
        else:
            momentum_weight = self.momentum_factor
            new_weight = 1.0 - self.momentum_factor
        
        # Blend with momentum
        if current_velocity.length() > 0:
            momentum_dir = current_velocity.normalize()
            blended_x = momentum_dir.x * momentum_weight + intended_direction.x * new_weight
            blended_y = momentum_dir.y * momentum_weight + intended_direction.y * new_weight
            intended_direction = Vector2D(blended_x, blended_y)
        
        # Add aiming error
        error_angle = random.gauss(0, self.aim_error)
        cos_err = math.cos(error_angle)
        sin_err = math.sin(error_angle)
        
        rotated_x = intended_direction.x * cos_err - intended_direction.y * sin_err
        rotated_y = intended_direction.x * sin_err + intended_direction.y * cos_err
        
        direction_with_error = Vector2D(rotated_x, rotated_y)
        
        # Occasional hesitation
        if random.random() < self.hesitation_chance:
            # Reduce speed when hesitating
            speed_factor = random.uniform(0.3, 0.7)
        else:
            speed_factor = 1.0
        
        # Normalize and apply velocity
        if direction_with_error.length() > 0:
            return direction_with_error.normalize().times(self.max_velocity * speed_factor)
        
        return Vector2D(0, 0)