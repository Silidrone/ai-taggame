"""
Deterministic tag game simulation using steering behaviors for both agents.
Demonstrates near-optimal evasion and pursuit strategies.
"""

import pygame
import time
from environments.taggame.taggame import TagGame
from environments.taggame.dumb_tag_steering import DumbTagSteering
from environments.taggame.static_info import Vector2D
from environments.taggame.config import (
    WIDTH, HEIGHT, FRAME_RATE_CAP, TIME_COEFFICIENT, TAG_COOLDOWN_MS
)


def run_deterministic_simulation(render=True, max_steps=10000, fps_limit=None):
    """
    Run tag game with steering behaviors for both players.

    Args:
        render: Whether to display the game visually
        max_steps: Maximum simulation steps before stopping
        fps_limit: Optional FPS limit for rendering

    Returns:
        total_steps: Number of steps the evader survived
    """
    env = TagGame(render=render)
    env.initialize()

    # Get players
    evader = env._get_rl_player()  # The agent we normally train
    tagger = env.tag_player  # The tagger

    # Steering controllers for both
    evader_steering = DumbTagSteering(evader, env, WIDTH, HEIGHT, env.max_velocity)
    tagger_steering = DumbTagSteering(tagger, env, WIDTH, HEIGHT, env.max_velocity)

    steps = 0
    start_time = time.time()

    print("Running deterministic simulation...")
    print("Evader (blue): Using evasion steering")
    print("Tagger (red): Using pursuit steering")
    print(f"Max steps: {max_steps}")
    if fps_limit:
        print(f"FPS limit: {fps_limit}")
    print("-" * 50)

    while steps < max_steps:
        if fps_limit:
            step_start = time.time()

        # Handle pygame events
        if render and pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print(f"\nSimulation ended by user at step {steps}")
                    env.close()
                    return steps

        # Get current time for cooldown check
        current_time = pygame.time.get_ticks() if pygame.get_init() else int(time.time() * 1000)
        tagger_sleeping = (current_time - env.tag_changed_time < TAG_COOLDOWN_MS)

        # Evader uses steering behavior (flee)
        evade_action = evader_steering(evader.static_info, evader.velocity)
        evader.set_velocity(evade_action)

        # Tagger uses steering behavior (seek) - unless on cooldown
        if not tagger_sleeping:
            pursue_action = tagger_steering(tagger.static_info, tagger.velocity)
            tagger.set_velocity(pursue_action)

        # Update physics for all players
        for player in env.players:
            player.update(1.0 * TIME_COEFFICIENT)

        # Check for tag
        distance = evader.static_info.pos.distance(tagger.static_info.pos)
        if distance < (evader.radius + tagger.radius) and not tagger_sleeping:
            print(f"\nEvader was caught at step {steps}")
            print(f"Survival time: {steps * TIME_COEFFICIENT:.2f}s")
            env.close()
            return steps

        # Render
        if render:
            env._render()

        steps += 1

        if fps_limit:
            elapsed = time.time() - step_start
            target_time = 1.0 / fps_limit
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

        # Print progress every 1000 steps
        if steps % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Step {steps} | Distance: {distance:.1f} | Elapsed: {elapsed:.1f}s")

    print(f"\nSimulation completed: {max_steps} steps without being caught!")
    print(f"Survival time: {max_steps * TIME_COEFFICIENT:.2f}s")
    env.close()
    return steps


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run deterministic tag game simulation')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (faster)')
    parser.add_argument('--max-steps', type=int, default=10000,
                        help='Maximum simulation steps')
    parser.add_argument('--fps', type=int, default=None,
                        help='FPS limit for rendering')

    args = parser.parse_args()

    steps = run_deterministic_simulation(
        render=not args.no_render,
        max_steps=args.max_steps,
        fps_limit=args.fps
    )

    print(f"\nFinal result: {steps} steps")
