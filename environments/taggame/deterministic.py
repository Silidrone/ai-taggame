
import pygame
import time
import logging
from environments.taggame.taggame import TagGame
from environments.taggame.deterministic_policies import ALL_POLICIES
from environments.taggame.deterministic_policies.evader_policy import EvaderPolicy
from environments.taggame.static_info import Vector2D
from environments.taggame.config import (
    WIDTH, HEIGHT, FRAME_RATE_CAP, TIME_COEFFICIENT, TAG_COOLDOWN_MS, CURRENT_CHASER_POLICY_IDX
)
from environments.taggame import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_deterministic_simulation(render=True, max_steps=10000, fps_limit=None):
    # Setup simulation logging
    sim_dir = 'data/taggame/simulation'
    os.makedirs(sim_dir, exist_ok=True)
    log_file = os.path.join(sim_dir, 'simulation.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    env = TagGame(render=render)
    env.initialize()

    evader = env._get_rl_player()
    tagger = env.tag_player

    evader_steering = EvaderPolicy(evader, env, WIDTH, HEIGHT, env.max_velocity)

    policy_class = ALL_POLICIES[config.CURRENT_CHASER_POLICY_IDX]
    tagger_steering = policy_class(tagger, env, WIDTH, HEIGHT, env.max_velocity)

    steps = 0
    start_time = time.time()

    policy_name = policy_class.__name__
    logger.info("Running deterministic simulation...")
    logger.info("Evader (blue): Using EvaderPolicy")
    logger.info(f"Tagger (red): Using {policy_name}")
    logger.info(f"Max steps: {max_steps}")
    if fps_limit:
        logger.info(f"FPS limit: {fps_limit}")
    logger.info("-" * 50)

    while steps < max_steps:
        if fps_limit:
            step_start = time.time()

        if render and pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info(f"\nSimulation ended by user at step {steps}")
                    env.close()
                    return steps

        current_time = pygame.time.get_ticks() if pygame.get_init() else int(time.time() * 1000)
        tagger_sleeping = (current_time - env.tag_changed_time < TAG_COOLDOWN_MS)

        evade_action = evader_steering(evader.static_info, evader.velocity)
        evader.set_velocity(evade_action)

        if not tagger_sleeping:
            pursue_action = tagger_steering(tagger.static_info, tagger.velocity)
            tagger.set_velocity(pursue_action)

        for player in env.players:
            player.update(1.0 * TIME_COEFFICIENT)

        distance = evader.static_info.pos.distance(tagger.static_info.pos)
        if distance < (evader.radius + tagger.radius) and not tagger_sleeping:
            logger.info(f"\nEvader was caught at step {steps}")
            logger.info(f"Survival time: {steps * TIME_COEFFICIENT:.2f}s")
            env.close()
            return steps

        if render:
            env._render()

        steps += 1

        if fps_limit:
            elapsed = time.time() - step_start
            target_time = 1.0 / fps_limit
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

        if steps % 1000 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Step {steps} | Distance: {distance:.1f} | Elapsed: {elapsed:.1f}s")

    logger.info(f"\nSimulation completed: {max_steps} steps without being caught!")
    logger.info(f"Survival time: {max_steps * TIME_COEFFICIENT:.2f}s")
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

    logger.info(f"\nFinal result: {steps} steps")
