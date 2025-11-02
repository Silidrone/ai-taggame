import argparse
import logging
import os
import sys
import random
import string
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def setup_logger(run_id, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger()
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Log directory: {log_dir}")

    return logger

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate DQN agent')
    parser.add_argument('--env', type=str, required=True, choices=['taggame', 'gridworld'],
                        help='Environment to use')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID (to continue training or evaluate from checkpoint, auto-generated for new training)')
    parser.add_argument('--n-episodes', type=int, required=True,
                        help='Number of episodes')
    parser.add_argument('--save-freq', type=int, default=None,
                        help='Save model every N episodes (train mode only, defaults to 5%% of n-episodes)')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')

    args = parser.parse_args()

    if args.save_freq is None:
        args.save_freq = max(1, int(args.n_episodes * 0.05))

    if args.mode == 'evaluate' and args.run_id is None:
        parser.error("--run-id required for evaluate mode")

    if args.mode == 'train' and args.run_id is None:
        args.run_id = generate_run_id()

    log_dir = os.path.join('./runs', args.run_id)
    logger = setup_logger(args.run_id, log_dir)

    try:
        if args.env == 'taggame':
            from environments.taggame.run import run
            run(args.mode, args.n_episodes, args.save_freq, args.render, logger, log_dir)
        elif args.env == 'gridworld':
            from environments.windy_grid_world.run import run
            run(args.mode, args.n_episodes, args.save_freq, args.render, logger, log_dir)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        import pygame
        if isinstance(e, pygame.error) and "display Surface quit" in str(e):
            logger.info("Training interrupted (display closed)")
        else:
            logger.error(f"Error during {args.mode}: {e}")
            raise
    finally:
        logger.info(f"Run ID: {args.run_id}")
        logger.info(f"Log directory: {log_dir}")

if __name__ == '__main__':
    main()
