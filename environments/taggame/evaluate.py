import os
import glob
import numpy as np
import pygame
import time
import logging
from stable_baselines3 import PPO
from environments.taggame.gym_wrapper import TagGameGymEnv
from environments.taggame.deterministic_policies import ALL_POLICIES
from environments.taggame import config
from environments.taggame.taggame import TagGame
from environments.taggame.deterministic_policies.evader_policy import EvaderPolicy
from environments.taggame.config import WIDTH, HEIGHT, TIME_COEFFICIENT, TAG_COOLDOWN_MS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def evaluate_against_policy(model, policy_idx, n_episodes=10, max_steps=None, render=False, fps_limit=60, deterministic_evader=False):
    """
    Evaluate model or deterministic evader against a specific chaser policy.
    """
    if max_steps is None:
        max_steps = config.MAX_EPISODE_STEPS
    
    episode_lengths = []
    
    for episode in range(n_episodes):
        episode_length = evaluate_single_episode(model, policy_idx, max_steps, render, fps_limit, deterministic_evader)
        episode_lengths.append(episode_length)
    
    return episode_lengths


def evaluate_single_episode(model, policy_idx, max_steps, render, fps_limit, deterministic_evader):
    """
    Evaluate a single episode using either RL model or deterministic evader.
    """
    if deterministic_evader:
        env = TagGame(render=render)
        env.initialize()
        
        evader = env._get_rl_player()
        tagger = env.tag_player
        
        policy_class = ALL_POLICIES[policy_idx]
        evader_steering = EvaderPolicy(evader, env, WIDTH, HEIGHT, env.max_velocity)
        tagger_steering = policy_class(tagger, env, WIDTH, HEIGHT, env.max_velocity)
        
        episode_length = 0
        caught = False
        
        while episode_length < max_steps and not caught:
            if fps_limit and render:
                step_start = time.time()
            
            if render and pygame.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return episode_length
            
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
                caught = True
                break
            
            if render:
                env._render()
            
            episode_length += 1
            
            if fps_limit and render:
                elapsed = time.time() - step_start
                target_time = 1.0 / fps_limit
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
    else:
        env = TagGameGymEnv(render=render, chaser_policy_idx=policy_idx)
        obs, _ = env.reset()
        done = False
        episode_length = 0
        
        while not done and episode_length < max_steps:
            if fps_limit and render:
                step_start = time.time()
            
            if render and pygame.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return episode_length
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_length += 1
            
            if fps_limit and render:
                elapsed = time.time() - step_start
                target_time = 1.0 / fps_limit
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
    
    env.close()
    return episode_length


def evaluate_comprehensive(model_path=None, policy_indices=None, n_episodes=10, max_steps=None, render=False, fps_limit=60, deterministic_evader=False):
    """
    Comprehensive evaluation against specified chaser policies.
    """
    if max_steps is None:
        max_steps = config.MAX_EPISODE_STEPS
    
    # Default to all policies if none specified
    if policy_indices is None:
        policy_indices = list(range(len(ALL_POLICIES)))
    
    # Setup evaluation logging with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    eval_dir = f'data/taggame/eval_{timestamp}'
    os.makedirs(eval_dir, exist_ok=True)
    log_file = os.path.join(eval_dir, 'evaluation.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Load model if using RL evader
    model = None
    if not deterministic_evader:
        if model_path is None:
            # Find most recent training directory
            train_dirs = glob.glob('data/taggame/train_*')
            if train_dirs:
                latest_dir = max(train_dirs, key=os.path.getctime)
                model_path = os.path.join(latest_dir, 'best_model.zip')
                if not os.path.exists(model_path):
                    model_path = os.path.join(latest_dir, 'final.zip')
            else:
                # Fallback to old structure
                model_path = 'data/taggame/best_model.zip'
                if not os.path.exists(model_path):
                    model_path = 'data/taggame/final.zip'
            
            if not os.path.exists(model_path):
                raise ValueError(f"No model found. Train a model first or specify --model path")
        model = PPO.load(model_path)
    
    # Log header
    evader_type = "EvaderPolicy (deterministic)" if deterministic_evader else f"PPO Model ({model_path})"
    logger.info(f"Comprehensive Evaluation")
    logger.info(f"Evader: {evader_type}")
    logger.info(f"Episodes per policy: {n_episodes}")
    logger.info(f"Max steps: {max_steps}")
    if render and fps_limit:
        logger.info(f"FPS limit: {fps_limit}")
    logger.info("=" * 60)
    
    results = {}
    all_episode_lengths = []
    
    for policy_idx in policy_indices:
        policy_class = ALL_POLICIES[policy_idx]
        policy_name = policy_class.__name__
        logger.info(f"\nTesting against {policy_name} (Policy {policy_idx})...")
        
        episode_lengths = evaluate_against_policy(
            model, policy_idx, n_episodes, max_steps, render, fps_limit, deterministic_evader
        )
        
        avg_length = np.mean(episode_lengths)
        total_steps_policy = sum(episode_lengths)
        
        results[policy_name] = {
            'avg_length': avg_length,
            'episode_lengths': episode_lengths,
            'policy_idx': policy_idx
        }
        
        all_episode_lengths.extend(episode_lengths)
        
        logger.info(f"  Episode lengths: {episode_lengths}")
        logger.info(f"  Average: {avg_length:.1f} - Total: {total_steps_policy}/{n_episodes * max_steps}")
    
    if len(policy_indices) > 1:
        total_steps = sum(episode_lengths for episode_lengths in all_episode_lengths)
        max_possible_steps = len(policy_indices) * n_episodes * max_steps
        logger.info(f"\nTotal: {total_steps}/{max_possible_steps}")
        
        policy_averages = [results[ALL_POLICIES[idx].__name__]['avg_length'] for idx in policy_indices]
        logger.info(f"Policy averages: {policy_averages}")
    else:
        policy_name = ALL_POLICIES[policy_indices[0]].__name__
        avg = results[policy_name]['avg_length']
        logger.info(f"Policy average: [{avg:.1f}]")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive evaluation against chaser policies')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: auto-find best model)')
    parser.add_argument('--policies', type=str, default=None,
                        help='Comma-separated policy indices (e.g., "0,1,7" or "all" for all policies)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per policy (default: 10)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Max steps per episode (default: from config)')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit for evaluation (default: 60)')
    parser.add_argument('--deterministic-evader', action='store_true',
                        help='Use deterministic evader instead of RL model')
    
    args = parser.parse_args()
    
    # Parse policy indices
    if args.policies is None:
        # Default to current policy for single evaluation
        policy_indices = [config.CURRENT_CHASER_POLICY_IDX]
    elif args.policies.lower() == 'all':
        policy_indices = list(range(len(ALL_POLICIES)))
    else:
        try:
            policy_indices = [int(idx.strip()) for idx in args.policies.split(',')]
        except ValueError:
            raise ValueError(f"Invalid policy indices: {args.policies}. Use comma-separated integers or 'all'")
    
    # Validate policy indices
    for idx in policy_indices:
        if idx < 0 or idx >= len(ALL_POLICIES):
            raise ValueError(f"Policy index {idx} out of range. Available: 0-{len(ALL_POLICIES)-1}")
    
    evaluate_comprehensive(
        model_path=args.model,
        policy_indices=policy_indices,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        fps_limit=args.fps,
        deterministic_evader=args.deterministic_evader
    )