import os
import numpy as np
from stable_baselines3 import PPO
from environments.taggame.gym_wrapper import TagGameGymEnv
from environments.taggame.deterministic_policies import ALL_POLICIES
from environments.taggame import config


def evaluate_against_policy(model, policy_idx, n_episodes=10, max_steps=None, render=False, fps_limit=60):
    """
    Evaluate model against a specific chaser policy.
    Returns success rate (episodes that reach max_steps).
    """
    if max_steps is None:
        max_steps = config.MAX_EPISODE_STEPS
    
    successes = 0
    episode_lengths = []
    
    for episode in range(n_episodes):
        env = TagGameGymEnv(render=render, chaser_policy_idx=policy_idx)
        obs, _ = env.reset()
        done = False
        episode_length = 0
        
        while not done and episode_length < max_steps:
            if fps_limit:
                import time
                step_start = time.time()
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_length += 1
            
            if fps_limit:
                elapsed = time.time() - step_start
                target_time = 1.0 / fps_limit
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
        
        episode_lengths.append(episode_length)
        
        if episode_length >= max_steps:
            successes += 1
        
        env.close()
    
    return successes, episode_lengths


def rigorous_evaluate(model_path, n_episodes_per_policy=10, max_steps=None, render=False, fps_limit=60):
    """
    Evaluate model against each chaser policy and report success rates.
    """
    if max_steps is None:
        max_steps = config.MAX_EPISODE_STEPS
        
    model = PPO.load(model_path)
    
    print(f"Rigorous Evaluation")
    print(f"Model: {model_path}")
    print(f"Episodes per policy: {n_episodes_per_policy}")
    print(f"Success threshold: {max_steps} steps")
    print("=" * 60)
    
    results = {}
    total_successes = 0
    total_episodes = 0
    
    for policy_idx, policy_class in enumerate(ALL_POLICIES):
        policy_name = policy_class.__name__
        print(f"\nTesting against {policy_name} (Policy {policy_idx})...")
        
        successes, episode_lengths = evaluate_against_policy(
            model, policy_idx, n_episodes_per_policy, max_steps, render, fps_limit
        )
        
        success_rate = successes / n_episodes_per_policy
        avg_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        results[policy_name] = {
            'success_rate': success_rate,
            'successes': successes,
            'avg_length': avg_length,
            'std_length': std_length,
            'episode_lengths': episode_lengths
        }
        
        total_successes += successes
        total_episodes += n_episodes_per_policy
        
        total_steps_policy = sum(episode_lengths)
        print(f"  Episode lengths: {episode_lengths}")
        print(f"  Average: {avg_length:.1f} - Total: {total_steps_policy}/{n_episodes_per_policy * max_steps}")
    
    total_steps = sum(data['avg_length'] * n_episodes_per_policy for data in results.values())
    max_possible_steps = len(ALL_POLICIES) * n_episodes_per_policy * max_steps
    
    print(f"\nTotal: {total_steps:.0f}/{max_possible_steps}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Rigorous evaluation against all chaser policies')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: tries to find best_model.zip or ppo_taggame_final.zip)')
    parser.add_argument('--episodes-per-policy', type=int, default=10,
                        help='Episodes to test per policy (default: 10)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Success threshold in steps (default: from config)')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit for evaluation (default: 60)')
    
    args = parser.parse_args()
    
    if not args.model:
        model_path = 'data/ppo_taggame/best_model.zip'
        if not os.path.exists(model_path):
            model_path = 'data/ppo_taggame/ppo_taggame_final.zip'
        if not os.path.exists(model_path):
            raise ValueError(f"No model found. Specify --model path")
    else:
        model_path = args.model
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
    
    rigorous_evaluate(model_path, args.episodes_per_policy, args.max_steps, args.render, args.fps)