import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environments.taggame.gym_wrapper import TagGameGymEnv
from environments.taggame import config


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.last_episode_count = 0
        self.current_policy_idx = 0

    def _on_step(self) -> bool:
        if 'dones' in self.locals and any(self.locals['dones']):
            self.episode_count += sum(self.locals['dones'])

        if config.ENABLE_POLICY_ROTATION and self.episode_count // config.ROTATION_FREQUENCY > self.last_episode_count // config.ROTATION_FREQUENCY:
            from environments.taggame.deterministic_policies import ALL_POLICIES
            import random
            
            self.current_policy_idx = random.choices(
                range(len(ALL_POLICIES)), 
                weights=config.POLICY_WEIGHTS, 
                k=1
            )[0]
            config.CURRENT_CHASER_POLICY_IDX = self.current_policy_idx
            policy_name = ALL_POLICIES[self.current_policy_idx].__name__
            self.last_episode_count = self.episode_count

        return True


def distribute_policies_to_envs(n_envs, policy_weights):
    import numpy as np
    
    weights = np.array(policy_weights)
    probabilities = weights / weights.sum()
    
    target_counts = probabilities * n_envs
    env_counts = np.round(target_counts).astype(int)
    
    diff = n_envs - env_counts.sum()
    if diff > 0:
        fractional_parts = target_counts - np.floor(target_counts)
        indices_to_increment = np.argsort(fractional_parts)[-diff:]
        env_counts[indices_to_increment] += 1
    elif diff < 0:
        fractional_parts = target_counts - np.floor(target_counts)
        indices_to_decrement = np.argsort(fractional_parts)[:abs(diff)]
        env_counts[indices_to_decrement] -= 1
    
    policy_assignments = []
    for policy_idx, count in enumerate(env_counts):
        policy_assignments.extend([policy_idx] * count)
    
    np.random.shuffle(policy_assignments)
    
    return policy_assignments


def make_env(chaser_policy_idx):
    def _init():
        env = TagGameGymEnv(render=False, chaser_policy_idx=chaser_policy_idx)
        env = Monitor(env)
        return env
    return _init


def train_ppo(log_dir, n_timesteps=1000000, render=False, n_envs=8):

    os.makedirs(log_dir, exist_ok=True)

    policy_assignments = distribute_policies_to_envs(n_envs, config.POLICY_WEIGHTS)
    print(f"Policy distribution across {n_envs} environments:")
    from collections import Counter
    from environments.taggame.deterministic_policies import ALL_POLICIES
    counts = Counter(policy_assignments)
    for policy_idx, count in sorted(counts.items()):
        policy_name = ALL_POLICIES[policy_idx].__name__
        print(f"  Policy {policy_idx} ({policy_name}): {count} envs")

    env = SubprocVecEnv([make_env(policy_idx) for policy_idx in policy_assignments])

    eval_env = TagGameGymEnv(render=False)
    eval_env = Monitor(eval_env, os.path.join(log_dir, 'eval'))

    curriculum_callback = CurriculumCallback(verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix='ppo_taggame'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    policy_kwargs = dict(
        net_arch=[256, 256]
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=16384,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cuda",
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    print(f"Training PPO for {n_timesteps} timesteps...")
    print(f"Parallel environments: {n_envs}")
    print(f"Log directory: {log_dir}")
    print("-" * 60)

    model.learn(
        total_timesteps=n_timesteps,
        callback=[curriculum_callback, checkpoint_callback, eval_callback],
        progress_bar=False
    )
    final_path = os.path.join(log_dir, 'ppo_taggame_final')
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    env.close()
    eval_env.close()

    return model


def setup_eval_logger(log_file="evaluation.txt"):
    logger = logging.getLogger('eval_logger')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def evaluate_ppo(model_path, n_episodes=100, render=True, fps_limit=60):
    logger = setup_eval_logger()
    model = PPO.load(model_path)

    env = TagGameGymEnv(render=render)
    episode_rewards = []
    episode_lengths = []

    logger.info(f"Evaluating for {n_episodes} episodes...")
    if fps_limit:
        logger.info(f"FPS limit: {fps_limit}")
    logger.info("-" * 60)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        if config.ENABLE_DEBUG_LOGS:
            logger.info(f"\nDEBUG: Episode {episode}")
            logger.info(f"Chaser policy index: {env.chaser_policy_idx}")
            from environments.taggame.deterministic_policies import ALL_POLICIES
            logger.info(f"Chaser policy name: {ALL_POLICIES[env.chaser_policy_idx].__name__}")
            
            prev_rl_vel = None
            velocity_reversals = []
            
            logger.info("\nTracking velocity reversals and predictions...")
            logger.info("-" * 100)

        while not done:
            if fps_limit:
                import time
                step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            
            if config.ENABLE_DEBUG_LOGS:
                state = env.current_state
                rl_pos = state[0]
                chaser_pos = state[2]
                rl_vel = state[1]
                chaser_vel = state[3]
                
                import math
                
                if prev_rl_vel is not None:
                    vel_dot = rl_vel[0] * prev_rl_vel[0] + rl_vel[1] * prev_rl_vel[1]
                    
                    if vel_dot < -5000:
                        velocity_reversals.append(episode_length)
                        
                        predicted_rl_x = rl_pos[0] + prev_rl_vel[0] * 1.0
                        predicted_rl_y = rl_pos[1] + prev_rl_vel[1] * 1.0
                        
                        actual_future_x = rl_pos[0] + rl_vel[0] * 1.0
                        actual_future_y = rl_pos[1] + rl_vel[1] * 1.0
                        
                        prediction_error = math.sqrt((predicted_rl_x - actual_future_x)**2 + 
                                                   (predicted_rl_y - actual_future_y)**2)
                        
                        logger.info(f"\nREVERSAL at step {episode_length}:")
                        logger.info(f"   RL velocity: ({prev_rl_vel[0]:5.0f},{prev_rl_vel[1]:5.0f}) → ({rl_vel[0]:5.0f},{rl_vel[1]:5.0f})")
                        logger.info(f"   Chaser predicted RL at: ({predicted_rl_x:5.0f},{predicted_rl_y:5.0f})")
                        logger.info(f"   RL actually going to: ({actual_future_x:5.0f},{actual_future_y:5.0f})")
                        logger.info(f"   Prediction error: {prediction_error:5.0f} units")
                        logger.info(f"   Chaser velocity: ({chaser_vel[0]:5.0f},{chaser_vel[1]:5.0f})")
                
                prev_rl_vel = rl_vel
                
                if episode_length % 50 == 0:
                    distance = math.sqrt((rl_pos[0] - chaser_pos[0])**2 + (rl_pos[1] - chaser_pos[1])**2)
                    
                    to_rl_x = rl_pos[0] - chaser_pos[0]
                    to_rl_y = rl_pos[1] - chaser_pos[1]
                    
                    dot_product = chaser_vel[0] * to_rl_x + chaser_vel[1] * to_rl_y
                    
                    if dot_product < 0:
                        direction_status = "AWAY"
                    elif dot_product > 0:
                        direction_status = "TOWARD"
                    else:
                        direction_status = "PERPENDICULAR"
                    
                    logger.info(f"\nStep {episode_length:4d}: Distance={distance:5.0f} | Chaser moving {direction_status}")
                    
                    recent_reversals = len([r for r in velocity_reversals if r > episode_length - 50])
                    if recent_reversals > 0:
                        logger.info(f"         {recent_reversals} reversals in last 50 steps!")

            if fps_limit:
                elapsed = time.time() - step_start
                target_time = 1.0 / fps_limit
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if config.ENABLE_DEBUG_LOGS:
            logger.info(f"\nEPISODE SUMMARY:")
            logger.info(f"   Total steps: {episode_length}")
            logger.info(f"   Total velocity reversals: {len(velocity_reversals)}")
            logger.info(f"   Reversal rate: {len(velocity_reversals)/episode_length*100:.1f}% of steps")
            logger.info(f"   Average steps between reversals: {episode_length/max(1,len(velocity_reversals)):.1f}")

        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{n_episodes} | "
                       f"Reward: {episode_reward:.2f} | "
                       f"Length: {episode_length}")

    import numpy as np
    logger.info("\nEvaluation Results:")
    logger.info(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    logger.info(f"Max Length: {np.max(episode_lengths)}")
    
    logger.info("\nLogs saved to evaluation.txt")

    env.close()

    return episode_rewards, episode_lengths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate PPO on TagGame')
    parser.add_argument('mode', choices=['train', 'evaluate'], help='Mode: train or evaluate')
    parser.add_argument('--log-dir', type=str, default='data/ppo_taggame',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Number of timesteps to train')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation')
    parser.add_argument('--fps', type=int, default=60,
                        help='FPS limit for evaluation')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='Number of parallel environments for training')

    args = parser.parse_args()

    if args.mode == 'train':
        train_ppo(args.log_dir, args.timesteps, n_envs=args.n_envs)
    elif args.mode == 'evaluate':
        if not args.model:
            model_path = os.path.join(args.log_dir, 'best_model.zip')
            if not os.path.exists(model_path):
                model_path = os.path.join(args.log_dir, 'ppo_taggame_final.zip')
            if not os.path.exists(model_path):
                raise ValueError(f"No model found. Specify --model path")
        else:
            model_path = args.model

        evaluate_ppo(model_path, args.episodes, args.render, args.fps)
