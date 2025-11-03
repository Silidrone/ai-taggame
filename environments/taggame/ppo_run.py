import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environments.taggame.gym_wrapper import TagGameGymEnv
from environments.taggame import config


class CurriculumCallback(BaseCallback):
    """
    Callback to gradually increase tagger noise level during training.
    Goes from 0.0 to 0.7 over 500k steps.
    """
    def __init__(self, max_noise=0.7, max_steps=500000, verbose=0):
        super().__init__(verbose)
        self.max_noise = max_noise
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        # Calculate current noise level based on num_timesteps
        progress = min(1.0, self.num_timesteps / self.max_steps)
        current_noise = progress * self.max_noise

        # Update global config
        config.TAGGER_NOISE_LEVEL = current_noise

        # Log every 10k steps
        if self.num_timesteps % 10000 == 0:
            print(f"Timesteps: {self.num_timesteps} | Tagger Noise: {current_noise:.3f}")

        return True


def make_env():
    """Create a single environment instance"""
    def _init():
        env = TagGameGymEnv(render=False)
        env = Monitor(env)
        return env
    return _init


def train_ppo(log_dir, n_timesteps=1000000, render=False, n_envs=8):
    """Train PPO agent on TagGame with parallel environments."""

    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environment (parallel)
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # Create evaluation environment (single)
    eval_env = TagGameGymEnv(render=False)
    eval_env = Monitor(eval_env, os.path.join(log_dir, 'eval'))

    # Curriculum callback - gradually increase noise
    curriculum_callback = CurriculumCallback(max_noise=0.7, max_steps=500000)

    # Checkpoint callback - save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix='ppo_taggame'
    )

    # Eval callback - evaluate every 10k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )

    print(f"Training PPO for {n_timesteps} timesteps...")
    print(f"Parallel environments: {n_envs}")
    print(f"Log directory: {log_dir}")
    print("-" * 60)

    # Train the model
    model.learn(
        total_timesteps=n_timesteps,
        callback=[curriculum_callback, checkpoint_callback, eval_callback],
        progress_bar=False
    )

    # Save final model
    final_path = os.path.join(log_dir, 'ppo_taggame_final')
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    env.close()
    eval_env.close()

    return model


def evaluate_ppo(model_path, n_episodes=100, render=True, fps_limit=60):
    """Evaluate trained PPO agent."""

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = TagGameGymEnv(render=render)

    episode_rewards = []
    episode_lengths = []

    print(f"Evaluating for {n_episodes} episodes...")
    if fps_limit:
        print(f"FPS limit: {fps_limit}")
    print("-" * 60)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            if fps_limit:
                import time
                step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if fps_limit:
                elapsed = time.time() - step_start
                target_time = 1.0 / fps_limit
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {episode_length}")

    import numpy as np
    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max Length: {np.max(episode_lengths)}")

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
            # Try to find best model
            model_path = os.path.join(args.log_dir, 'best_model.zip')
            if not os.path.exists(model_path):
                model_path = os.path.join(args.log_dir, 'ppo_taggame_final.zip')
            if not os.path.exists(model_path):
                raise ValueError(f"No model found. Specify --model path")
        else:
            model_path = args.model

        evaluate_ppo(model_path, args.episodes, args.render, args.fps)
