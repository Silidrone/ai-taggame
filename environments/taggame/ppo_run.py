import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from environments.taggame.gym_wrapper import TagGameGymEnv


def train_ppo(log_dir, n_timesteps=1000000, render=False):
    """Train PPO agent on TagGame with continuous actions."""

    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = TagGameGymEnv(render=render)
    env = Monitor(env, log_dir)

    # Create evaluation environment
    eval_env = TagGameGymEnv(render=False)
    eval_env = Monitor(eval_env, os.path.join(log_dir, 'eval'))

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
    print(f"Log directory: {log_dir}")
    print("-" * 60)

    # Train the model
    model.learn(
        total_timesteps=n_timesteps,
        callback=[checkpoint_callback, eval_callback],
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

    args = parser.parse_args()

    if args.mode == 'train':
        train_ppo(args.log_dir, args.timesteps)
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
