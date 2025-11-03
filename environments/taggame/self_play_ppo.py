import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environments.taggame.gym_wrapper import TagGameGymEnv
from environments.taggame.config import MAX_VELOCITY
from environments.taggame.run import feature_extractor


class SelfPlayEnv(TagGameGymEnv):
    def __init__(self, render=False, chaser_model=None):
        super().__init__(render=render)
        self.chaser_model = chaser_model

    def set_chaser_model(self, model):
        """Update the chaser model."""
        self.chaser_model = model

    def step(self, evader_action):
        evader_angle = float(evader_action[0])
        evader_vx = np.cos(evader_angle) * MAX_VELOCITY
        evader_vy = np.sin(evader_angle) * MAX_VELOCITY
        evader_game_action = (evader_vx, evader_vy)

        chaser_game_action = None
        if self.chaser_model is not None:
            chaser_obs = self._get_chaser_observation()
            chaser_action, _ = self.chaser_model.predict(chaser_obs, deterministic=False)

            chaser_angle = float(chaser_action[0])
            chaser_vx = np.cos(chaser_angle) * MAX_VELOCITY
            chaser_vy = np.sin(chaser_angle) * MAX_VELOCITY
            chaser_game_action = (chaser_vx, chaser_vy)

        next_state, reward = self.env.step(self.current_state, evader_game_action, chaser_game_action)

        terminated = self.env.is_terminal(next_state)
        truncated = False
        obs = feature_extractor(next_state)
        self.current_state = next_state

        return obs, reward, terminated, truncated, {}

    def _get_chaser_observation(self):
        rl_pos, rl_vel, tag_pos, tag_vel, is_tagged = self.current_state
        chaser_state = (tag_pos, tag_vel, rl_pos, rl_vel, is_tagged)
        return feature_extractor(chaser_state)


def make_env(chaser_model=None):
    def _init():
        env = SelfPlayEnv(render=False, chaser_model=chaser_model)
        env = Monitor(env)
        return env
    return _init


class SelfPlayCallback(BaseCallback):
    def __init__(self, evader_model, chaser_model, update_freq=50000, verbose=0):
        super().__init__(verbose)
        self.evader_model = evader_model
        self.chaser_model = chaser_model
        self.update_freq = update_freq
        self.last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update >= self.update_freq:
            if self.verbose > 0:
                print(f"Timesteps: {self.num_timesteps} | Updating chaser model")

            self.chaser_model.policy.load_state_dict(self.evader_model.policy.state_dict())

            self.last_update = self.num_timesteps

        return True


def train_ppo(log_dir, n_timesteps=1000000, render=False, n_envs=8):
    """Train PPO agent with self-play."""

    os.makedirs(log_dir, exist_ok=True)

    dummy_env = SelfPlayEnv(render=False)
    chaser_model = PPO(
        "MlpPolicy",
        dummy_env,
        learning_rate=3e-4,
        verbose=0
    )
    dummy_env.close()

    env = SubprocVecEnv([make_env(chaser_model=chaser_model) for _ in range(n_envs)])

    eval_env = SelfPlayEnv(render=False, chaser_model=chaser_model)
    eval_env = Monitor(eval_env, os.path.join(log_dir, 'eval'))

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix='selfplay_evader'
    )

    evader_model = PPO(
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

    selfplay_callback = SelfPlayCallback(
        evader_model=evader_model,
        chaser_model=chaser_model,
        update_freq=50000,
        verbose=1
    )

    print(f"Training self-play PPO for {n_timesteps} timesteps...")
    print(f"Parallel environments: {n_envs}")
    print(f"Log directory: {log_dir}")
    print("-" * 60)

    # Train the model
    evader_model.learn(
        total_timesteps=n_timesteps,
        callback=[checkpoint_callback, selfplay_callback],
        progress_bar=False
    )

    evader_path = os.path.join(log_dir, 'selfplay_evader_final')
    chaser_path = os.path.join(log_dir, 'selfplay_chaser_final')
    evader_model.save(evader_path)
    chaser_model.save(chaser_path)
    print(f"Evader model saved to {evader_path}")
    print(f"Chaser model saved to {chaser_path}")

    env.close()
    eval_env.close()

    return evader_model, chaser_model


def evaluate_ppo(evader_model_path, chaser_model_path, n_episodes=100, render=True, fps_limit=60):
    """Evaluate trained self-play agents."""

    evader_model = PPO.load(evader_model_path)
    chaser_model = PPO.load(chaser_model_path)

    env = SelfPlayEnv(render=render, chaser_model=chaser_model)

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

            action, _ = evader_model.predict(obs, deterministic=True)
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

    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max Length: {np.max(episode_lengths)}")

    env.close()

    return episode_rewards, episode_lengths


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate self-play PPO on TagGame')
    parser.add_argument('mode', choices=['train', 'evaluate'], help='Mode: train or evaluate')
    parser.add_argument('--log-dir', type=str, default='data/selfplay_ppo',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Number of timesteps to train')
    parser.add_argument('--evader-model', type=str, default=None,
                        help='Path to evader model for evaluation')
    parser.add_argument('--chaser-model', type=str, default=None,
                        help='Path to chaser model for evaluation')
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
        evader_path = args.evader_model or os.path.join(args.log_dir, 'selfplay_evader_final.zip')
        chaser_path = args.chaser_model or os.path.join(args.log_dir, 'selfplay_chaser_final.zip')

        if not os.path.exists(evader_path) or not os.path.exists(chaser_path):
            raise ValueError(f"Models not found at {evader_path} and {chaser_path}")

        evaluate_ppo(evader_path, chaser_path, args.episodes, args.render, args.fps)
