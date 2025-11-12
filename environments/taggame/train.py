import os
import logging
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environments.taggame import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

N_FEATURES = 22

def feature_extractor(state, chaser_policy_idx=0):
    import math
    from environments.taggame.config import WIDTH, HEIGHT, MAX_VELOCITY
    
    my_pos, my_vel, tag_pos, tag_vel, is_tagged = state

    dist_left = my_pos[0] / WIDTH
    dist_right = (WIDTH - my_pos[0]) / WIDTH
    dist_top = my_pos[1] / HEIGHT
    dist_bottom = (HEIGHT - my_pos[1]) / HEIGHT

    norm_vel_x = my_vel[0] / MAX_VELOCITY
    norm_vel_y = my_vel[1] / MAX_VELOCITY

    norm_tagger_vel_x = tag_vel[0] / MAX_VELOCITY
    norm_tagger_vel_y = tag_vel[1] / MAX_VELOCITY

    dx = my_pos[0] - tag_pos[0]
    dy = my_pos[1] - tag_pos[1]
    distance = math.sqrt(dx * dx + dy * dy) / math.sqrt(WIDTH**2 + HEIGHT**2)

    # Normalized distance components to tagger
    norm_dx = dx / WIDTH
    norm_dy = dy / HEIGHT

    angle = math.atan2(dy, dx)
    normalized_angle = (angle + math.pi) / (2 * math.pi)

    # Corner positions
    corners = [(0, 0), (0, HEIGHT), (WIDTH, 0), (WIDTH, HEIGHT)]
    max_corner_dist = math.sqrt(WIDTH**2 + HEIGHT**2)

    my_corner_dists = [
        math.sqrt((my_pos[0] - cx)**2 + (my_pos[1] - cy)**2) / max_corner_dist
        for cx, cy in corners
    ]

    tagger_corner_dists = [
        math.sqrt((tag_pos[0] - cx)**2 + (tag_pos[1] - cy)**2) / max_corner_dist
        for cx, cy in corners
    ]
    
    # Add chaser type feature
    from environments.taggame.deterministic_policies import ALL_POLICIES
    norm_chaser_type = chaser_policy_idx / len(ALL_POLICIES)

    bias = 1.0

    return np.array([
        dist_left, dist_right, dist_top, dist_bottom,
        norm_vel_x, norm_vel_y,
        norm_tagger_vel_x, norm_tagger_vel_y,
        distance, normalized_angle,
        norm_dx, norm_dy,
        my_corner_dists[0], my_corner_dists[1], my_corner_dists[2], my_corner_dists[3],
        tagger_corner_dists[0], tagger_corner_dists[1], tagger_corner_dists[2], tagger_corner_dists[3],
        bias,
        norm_chaser_type
    ], dtype=np.float32)


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
        from environments.taggame.gym_wrapper import TagGameGymEnv
        env = TagGameGymEnv(render=False, chaser_policy_idx=chaser_policy_idx)
        env = Monitor(env)
        return env
    return _init


def train_ppo(log_dir, n_timesteps=1000000, render=False, n_envs=8):

    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logging for training
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    policy_assignments = distribute_policies_to_envs(n_envs, config.POLICY_WEIGHTS)
    logger.info(f"Policy distribution across {n_envs} environments:")
    from collections import Counter
    from environments.taggame.deterministic_policies import ALL_POLICIES
    counts = Counter(policy_assignments)
    for policy_idx, count in sorted(counts.items()):
        policy_name = ALL_POLICIES[policy_idx].__name__
        logger.info(f"  Policy {policy_idx} ({policy_name}): {count} envs")

    env = SubprocVecEnv([make_env(policy_idx) for policy_idx in policy_assignments])

    from environments.taggame.gym_wrapper import TagGameGymEnv
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
        device="cpu",
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )
    logger.info(f"Training PPO for {n_timesteps} timesteps...")
    logger.info(f"Parallel environments: {n_envs}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("-" * 60)

    model.learn(
        total_timesteps=n_timesteps,
        callback=[curriculum_callback, checkpoint_callback, eval_callback],
        progress_bar=False
    )
    final_path = os.path.join(log_dir, 'final')
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}.zip")

    env.close()
    eval_env.close()

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PPO on TagGame')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for logs and checkpoints (default: auto-generate with timestamp)')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Number of timesteps to train')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='Number of parallel environments for training')

    args = parser.parse_args()
    
    # Generate timestamped directory if not specified
    if args.log_dir is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        args.log_dir = f'data/taggame/train_{timestamp}'
    
    train_ppo(args.log_dir, args.timesteps, n_envs=args.n_envs)
