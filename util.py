import os
import numpy as np
import matplotlib.pyplot as plt

def standard_saver(agent, save_freq, log_dir, logger):
    def saver(episode):
        if episode % save_freq == 0:
            checkpoint_path = os.path.join(log_dir, 'checkpoint.pt')
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint at episode {episode}")
    return saver

def plot_training_progress(episode_rewards, episode_durations, log_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= 100:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(episode_rewards)), smoothed_rewards, label='Moving Average (100 eps)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(episode_durations, alpha=0.3, label='Episode Duration')
    if len(episode_durations) >= 100:
        smoothed_durations = np.convolve(episode_durations, np.ones(100)/100, mode='valid')
        ax2.plot(range(99, len(episode_durations)), smoothed_durations, label='Moving Average (100 eps)', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Duration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
