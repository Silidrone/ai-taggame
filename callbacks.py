import time
import os
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Union

from environments.taggame.constants import PLOT_DIR, LOG_FREQ



def benchmark(func: Callable[[], Any]) -> float:
    start_time = time.time()
    func()
    end_time = time.time()
    return end_time - start_time


class EpisodeLogger:
    def __init__(self, log_freq: int = LOG_FREQ):
        self.log_freq = log_freq
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.episode_times: List[float] = []
        self.current_episode = 0
        self.start_time = None
    
    def start_episode(self) -> None:
        self.start_time = time.time()
        self.current_episode += 1
    
    def end_episode(self, reward: float, steps: int) -> None:
        if self.start_time is None:
            raise RuntimeError("Cannot end episode before starting one")
        
        duration = time.time() - self.start_time
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_times.append(duration)
        
        if self.current_episode % self.log_freq == 0:
            self.print_stats()
    
    def print_stats(self) -> None:
        if not self.episode_rewards:
            return
        
        last_rewards = self.episode_rewards[-self.log_freq:]
        last_steps = self.episode_steps[-self.log_freq:]
        last_times = self.episode_times[-self.log_freq:]
        
        avg_reward = sum(last_rewards) / len(last_rewards)
        avg_steps = sum(last_steps) / len(last_steps)
        avg_time = sum(last_times) / len(last_times)
        
        print(f"Episode {self.current_episode}:")
        print(f"  Avg reward: {avg_reward:.2f}")
        print(f"  Avg steps: {avg_steps:.2f}")
        print(f"  Avg time: {avg_time:.4f}s")
        if hasattr(self, 'get_current_epsilon'):
            print(f"  Epsilon: {self.get_current_epsilon():.4f}")
        if hasattr(self, 'get_current_learning_rate'):
            print(f"  Learning Rate: {self.get_current_learning_rate():.6f}")
    
    def set_epsilon_getter(self, getter: Callable[[], float]) -> None:
        self.get_current_epsilon = getter
    
    def set_learning_rate_getter(self, getter: Callable[[], float]) -> None:
        self.get_current_learning_rate = getter
    
    def get_stats(self) -> Dict[str, Union[List[float], List[int]]]:
        return {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'times': self.episode_times,
            'episodes': self.current_episode
        }
    
    def plot_training_progress(self, filename: str = "training_progress.png") -> None:
        if len(self.episode_rewards) < self.log_freq:
            print(f"Plot generation unsuccessful: Not enough data points (need at least {self.log_freq} episodes)")
            return
        
        episodes = []
        avg_rewards = []
        avg_steps = []
        
        for i in range(self.log_freq, len(self.episode_rewards) + 1, self.log_freq):
            episode_num = i
            recent_rewards = self.episode_rewards[i-self.log_freq:i]
            recent_steps = self.episode_steps[i-self.log_freq:i]
            
            episodes.append(episode_num)
            avg_rewards.append(sum(recent_rewards) / len(recent_rewards))
            avg_steps.append(sum(recent_steps) / len(recent_steps))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(episodes, avg_rewards, 'b-', linewidth=2, label='Average Reward')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Training Progress: Average Reward over Episodes')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(episodes, avg_steps, 'gray', linewidth=2, label='Average Steps')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Average Steps')
        ax2.set_title('Training Progress: Average Steps per Episode')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(PLOT_DIR, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()