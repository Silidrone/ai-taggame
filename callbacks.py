import time
from typing import Any, Callable, Dict, List, Optional, Union


def benchmark(func: Callable[[], Any]) -> float:
    start_time = time.time()
    func()
    end_time = time.time()
    return end_time - start_time


class EpisodeLogger:
    def __init__(self, log_freq: int = 100):
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
        print(f"  Epsilon: {self.get_current_epsilon():.4f}" if hasattr(self, 'get_current_epsilon') else "")
    
    def set_epsilon_getter(self, getter: Callable[[], float]) -> None:
        self.get_current_epsilon = getter
    
    def get_stats(self) -> Dict[str, Union[List[float], List[int]]]:
        return {
            'rewards': self.episode_rewards,
            'steps': self.episode_steps,
            'times': self.episode_times,
            'episodes': self.current_episode
        }