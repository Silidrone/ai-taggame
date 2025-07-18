import time
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from mdp import MDP
from policy import EpsilonGreedyPolicy, Policy
from callbacks import EpisodeLogger, benchmark
from value_strategy import ValueStrategy
from mdp_solver import MDPSolver

S = TypeVar('S')
A = TypeVar('A')

class TD(MDPSolver[S, A], ABC):
    def __init__(self, mdp: MDP[S, A], policy: Policy[S, A],
                 value_strategy: ValueStrategy[S, A], discount_rate: float,
                 policy_threshold: float, decay_epsilon: bool = False):
        super().__init__(mdp, policy, discount_rate, policy_threshold)
        self._value_strategy = value_strategy
        self._decay_epsilon = decay_epsilon
        self._print_freq = 100
        self._logger = EpisodeLogger(self._print_freq)
        self._episode_end_callback = None
        
        policy.initialize(mdp, value_strategy)
        
        eps_policy = self._get_epsilon_policy()
        if eps_policy:
            self._logger.set_epsilon_getter(lambda: eps_policy.epsilon)
        
        if hasattr(self._value_strategy, 'get_current_learning_rate'):
            self._logger.set_learning_rate_getter(lambda: self._value_strategy.get_current_learning_rate())
    
    def _get_epsilon_policy(self) -> Optional[EpsilonGreedyPolicy[S, A]]:
        if isinstance(self._policy, EpsilonGreedyPolicy):
            return self._policy
        return None
    
    def set_episode_end_callback(self, callback):
        self._episode_end_callback = callback
    
    @abstractmethod
    def compute_target_q(self, reward: float, next_state: S, done: bool, next_action: Optional[A] = None) -> float:
        """Compute the target Q-value for the TD update.
        
        Args:
            reward: The immediate reward received
            next_state: The next state
            done: Whether the episode is terminal
            next_action: The next action (used by SARSA, ignored by Q-learning)
            
        Returns:
            The target Q-value for the update
        """
        pass
    
    def td_main(self) -> None:
        episode = 0
        eps_policy = self._get_epsilon_policy()
        
        try:
            while episode < self._policy_threshold:
                episode += 1
                self._logger.start_episode()
                
                state = self._mdp.reset()
                action = self._policy.sample(state)
                
                episode_reward = 0.0
                steps = 0
                
                while not self._mdp.is_terminal(state):
                    steps += 1
                    
                    next_state, reward = self._mdp.step(state, action)
                    episode_reward += reward
                    
                    done = self._mdp.is_terminal(next_state)
                    
                    next_action = self._policy.sample(next_state) if not done else None
                    target_q = self.compute_target_q(reward, next_state, done, next_action)
                    
                    self._value_strategy.update(state, action, target_q)
                    
                    state = next_state
                    action = next_action
                
                if self._decay_epsilon and eps_policy:
                    eps_policy.decay_epsilon()
                
                if hasattr(self._value_strategy, 'decay_learning_rate'):
                    self._value_strategy.decay_learning_rate()
                
                self._logger.end_episode(episode_reward, steps)
                
                if self._episode_end_callback:
                    self._episode_end_callback(episode)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        
        print(f"Training completed after {episode} episodes")
    
    def policy_iteration(self) -> None:
        def run_td():
            self.td_main()
        
        total_time = benchmark(run_td)
        print(f"Policy iteration completed in {total_time:.2f} seconds")


class SARSA(TD[S, A]):
    def __init__(self, mdp: MDP[S, A], policy: Policy[S, A],
                 value_strategy: ValueStrategy[S, A], discount_rate: float,
                 policy_threshold: float, decay_epsilon: bool = False):
        super().__init__(mdp, policy, value_strategy, discount_rate, policy_threshold, decay_epsilon)
    
    def compute_target_q(self, reward: float, next_state: S, done: bool, next_action: Optional[A] = None) -> float:
        if done:
            return reward
        
        next_q = self._value_strategy.Q(next_state, next_action)
        return reward + self._discount_rate * next_q
    

class QLearning(TD[S, A]):
    def __init__(self, mdp: MDP[S, A], policy: Policy[S, A],
                 value_strategy: ValueStrategy[S, A], discount_rate: float,
                 policy_threshold: float, decay_epsilon: bool = False):
        super().__init__(mdp, policy, value_strategy, discount_rate, policy_threshold, decay_epsilon)
    
    def compute_target_q(self, reward: float, next_state: S, done: bool, next_action: Optional[A] = None) -> float:
        if done:
            return reward
        
        best_action, best_q = self._value_strategy.get_best_action(next_state)
        return reward + self._discount_rate * best_q