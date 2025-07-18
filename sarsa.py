import time
from typing import Optional, TypeVar

from mdp import MDP
from policy import EpsilonGreedyPolicy, Policy
from callbacks import EpisodeLogger, benchmark
from value_strategy import ValueStrategy
from mdp_solver import MDPSolver
from environments.taggame.constants import REPLAY_BATCH_SIZE, REPLAY_MIN_SIZE

S = TypeVar('S')
A = TypeVar('A')

class SARSA(MDPSolver[S, A]):
    def __init__(self, mdp: MDP[S, A], policy: Policy[S, A],
                 value_strategy: ValueStrategy[S, A], discount_rate: float,
                 policy_threshold: float, decay_epsilon: bool = False, 
                 replay_buffer=None):
        super().__init__(mdp, policy, discount_rate, policy_threshold)
        self._value_strategy = value_strategy
        self._decay_epsilon = decay_epsilon
        self._logger = EpisodeLogger()
        self._episode_end_callback = None
        self._replay_buffer = replay_buffer
        
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
    
    def _calculate_target_q(self, reward: float, next_state: S, next_action: A, done: bool) -> float:
        if done:
            return reward
        else:
            return reward + self._discount_rate * self._value_strategy.Q(next_state, next_action)
    
    def _train_on_batch(self, batch):
        for state, action, reward, next_state, done in batch:
            next_action = self._policy.sample(next_state)
            target_q = self._calculate_target_q(reward, next_state, next_action, done)
            self._value_strategy.update(state, action, target_q)
            
    def _train_from_replay_buffer(self, state, action, reward, next_state, done):
        self._replay_buffer.add(state, action, reward, next_state, done)
        
        if self._replay_buffer.is_ready(REPLAY_MIN_SIZE):
            batch = self._replay_buffer.sample(REPLAY_BATCH_SIZE)
            self._train_on_batch(batch)
    
    def sarsa_main(self) -> None:
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
                    
                    next_action = self._policy.sample(next_state) if not done else action
                    
                    if self._replay_buffer is not None:
                        self._train_from_replay_buffer(state, action, reward, next_state, done)
                    else:
                        target_q = self._calculate_target_q(reward, next_state, next_action, done)
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
        def run_sarsa():
            self.sarsa_main()
        
        total_time = benchmark(run_sarsa)
        print(f"Policy iteration completed in {total_time:.2f} seconds")