import time
from typing import Optional, TypeVar

from mdp import MDP
from policy import EpsilonGreedyPolicy, Policy
from callbacks import EpisodeLogger, benchmark
from value_strategy import ValueStrategy
from mdp_solver import MDPSolver

S = TypeVar('S')
A = TypeVar('A')

class SARSA(MDPSolver[S, A]):
    def __init__(self, mdp: MDP[S, A], policy: Policy[S, A],
                 value_strategy: ValueStrategy[S, A], discount_rate: float,
                 policy_threshold: float, decay_epsilon: bool = False):
        super().__init__(mdp, policy, discount_rate, policy_threshold)
        self._value_strategy = value_strategy
        self._decay_epsilon = decay_epsilon
        self._print_freq = 100
        self._logger = EpisodeLogger(self._print_freq)
        
        policy.initialize(mdp, value_strategy)
        
        eps_policy = self._get_epsilon_policy()
        if eps_policy:
            self._logger.set_epsilon_getter(lambda: eps_policy.epsilon)
    
    def _get_epsilon_policy(self) -> Optional[EpsilonGreedyPolicy[S, A]]:
        if isinstance(self._policy, EpsilonGreedyPolicy):
            return self._policy
        return None
    
    def _timed_update(self, state: S, action: A, target_q: float, current_episode: int = 0) -> None:
        if current_episode % self._print_freq == 0:
            start = time.time()
            self._value_strategy.update(state, action, target_q)
            end = time.time()
            print(f"Episode {current_episode} - Update time: {(end - start) * 1000:.2f} ms")
        else:
            self._value_strategy.update(state, action, target_q)
    
    def _timed_q(self, state: S, action: A, current_episode: int = 0) -> float:
        if current_episode % self._print_freq == 0:
            start = time.time()
            q_value = self._value_strategy.Q(state, action)
            end = time.time()
            print(f"Episode {current_episode} - Q lookup time: {(end - start) * 1000:.2f} ms")
            return q_value
        else:
            return self._value_strategy.Q(state, action)
    
    def sarsa_main(self) -> None:
        episode = 0
        eps_policy = self._get_epsilon_policy()
        
        print("Starting SARSA training...")
        
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
                    
                    if done:
                        target_q = reward
                    else:
                        next_q = self._value_strategy.Q(next_state, next_action)
                        target_q = reward + self._discount_rate * next_q
                    
                    self._value_strategy.update(state, action, target_q)
                    
                    state = next_state
                    action = next_action
                
                if self._decay_epsilon and eps_policy:
                    eps_policy.decay_epsilon()
                
                self._logger.end_episode(episode_reward, steps)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        
        print(f"Training completed after {episode} episodes")
    
    def policy_iteration(self) -> None:
        def run_sarsa():
            self.sarsa_main()
        
        total_time = benchmark(run_sarsa)
        print(f"Policy iteration completed in {total_time:.2f} seconds")