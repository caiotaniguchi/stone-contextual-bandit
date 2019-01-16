# Explore-then-Exploit Multi-Armed-Bandit Strategy
import numpy as np


class EpsilonGreedy:
    def __init__(self, bandits):
        self.bandits = bandits
        self.reset()

    def reset(self):
        self.bandits_rewards = [[] for i in range(len(self.bandits))]
        self.avg_estimates = [0] * len(self.bandits)
        self.num_activations = [0] * len(self.bandits)
        self.experiment_rewards = []
        self.experiment_bandits_selected = []
        self.best_bandit = 0

    def run_experiment(self, epsilon=0.1, experiment_length=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.reset()

        for _ in range(experiment_length):
            if np.random.uniform() < epsilon:
                while True:
                    chosen_bandit = np.random.randint(0, len(self.bandits))
                    if chosen_bandit != self.best_bandit:
                        break
            else:
                chosen_bandit = self.best_bandit

            reward = self.bandits[chosen_bandit].sample_reward()
            self.avg_estimates[chosen_bandit] = (
                (self.avg_estimates[chosen_bandit] * self.num_activations[chosen_bandit] + reward) /
                (self.num_activations[chosen_bandit] + 1)
            )
            self.num_activations[chosen_bandit] += 1
            self.experiment_rewards.append(reward)
            self.experiment_bandits_selected.append(chosen_bandit)
            self.best_bandit = np.argmax(self.avg_estimates)
