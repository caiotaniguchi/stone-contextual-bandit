# A Bernoulli one-armed bandit
import numpy as np


class OneArmedBandit:
    def __init__(self, reward_probability=0.5):
        self.reward_probability = reward_probability

    def sample_reward(self, n_samples=1):
        return np.random.binomial(1, self.reward_probability)
