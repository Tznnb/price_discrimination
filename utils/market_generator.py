# utils/market_generator.py

import numpy as np
from scipy.stats import truncnorm

class MarketGenerator:
    """
    用于构造不同分布的 market (values + masses)，支持基础与复杂分布。
    """

    @staticmethod
    def uniform(n=5, low=1, high=10):
        values = np.linspace(low, high, n, dtype=int)
        masses = np.ones_like(values) / n
        return values.tolist(), masses.tolist()

    @staticmethod
    def binary(low=1, high=10, p=0.5):
        values = [low, high]
        masses = [p, 1 - p]
        return values, masses

    @staticmethod
    def geometric(start=1, ratio=2, n=5, q=0.5):
        values = [start * (ratio ** i) for i in range(n)]
        masses = [q ** i for i in range(n)]
        masses = masses / np.sum(masses)
        return values, masses.tolist()

    @staticmethod
    def truncated_normal(mu=5, sigma=2, n=5, low=1, high=10):
        a, b = (low - mu) / sigma, (high - mu) / sigma
        dist = truncnorm(a, b, loc=mu, scale=sigma)
        values = np.linspace(low, high, n)
        pdf = dist.pdf(values)
        masses = pdf / pdf.sum()
        return values.tolist(), masses.tolist()

    @staticmethod
    def bimodal(values1, values2, weight=0.5):
        v = values1 + values2
        m1 = np.ones(len(values1)) * weight / len(values1)
        m2 = np.ones(len(values2)) * (1 - weight) / len(values2)
        masses = np.concatenate([m1, m2])
        return v, masses.tolist()

    @staticmethod
    def powerlaw(alpha=2.0, n=5, scale=1.0):
        values = np.array([scale * (i + 1) for i in range(n)])
        probs = 1.0 / (values ** alpha)
        masses = probs / probs.sum()
        return values.tolist(), masses.tolist()
