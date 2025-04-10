# algorithms/market.py
import numpy as np


class Market:
    """表示市场的基本类"""

    def __init__(self, values, masses):
        """
        初始化市场

        参数:
        values: 数组，表示不同的价值点
        masses: 数组，表示每个价值点的质量
        """
        # 先转换为NumPy数组，再进行操作
        self.values = np.array(values)
        self.masses = np.array(masses)

        assert len(self.values) == len(self.masses), "Values and masses must have the same length"
        assert np.all(self.masses >= 0), "Masses must be non-negative"

        # 确保values是升序排列的
        if not np.all(np.diff(self.values) > 0):
            sort_idx = np.argsort(self.values)
            self.values = self.values[sort_idx]
            self.masses = self.masses[sort_idx]

    def total_mass(self):
        """返回市场总质量"""
        return np.sum(self.masses)

    def normalize(self):
        """归一化市场质量为1"""
        if self.total_mass() > 0:
            self.masses = self.masses / self.total_mass()
        return self

    def revenue(self, price):
        """
        计算给定价格下的收入

        参数:
        price: 价格点

        返回:
        对应的收入
        """
        if price not in self.values:
            raise ValueError(f"Price {price} not in values {self.values}")

        price_idx = np.where(self.values == price)[0][0]
        return price * np.sum(self.masses[price_idx:])

    def optimal_price(self):
        """
        计算最优统一价格

        返回:
        最优价格集合
        """
        revenues = [self.revenue(p) for p in self.values]
        max_revenue = max(revenues)
        opt_prices = [p for i, p in enumerate(self.values) if revenues[i] == max_revenue]
        return opt_prices

    def copy(self):
        """创建市场的副本"""
        return Market(self.values.copy(), self.masses.copy())

    def __str__(self):
        return f"Market(values={self.values}, masses={self.masses})"

    def __repr__(self):
        return self.__str__()


class MarketScheme:
    """表示市场方案的类"""

    def __init__(self, segments=None, prices=None):
        """
        初始化市场方案

        参数:
        segments: 市场段列表
        prices: 对应的价格列表
        """
        if segments is None:
            segments = []
        if prices is None:
            prices = []

        self.segments = segments
        self.prices = prices
        assert len(segments) == len(prices), "Segments and prices must have the same length"

    def add_segment(self, segment, price):
        """添加市场段和价格"""
        self.segments.append(segment)
        self.prices.append(price)

    def consumer_surplus(self):
        """计算消费者剩余"""
        cs = 0
        for segment, price in zip(self.segments, self.prices):
            for i, v in enumerate(segment.values):
                if v >= price:
                    cs += segment.masses[i] * (v - price)
        return cs

    def producer_surplus(self):
        """计算生产者剩余"""
        ps = 0
        for segment, price in zip(self.segments, self.prices):
            for i, v in enumerate(segment.values):
                if v >= price:
                    ps += segment.masses[i] * price
        return ps

    def social_welfare(self):
        """计算社会福利"""
        return self.consumer_surplus() + self.producer_surplus()

    def total_revenue(self):
        """计算方案的总收入"""
        return sum(segment.total_mass() * price for segment, price in zip(self.segments, self.prices))

    def __str__(self):
        return f"MarketScheme with {len(self.segments)} segments"