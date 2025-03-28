# algorithms/bbm.py
import numpy as np
from algorithms.market import Market, MarketScheme


def create_equal_revenue_market(values, support, unit_mass=True):
    """
    创建等收入市场

    参数:
    values: 完整的价值数组
    support: 支持集（价值点子集）
    unit_mass: 是否归一化为总质量为1

    返回:
    等收入市场
    """
    support = sorted(support)
    min_val = support[0]

    # 创建与原始市场相同维度的质量数组
    masses = np.zeros(len(values))

    # 找到支持集中每个值在原始values中的索引
    support_indices = [np.where(values == v)[0][0] for v in support]

    # 计算最大值对应的质量
    max_idx = support_indices[-1]
    masses[max_idx] = min_val / values[max_idx]

    # 计算其他值对应的质量
    for j in range(len(support_indices) - 2, -1, -1):
        idx = support_indices[j]
        next_idx = support_indices[j + 1]
        masses[idx] = min_val * (1 / values[idx] - 1 / values[next_idx])

    # 归一化
    if unit_mass and np.sum(masses) > 0:
        masses = masses / np.sum(masses)

    return Market(values, masses)


def bbm(market):
    """
    实现Bergemann Brooks Morris算法

    参数:
    market: Market对象

    返回:
    MarketScheme对象
    """
    remaining = market.copy()
    scheme = MarketScheme()

    while remaining.total_mass() > 1e-10:  # 避免浮点精度问题
        # 找出支持集
        support_indices = np.where(remaining.masses > 0)[0]
        if len(support_indices) == 0:
            break

        support = remaining.values[support_indices].tolist()

        # 创建等收入市场
        eq_market = create_equal_revenue_market(remaining.values, support)

        # 找到最大可能的gamma
        gamma_candidates = []
        for i in support_indices:
            if remaining.masses[i] > 0 and eq_market.masses[i] > 0:
                gamma_candidates.append(remaining.masses[i] / eq_market.masses[i])

        gamma = min(gamma_candidates) if gamma_candidates else 0

        if gamma <= 0:
            break

        # 创建新的市场段
        new_segment = Market(remaining.values, eq_market.masses * gamma)

        # 更新剩余市场
        remaining.masses = remaining.masses - new_segment.masses

        # 将新段添加到方案中，使用support中的最小价值作为价格
        opt_price = min(support)
        scheme.add_segment(new_segment, opt_price)

    return scheme