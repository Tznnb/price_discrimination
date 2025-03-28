# algorithms/passive_ps_max.py
import numpy as np
from algorithms.market import Market, MarketScheme


def find_optimal_price(market, price_set):
    """找到给定价格集中的最优价格"""
    max_revenue = -1
    opt_price = None

    for price in price_set:
        if price in market.values:
            revenue = market.revenue(price)
            if revenue > max_revenue:
                max_revenue = revenue
                opt_price = price

    return opt_price


def create_er_market(values, masses, target_support, price, epsilon=1e-10):
    """创建等收入市场段"""
    er_masses = np.zeros_like(masses)
    support = sorted(target_support)

    # 特殊情况：支持集只有一个元素
    if len(support) == 1:
        single_val = support[0]
        single_idx = np.where(values == single_val)[0][0]

        if single_val > price:
            # 如果单一值点大于价格，可以分配
            er_masses[single_idx] = 1.0  # 单点支持集分配100%质量
        else:
            # 值点小于等于价格，不能购买
            return er_masses

        return er_masses

    # 正常情况：多个值点
    min_val = min(support)

    # 对支持集中的每个值计算质量
    for i, val in enumerate(support):
        idx = np.where(values == val)[0][0]

        if val == price:
            # 指导价格的点
            er_masses[idx] = min_val / val
        elif i < len(support) - 1:
            # 中间点
            next_val = support[i + 1]
            er_masses[idx] = min_val * (1 / val - 1 / next_val)
        else:
            # 最大点如果不是价格点
            max_val = val
            if max_val > price:  # 只有最大值大于价格时才分配
                er_masses[idx] = min_val / max_val

    # 归一化
    total = np.sum(er_masses)
    if total > 0:
        er_masses = er_masses / total

    return er_masses


def is_price_optimal_for_values(values, masses, price, epsilon=1e-8):
    """检查价格是否是给定值集合的最优价格"""
    if price not in values:
        return False

    temp_market = Market(values, masses)
    opt_prices = temp_market.optimal_price()

    return price in opt_prices


def passive_ps_max(market, price_set, epsilon=1e-8, debug=False):
    """实现PassivePSMax算法"""
    if debug:
        print("\nExecuting PassivePSMax algorithm:")
        print(f"Market: {market}")
        print(f"Price set F: {price_set}")

    # 过滤掉不在市场值域中的价格
    price_set = sorted([p for p in price_set if p in market.values])

    if not price_set:
        if debug:
            print("No valid prices in F, not feasible")
        return None

    remaining = market.copy()
    scheme = MarketScheme()
    iteration = 0

    while remaining.total_mass() > epsilon:
        iteration += 1
        if debug:
            print(f"\nIteration {iteration}:")
            print(f"Remaining market: {remaining}")

        # 找出所有剩余点
        valid_indices = np.where(remaining.masses > epsilon)[0]
        if len(valid_indices) == 0:
            break

        # 如果只剩下一个值点，特殊处理
        if len(valid_indices) == 1:
            single_idx = valid_indices[0]
            single_val = remaining.values[single_idx]
            single_mass = remaining.masses[single_idx]

            # 找到最大的小于该值的价格点
            suitable_prices = [p for p in price_set if p < single_val]

            if not suitable_prices:
                if debug:
                    print(f"No price in F below remaining value {single_val}")
                return None

            best_price = max(suitable_prices)

            # 创建新段
            new_segment_masses = np.zeros_like(remaining.masses)
            new_segment_masses[single_idx] = single_mass
            new_segment = Market(remaining.values, new_segment_masses)

            # 添加到方案
            scheme.add_segment(new_segment, best_price)

            # 更新剩余质量
            remaining.masses[single_idx] = 0

            if debug:
                print(f"Single point case: assigned value {single_val} to price {best_price}")
                print(f"Remaining mass: {remaining.total_mass()}")

            continue

        # 正常情况：多个点
        # 找到所有F中价格点，检查哪个最优
        best_price = None
        max_revenue = -1

        for p in price_set:
            # 计算对当前剩余市场的收入
            if p in remaining.values:
                p_idx = np.where(remaining.values == p)[0][0]
                revenue = p * np.sum(remaining.masses[p_idx:])

                if revenue > max_revenue:
                    max_revenue = revenue
                    best_price = p

        if not best_price:
            if debug:
                print("No price in F is suitable for remaining market")
            return None

        if debug:
            print(f"Selected optimal F price: {best_price}")

        # 构建支持集：包括价格点加所有非F点
        support_indices = []
        for i in valid_indices:
            val = remaining.values[i]
            if val == best_price or val not in price_set:
                support_indices.append(i)

        support_values = [remaining.values[i] for i in support_indices]

        if not support_values:
            if debug:
                print("Empty support set, stopping")
            break

        if debug:
            print(f"Support set: {support_values}")

        # 创建等收入市场
        er_masses = create_er_market(
            remaining.values,
            remaining.masses,
            support_values,
            best_price
        )

        # 计算gamma
        gamma_candidates = []
        for i in support_indices:
            if remaining.masses[i] > epsilon and er_masses[i] > epsilon:
                gamma_candidates.append(remaining.masses[i] / er_masses[i])

        if not gamma_candidates:
            if debug:
                print("No valid gamma found, stopping")
            return None

        gamma = min(gamma_candidates)

        if debug:
            print(f"Calculated gamma: {gamma}")

        # 创建新市场段
        new_segment_masses = np.zeros_like(remaining.masses)
        for i in support_indices:
            new_segment_masses[i] = min(er_masses[i] * gamma, remaining.masses[i])

        # 验证价格对这个段是最优的
        if not is_price_optimal_for_values(remaining.values, new_segment_masses, best_price):
            if debug:
                print(f"Price {best_price} is not optimal for constructed segment")
            return None

        new_segment = Market(remaining.values, new_segment_masses)

        # 更新剩余市场
        for i in support_indices:
            remaining.masses[i] = max(0, remaining.masses[i] - new_segment_masses[i])
            if remaining.masses[i] < epsilon:
                remaining.masses[i] = 0

        # 将新段添加到方案中
        scheme.add_segment(new_segment, best_price)

        if debug:
            print(f"Added segment with price {best_price}")
            print(f"Remaining mass: {remaining.total_mass()}")

    if debug:
        print(f"Success! Created scheme with {len(scheme.segments)} segments")

    # 验证结果
    total_scheme_mass = sum(segment.total_mass() for segment in scheme.segments)
    if abs(total_scheme_mass - market.total_mass()) > epsilon:
        if debug:
            print(f"Failed: Total mass mismatch. Scheme: {total_scheme_mass:.6f}, Market: {market.total_mass():.6f}")
        return None

    return scheme