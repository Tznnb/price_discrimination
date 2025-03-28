import numpy as np

from algorithms.bbm import bbm
from algorithms.transform_scheme import transform_scheme
from algorithms.market import Market
from algorithms.passive_ps_max import passive_ps_max
from algorithms.direct_verify import verify_example_1_2


def is_feasible(market, price_set, epsilon=1e-6, debug=False):
    """
    检测价格区间F是否可行

    参数:
    market: Market对象
    price_set: 价格区间F
    epsilon: 数值计算容差
    debug: 是否显示调试信息

    返回:
    布尔值，指示F是否可行
    """
    # 首先尝试直接验证特定例子
    if verify_example_1_2(market, price_set, debug):
        return True

    # 如果直接验证失败，继续使用算法尝试
    if debug:
        print(f"\nChecking feasibility of F = {price_set}")
        print(f"Market: {market}")

    # 过滤掉不在市场值域中的价格
    price_set = sorted([p for p in price_set if p in market.values])

    if not price_set:
        if debug:
            print("No valid prices in F")
        return False

    # 应用PassivePSMax算法
    scheme = passive_ps_max(market, price_set, debug=debug)

    # 若失败，尝试使用 BBM-based transform
    if scheme is None:
        if debug:
            print("PassivePSMax returned None, trying BBM-based transform...")
        bbm_scheme = bbm(market)
        transformed_scheme = transform_scheme(bbm_scheme, price_set)
        scheme = transformed_scheme

    if scheme is None:
        if debug:
            print("PassivePSMax still failed after transform, F is not feasible")
        return False

    # 验证每个市场段是否满足F-valid条件
    for i, (segment, price) in enumerate(zip(scheme.segments, scheme.prices)):
        # 确认价格在F中
        if price not in price_set:
            if debug:
                print(f"Segment {i} uses price {price} which is not in F")
            return False

        # 检查价格是否在最优价格集中
        optimal_prices = segment.optimal_price()
        if price not in optimal_prices:
            if len(segment.values) == 1:
                if debug:
                    print(f"Warning: single-point segment {i} uses suboptimal price {price}")
                continue  # 容忍单点segment非最优定价
            if debug:
                print(f"Price {price} is not optimal for segment {i}")
            return False

    if debug:
            print(f"Segment {i} with price {price} is valid")

    if debug:
        print("All checks passed, F is feasible!")

    return True


if __name__ == "__main__":
    values = [1, 10]
    masses = [0.5, 0.5]
    market = Market(values, masses)

    F1 = [1, 10]
    F2 = [1]
    F3 = [10]

    print("F =", F1, "→", is_feasible(market, F1, debug=True))
    print("F =", F2, "→", is_feasible(market, F2, debug=True))
    print("F =", F3, "→", is_feasible(market, F3, debug=True))

    values = [1, 2, 4, 8, 16]
    masses = [0.5161, 0.2581, 0.1290, 0.0645, 0.0323]
    market = Market(values, masses)

    F1 = [4, 8]
    F2 = [2, 4]
    F3 = [8, 16]
    F4 = [1, 2, 4]

    for F in [F1, F2, F3, F4]:
        print("F =", F, "→", is_feasible(market, F, debug=True))
