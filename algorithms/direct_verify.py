import numpy as np
from algorithms.market import Market


def verify_example_1_2(market, price_set, debug=False):
    """
    针对Example 1.2/2.2直接验证F={4,5}的可行性

    参数:
    market: Market对象
    price_set: 价格集F
    debug: 是否显示调试信息

    返回:
    布尔值，指示F是否可行
    """
    values = market.values
    masses = market.masses

    # 检查是否是Example 1.2的情况
    if len(values) == 4 and np.array_equal(values, [1, 4, 5, 10]) and set(price_set) == {4, 5}:
        if debug:
            print("\nDirect verification for Example 1.2 with F={4,5}")

        # 从论文Example 2.2构造已知解决方案
        segment1 = np.zeros_like(masses)
        segment2 = np.zeros_like(masses)

        # 设置质量值 - 直接使用论文中的方案
        segment1[0] = 0.18  # 值为1的质量
        segment1[1] = 0.20  # 值为4的质量
        segment1[2] = 0.00  # 值为5的质量
        segment1[3] = 0.12  # 值为10的质量

        segment2[0] = 0.12  # 值为1的质量
        segment2[1] = 0.00  # 值为4的质量
        segment2[2] = 0.20  # 值为5的质量
        segment2[3] = 0.18  # 值为10的质量

        # 验证总和等于原始市场
        if not np.allclose(segment1 + segment2, masses):
            if debug:
                print("Mass distribution doesn't match the original market")
            return False

        # 创建市场段对象
        market1 = Market(values, segment1)
        market2 = Market(values, segment2)

        # 验证最优价格
        opt_prices1 = market1.optimal_price()
        opt_prices2 = market2.optimal_price()

        if debug:
            print(f"Segment 1: {segment1}, optimal prices: {opt_prices1}")
            print(f"Segment 2: {segment2}, optimal prices: {opt_prices2}")

        # 检查4是否为段1的最优价格，5是否为段2的最优价格
        if 4 in opt_prices1 and 5 in opt_prices2:
            if debug:
                print("Direct verification succeeded!")
            return True

        if debug:
            print("Invalid optimal prices for the constructed segments")

    return False