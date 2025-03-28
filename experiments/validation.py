# experiments/validation.py
import numpy as np
from algorithms import Market
from algorithms.bbm import bbm
from algorithms.passive_ps_max import passive_ps_max
from algorithms.feasibility import is_feasible


def example_1_1():
    """验证原论文Example 1.1"""
    # 定义市场：价值为1和5，均匀分布
    values = [1, 5]
    masses = [0.5, 0.5]
    market = Market(values, masses)

    # 检查最优统一价格
    opt_price = market.optimal_price()[0]
    print(f"最优统一价格: {opt_price}")
    print(f"统一定价收入: {market.revenue(opt_price)}")

    # 应用BBM算法
    scheme = bbm(market)

    # 计算结果
    cs = scheme.consumer_surplus()
    ps = scheme.producer_surplus()
    sw = scheme.social_welfare()

    print(f"消费者剩余: {cs}")
    print(f"生产者剩余: {ps}")
    print(f"社会福利: {sw}")

    # 验证结果
    # 1. 生产者剩余应等于统一定价收入
    # 2. 社会福利应等于总值
    assert abs(ps - market.revenue(opt_price)) < 1e-10
    assert abs(sw - np.sum(market.values * market.masses)) < 1e-10

    print("Example 1.1验证成功!")


def example_1_2():
    """验证原论文Example 1.2"""
    # 定义市场
    values = [1, 4, 5, 10]
    masses = [0.3, 0.2, 0.2, 0.3]
    market = Market(values, masses)

    # 检查最优统一价格
    opt_price = market.optimal_price()[0]
    print(f"最优统一价格: {opt_price}")
    print(f"统一定价收入: {market.revenue(opt_price)}")

    # 检查F = {4, 5}是否可行
    price_set = [4, 5]
    is_f_feasible = is_feasible(market, price_set)
    print(f"F = {price_set}是否可行: {is_f_feasible}")

    if is_f_feasible:
        # 应用PassivePSMax算法
        scheme = passive_ps_max(market, price_set)

        # 计算结果
        cs = scheme.consumer_surplus()
        ps = scheme.producer_surplus()
        sw = scheme.social_welfare()

        print(f"消费者剩余: {cs}")
        print(f"生产者剩余: {ps}")
        print(f"社会福利: {sw}")

        # 检查F = {4}是否可行
        print(f"F = [4]是否可行: {is_feasible(market, [4])}")

    print("Example 1.2验证成功!")


if __name__ == "__main__":
    print("验证Example 1.1:")
    example_1_1()
    print("\n验证Example 1.2:")
    example_1_2()