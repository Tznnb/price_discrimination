# examples/example_1_1.py
import numpy as np
from algorithms.market import Market
from algorithms.bbm import bbm
from utils.visualization import plot_market, plot_surplus_triangle


def run_example_1_1(show_plots=True):
    """
    运行论文Example 1.1

    参数:
    show_plots: 是否显示图表

    返回:
    包含结果的字典
    """
    print("-" * 40)
    print("Running Example 1.1 from the paper")
    print("-" * 40)

    # 定义市场：价值为1和5，均匀分布
    values = [1, 5]
    masses = [0.5, 0.5]
    market = Market(values, masses)

    print(f"Market: values={values}, masses={masses}")

    # 检查最优统一价格
    opt_price = market.optimal_price()[0]
    uniform_revenue = market.revenue(opt_price)
    print(f"Optimal uniform price: {opt_price}")
    print(f"Uniform pricing revenue: {uniform_revenue}")

    # 应用BBM算法
    scheme = bbm(market)

    # 计算结果
    cs = scheme.consumer_surplus()
    ps = scheme.producer_surplus()
    sw = scheme.social_welfare()

    print(f"Consumer surplus: {cs}")
    print(f"Producer surplus: {ps}")
    print(f"Social welfare: {sw}")

    # 验证结果是否符合论文描述
    total_value = sum(v * m for v, m in zip(values, masses))
    print(f"Total value: {total_value}")
    print(f"Producer surplus equals uniform revenue: {abs(ps - uniform_revenue) < 1e-10}")
    print(f"Social welfare equals total value: {abs(sw - total_value) < 1e-10}")

    # 可视化
    if show_plots:
        plot_market(market, title="Market in Example 1.1")
        plot_surplus_triangle(market, scheme, values,
                              title="Surplus Triangle for Example 1.1")

    return {
        "market": market,
        "scheme": scheme,
        "opt_price": opt_price,
        "uniform_revenue": uniform_revenue,
        "consumer_surplus": cs,
        "producer_surplus": ps,
        "social_welfare": sw
    }


if __name__ == "__main__":
    run_example_1_1()