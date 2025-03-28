# examples/example_1_2.py
import numpy as np
from algorithms.market import Market
from algorithms.passive_ps_max import passive_ps_max
from algorithms.feasibility import is_feasible
from utils.visualization import plot_market, plot_surplus_triangle


def run_example_1_2(show_plots=True, debug=False):
    """
    运行论文Example 1.2

    参数:
    show_plots: 是否显示图表
    debug: 是否显示详细调试信息

    返回:
    包含结果的字典
    """
    print("-" * 40)
    print("Running Example 1.2 from the paper")
    print("-" * 40)

    # 定义市场
    values = [1, 4, 5, 10]
    masses = [0.3, 0.2, 0.2, 0.3]
    market = Market(values, masses)

    print(f"Market: values={values}, masses={masses}")

    # 检查最优统一价格
    opt_price = market.optimal_price()[0]
    uniform_revenue = market.revenue(opt_price)
    print(f"Optimal uniform price: {opt_price}")
    print(f"Uniform pricing revenue: {uniform_revenue}")

    # 检查F = {4, 5}是否可行
    price_set_1 = [4, 5]
    is_f1_feasible = is_feasible(market, price_set_1, debug=debug)
    print(f"F = {price_set_1} is feasible: {is_f1_feasible}")

    results = {}

    if is_f1_feasible:
        # 应用PassivePSMax算法
        scheme = passive_ps_max(market, price_set_1)

        # 计算结果
        cs = scheme.consumer_surplus()
        ps = scheme.producer_surplus()
        sw = scheme.social_welfare()

        print(f"Consumer surplus: {cs}")
        print(f"Producer surplus: {ps}")
        print(f"Social welfare: {sw}")

        # 可视化
        if show_plots:
            plot_market(market, title="Market in Example 1.2")
            plot_surplus_triangle(market, scheme, price_set_1,
                                  title=f"Surplus Triangle for Example 1.2 with F={price_set_1}")

        results["scheme"] = scheme
        results["consumer_surplus"] = cs
        results["producer_surplus"] = ps
        results["social_welfare"] = sw

    # 检查F = {4}是否可行
    price_set_2 = [4]
    is_f2_feasible = is_feasible(market, price_set_2)
    print(f"F = {price_set_2} is feasible: {is_f2_feasible}")

    # 返回结果
    results.update({
        "market": market,
        "opt_price": opt_price,
        "uniform_revenue": uniform_revenue,
        "F1": price_set_1,
        "F1_feasible": is_f1_feasible,
        "F2": price_set_2,
        "F2_feasible": is_f2_feasible
    })

    return results


if __name__ == "__main__":
    run_example_1_2()