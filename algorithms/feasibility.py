import pulp
import math

def is_feasible_lp(market, price_set, epsilon=1e-6, debug=False):
    """
    使用线性规划来检测价格集 F 的可行性（Passive Intermediary 情形）。
    若可行，则返回 True，否则返回 False。

    其中:
    - market: 具有 market.values (升序列表) 和 market.masses (同长度列表) 的简单类/结构
    - price_set: 可用价格列表 F
    - epsilon: 允许在“最优”对比中有少量数值容差
    - debug: 是否打印一些调试信息
    """

    values = market.values  # v_1, v_2, ..., v_n
    masses = market.masses  # x^*_1, x^*_2, ..., x^*_n
    n = len(values)
    m = len(price_set)

    # 若价格集为空，直接不可行
    if m == 0:
        if debug:
            print("[LP] price_set 为空，不可行")
        return False

    # 1) 创建 LP 问题 - 这里做可行性判定，所以令 Minimize 0
    # 用旧版 pulp 语法:
    prob = pulp.LpProblem("FeasibilityCheck", pulp.LpMinimize)
    # 给一个dummy的目标
    prob += 0, "dummy_objective"

    # 2) 定义 x_{i,j} 变量: 每个(价值 i, 段 j)对应的质量
    # 允许取值 [0, ∞)
    x_vars = pulp.LpVariable.dicts("x",(range(n), range(m)),
                                   lowBound=0, cat=pulp.LpContinuous)

    # 3) 质量守恒: 每个 i 的分配之和 = masses[i]
    for i in range(n):
        prob += (
            pulp.lpSum(x_vars[i][j] for j in range(m)) == masses[i],
            f"mass_allocation_for_value_{i}"
        )

    # 4) 卖家最优定价约束:
    #    对于第 j 段, 定价 = price_set[j], 需保证它不差于其它可能价 p'
    for j in range(m):
        pj = price_set[j]

        # 收益表达式: p_j * sum_{i : v_i >= p_j} x_{i,j}
        # pulp 表达式:
        revenue_pj = pj * pulp.lpSum(
            x_vars[i][j] for i in range(n) if values[i] >= pj
        )

        # 与其它价 p' 做比较
        # 如果只比较 price_set 内其它价:
        # for pprime in price_set:
        #     if abs(pprime - pj) < 1e-15:  # 或 == pj
        #         continue
        #     revenue_pprime = pprime * pulp.lpSum(
        #         x_vars[i][j] for i in range(n) if values[i] >= pprime
        #     )
        #     prob += revenue_pj >= revenue_pprime - epsilon, ...

        # 如果要和所有 v_i 都比:
        for pprime in values:
            if abs(pprime - pj) < 1e-15:
                continue  # 跳过同一个价
            revenue_pprime = pprime * pulp.lpSum(
                x_vars[i][j] for i in range(n) if values[i] >= pprime
            )
            # 要求 revenue_pj >= revenue_pprime - epsilon
            prob += (
                revenue_pj >= revenue_pprime - epsilon,
                f"optimality_j{j}_p{pj}_vs_{pprime}"
            )

    # 5) 求解
    # 默认 solver 是 CBC，若没有其他需求，直接 prob.solve() 即可
    solve_status = prob.solve(pulp.PULP_CBC_CMD(msg=debug))
    # solve_status 是一个 int, 可以用 pulp.LpStatus[solve_status] 查看文字

    # 6) 返回是否可行
    status_str = pulp.LpStatus[solve_status]
    if debug:
        print(f"[LP] Problem status code = {solve_status}, status = {status_str}")

    # "Optimal" 表示找到可行解并且把0的目标成功最小化(=0) —— 即可行
    # "Infeasible" / "Undefined" / "Unbounded" 等则不可行
    return (status_str == "Optimal")



def is_feasible(market, price_set, epsilon=1e-6, debug=False):
    """
    基于 LP 来检测可行性，若可行则返回 True，否则 False
    """
    # 0) 若要保留旧的 verify_example_1_2 测试，可先行尝试：
    from algorithms.direct_verify import verify_example_1_2
    if verify_example_1_2(market, price_set, debug):
        return True

    if debug:
        print(f"Checking feasibility of F = {price_set} with LP approach...")

    # 调用上面写的 LP 函数
    return is_feasible_lp(market, price_set, epsilon=epsilon, debug=debug)


# 老版本

#
# import numpy as np
#
# from algorithms.bbm import bbm
# from algorithms.transform_scheme import transform_scheme
# from algorithms.market import Market
# from algorithms.passive_ps_max import passive_ps_max
# from algorithms.direct_verify import verify_example_1_2
#
#
# def is_feasible(market, price_set, epsilon=1e-6, debug=False):
#     """
#     检测价格区间F是否可行
#
#     参数:
#     market: Market对象
#     price_set: 价格区间F
#     epsilon: 数值计算容差
#     debug: 是否显示调试信息
#
#     返回:
#     布尔值，指示F是否可行
#     """
#     # 首先尝试直接验证特定例子
#     if verify_example_1_2(market, price_set, debug):
#         return True
#
#     # 如果直接验证失败，继续使用算法尝试
#     if debug:
#         print(f"\nChecking feasibility of F = {price_set}")
#         print(f"Market: {market}")
#
#     # 过滤掉不在市场值域中的价格
#     price_set = sorted([p for p in price_set if p in market.values])
#
#     if not price_set:
#         if debug:
#             print("No valid prices in F")
#         return False
#
#     # 应用PassivePSMax算法
#     scheme = passive_ps_max(market, price_set, debug=debug)
#
#     # 若失败，尝试使用 BBM-based transform
#     if scheme is None:
#         if debug:
#             print("PassivePSMax returned None, trying BBM-based transform...")
#         bbm_scheme = bbm(market)
#         transformed_scheme = transform_scheme(bbm_scheme, price_set)
#         scheme = transformed_scheme
#
#     if scheme is None:
#         if debug:
#             print("PassivePSMax still failed after transform, F is not feasible")
#         return False
#
#     # 验证每个市场段是否满足F-valid条件
#     for i, (segment, price) in enumerate(zip(scheme.segments, scheme.prices)):
#         # 确认价格在F中
#         if price not in price_set:
#             if debug:
#                 print(f"Segment {i} uses price {price} which is not in F")
#             return False
#
#         # 检查价格是否在最优价格集中
#         optimal_prices = segment.optimal_price()
#         if price not in optimal_prices:
#             if len(segment.values) == 1:
#                 if debug:
#                     print(f"Warning: single-point segment {i} uses suboptimal price {price}")
#                 continue  # 容忍单点segment非最优定价
#             if debug:
#                 print(f"Price {price} is not optimal for segment {i}")
#             return False
#
#     if debug:
#             print(f"Segment {i} with price {price} is valid")
#
#     if debug:
#         print("All checks passed, F is feasible!")
#
#     return True
#
#
# if __name__ == "__main__":
#     values = [1, 10]
#     masses = [0.5, 0.5]
#     market = Market(values, masses)
#
#     F1 = [1, 10]
#     F2 = [1]
#     F3 = [10]
#
#     print("F =", F1, "→", is_feasible(market, F1, debug=True))
#     print("F =", F2, "→", is_feasible(market, F2, debug=True))
#     print("F =", F3, "→", is_feasible(market, F3, debug=True))
#
#     values = [1, 2, 4, 8, 16]
#     masses = [0.5161, 0.2581, 0.1290, 0.0645, 0.0323]
#     market = Market(values, masses)
#
#     F1 = [4, 8]
#     F2 = [2, 4]
#     F3 = [8, 16]
#     F4 = [1, 2, 4]
#
#     for F in [F1, F2, F3, F4]:
#         print("F =", F, "→", is_feasible(market, F, debug=True))
