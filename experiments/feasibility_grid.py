
import numpy as np
from algorithms.market import Market
from algorithms.feasibility import is_feasible
from utils.visualization import plot_feasibility_heatmap


def generate_feasibility_grid(market, debug=False):
    """
    构建 n x n 布尔矩阵，表示每个可能的价格区间是否可行

    参数:
    market: Market 对象
    debug: 是否输出调试信息

    返回:
    n x n 布尔矩阵，(i, j) 表示 [values[i], values[j]] 是否为可行区间
    """
    values = market.values
    n = len(values)
    grid = np.full((n, n), False)

    for i in range(n):
        for j in range(i, n):
            F = list(values[i:j + 1])
            try:
                feasible = is_feasible(market, F, debug=debug)
                grid[i, j] = feasible
                if debug:
                    print(f"F = {F}, feasible = {feasible}")
            except Exception as e:
                if debug:
                    print(f"F = {F} raised error: {e}")
                grid[i, j] = False

    return grid


def run_feasibility_grid_experiment(values, masses, debug=False, save_path=None):
    """
    构建市场并运行可行性扫描 + 可视化热图

    参数:
    values: 价值点列表
    masses: 对应质量列表
    debug: 是否调试
    save_path: 图片保存路径（可选）
    """
    market = Market(values, masses)
    result_grid = generate_feasibility_grid(market, debug=debug)
    plot_feasibility_heatmap(values, result_grid, save_path=save_path)
    return result_grid
