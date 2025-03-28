
import numpy as np
import matplotlib.pyplot as plt
import os

from algorithms.market import Market
from experiments.feasibility_grid import generate_feasibility_grid
from utils.market_generator import MarketGenerator
from utils.feasibility_analysis import analyze_feasibility_matrix

def run_parameter_sweep(distribution, param_grid, fixed_args=None, label="sweep", save_dir="outputs"):
    """
    运行分布参数扫描实验，自动分析可行性结构并生成趋势图

    参数:
        distribution: str，分布类型，支持 "normal", "powerlaw", "binary"
        param_grid: list，扫描参数值
        fixed_args: dict，固定参数（如 normal 中的 mu）
        label: str，实验标签
        save_dir: str，结果保存目录
    """
    results = []

    for param in param_grid:
        if distribution == "normal":
            args = fixed_args or {}
            values, masses = MarketGenerator.truncated_normal(sigma=param, **args)
        elif distribution == "powerlaw":
            args = fixed_args or {}
            values, masses = MarketGenerator.powerlaw(alpha=param, **args)
        elif distribution == "binary":
            args = fixed_args or {}
            values, masses = MarketGenerator.binary(p=param, **args)
        else:
            raise ValueError("Unsupported distribution type")

        market = Market(values, masses)
        matrix = generate_feasibility_grid(market)
        metrics = analyze_feasibility_matrix(values, matrix, verbose=False)
        results.append(metrics)

    # 提取各指标序列
    counts = [r["count"] for r in results]
    min_widths = [r["min_width"] if r["min_width"] is not None else 0 for r in results]
    avg_widths = [r["avg_width"] if r["avg_width"] is not None else 0 for r in results]
    conn_ratios = [r["row_connected_ratio"] for r in results]

    # 绘图
    os.makedirs(save_dir, exist_ok=True)
    x = param_grid
    plt.figure(figsize=(10, 6))
    plt.plot(x, counts, label="Feasible Count")
    plt.plot(x, min_widths, label="Min Width")
    plt.plot(x, avg_widths, label="Avg Width")
    plt.plot(x, conn_ratios, label="Row Connected Ratio")
    plt.xlabel("Parameter")
    plt.ylabel("Metric")
    plt.title(f"Feasibility Metrics vs Parameter ({distribution})")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"{distribution}_{label}.png")
    plt.savefig(plot_path)
    plt.close()

    return results
