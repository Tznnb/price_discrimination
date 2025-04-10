# experiments/sensitivity_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib as mpl

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.triangle_visualizer import TriangleVisualizer
from utils.market_generator import MarketGenerator
from algorithms.market import Market


class SensitivityAnalysis:
    """敏感性分析：研究分布参数变化对三角形特性的影响"""

    def __init__(self):
        self.market_generator = MarketGenerator()

    def run_experiment(self, save_dir=None):
        """
        运行敏感性分析实验

        参数:
        save_dir: 保存结果的目录

        返回:
        实验结果字典
        """
        results = {}

        # 1. 均匀分布敏感性分析
        uniform_results = self._analyze_uniform_sensitivity(save_dir)
        results["uniform"] = uniform_results

        # 2. 截断正态分布敏感性分析
        normal_results = self._analyze_normal_sensitivity(save_dir)
        results["truncated_normal"] = normal_results

        return results

    def _analyze_uniform_sensitivity(self, save_dir):
        """分析均匀分布参数敏感性"""
        print("\n===== 均匀分布敏感性分析 =====")

        # 变化分段数量
        n_values = [3, 5, 7, 9]

        # 选择代表性F值
        F_values = [
            [5, 5],  # 中价点
            [3, 7],  # 中等宽度F
            [1, 10]  # 全范围F
        ]

        results = []

        # 创建结果存储结构
        passive_areas = {str(F): [] for F in F_values}
        active_areas = {str(F): [] for F in F_values}
        ratios = {str(F): [] for F in F_values}
        feasible_flags = {str(F): [] for F in F_values}

        for n in n_values:
            print(f"\n--- n = {n} ---")
            values, masses = self.market_generator.uniform(n=n, low=1, high=10)

            # 计算统一最优价格
            market = Market(values, masses)
            optimal_prices = market.optimal_price()
            uniform_price = optimal_prices[0] if optimal_prices else None
            print(f"    统一最优价格: {uniform_price}")

            # 创建可视化器
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # 安全计算比率
                if active_area < 1e-6:
                    ratio = 50.0 if passive_area > 0 else 1.0
                else:
                    ratio = passive_area / active_area

                ratio_str = f"{ratio:.4f}" if ratio < 50 else ">50"
                print(f"    F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                      f"Ratio: {ratio_str}"
                      + (" (近似计算)" if not is_feasible else ""))

                # 存储结果
                results.append({
                    "distribution": "uniform",
                    "param_name": "n",
                    "param_value": n,
                    "F": F,
                    "uniform_price": float(uniform_price),
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio
                })

                F_str = str(F)
                passive_areas[F_str].append(passive_area)
                active_areas[F_str].append(active_area)
                ratios[F_str].append(ratio)
                feasible_flags[F_str].append(is_feasible)

        # 创建可视化
        self._create_sensitivity_plots(
            n_values, passive_areas, active_areas, ratios, feasible_flags,
            "uniform_n", "Uniform Distribution - Parameter n", "n", save_dir
        )

        # 创建热图
        self._create_heatmaps(
            passive_areas, active_areas, ratios, feasible_flags, n_values,
            "uniform_n_heatmap", "Uniform Distribution - Parameter n", save_dir
        )

        return results

    def _analyze_normal_sensitivity(self, save_dir):
        """分析截断正态分布参数敏感性"""
        print("\n===== 截断正态分布敏感性分析 =====")

        # 固定均值，变化标准差
        sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        mu = 5  # 固定均值为5

        # 选择代表性F值
        F_values = [
            [5.5, 5.5],  # 均值点
            [3, 7],  # 中等宽度F
            [1, 10]  # 全范围F
        ]

        results = []

        # 创建结果存储结构
        passive_areas = {str(F): [] for F in F_values}
        active_areas = {str(F): [] for F in F_values}
        ratios = {str(F): [] for F in F_values}
        feasible_flags = {str(F): [] for F in F_values}

        for sigma in sigma_values:
            print(f"\n--- sigma = {sigma:.1f}, mu = {mu} ---")
            values, masses = self.market_generator.truncated_normal(
                mu=mu, sigma=sigma, n=5, low=1, high=10)

            # 计算统一最优价格
            market = Market(values, masses)
            optimal_prices = market.optimal_price()
            uniform_price = optimal_prices[0] if optimal_prices else None
            print(f"    统一最优价格: {uniform_price}")

            # 创建可视化器
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # 安全计算比率
                if active_area < 1e-6:
                    ratio = 50.0 if passive_area > 0 else 1.0
                else:
                    ratio = passive_area / active_area

                ratio_str = f"{ratio:.4f}" if ratio < 50 else ">50"
                print(f"    F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                      f"Ratio: {ratio_str}"
                      + (" (近似计算)" if not is_feasible else ""))

                # 存储结果
                results.append({
                    "distribution": "truncated_normal",
                    "param_name": "sigma",
                    "param_value": sigma,
                    "mu": mu,
                    "F": F,
                    "uniform_price": float(uniform_price),
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio
                })

                F_str = str(F)
                passive_areas[F_str].append(passive_area)
                active_areas[F_str].append(active_area)
                ratios[F_str].append(ratio)
                feasible_flags[F_str].append(is_feasible)

        # 创建可视化
        self._create_sensitivity_plots(
            sigma_values, passive_areas, active_areas, ratios, feasible_flags,
            "normal_sigma", f"Normal Distribution - Parameter sigma (mu={mu})", "sigma", save_dir
        )

        # 创建热图
        self._create_heatmaps(
            passive_areas, active_areas, ratios, feasible_flags, sigma_values,
            "normal_sigma_heatmap", f"Normal Distribution - Parameter sigma (mu={mu})", save_dir
        )

        # 固定标准差，变化均值
        mu_values = [3, 4, 5, 6, 7]
        sigma = 1.5  # 固定标准差为1.5

        # 重置结果存储
        passive_areas = {str(F): [] for F in F_values}
        active_areas = {str(F): [] for F in F_values}
        ratios = {str(F): [] for F in F_values}
        feasible_flags = {str(F): [] for F in F_values}

        for mu in mu_values:
            print(f"\n--- mu = {mu}, sigma = {sigma} ---")
            values, masses = self.market_generator.truncated_normal(
                mu=mu, sigma=sigma, n=5, low=1, high=10)

            # 计算统一最优价格
            market = Market(values, masses)
            optimal_prices = market.optimal_price()
            uniform_price = optimal_prices[0] if optimal_prices else None
            print(f"    统一最优价格: {uniform_price}")

            # 创建可视化器
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            for F in F_values:
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # 安全计算比率
                if active_area < 1e-6:
                    ratio = 50.0 if passive_area > 0 else 1.0
                else:
                    ratio = passive_area / active_area

                ratio_str = f"{ratio:.4f}" if ratio < 50 else ">50"
                print(f"    F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                      f"Ratio: {ratio_str}"
                      + (" (近似计算)" if not is_feasible else ""))

                # 存储结果
                results.append({
                    "distribution": "truncated_normal",
                    "param_name": "mu",
                    "param_value": mu,
                    "sigma": sigma,
                    "F": F,
                    "uniform_price": float(uniform_price),
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio
                })

                F_str = str(F)
                passive_areas[F_str].append(passive_area)
                active_areas[F_str].append(active_area)
                ratios[F_str].append(ratio)
                feasible_flags[F_str].append(is_feasible)

        # 创建可视化
        self._create_sensitivity_plots(
            mu_values, passive_areas, active_areas, ratios, feasible_flags,
            "normal_mu", f"Normal Distribution - Parameter mu (sigma={sigma})", "mu", save_dir
        )

        # 创建热图
        self._create_heatmaps(
            passive_areas, active_areas, ratios, feasible_flags, mu_values,
            "normal_mu_heatmap", f"Normal Distribution - Parameter mu (sigma={sigma})", save_dir
        )

        return results

    def _create_sensitivity_plots(self, param_values, passive_areas, active_areas, ratios, feasible_flags,
                                  name_suffix, title_prefix, param_name, save_dir):
        """创建简化版的敏感性图表"""

        # 1. 面积对比图
        plt.figure(figsize=(10, 6))

        for F_str in sorted(passive_areas.keys()):
            # 使用不同的线型和标记区分不同的F值
            if F_str == "[5, 5]" or F_str == "[5.5, 5.5]":
                p_style, a_style = 'ro-', 'bo-'  # 单点F值使用圆形标记
            elif F_str == "[3, 7]":
                p_style, a_style = 'rs-', 'bs-'  # 中等宽度F使用方形标记
            else:
                p_style, a_style = 'r^-', 'b^-'  # 全范围F使用三角形标记

            # 增加标记大小，线宽保持默认，以便标记可以更清晰地显示
            plt.plot(param_values, passive_areas[F_str], p_style, label=f"Passive (F={F_str})", markersize=8)
            plt.plot(param_values, active_areas[F_str], a_style, label=f"Active (F={F_str})", markersize=8)

            # 标记不可行的点
            for i, is_feasible in enumerate(feasible_flags[F_str]):
                if not is_feasible:
                    plt.plot(param_values[i], passive_areas[F_str][i], 'rx', markersize=10)
                    plt.plot(param_values[i], active_areas[F_str][i], 'bx', markersize=10)

        plt.title(f"{title_prefix} - Area Comparison")
        plt.xlabel(param_name)
        plt.ylabel("Triangle Area")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / f"{name_suffix}_areas.png", dpi=300)
        plt.close()

        # 2. 比率图 (被动/主动)
        plt.figure(figsize=(10, 6))

        for F_str in sorted(ratios.keys()):
            # 使用与面积图相同的线型和标记，但颜色改为绿色
            if F_str == "[5, 5]" or F_str == "[5.5, 5.5]":
                style = 'go-'  # 单点F值
            elif F_str == "[3, 7]":
                style = 'gs-'  # 中等宽度F
            else:
                style = 'g^-'  # 全范围F

            # 限制比率最大值为5，方便可视化
            limited_ratios = [min(r, 5) for r in ratios[F_str]]
            plt.plot(param_values, limited_ratios, style, label=f"Ratio (F={F_str})", markersize=8)

            # 标记不可行的点
            for i, is_feasible in enumerate(feasible_flags[F_str]):
                if not is_feasible:
                    plt.plot(param_values[i], min(ratios[F_str][i], 5), 'rx', markersize=10)

        plt.title(f"{title_prefix} - Passive/Active Area Ratio (max=5)")
        plt.xlabel(param_name)
        plt.ylabel("Area Ratio (Passive/Active)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dir:
            plt.savefig(save_path / f"{name_suffix}_ratios.png", dpi=300)
        plt.close()

        # 3. 可行性图
        plt.figure(figsize=(10, 6))

        for F_str in sorted(feasible_flags.keys()):
            # 使用不同的线型和颜色
            if F_str == "[5, 5]" or F_str == "[5.5, 5.5]":
                style = 'mo-'  # 单点F值，紫色
            elif F_str == "[3, 7]":
                style = 'cs-'  # 中等宽度F，青色
            else:
                style = 'y^-'  # 全范围F，黄色

            # 将布尔值转换为0/1
            feasible_values = [1 if f else 0 for f in feasible_flags[F_str]]
            plt.plot(param_values, feasible_values, style, label=f"F={F_str}", markersize=8)

        plt.title(f"{title_prefix} - F Value Feasibility")
        plt.xlabel(param_name)
        plt.ylabel("Feasible(1) / Not Feasible(0)")
        plt.yticks([0, 1], ["Not Feasible", "Feasible"])
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dir:
            plt.savefig(save_path / f"{name_suffix}_feasibility.png", dpi=300)
        plt.close()

        # 4. 总面积和比率汇总图
        plt.figure(figsize=(10, 6))

        # 计算每个参数值下所有可行F的总面积
        total_passive = []
        total_active = []
        total_ratios = []
        feasible_counts = []

        for i, param in enumerate(param_values):
            p_sum = sum(passive_areas[F_str][i] for F_str in passive_areas
                        if feasible_flags[F_str][i])
            a_sum = sum(active_areas[F_str][i] for F_str in active_areas
                        if feasible_flags[F_str][i])

            total_passive.append(p_sum)
            total_active.append(a_sum)

            if a_sum > 1e-6:
                total_ratios.append(p_sum / a_sum)
            else:
                total_ratios.append(0)

            feasible_counts.append(sum(1 for F_str in feasible_flags
                                       if feasible_flags[F_str][i]))

        width = (param_values[-1] - param_values[0]) / (len(param_values) * 3)

        # 绘制总面积柱状图
        plt.bar(param_values, total_passive, width=width, color='red', alpha=0.6, label='Total Passive Area')
        plt.bar([p + width for p in param_values], total_active, width=width, color='blue', alpha=0.6,
                label='Total Active Area')

        # 绘制总比率线
        ax2 = plt.twinx()
        ax2.plot([p + width * 0.5 for p in param_values], total_ratios, 'g-', marker='o', label='Total Ratio')
        ax2.set_ylabel('Total Ratio (Passive/Active)')

        plt.title(f"{title_prefix} - Total Areas and Ratio")
        plt.xlabel(param_name)
        plt.ylabel('Area')
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        if save_dir:
            plt.savefig(save_path / f"{name_suffix}_total_summary.png", dpi=300)
        plt.close()

    def _create_heatmaps(self, passive_areas, active_areas, ratios, feasible_flags, param_values,
                         name_suffix, title_prefix, save_dir):
        """创建热力图可视化参数敏感性"""

        # 转换数据为DataFrame格式
        F_values = sorted(passive_areas.keys())

        # 创建热力图矩阵
        passive_matrix = np.zeros((len(F_values), len(param_values)))
        active_matrix = np.zeros((len(F_values), len(param_values)))
        ratio_matrix = np.zeros((len(F_values), len(param_values)))
        feasible_matrix = np.zeros((len(F_values), len(param_values)), dtype=bool)

        for i, F_str in enumerate(F_values):
            for j, _ in enumerate(param_values):
                passive_matrix[i, j] = passive_areas[F_str][j]
                active_matrix[i, j] = active_areas[F_str][j]
                ratio_matrix[i, j] = min(ratios[F_str][j], 5)  # 限制最大比率为5
                feasible_matrix[i, j] = feasible_flags[F_str][j]

        # 1. 被动中介面积热力图
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # 创建热力图
        sns.heatmap(passive_matrix, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=[f"{p}" for p in param_values],
                    yticklabels=F_values,
                    ax=ax)

        # 标记不可行的单元格
        for i in range(len(F_values)):
            for j in range(len(param_values)):
                if not feasible_matrix[i, j]:
                    # 不可行的单元格用红框标记
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.title(f"{title_prefix} - Passive Area")
        plt.xlabel("Parameter Value")
        plt.ylabel("F Value")

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / f"{name_suffix}_passive.png", dpi=300)
        plt.close()

        # 2. 主动中介面积热力图
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # 创建热力图
        sns.heatmap(active_matrix, annot=True, fmt=".2f", cmap="plasma",
                    xticklabels=[f"{p}" for p in param_values],
                    yticklabels=F_values,
                    ax=ax)

        # 标记不可行的单元格
        for i in range(len(F_values)):
            for j in range(len(param_values)):
                if not feasible_matrix[i, j]:
                    # 不可行的单元格用红框标记
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.title(f"{title_prefix} - Active Area")
        plt.xlabel("Parameter Value")
        plt.ylabel("F Value")

        if save_dir:
            plt.savefig(save_path / f"{name_suffix}_active.png", dpi=300)
        plt.close()

        # 3. 面积比热力图
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # 创建热力图
        sns.heatmap(ratio_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=[f"{p}" for p in param_values],
                    yticklabels=F_values,
                    ax=ax)

        # 标记不可行的单元格
        for i in range(len(F_values)):
            for j in range(len(param_values)):
                if not feasible_matrix[i, j]:
                    # 不可行的单元格用红框标记
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.title(f"{title_prefix} - Area Ratio (Passive/Active, max=5)")
        plt.xlabel("Parameter Value")
        plt.ylabel("F Value")

        if save_dir:
            plt.savefig(save_path / f"{name_suffix}_ratio.png", dpi=300)
        plt.close()

        # 4. 可行性热力图
        plt.figure(figsize=(10, 6))

        # 创建热力图
        sns.heatmap(feasible_matrix.astype(int), annot=True, fmt="d", cmap="YlGn",
                    xticklabels=[f"{p}" for p in param_values],
                    yticklabels=F_values)

        plt.title(f"{title_prefix} - F Value Feasibility")
        plt.xlabel("Parameter Value")
        plt.ylabel("F Value")

        if save_dir:
            plt.savefig(save_path / f"{name_suffix}_feasibility_heatmap.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    # 当前用户和时间信息
    current_user = "Tznnb"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"敏感性分析实验 - 用户: {current_user} - 时间: {current_time}")

    parser = argparse.ArgumentParser(description='运行敏感性分析实验')
    parser.add_argument('--save-dir', type=str, default='results/sensitivity_analysis',
                        help='结果保存目录')
    parser.add_argument('--uniform-only', action='store_true',
                        help='只运行均匀分布分析')
    parser.add_argument('--normal-only', action='store_true',
                        help='只运行正态分布分析')

    args = parser.parse_args()

    # 创建结果目录
    result_dir = Path(args.save_dir)
    result_dir.mkdir(exist_ok=True, parents=True)

    # 实例化实验类
    experiment = SensitivityAnalysis()

    # 根据参数决定运行哪些分析
    if args.uniform_only:
        uniform_results = experiment._analyze_uniform_sensitivity(result_dir)
        print(f"\n均匀分布敏感性分析完成！结果保存在: {result_dir}")
    elif args.normal_only:
        normal_results = experiment._analyze_normal_sensitivity(result_dir)
        print(f"\n正态分布敏感性分析完成！结果保存在: {result_dir}")
    else:
        # 运行所有分析
        results = experiment.run_experiment(save_dir=result_dir)
        print(f"\n敏感性分析完成！结果保存在: {result_dir}")

    print("\n提示: 热力图中红框标记的单元格表示F值不可行")