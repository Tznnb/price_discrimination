# experiments/sensitivity_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
from itertools import product

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

        # 1. 二项分布敏感性分析 - 变化p值
        binary_results = self._analyze_binary_sensitivity(save_dir)
        results["binary"] = binary_results

        # 2. 截断正态分布敏感性分析 - 变化sigma和mu
        normal_results = self._analyze_normal_sensitivity(save_dir)
        results["truncated_normal"] = normal_results

        # 3. 多项分布敏感性分析 (选择性，如果需要)
        # multinomial_results = self._analyze_multinomial_sensitivity(save_dir)
        # results["multinomial"] = multinomial_results

        return results

    def _analyze_binary_sensitivity(self, save_dir):
        """分析二项分布参数敏感性"""
        print("\n===== 二项分布敏感性分析 =====")

        # 变化p值从0.05到0.95
        p_values = np.linspace(0.05, 0.95, 10)

        # 选择3组代表性F值
        F_values = [
            [5, 5],  # 单点F
            [3, 7],  # 中等宽度F
            [1, 10]  # 全范围F
        ]

        results = []

        # 为热图准备数据
        passive_data = {F_str: [] for F_str in [str(F) for F in F_values]}
        active_data = {F_str: [] for F_str in [str(F) for F in F_values]}
        ratio_data = {F_str: [] for F_str in [str(F) for F in F_values]}

        for p in p_values:
            print(f"\n--- p = {p:.2f} ---")
            values, masses = self.market_generator.binary(p=p)

            # 创建市场和可视化器
            market = Market(values, masses)
            visualizer = TriangleVisualizer(np.array(masses), np.array(values))

            for F in F_values:
                F_str = str(F)
                is_feasible = visualizer.check_F_feasibility(F)
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)

                passive_area = features["passive_intermediary"]["area"]
                active_area = features["active_intermediary"]["area"]

                # 计算比率，避免除零错误
                ratio = passive_area / active_area if active_area > 0 else float('inf')
                if ratio == float('inf'):
                    # 用一个大数代替无穷
                    ratio = 1000

                print(f"F={F}, Feasible: {is_feasible}, "
                      f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                      f"Ratio: {ratio if ratio < 1000 else 'Inf'}")

                results.append({
                    "p": p,
                    "F": F,
                    "is_feasible": is_feasible,
                    "passive_area": passive_area,
                    "active_area": active_area,
                    "area_ratio": ratio
                })

                # 添加到热图数据
                passive_data[F_str].append(passive_area)
                active_data[F_str].append(active_area)
                ratio_data[F_str].append(ratio)

        # 创建热图
        self._create_heatmaps(passive_data, active_data, ratio_data, p_values,
                              "binary_p", "Binary Distribution - p value", save_dir)

        # 创建敏感性曲线图
        self._create_sensitivity_curves(results, "p", "binary", save_dir)

        return results

    def _analyze_normal_sensitivity(self, save_dir):
        """分析截断正态分布参数敏感性"""
        print("\n===== 截断正态分布敏感性分析 =====")

        # 参数组合
        mu_values = [3, 5, 7]
        sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5]

        # 固定其他参数
        n_points = 5
        low, high = 1, 10

        # 选择3组代表性F值
        F_values = [
            [5, 5],  # 单点F
            [3, 7],  # 中等宽度F
            [1, 10]  # 全范围F
        ]

        results = []

        # 为mu热图准备数据
        for sigma in sigma_values:
            print(f"\n--- sigma = {sigma:.1f} ---")

            # 为热图准备数据
            passive_data = {F_str: [] for F_str in [str(F) for F in F_values]}
            active_data = {F_str: [] for F_str in [str(F) for F in F_values]}
            ratio_data = {F_str: [] for F_str in [str(F) for F in F_values]}

            for mu in mu_values:
                print(f"  mu = {mu}")
                values, masses = self.market_generator.truncated_normal(
                    mu=mu, sigma=sigma, n=n_points, low=low, high=high)

                # 创建市场和可视化器
                market = Market(values, masses)
                visualizer = TriangleVisualizer(np.array(masses), np.array(values))

                for F in F_values:
                    F_str = str(F)
                    is_feasible = visualizer.check_F_feasibility(F)
                    features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)

                    passive_area = features["passive_intermediary"]["area"]
                    active_area = features["active_intermediary"]["area"]

                    # 计算比率，避免除零错误
                    ratio = passive_area / active_area if active_area > 0 else float('inf')
                    if ratio == float('inf'):
                        # 用一个大数代替无穷
                        ratio = 1000

                    print(f"    F={F}, Feasible: {is_feasible}, "
                          f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
                          f"Ratio: {ratio if ratio < 1000 else 'Inf'}")

                    results.append({
                        "mu": mu,
                        "sigma": sigma,
                        "F": F,
                        "is_feasible": is_feasible,
                        "passive_area": passive_area,
                        "active_area": active_area,
                        "area_ratio": ratio
                    })

                    # 添加到热图数据
                    passive_data[F_str].append(passive_area)
                    active_data[F_str].append(active_area)
                    ratio_data[F_str].append(ratio)

            # 创建当前sigma的热图
            self._create_heatmaps(passive_data, active_data, ratio_data, mu_values,
                                  f"normal_mu_sigma{sigma}",
                                  f"Normal Distribution - mu (sigma={sigma})", save_dir)

        # 创建敏感性曲线图 - sigma
        for mu in mu_values:
            # 筛选当前mu的数据
            mu_results = [r for r in results if r["mu"] == mu]
            self._create_sensitivity_curves(mu_results, "sigma", f"normal_mu{mu}", save_dir)

        return results

    def _create_heatmaps(self, passive_data, active_data, ratio_data, x_values,
                         name_suffix, title_prefix, save_dir):
        """创建热图"""
        # 转换数据为DataFrame
        passive_df = pd.DataFrame(passive_data, index=x_values)
        active_df = pd.DataFrame(active_data, index=x_values)
        ratio_df = pd.DataFrame(ratio_data, index=x_values)

        # 对比度颜色映射
        cmap = "viridis"

        # 创建热图 - 被动中介
        plt.figure(figsize=(12, 8))
        sns.heatmap(passive_df, cmap=cmap, annot=True, fmt=".2f")
        plt.title(f"{title_prefix} - Passive Intermediary Area")
        plt.xlabel("F Value")
        plt.ylabel("Parameter Value")

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / f"heatmap_passive_{name_suffix}.png", dpi=300)
        plt.close()

        # 创建热图 - 主动中介
        plt.figure(figsize=(12, 8))
        sns.heatmap(active_df, cmap=cmap, annot=True, fmt=".2f")
        plt.title(f"{title_prefix} - Active Intermediary Area")
        plt.xlabel("F Value")
        plt.ylabel("Parameter Value")

        if save_dir:
            plt.savefig(save_path / f"heatmap_active_{name_suffix}.png", dpi=300)
        plt.close()

        # 创建热图 - 比率
        plt.figure(figsize=(12, 8))
        sns.heatmap(ratio_df, cmap="coolwarm", annot=True, fmt=".2f", vmax=10)
        plt.title(f"{title_prefix} - Passive/Active Area Ratio")
        plt.xlabel("F Value")
        plt.ylabel("Parameter Value")

        if save_dir:
            plt.savefig(save_path / f"heatmap_ratio_{name_suffix}.png", dpi=300)
        plt.close()

    def _create_sensitivity_curves(self, results, param_name, dist_name, save_dir):
        """创建敏感性曲线"""
        # 按F值分组
        F_grouped = {}
        for r in results:
            F_str = str(r["F"])
            if F_str not in F_grouped:
                F_grouped[F_str] = []
            F_grouped[F_str].append(r)

        # 创建曲线图
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # 面积图
        for F_str, F_results in F_grouped.items():
            param_values = [r[param_name] for r in F_results]
            passive_areas = [r["passive_area"] for r in F_results]
            active_areas = [r["active_area"] for r in F_results]

            axes[0].plot(param_values, passive_areas, 'ro-', label=f"Passive ({F_str})")
            axes[0].plot(param_values, active_areas, 'bo-', label=f"Active ({F_str})")

        axes[0].set_title(f"Triangle Area vs {param_name}")
        axes[0].set_xlabel(param_name)
        axes[0].set_ylabel("Area")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 比率图
        for F_str, F_results in F_grouped.items():
            param_values = [r[param_name] for r in F_results]
            ratios = [min(r["area_ratio"], 10) for r in F_results]  # 限制最大值为10，以便可视化

            axes[1].plot(param_values, ratios, 'go-', label=f"Ratio ({F_str})")

        axes[1].set_title(f"Passive/Active Area Ratio vs {param_name}")
        axes[1].set_xlabel(param_name)
        axes[1].set_ylabel("Ratio (max 10)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # 可行性图
        for F_str, F_results in F_grouped.items():
            param_values = [r[param_name] for r in F_results]
            feasibility = [1 if r["is_feasible"] else 0 for r in F_results]

            axes[2].plot(param_values, feasibility, 's-', label=f"F={F_str}")

        axes[2].set_title(f"F Feasibility vs {param_name}")
        axes[2].set_xlabel(param_name)
        axes[2].set_ylabel("Feasible (1) / Not Feasible (0)")
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["Not Feasible", "Feasible"])
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / f"sensitivity_{dist_name}_{param_name}.png", dpi=300)


if __name__ == "__main__":
    # run_sensitivity_analysis.py

    from pathlib import Path
    from experiments.sensitivity_analysis import SensitivityAnalysis

    # 创建结果目录
    result_dir = Path("results/sensitivity_analysis")
    result_dir.mkdir(exist_ok=True, parents=True)

    # 运行敏感性分析
    experiment = SensitivityAnalysis()
    results = experiment.run_experiment(save_dir=result_dir)

    print("敏感性分析完成！结果保存在:", result_dir)