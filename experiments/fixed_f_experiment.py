# experiments/fixed_f_experiment.py

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 配置matplotlib支持中文显示（如果需要中文）
import matplotlib

# 方案1：切换到英文界面
USE_ENGLISH = True  # 设为False则使用中文界面

if not USE_ENGLISH:
    # 方案2：设置中文字体支持
    try:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except:
        print("警告：无法设置中文字体，将使用英文界面")
        USE_ENGLISH = True

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.triangle_visualizer import TriangleVisualizer
from utils.market_generator import MarketGenerator
from algorithms.market import Market


class FixedFExperiment:
    """固定F值，变化分布的实验"""

    def __init__(self):
        self.market_generator = MarketGenerator()

    def run_experiment(self, F_values=None, n_points=5, save_dir=None):
        """
        运行固定F变化分布的实验

        参数:
        F_values: 要测试的F值列表，每个元素是一个列表[min, max]
        n_points: 生成的分布中的点数
        save_dir: 保存结果的目录，如果为None则不保存

        返回:
        实验结果字典
        """
        if F_values is None:
            F_values = [[4, 5], [1, 10]]

        # 要测试的分布列表
        distributions = [
            {"name": "uniform", "params": {}},
            {"name": "binary", "params": {"p": 0.5}},
            {"name": "truncated_normal", "params": {"mu": 5, "sigma": 1.5}},
            {"name": "geometric", "params": {"q": 0.5}},
            {"name": "powerlaw", "params": {"alpha": 2.0}}
        ]

        # 存储结果
        results = {}

        # 对每个F值运行实验
        for F in F_values:
            F_key = f"F={F}"
            results[F_key] = []

            # 创建包含多个子图的图表
            fig, axes = plt.subplots(len(distributions), 3, figsize=(18, 5 * len(distributions)))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)

            if USE_ENGLISH:
                fig_title = f"Triangle Features Comparison: Fixed F={F}"
            else:
                fig_title = f"三角形特性对比: 固定F={F}"

            fig.suptitle(fig_title, fontsize=16)

            # 对每个分布运行实验
            for i, dist in enumerate(distributions):
                dist_name = dist["name"]
                params = dist["params"]

                # 生成分布
                if dist_name == "uniform":
                    values, masses = self.market_generator.uniform(n=n_points, low=1, high=10)
                elif dist_name == "binary":
                    values, masses = self.market_generator.binary(**params)
                elif dist_name == "truncated_normal":
                    values, masses = self.market_generator.truncated_normal(n=n_points, **params)
                elif dist_name == "geometric":
                    values, masses = self.market_generator.geometric(n=n_points, **params)
                elif dist_name == "powerlaw":
                    values, masses = self.market_generator.powerlaw(n=n_points, **params)

                # 创建市场和可视化器
                market = Market(values, masses)
                visualizer = TriangleVisualizer(np.array(masses), np.array(values))

                # 检查F可行性
                is_feasible_result = visualizer.check_F_feasibility(F)

                # 分析三角形特性，确保使用正确的算法
                features = visualizer.analyze_triangle_features(
                    F,
                    use_exact_algorithm=is_feasible_result
                )

                # 调试信息：打印三角形顶点和面积
                print(f"Debug - Dist: {dist_name}, F={F}, Feasible: {is_feasible_result}")
                for tri_type in ["no_regulation", "passive_intermediary", "active_intermediary"]:
                    area = features[tri_type]["area"]
                    min_vertex = min(v[0] for v in features[tri_type]["vertices"])
                    max_vertex = max(v[1] for v in features[tri_type]["vertices"])
                    print(f"  {tri_type}: Area={area:.4f}, Min X={min_vertex:.4f}, Max Y={max_vertex:.4f}")

                # 绘制三角形
                ax_triangle = axes[i, 0]
                visualizer.draw_triangles(F, ax=ax_triangle, fixed_axes=False)

                # 设置标题
                if USE_ENGLISH:
                    tri_title = f"{dist_name} (F Feasible: {'Yes' if is_feasible_result else 'No'})"
                else:
                    tri_title = f"{dist_name} (F可行: {'是' if is_feasible_result else '否'})"

                ax_triangle.set_title(tri_title)

                # 绘制市场分布
                ax_dist = axes[i, 1]
                ax_dist.bar(values, masses, width=0.3)

                if USE_ENGLISH:
                    dist_title = f"{dist_name} Distribution"
                    ax_dist.set_xlabel("Value")
                    ax_dist.set_ylabel("Mass")
                else:
                    dist_title = f"{dist_name} 分布"
                    ax_dist.set_xlabel("价值")
                    ax_dist.set_ylabel("质量")

                ax_dist.set_title(dist_title)
                ax_dist.grid(alpha=0.3)

                # 绘制三角形面积比较
                ax_area = axes[i, 2]

                # 确保面积非负
                areas = [
                    max(0, features["no_regulation"]["area"]),
                    max(0, features["passive_intermediary"]["area"]),
                    max(0, features["active_intermediary"]["area"])
                ]

                if USE_ENGLISH:
                    labels = ["No Regulation", "Passive", "Active"]
                else:
                    labels = ["无监管", "被动中介", "主动中介"]

                ax_area.bar(labels, areas, color=["gray", "red", "blue"])

                if USE_ENGLISH:
                    area_title = "Triangle Area Comparison"
                    ax_area.set_ylabel("Area")
                else:
                    area_title = "三角形面积对比"
                    ax_area.set_ylabel("面积")

                ax_area.set_title(area_title)

                # 记录结果
                result = {
                    "distribution": dist_name,
                    "params": params,
                    "values": values,
                    "masses": masses,
                    "is_feasible": is_feasible_result,
                    "features": features
                }
                results[F_key].append(result)

            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # 保存结果
            if save_dir is not None:
                save_path = Path(save_dir)
                save_path.mkdir(exist_ok=True, parents=True)
                plt.savefig(save_path / f"fixed_f_{F[0]}_{F[1]}.svg", dpi=300)

        # 生成汇总比较图
        self._create_summary_plot(results, F_values, save_dir)

        return results

    def _create_summary_plot(self, results, F_values, save_dir):
        """创建汇总比较图"""
        # 创建一个包含所有F值的图表
        fig, axes = plt.subplots(len(F_values), 1, figsize=(12, 6 * len(F_values)))
        if len(F_values) == 1:
            axes = [axes]

        for i, F in enumerate(F_values):
            F_key = f"F={F}"
            F_results = results[F_key]

            ax = axes[i]

            # 准备数据
            dist_names = [r["distribution"] for r in F_results]
            passive_areas = [max(0, r["features"]["passive_intermediary"]["area"]) for r in F_results]
            active_areas = [max(0, r["features"]["active_intermediary"]["area"]) for r in F_results]
            no_reg_areas = [max(0, r["features"]["no_regulation"]["area"]) for r in F_results]

            # 分组条形图
            x = np.arange(len(dist_names))
            width = 0.25

            ax.bar(x - width, no_reg_areas, width, label="No Regulation" if USE_ENGLISH else "无监管", color="gray",
                   alpha=0.6)
            ax.bar(x, passive_areas, width, label="Passive" if USE_ENGLISH else "被动中介", color="red", alpha=0.6)
            ax.bar(x + width, active_areas, width, label="Active" if USE_ENGLISH else "主动中介", color="blue", alpha=0.6)

            if USE_ENGLISH:
                ax.set_title(f"Triangle Areas in Different Distributions (F={F})")
                ax.set_xlabel("Distribution Type")
                ax.set_ylabel("Triangle Area")
            else:
                ax.set_title(f"不同分布下的三角形面积对比 (F={F})")
                ax.set_xlabel("分布类型")
                ax.set_ylabel("三角形面积")

            ax.set_xticks(x)
            ax.set_xticklabels(dist_names)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / "summary_comparison.svg", dpi=300)


if __name__ == "__main__":
    # run_fixed_f_experiment.py

    import os
    from pathlib import Path
    from experiments.fixed_f_experiment import FixedFExperiment

    # 创建结果目录
    result_dir = Path("results/fixed_f_experiment")
    result_dir.mkdir(exist_ok=True, parents=True)

    # 定义要测试的F值
    F_values = [
        [4, 5],  # 典型区间
        [1, 10],  # 全范围
        [5, 5]  # 单点区间
    ]

    # 运行实验
    experiment = FixedFExperiment()
    results = experiment.run_experiment(
        F_values=F_values,
        n_points=5,
        save_dir=result_dir
    )

    print("实验完成！结果保存在:", result_dir)