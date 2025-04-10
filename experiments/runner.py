#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行F值变化实验的脚本
用法:
    python run_varying_f_experiment.py

可选参数:
    --result-dir: 结果保存目录
    --distributions: 要测试的分布类型，逗号分隔（uniform,binary,truncated_normal）
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import numpy as np

from experiments.varying_f_experiment import VaryingFExperiment


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行F值变化实验')

    parser.add_argument('--result-dir', type=str, default='results/varying_f_experiment',
                        help='结果保存目录')

    parser.add_argument('--distributions', type=str, default='uniform,binary,truncated_normal',
                        help='要测试的分布类型，逗号分隔')

    parser.add_argument('--save-results', action='store_true',
                        help='是否保存结果为JSON格式')

    parser.add_argument('--show-plots', action='store_true',
                        help='是否显示图表')

    args = parser.parse_args()
    return args


def get_distribution_configs(dist_names):
    """获取分布配置"""
    all_configs = {
        "uniform": {"name": "uniform", "params": {"n": 5, "low": 1, "high": 10}},
        "binary": {"name": "binary", "params": {"p": 0.5, "low": 1, "high": 10}},
        "truncated_normal": {"name": "truncated_normal",
                             "params": {"mu": 5, "sigma": 1.5, "n": 5, "low": 1, "high": 10}},
        "geometric": {"name": "geometric", "params": {"start": 1, "ratio": 2, "n": 5, "q": 0.5}},
        "powerlaw": {"name": "powerlaw", "params": {"alpha": 2.0, "n": 5, "scale": 1.0}}
    }

    selected_configs = []
    for name in dist_names:
        if name.strip() in all_configs:
            selected_configs.append(all_configs[name.strip()])
        else:
            print(f"警告: 不支持的分布类型 '{name}'，将被忽略")

    return selected_configs


def save_experiment_metadata(result_dir, distributions):
    """保存实验元数据"""
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "distributions": distributions,
    }

    with open(Path(result_dir) / "experiment_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()

        # 创建结果目录
        result_dir = Path(args.result_dir)
        result_dir.mkdir(exist_ok=True, parents=True)

        # 获取分布配置
        dist_names = args.distributions.split(',')
        distributions = get_distribution_configs(dist_names)

        if not distributions:
            print("错误: 没有有效的分布类型！")
            return

        print(f"将测试以下分布: {[d['name'] for d in distributions]}")

        # 保存实验元数据
        save_experiment_metadata(result_dir, distributions)

        # 运行实验
        experiment = VaryingFExperiment()
        results = experiment.run_experiment(
            distributions=distributions,
            save_dir=result_dir
        )

        # 保存结果为JSON（只保存可序列化的部分）
        if args.save_results and results:
            serializable_results = {}
            for dist_name, result in results.items():
                serializable_results[dist_name] = {
                    "values": result.get("values", []),
                    "masses": result.get("masses", []),
                    "uniform_optimal_price": float(result.get("uniform_optimal_price", 0)),
                    # 保存F值和面积信息
                    "feasible_F": {
                        "width": [r["F"] for r in result.get("width_results", [])],
                        "position": [r["F"] for r in result.get("position_results", [])],
                        "optimal": [r["F"] for r in result.get("optimal_results", [])],
                        "micro": [r["F"] for r in result.get("micro_results", [])],
                    },
                    "triangle_areas": {
                        "width": [
                            {
                                "F": r["F"],
                                "passive": r["features"]["passive_intermediary"]["area"],
                                "active": r["features"]["active_intermediary"]["area"]
                            }
                            for r in result.get("width_results", [])
                        ],
                        "position": [
                            {
                                "F": r["F"],
                                "passive": r["features"]["passive_intermediary"]["area"],
                                "active": r["features"]["active_intermediary"]["area"]
                            }
                            for r in result.get("position_results", [])
                        ]
                    }
                }

            try:
                with open(result_dir / "experiment_results.json", 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)

                print(f"结果已保存到: {result_dir / 'experiment_results.json'}")
            except Exception as e:
                print(f"保存JSON结果时出错: {e}")

        # 显示图表
        if args.show_plots:
            plt.show()
        else:
            plt.close('all')

        print(f"\n实验完成！结果保存在: {result_dir}")

        # 生成统计摘要
        print("\n--- 实验统计摘要 ---")
        for dist_name, result in results.items():
            all_feasible_count = 0
            for key in ["width_results", "position_results", "optimal_results",
                        "micro_results", "dist_specific_results"]:
                all_feasible_count += len(result.get(key, []))

            candidate_counts = 0
            feasible_counts = 0

            for key in result.get("candidate_F_info", {}):
                candidate_counts += len(result["candidate_F_info"][key].get("candidates", []))
                feasible_counts += len(result["candidate_F_info"][key].get("feasible", []))

            print(f"\n分布: {dist_name}")
            print(f"  统一最优价格: {result.get('uniform_optimal_price')}")
            print(f"  候选F值总数: {candidate_counts}, 可行F值总数: {feasible_counts}")
            if candidate_counts > 0:
                print(f"  可行率: {feasible_counts / candidate_counts * 100:.2f}%")

            # 检查是否有有意义的三角形
            meaningful_results = [r for r in result.get("all_results", []) if r.get("meaningful", False)]

            if meaningful_results:
                passive_areas = [r["features"]["passive_intermediary"]["area"] for r in meaningful_results]
                active_areas = [r["features"]["active_intermediary"]["area"] for r in meaningful_results]

                print(f"  有意义的三角形数量: {len(meaningful_results)}")

                # 安全计算有效比率
                valid_ratios = []
                for p, a in zip(passive_areas, active_areas):
                    if a > 1e-6:
                        valid_ratios.append(p / a)
                    elif p > 1e-6:
                        valid_ratios.append(min(50.0, p * 10))  # 限制极端值

                # 计算总面积比
                total_passive = sum(passive_areas)
                total_active = sum(active_areas)
                total_ratio = total_passive / total_active if total_active > 1e-6 else 0

                if valid_ratios:
                    # 计算多种统计指标
                    median_ratio = np.median(valid_ratios)
                    mean_ratio = np.mean(valid_ratios)

                    print(f"  被动/主动三角形面积比:")
                    print(f"    - 总面积比: {total_ratio:.4f}")
                    print(f"    - 中位数比率: {median_ratio:.4f}")
                    print(f"    - 平均比率: {mean_ratio:.4f}")

            print(f"  生成的图表数量: {len(list(result_dir.glob(f'{dist_name}_*.png')))}")

    except Exception as e:
        print(f"实验运行时发生错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()


# # experiments/varying_f_experiment.py
#
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# from pathlib import Path
# from itertools import product
#
# # 确保项目根目录在路径中
# project_root = Path(__file__).parent.parent
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))
#
# from utils.triangle_visualizer import TriangleVisualizer
# from utils.market_generator import MarketGenerator
# from algorithms.market import Market
#
#
# class VaryingFExperiment:
#     """固定分布，变化F值的实验"""
#
#     def __init__(self):
#         self.market_generator = MarketGenerator()
#
#     def run_experiment(self, distributions=None, save_dir=None):
#         """
#         运行固定分布变化F的实验
#
#         参数:
#         distributions: 要测试的分布列表，每个元素是一个字典包含名称和参数
#         save_dir: 保存结果的目录
#
#         返回:
#         实验结果字典
#         """
#         if distributions is None:
#             distributions = [
#                 {"name": "uniform", "params": {"n": 5, "low": 1, "high": 10}},
#                 {"name": "binary", "params": {"p": 0.5, "low": 1, "high": 10}},
#                 {"name": "truncated_normal", "params": {"mu": 5, "sigma": 1.5, "n": 5, "low": 1, "high": 10}}
#             ]
#
#         results = {}
#
#         # 对每个分布运行F变化实验
#         for dist in distributions:
#             dist_name = dist["name"]
#             params = dist["params"]
#
#             print(f"\n===== 实验: {dist_name} 分布 =====")
#
#             # 生成市场
#             if dist_name == "uniform":
#                 values, masses = self.market_generator.uniform(**params)
#             elif dist_name == "binary":
#                 values, masses = self.market_generator.binary(**params)
#             elif dist_name == "truncated_normal":
#                 values, masses = self.market_generator.truncated_normal(**params)
#
#             # 创建市场和可视化器
#             market = Market(values, masses)
#             visualizer = TriangleVisualizer(np.array(masses), np.array(values))
#
#             # 获取统一最优价格和值域
#             uniform_optimal_price = visualizer.uniform_optimal_price
#             min_value, max_value = min(values), max(values)
#
#             # 系统性变化F值
#             # 1. 变化F宽度
#             F_widths = self._generate_f_widths(min_value, max_value, uniform_optimal_price, n=5)
#             width_results = self._analyze_f_variations(visualizer, F_widths, "Width Variation")
#
#             # 2. 变化F位置
#             F_positions = self._generate_f_positions(min_value, max_value, uniform_optimal_price, n=5)
#             position_results = self._analyze_f_variations(visualizer, F_positions, "Position Variation")
#
#             # 3. 包含/不包含统一最优价格的F
#             F_optimal_tests = self._generate_f_optimal_tests(min_value, max_value, uniform_optimal_price)
#             optimal_results = self._analyze_f_variations(visualizer, F_optimal_tests, "Optimal Price Tests")
#
#             # 存储结果
#             results[dist_name] = {
#                 "values": values,
#                 "masses": masses,
#                 "uniform_optimal_price": uniform_optimal_price,
#                 "width_results": width_results,
#                 "position_results": position_results,
#                 "optimal_results": optimal_results
#             }
#
#             # 创建并保存图表
#             self._create_plots(results[dist_name], dist_name, save_dir)
#
#         return results
#
#     def _generate_f_widths(self, min_value, max_value, uniform_optimal_price, n=5):
#         """生成不同宽度的F值，从单点到全范围"""
#         result = []
#
#         # 先添加单点F
#         result.append([uniform_optimal_price, uniform_optimal_price])
#
#         # 然后添加不同宽度的F，中心是统一最优价格
#         range_size = max_value - min_value
#         for i in range(1, n):
#             width = range_size * (i / n)
#             lower = max(min_value, uniform_optimal_price - width / 2)
#             upper = min(max_value, uniform_optimal_price + width / 2)
#             result.append([lower, upper])
#
#         # 最后添加全范围
#         result.append([min_value, max_value])
#
#         return result
#
#     def _generate_f_positions(self, min_value, max_value, uniform_optimal_price, n=5):
#         """生成不同位置的F值，保持相同宽度"""
#         result = []
#
#         # 固定的F宽度，大约是全范围的1/3
#         width = (max_value - min_value) / 3
#
#         # 从最小值到最大值移动F的中心点
#         for i in range(n):
#             center = min_value + (max_value - min_value) * (i / (n - 1))
#             lower = max(min_value, center - width / 2)
#             upper = min(max_value, center + width / 2)
#             result.append([lower, upper])
#
#         return result
#
#     def _generate_f_optimal_tests(self, min_value, max_value, uniform_optimal_price):
#         """生成包含和不包含统一最优价格的F测试"""
#         result = []
#
#         # 包含统一最优价格的F
#         width = (max_value - min_value) / 3
#         result.append([max(min_value, uniform_optimal_price - width / 2),
#                        min(max_value, uniform_optimal_price + width / 2)])
#
#         # 不包含统一最优价格但接近的F (低侧)
#         if uniform_optimal_price > min_value + width:
#             result.append([max(min_value, uniform_optimal_price - width * 1.5),
#                            uniform_optimal_price - 0.01])
#
#         # 不包含统一最优价格但接近的F (高侧)
#         if uniform_optimal_price < max_value - width:
#             result.append([uniform_optimal_price + 0.01,
#                            min(max_value, uniform_optimal_price + width * 1.5)])
#
#         return result
#
#     def _analyze_f_variations(self, visualizer, F_list, variation_type):
#         """分析一系列F值的三角形特征"""
#         results = []
#
#         print(f"\n--- {variation_type} ---")
#         for F in F_list:
#             is_feasible = visualizer.check_F_feasibility(F)
#             features = visualizer.analyze_triangle_features(F, use_exact_algorithm=is_feasible)
#
#             # 提取关键指标
#             passive_area = features["passive_intermediary"]["area"]
#             active_area = features["active_intermediary"]["area"]
#             no_reg_area = features["no_regulation"]["area"]
#
#             # 计算面积比率，避免除零错误
#             ratio = passive_area / active_area if active_area > 0 else float('inf')
#             if ratio == float('inf'):
#                 ratio_str = 'Inf'
#             else:
#                 ratio_str = f"{ratio:.4f}"
#
#             print(f"F={F}, Feasible: {is_feasible}, "
#                   f"Passive: {passive_area:.4f}, Active: {active_area:.4f}, "
#                   f"Ratio: {ratio_str}")
#
#             results.append({
#                 "F": F,
#                 "is_feasible": is_feasible,
#                 "features": features,
#                 "F_width": F[1] - F[0],
#                 "F_center": (F[0] + F[1]) / 2
#             })
#
#         return results
#
#     def _create_plots(self, result_dict, dist_name, save_dir=None):
#         """为一个分布创建各种图表"""
#         # 1. 宽度变化图
#         self._plot_area_vs_parameter(result_dict["width_results"], "F_width",
#                                      f"{dist_name} - Triangle Area vs F Width",
#                                      "F Width", dist_name, save_dir)
#
#         # 2. 位置变化图
#         self._plot_area_vs_parameter(result_dict["position_results"], "F_center",
#                                      f"{dist_name} - Triangle Area vs F Position",
#                                      "F Center", dist_name, save_dir)
#
#         # 3. 三角形可视化图
#         self._plot_triangle_matrix(result_dict, dist_name, save_dir)
#
#         # 4. 三角形面积关系图 (Passive vs Active)
#         self._plot_triangle_relationship(result_dict, dist_name, save_dir)
#
#     def _plot_area_vs_parameter(self, results, param_name, title, x_label, dist_name, save_dir=None):
#         """绘制三角形面积与参数的关系图"""
#         fig, ax = plt.subplots(figsize=(10, 6))
#
#         x_values = [r[param_name] for r in results]
#         passive_areas = [r["features"]["passive_intermediary"]["area"] for r in results]
#         active_areas = [r["features"]["active_intermediary"]["area"] for r in results]
#         is_feasible = [r["is_feasible"] for r in results]
#
#         # 用不同样式标记可行与不可行点
#         feasible_indices = [i for i, f in enumerate(is_feasible) if f]
#         infeasible_indices = [i for i, f in enumerate(is_feasible) if not f]
#
#         # 绘制可行点
#         if feasible_indices:
#             ax.plot([x_values[i] for i in feasible_indices],
#                     [passive_areas[i] for i in feasible_indices],
#                     'ro-', label="Passive (Feasible)", markersize=8)
#             ax.plot([x_values[i] for i in feasible_indices],
#                     [active_areas[i] for i in feasible_indices],
#                     'bo-', label="Active (Feasible)", markersize=8)
#
#         # 绘制不可行点
#         if infeasible_indices:
#             ax.plot([x_values[i] for i in infeasible_indices],
#                     [passive_areas[i] for i in infeasible_indices],
#                     'r*--', label="Passive (Not Feasible)", markersize=10)
#             ax.plot([x_values[i] for i in infeasible_indices],
#                     [active_areas[i] for i in infeasible_indices],
#                     'b*--', label="Active (Not Feasible)", markersize=10)
#
#         ax.set_title(title)
#         ax.set_xlabel(x_label)
#         ax.set_ylabel("Triangle Area")
#         ax.legend()
#         ax.grid(alpha=0.3)
#
#         if save_dir:
#             save_path = Path(save_dir)
#             save_path.mkdir(exist_ok=True, parents=True)
#             fig_name = f"{dist_name}_{param_name}_analysis.png"
#             plt.savefig(save_path / fig_name, dpi=300)
#
#     def _plot_triangle_matrix(self, result_dict, dist_name, save_dir=None):
#         """绘制三角形矩阵图"""
#         # 为每种F变化生成三角形
#         variations = ["width_results", "position_results", "optimal_results"]
#
#         rows = len(variations)
#         cols = max(len(result_dict[var]) for var in variations)
#
#         fig = plt.figure(figsize=(cols * 5, rows * 4))
#         fig.suptitle(f"Triangle Visualization Matrix - {dist_name}", fontsize=16)
#
#         # 创建可视化器
#         visualizer = TriangleVisualizer(np.array(result_dict["masses"]), np.array(result_dict["values"]))
#
#         for i, var in enumerate(variations):
#             results = result_dict[var]
#             for j, r in enumerate(results):
#                 if j < cols:
#                     ax = fig.add_subplot(rows, cols, i * cols + j + 1)
#                     F = r["F"]
#                     is_feasible = r["is_feasible"]
#                     visualizer.draw_triangles(F, ax=ax, fixed_axes=True, use_exact_algorithm=is_feasible)
#
#                     # 简化标题
#                     ax.set_title(f"F={F}\nFeasible: {is_feasible}")
#
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#         if save_dir:
#             save_path = Path(save_dir)
#             save_path.mkdir(exist_ok=True, parents=True)
#             fig_name = f"{dist_name}_triangle_matrix.png"
#             plt.savefig(save_path / fig_name, dpi=300)
#
#     def _plot_triangle_relationship(self, result_dict, dist_name, save_dir=None):
#         """绘制三角形面积关系图"""
#         fig, ax = plt.subplots(figsize=(10, 10))
#
#         # 合并所有结果
#         all_results = []
#         for var in ["width_results", "position_results", "optimal_results"]:
#             all_results.extend(result_dict[var])
#
#         # 提取数据
#         passive_areas = [r["features"]["passive_intermediary"]["area"] for r in all_results]
#         active_areas = [r["features"]["active_intermediary"]["area"] for r in all_results]
#         is_feasible = [r["is_feasible"] for r in all_results]
#         F_values = [r["F"] for r in all_results]
#
#         # 如果所有面积都为0，设置一个默认的最大值
#         if max(passive_areas + active_areas, default=0) == 0:
#             max_val = 1.0
#         else:
#             max_val = max(max(passive_areas), max(active_areas)) * 1.1
#
#         # 对角线
#         ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="Equal Area")
#
#         # 分别绘制可行和不可行点
#         feasible_indices = [i for i, f in enumerate(is_feasible) if f]
#         infeasible_indices = [i for i, f in enumerate(is_feasible) if not f]
#
#         # 绘制可行点
#         if feasible_indices:
#             ax.scatter([passive_areas[i] for i in feasible_indices],
#                        [active_areas[i] for i in feasible_indices],
#                        c='g', marker='o', s=100, label="Feasible F")
#
#             # 为点添加标签
#             for i in feasible_indices:
#                 ax.annotate(f"{F_values[i]}",
#                             (passive_areas[i], active_areas[i]),
#                             xytext=(5, 5), textcoords='offset points')
#
#         # 绘制不可行点
#         if infeasible_indices:
#             ax.scatter([passive_areas[i] for i in infeasible_indices],
#                        [active_areas[i] for i in infeasible_indices],
#                        c='r', marker='x', s=100, label="Not Feasible F")
#
#             # 为点添加标签
#             for i in infeasible_indices:
#                 ax.annotate(f"{F_values[i]}",
#                             (passive_areas[i], active_areas[i]),
#                             xytext=(5, 5), textcoords='offset points')
#
#         ax.set_title(f"{dist_name} - Passive vs Active Triangle Areas")
#         ax.set_xlabel("Passive Intermediary Triangle Area")
#         ax.set_ylabel("Active Intermediary Triangle Area")
#         ax.grid(alpha=0.3)
#         ax.legend()
#
#         # 设置相等的坐标轴比例
#         ax.set_aspect('equal')
#
#         # 确保原点可见
#         ax.set_xlim(left=-0.01 * max_val)
#         ax.set_ylim(bottom=-0.01 * max_val)
#
#         if save_dir:
#             save_path = Path(save_dir)
#             save_path.mkdir(exist_ok=True, parents=True)
#             fig_name = f"{dist_name}_area_relationship.png"
#             plt.savefig(save_path / fig_name, dpi=300)
#
#
# if __name__ == "__main__":
#     # run_varying_f_experiment.py
#
#     from pathlib import Path
#     from experiments.varying_f_experiment import VaryingFExperiment
#
#     # 创建结果目录
#     result_dir = Path("results/varying_f_experiment")
#     result_dir.mkdir(exist_ok=True, parents=True)
#
#     # 定义要测试的分布
#     distributions = [
#         {"name": "uniform", "params": {"n": 5, "low": 1, "high": 10}},
#         {"name": "binary", "params": {"p": 0.5, "low": 1, "high": 10}},
#         {"name": "truncated_normal", "params": {"mu": 5, "sigma": 1.5, "n": 5, "low": 1, "high": 10}}
#     ]
#
#     # 运行实验
#     experiment = VaryingFExperiment()
#     results = experiment.run_experiment(
#         distributions=distributions,
#         save_dir=result_dir
#     )
#
#     print("实验完成！结果保存在:", result_dir)
#
#