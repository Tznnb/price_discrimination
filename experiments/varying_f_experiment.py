import numpy as np
import matplotlib.pyplot as plt
import sys
import platform
from pathlib import Path
from itertools import product
import traceback

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.triangle_visualizer import TriangleVisualizer
from utils.market_generator import MarketGenerator
from algorithms.market import Market
from algorithms.passive_ps_max import passive_ps_max

# 尝试导入其他可行性检查函数
try:
    from utils.feasibility_analysis import is_f_feasible
except ImportError:
    is_f_feasible = None


def setup_matplotlib_chinese():
    """配置matplotlib支持中文显示"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    system = platform.system()

    if system == 'Windows':
        # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system == 'Darwin':
        # macOS系统
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS']
    else:
        # Linux系统
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

    # 通用配置
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    mpl.rcParams['font.family'] = 'sans-serif'


class VaryingFExperiment:
    """固定分布，变化F值的实验"""

    def __init__(self):
        self.market_generator = MarketGenerator()
        # 配置matplotlib支持中文
        setup_matplotlib_chinese()

    def run_experiment(self, distributions=None, save_dir=None):
        """
        运行固定分布变化F的实验

        参数:
        distributions: 要测试的分布列表，每个元素是一个字典包含名称和参数
        save_dir: 保存结果的目录

        返回:
        实验结果字典
        """
        try:
            if distributions is None:
                distributions = [
                    {"name": "uniform", "params": {"n": 5, "low": 1, "high": 10}},
                    {"name": "binary", "params": {"p": 0.5, "low": 1, "high": 10}},
                    {"name": "truncated_normal", "params": {"mu": 5, "sigma": 1.5, "n": 5, "low": 1, "high": 10}}
                ]

            results = {}

            # 对每个分布运行F变化实验
            for dist in distributions:
                dist_name = dist["name"]
                params = dist["params"]

                print(f"\n===== 实验: {dist_name} 分布 =====")

                # 生成市场
                if dist_name == "uniform":
                    values, masses = self.market_generator.uniform(**params)
                elif dist_name == "binary":
                    values, masses = self.market_generator.binary(**params)
                elif dist_name == "truncated_normal":
                    values, masses = self.market_generator.truncated_normal(**params)
                else:
                    # 其他分布类型
                    method = getattr(self.market_generator, dist_name, None)
                    if method:
                        values, masses = method(**params)
                    else:
                        print(f"不支持的分布类型: {dist_name}")
                        continue

                # 创建市场和可视化器
                market = Market(values, masses)
                visualizer = TriangleVisualizer(np.array(masses), np.array(values))

                # 获取统一最优价格和值域
                uniform_optimal_price = visualizer.uniform_optimal_price
                min_value, max_value = min(values), max(values)

                print(f"统一最优价格: {uniform_optimal_price}, 值域: [{min_value}, {max_value}]")
                print(f"价值点: {values}")
                print(f"质量分布: {masses}")

                # 系统性生成更多候选F值
                candidate_F_widths = self._generate_f_widths(min_value, max_value, uniform_optimal_price, n=20)
                candidate_F_positions = self._generate_f_positions(min_value, max_value, uniform_optimal_price, n=20)
                candidate_F_optimal_tests = self._generate_f_optimal_tests(min_value, max_value, uniform_optimal_price)
                candidate_F_micro = self._generate_micro_f_values(min_value, max_value, uniform_optimal_price)
                candidate_F_dist_specific = self._generate_distribution_specific_F(
                    dist_name, min_value, max_value, uniform_optimal_price, values
                )

                # 针对价值点生成特殊F
                candidate_F_value_points = self._generate_value_point_f(values)

                # 生成稠密划分的F值
                candidate_F_dense = self._generate_dense_partitioning(min_value, max_value, n=30)

                # 合并所有候选F值并去重
                all_candidates = []
                for F in (candidate_F_widths + candidate_F_positions +
                          candidate_F_optimal_tests + candidate_F_micro +
                          candidate_F_dist_specific + candidate_F_value_points +
                          candidate_F_dense):
                    # 检查F是否已经存在于候选列表中，避免重复
                    is_duplicate = False
                    for existing_F in all_candidates:
                        if np.isclose(F[0], existing_F[0]) and np.isclose(F[1], existing_F[1]):
                            is_duplicate = True
                            break

                    # 确保上下界不同(除了单点F)，不是重复的，再添加
                    if not is_duplicate:
                        # 单点F的特殊情况
                        if np.isclose(F[0], F[1]):
                            all_candidates.append(F)
                        # 常规F的情况
                        elif F[0] < F[1]:
                            all_candidates.append(F)

                # 整理候选F值信息
                candidate_info = {
                    "widths": {"candidates": candidate_F_widths},
                    "positions": {"candidates": candidate_F_positions},
                    "optimal_tests": {"candidates": candidate_F_optimal_tests},
                    "micro": {"candidates": candidate_F_micro},
                    "dist_specific": {"candidates": candidate_F_dist_specific},
                    "value_points": {"candidates": candidate_F_value_points},
                    "dense": {"candidates": candidate_F_dense}
                }

                # 筛选出可行的F值
                print("正在筛选可行的F值...")
                feasible_Fs, debug_info = self._filter_feasible_F(visualizer, market, all_candidates)

                # 输出调试信息
                print(f"总共{len(all_candidates)}个候选F值，其中{len(feasible_Fs)}个可行")
                print(f"筛选详情: {debug_info}")

                # 如果没有可行的F值，使用松弛的可行性标准
                if not feasible_Fs:
                    print("没有找到严格可行的F值，使用松弛标准...")
                    feasible_Fs, alt_debug_info = self._filter_feasible_F(visualizer, market, all_candidates,
                                                                          strict=False)
                    print(f"松弛标准下找到{len(feasible_Fs)}个可行F值")
                    print(f"松弛筛选详情: {alt_debug_info}")

                # 如果仍然没有可行的F值，使用最优价格单点作为退化方案
                if not feasible_Fs:
                    print("警告: 即使使用松弛标准也没有找到可行的F值")
                    print("使用最优价格单点F作为退化方案")
                    single_point_F = [uniform_optimal_price, uniform_optimal_price]
                    feasible_Fs = [single_point_F]

                # 按照类型分类可行的F值
                feasible_F_by_type = self._categorize_feasible_F(
                    feasible_Fs,
                    candidate_F_widths,
                    candidate_F_positions,
                    candidate_F_optimal_tests,
                    candidate_F_micro,
                    candidate_F_dist_specific,
                    candidate_F_value_points,
                    candidate_F_dense
                )

                # 更新候选F值信息
                for key, candidates in candidate_info.items():
                    candidates["feasible"] = feasible_F_by_type.get(key, [])
                    print(f"{key}类型: 总共{len(candidates['candidates'])}个F值，"
                          f"其中{len(candidates['feasible'])}个可行")

                # 分析可行的F值
                feasible_width_results = self._analyze_f_variations(visualizer,
                                                                    feasible_F_by_type.get("widths", []),
                                                                    "Width Variation")
                feasible_position_results = self._analyze_f_variations(visualizer,
                                                                       feasible_F_by_type.get("positions", []),
                                                                       "Position Variation")
                feasible_optimal_results = self._analyze_f_variations(visualizer,
                                                                      feasible_F_by_type.get("optimal_tests", []),
                                                                      "Optimal Price Tests")
                feasible_micro_results = self._analyze_f_variations(visualizer,
                                                                    feasible_F_by_type.get("micro", []),
                                                                    "Micro F Variation")
                feasible_dist_results = self._analyze_f_variations(visualizer,
                                                                   feasible_F_by_type.get("dist_specific", []),
                                                                   "Distribution Specific")
                feasible_value_results = self._analyze_f_variations(visualizer,
                                                                    feasible_F_by_type.get("value_points", []),
                                                                    "Value Point F")
                feasible_dense_results = self._analyze_f_variations(visualizer,
                                                                    feasible_F_by_type.get("dense", []),
                                                                    "Dense Partitioning")

                # 合并所有分析结果
                all_results = (feasible_width_results + feasible_position_results +
                               feasible_optimal_results + feasible_micro_results +
                               feasible_dist_results + feasible_value_results +
                               feasible_dense_results)

                # 存储结果
                results[dist_name] = {
                    "values": values,
                    "masses": masses,
                    "uniform_optimal_price": uniform_optimal_price,
                    "width_results": feasible_width_results,
                    "position_results": feasible_position_results,
                    "optimal_results": feasible_optimal_results,
                    "micro_results": feasible_micro_results,
                    "dist_specific_results": feasible_dist_results,
                    "value_point_results": feasible_value_results,
                    "dense_results": feasible_dense_results,
                    "all_results": all_results,
                    # 添加候选F值和可行F值信息，用于后续分析
                    "candidate_F_info": candidate_info
                }

                # 创建并保存图表
                self._create_plots(results[dist_name], dist_name, save_dir)

                # 额外创建可行性统计图表
                self._create_feasibility_plots(results[dist_name], dist_name, save_dir)

            # 分析可行性因素
            self.analyze_feasibility_factors(results)

            return results

        except Exception as e:
            print(f"实验运行中发生错误: {e}")
            traceback.print_exc()
            return {}

    def _categorize_feasible_F(self, feasible_Fs, *candidate_lists):
        """将可行的F按照它们的来源分类"""
        result = {}
        for i, candidates in enumerate(candidate_lists):
            category_name = ["widths", "positions", "optimal_tests", "micro",
                             "dist_specific", "value_points", "dense"][i]
            result[category_name] = []

            for F in feasible_Fs:
                for candidate in candidates:
                    if (np.isclose(F[0], candidate[0]) and np.isclose(F[1], candidate[1])):
                        result[category_name].append(F)
                        break

        return result

    def _filter_feasible_F(self, visualizer, market, F_list, strict=True, max_tests=200):
        """
        筛选出可行的F值

        使用与先前实验相同的可行性判断标准
        """
        # 如果列表太长，只取前max_tests个
        if len(F_list) > max_tests:
            print(f"候选F值太多 ({len(F_list)}), 只测试前{max_tests}个")
            F_list = F_list[:max_tests]

        feasible_F = []
        debug_info = {
            "total": len(F_list),
            "fail_feasibility": 0,
            "success": 0,
            "fail_other": 0
        }

        # 尝试导入之前实现的可行性判断函数
        try:
            from algorithms.feasibility import is_feasible as orig_is_feasible
            use_original = True
            print("使用原始is_feasible函数进行可行性判断")
        except ImportError:
            use_original = False
            print("无法导入原始is_feasible函数，使用备选判断标准")

        for idx, F in enumerate(F_list):
            # 每20个F打印一次进度
            if idx % 20 == 0:
                print(f"已测试 {idx}/{len(F_list)} 个F值")

            # 打印一些调试信息
            if idx < 50:
                print(f"测试F = {F}")

            # 判断F是否可行
            if use_original:
                try:
                    # 将区间F转换为价值点列表
                    values = market.values
                    # 价值点在F区间内
                    F_as_values = [v for v in values if F[0] <= v <= F[1]]

                    # 如果区间内有价值点
                    if F_as_values:
                        # 使用原始的可行性判断函数
                        is_f_feasible = orig_is_feasible(market, F_as_values)
                        if is_f_feasible:
                            feasible_F.append(F)
                            debug_info["success"] += 1
                            continue
                        else:
                            debug_info["fail_feasibility"] += 1
                    else:
                        # 如果区间内没有价值点，检查是否为单点F
                        if np.isclose(F[0], F[1]) and any(np.isclose(F[0], v) for v in values):
                            # 单点F且匹配某个价值点
                            feasible_F.append(F)
                            debug_info["success"] += 1
                            continue
                        debug_info["fail_feasibility"] += 1
                except Exception as e:
                    debug_info["fail_other"] += 1
                    if idx < 10:
                        print(f"使用原始is_feasible检查失败: {e}，回退到默认方法")
                    # 回退到默认的可行性判断

            # 如果原始方法失败或不可用，使用当前的可行性判断
            try:
                # 检查F是否可行
                if not visualizer.check_F_feasibility(F):
                    debug_info["fail_feasibility"] += 1
                    if not strict:
                        # 在非严格模式下，单点F总是认为可行
                        if np.isclose(F[0], F[1]):
                            feasible_F.append(F)
                            debug_info["success"] += 1
                    continue

                # 尝试使用PassivePSMax
                scheme = passive_ps_max(visualizer.market, F)
                if scheme is None:
                    debug_info["fail_other"] += 1
                    if not strict and np.isclose(F[0], F[1]):
                        # 单点F特殊处理
                        feasible_F.append(F)
                        debug_info["success"] += 1
                    continue

                # 如果通过了所有检查，认为F是可行的
                feasible_F.append(F)
                debug_info["success"] += 1

            except Exception as e:
                debug_info["fail_other"] += 1
                if idx < 10:
                    print(f"检查F可行性时出错: {e}")

        return feasible_F, debug_info

    def _generate_f_widths(self, min_value, max_value, uniform_optimal_price, n=20):
        """生成不同宽度的F值，从单点到全范围"""
        result = []

        # 先添加单点F（统一最优价格）
        result.append([uniform_optimal_price, uniform_optimal_price])

        # 然后添加不同宽度的F，中心是统一最优价格
        range_size = max_value - min_value

        # 更细致的宽度渐变
        for i in range(1, n + 1):
            width = range_size * (i / n)
            lower = max(min_value, uniform_optimal_price - width / 2)
            upper = min(max_value, uniform_optimal_price + width / 2)
            result.append([lower, upper])

        # 更多细粒度变化
        for width_factor in np.linspace(0.01, 0.99, n):
            width = range_size * width_factor
            lower = max(min_value, uniform_optimal_price - width / 2)
            upper = min(max_value, uniform_optimal_price + width / 2)
            result.append([lower, upper])

        # 最后添加全范围
        result.append([min_value, max_value])

        return result

    def _generate_f_positions(self, min_value, max_value, uniform_optimal_price, n=20):
        """生成不同位置的F值，包括多种不同宽度"""
        result = []

        # 使用不同的宽度
        widths = [(max_value - min_value) * factor for factor in
                  [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]]

        # 对每个宽度，生成不同位置的F
        for width in widths:
            # 从最小值到最大值移动F的中心点
            for i in range(n):
                center = min_value + (max_value - min_value) * (i / (n - 1))
                lower = max(min_value, center - width / 2)
                upper = min(max_value, center + width / 2)
                # 确保上下界不同
                if lower < upper:
                    result.append([lower, upper])

        # 非对称宽度F
        for left_ratio in [0.1, 0.2, 0.3, 0.4]:
            for right_ratio in [0.1, 0.2, 0.3, 0.4]:
                if left_ratio != right_ratio:  # 非对称
                    lower = max(min_value, uniform_optimal_price - (max_value - min_value) * left_ratio)
                    upper = min(max_value, uniform_optimal_price + (max_value - min_value) * right_ratio)
                    if lower < upper:
                        result.append([lower, upper])

        return result

    def _generate_f_optimal_tests(self, min_value, max_value, uniform_optimal_price):
        """生成包含和不包含统一最优价格的F测试"""
        result = []
        range_size = max_value - min_value

        # 统一最优价格附近的多个宽度
        for width_factor in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
            width = range_size * width_factor
            lower = max(min_value, uniform_optimal_price - width / 2)
            upper = min(max_value, uniform_optimal_price + width / 2)
            if lower < upper:
                result.append([lower, upper])

        # 不包含统一最优价格的F (多个位置)
        for offset_factor in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            offset = range_size * offset_factor

            # 低侧
            if uniform_optimal_price > min_value + offset:
                lower = max(min_value, uniform_optimal_price - offset - range_size * 0.1)
                upper = uniform_optimal_price - 0.001  # 非常接近但不包含
                if lower < upper:
                    result.append([lower, upper])

            # 高侧
            if uniform_optimal_price < max_value - offset:
                lower = uniform_optimal_price + 0.001  # 非常接近但不包含
                upper = min(max_value, uniform_optimal_price + offset + range_size * 0.1)
                if lower < upper:
                    result.append([lower, upper])

        return result

    def _generate_micro_f_values(self, min_value, max_value, uniform_optimal_price):
        """生成非常小范围的F值，重点关注最优价格附近"""
        result = []

        # 最优价格单点
        result.append([uniform_optimal_price, uniform_optimal_price])

        # 添加以最优价格为中心的非常小范围F
        range_size = max_value - min_value
        for width_factor in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
            width = range_size * width_factor
            lower = max(min_value, uniform_optimal_price - width / 2)
            upper = min(max_value, uniform_optimal_price + width / 2)
            if lower < upper:
                result.append([lower, upper])

        # 最优价格单侧扩展，更精细的步长
        for width_factor in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            width = range_size * width_factor
            # 向右扩展
            if uniform_optimal_price + width <= max_value:
                result.append([uniform_optimal_price, uniform_optimal_price + width])
            # 向左扩展
            if uniform_optimal_price - width >= min_value:
                result.append([uniform_optimal_price - width, uniform_optimal_price])

        return result

    def _generate_value_point_f(self, values):
        """生成基于价值点的F值"""
        result = []

        # 添加所有价值点作为单点F
        for v in values:
            result.append([v, v])

        # 添加相邻价值点对构成的F
        sorted_values = sorted(list(set(values)))  # 去重并排序
        if len(sorted_values) > 1:
            for i in range(len(sorted_values) - 1):
                result.append([sorted_values[i], sorted_values[i + 1]])

        # 添加所有可能的价值点区间
        for i in range(len(sorted_values)):
            for j in range(i, len(sorted_values)):
                result.append([sorted_values[i], sorted_values[j]])

        return result

    def _generate_dense_partitioning(self, min_value, max_value, n=30):
        """生成更稠密的F值，覆盖整个值域"""
        result = []

        # 等分区间
        points = np.linspace(min_value, max_value, n)

        # 生成所有可能的区间组合
        for i in range(len(points)):
            for j in range(i, len(points)):
                if i != j:  # 排除单点
                    result.append([points[i], points[j]])

        return result

    def _generate_distribution_specific_F(self, dist_name, min_value, max_value, uniform_optimal_price, values):
        """针对特定分布生成可能更有效的F值"""
        result = []

        # 基础F值
        result.append([uniform_optimal_price, uniform_optimal_price])
        range_size = max_value - min_value

        if dist_name == "binary":
            # binary分布特定策略
            # 通常binary分布的最优价格是较高的值
            # 生成以最优价格为上界的F
            for lower in np.linspace(min_value, float(uniform_optimal_price) - 0.001, 20):
                result.append([lower, uniform_optimal_price])

            # 生成包含分布中实际价值点的F
            if len(values) >= 2:
                # binary分布通常只有两个不同价值
                distinct_values = sorted(list(set([float(v) for v in values])))
                if len(distinct_values) >= 2:
                    result.append([distinct_values[0], distinct_values[-1]])

            # 很小范围内包含最小值
            for width_factor in [0.001, 0.005, 0.01, 0.02, 0.05]:
                width = range_size * width_factor
                distinct_values = sorted(list(set([float(v) for v in values])))
                if len(distinct_values) >= 1:
                    min_val = distinct_values[0]
                    result.append([min_val, min_val + width])

        elif dist_name == "uniform":
            # uniform分布特定策略
            # 尝试包含多个价值点的小范围F
            for i in range(20):
                width = range_size * (0.05 + i * 0.02)
                center = uniform_optimal_price
                lower = max(min_value, center - width / 2)
                upper = min(max_value, center + width / 2)
                result.append([lower, upper])

            # 生成以最优价格为一端的F
            step = range_size / 20
            for i in range(1, 20):
                # 以最优价格为下界
                if uniform_optimal_price + i * step <= max_value:
                    result.append([uniform_optimal_price, uniform_optimal_price + i * step])
                # 以最优价格为上界
                if uniform_optimal_price - i * step >= min_value:
                    result.append([uniform_optimal_price - i * step, uniform_optimal_price])

            # 更多小范围F - 修复这里的错误
            for v in values:
                for width_factor in [0.01, 0.05, 0.1, 0.2]:
                    width = width_factor * range_size  # 修复：正确计算宽度
                    lower = max(min_value, v - width / 2)
                    upper = min(max_value, v + width / 2)
                    if lower < upper:
                        result.append([lower, upper])

        elif dist_name == "truncated_normal":
            # 针对正态分布的策略
            # 尝试以均值为中心的F
            mu = min_value + range_size / 2  # 假设均值在中间
            for width_factor in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]:
                width = range_size * width_factor
                lower = max(min_value, mu - width / 2)
                upper = min(max_value, mu + width / 2)
                result.append([lower, upper])

            # 尝试包含均值和最优价格的F
            if not np.isclose(mu, uniform_optimal_price):
                lower = min(mu, uniform_optimal_price)
                upper = max(mu, uniform_optimal_price)
                # 稍微扩大范围
                lower = max(min_value, lower - 0.05 * range_size)
                upper = min(max_value, upper + 0.05 * range_size)
                result.append([lower, upper])

            # 添加更多尝试 - 修复这里的错误
            for v in values:
                for width_factor in [0.01, 0.05, 0.1, 0.2]:
                    width = width_factor * range_size  # 修复：正确计算宽度
                    lower = max(min_value, v - width / 2)
                    upper = min(max_value, v + width / 2)
                    if lower < upper:
                        result.append([lower, upper])

        return result

    def _analyze_f_variations(self, visualizer, F_list, variation_type):
        """分析一系列F值的三角形特征"""
        results = []

        print(f"\n--- {variation_type} ---")
        for F in F_list:
            # 尝试使用精确算法，如果失败则使用近似计算
            try:
                features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)
            except Exception as e:
                print(f"警告: 使用精确算法分析F={F}失败: {str(e)[:50]}, 尝试近似计算")
                try:
                    features = visualizer.analyze_triangle_features(F, use_exact_algorithm=False)
                except Exception as e2:
                    print(f"错误: 无法分析F={F}: {str(e2)[:50]}")
                    continue

            # 提取关键指标
            passive_area = features["passive_intermediary"]["area"]
            active_area = features["active_intermediary"]["area"]
            no_reg_area = features["no_regulation"]["area"]

            # 检查三角形是否有效（有意义的面积）
            meaningful_result = passive_area > 1e-6 or active_area > 1e-6

            # 计算面积比率，避免除零错误
            ratio = passive_area / active_area if active_area > 1e-6 else float('inf')
            if ratio == float('inf'):
                ratio_str = 'Inf'
            else:
                ratio_str = f"{ratio:.4f}"

            print(f"F={F}, Passive: {passive_area:.4f}, Active: {active_area:.4f}, Ratio: {ratio_str}")

            results.append({
                "F": F,
                "features": features,
                "F_width": F[1] - F[0],
                "F_center": (F[0] + F[1]) / 2,
                "meaningful": meaningful_result
            })

        return results

    def analyze_feasibility_factors(self, results):
        """分析影响F值可行性的因素"""
        print("\n=== 可行性因素分析 ===")
        for dist_name, result in results.items():
            if not result:
                continue

            feasible_fs = []
            for key in result["candidate_F_info"]:
                if "feasible" in result["candidate_F_info"][key]:
                    feasible_fs.extend(result["candidate_F_info"][key]["feasible"])

            if not feasible_fs:
                print(f"{dist_name}: 没有找到可行的F")
                continue

            # 对可行的F去重
            unique_fs = []
            for f in feasible_fs:
                if not any(np.isclose(f[0], existing_f[0]) and np.isclose(f[1], existing_f[1]) for existing_f in
                           unique_fs):
                    unique_fs.append(f)
            feasible_fs = unique_fs

            # 分析可行F的特征
            widths = [f[1] - f[0] for f in feasible_fs]
            centers = [(f[0] + f[1]) / 2 for f in feasible_fs]
            contains_optimal = [f[0] <= result["uniform_optimal_price"] <= f[1] for f in feasible_fs]

            print(f"{dist_name}分布可行性分析:")
            if widths:
                print(f"  可行F数量: {len(feasible_fs)}")
                print(f"  可行F平均宽度: {np.mean(widths):.4f}")
                print(f"  可行F中包含最优价格的比例: {sum(contains_optimal) / len(contains_optimal) * 100:.1f}%")

                if len(widths) > 0:
                    min_idx = np.argmin(widths)
                    max_idx = np.argmax(widths)
                    print(f"  最窄的可行F: {feasible_fs[min_idx]}, 宽度: {min(widths):.4f}")
                    print(f"  最宽的可行F: {feasible_fs[max_idx]}, 宽度: {max(widths):.4f}")

                # 分析可行F中心点的分布
                opt_price = result["uniform_optimal_price"]
                distances = [abs(c - opt_price) for c in centers]
                print(f"  可行F中心点距最优价格的平均距离: {np.mean(distances):.4f}")

                # 寻找有意义面积的F
                meaningful_fs = []
                for r in result.get("all_results", []):
                    if r.get("meaningful", False):
                        meaningful_fs.append(r["F"])

                if meaningful_fs:
                    print(f"  产生有意义三角形的F数量: {len(meaningful_fs)}")

                    # 计算被动/主动三角形面积比
                    passive_areas = []
                    active_areas = []
                    ratios = []

                    for r in result.get("all_results", []):
                        if r.get("meaningful", False):
                            p_area = r["features"]["passive_intermediary"]["area"]
                            a_area = r["features"]["active_intermediary"]["area"]

                            if p_area > 0 and a_area > 0:
                                passive_areas.append(p_area)
                                active_areas.append(a_area)
                                ratios.append(p_area / a_area)

                    if ratios:
                        print(f"  三角形面积比率统计:")
                        print(f"    最小比率: {min(ratios):.4f}")
                        print(f"    最大比率: {max(ratios):.4f}")
                        print(f"    平均比率: {np.mean(ratios):.4f}")
                        print(f"    中位数比率: {np.median(ratios):.4f}")

    def _create_plots(self, result_dict, dist_name, save_dir=None):
        """为一个分布创建各种图表"""
        # 检查是否有可行的结果
        if not any(result_dict.get(key, []) for key in ["width_results", "position_results",
                                                        "optimal_results", "micro_results",
                                                        "dist_specific_results", "value_point_results",
                                                        "dense_results"]):
            print(f"警告：{dist_name}分布没有找到任何可行的F值，跳过绘图")
            return

        # 合并所有有意义的结果
        meaningful_results = [r for r in result_dict.get("all_results", []) if r.get("meaningful", False)]

        # 如果没有有意义的结果，使用所有结果
        if not meaningful_results:
            meaningful_results = result_dict.get("all_results", [])

        # 按类型绘制图表
        for key, title_part in [
            ("width_results", "宽度变化"),
            ("position_results", "位置变化"),
            ("micro_results", "微小范围"),
            ("dist_specific_results", "分布特定"),
            ("value_point_results", "价值点F"),
            ("dense_results", "稠密划分")
        ]:
            results = result_dict.get(key, [])
            if results:
                if key == "width_results":
                    self._plot_area_vs_parameter(
                        results, "F_width",
                        f"{dist_name} - 三角形面积 vs F{title_part}",
                        "F宽度", dist_name, save_dir
                    )
                elif key in ["position_results", "micro_results", "dist_specific_results",
                             "value_point_results", "dense_results"]:
                    self._plot_area_vs_parameter(
                        results, "F_center",
                        f"{dist_name} - 三角形面积 vs F{title_part}",
                        "F中心位置", dist_name, save_dir
                    )

        # 三角形可视化图
        self._plot_triangle_matrix(result_dict, dist_name, save_dir)

        # 三角形面积关系图 (Passive vs Active)
        self._plot_triangle_relationship(result_dict, dist_name, save_dir)

    def _create_feasibility_plots(self, result_dict, dist_name, save_dir=None):
        """创建F值可行性分析图表"""
        candidate_info = result_dict.get("candidate_F_info", {})

        # 1. 宽度与可行性关系图
        if "widths" in candidate_info:
            width_candidates = candidate_info["widths"].get("candidates", [])
            width_feasible = candidate_info["widths"].get("feasible", [])

            if width_candidates:
                self._plot_feasibility_analysis(
                    width_candidates, width_feasible,
                    x_key="width", x_label="F宽度",
                    title=f"{dist_name} - F宽度与可行性关系",
                    dist_name=dist_name, save_dir=save_dir
                )

        # 2. 位置与可行性关系图
        if "positions" in candidate_info:
            pos_candidates = candidate_info["positions"].get("candidates", [])
            pos_feasible = candidate_info["positions"].get("feasible", [])

            if pos_candidates:
                self._plot_feasibility_analysis(
                    pos_candidates, pos_feasible,
                    x_key="center", x_label="F中心位置",
                    title=f"{dist_name} - F位置与可行性关系",
                    dist_name=dist_name, save_dir=save_dir
                )

        # 3. 可行F值特征分布图
        self._plot_feasible_f_distribution(result_dict, dist_name, save_dir)

    def _plot_feasible_f_distribution(self, result_dict, dist_name, save_dir=None):
        """绘制可行F值特征分布图"""
        # 收集所有可行的F
        feasible_fs = []
        for key in result_dict.get("candidate_F_info", {}):
            if "feasible" in result_dict["candidate_F_info"][key]:
                feasible_fs.extend(result_dict["candidate_F_info"][key]["feasible"])

        if not feasible_fs:
            return

        # 对可行的F去重
        unique_fs = []
        for f in feasible_fs:
            if not any(np.isclose(f[0], existing_f[0]) and np.isclose(f[1], existing_f[1]) for existing_f in unique_fs):
                unique_fs.append(f)
        feasible_fs = unique_fs

        # 提取特征
        widths = np.array([f[1] - f[0] for f in feasible_fs])
        centers = np.array([(f[0] + f[1]) / 2 for f in feasible_fs])

        # 创建分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 宽度分布
        if len(widths) > 1:
            ax1.hist(widths, bins=min(10, len(widths)), alpha=0.7)
            ax1.axvline(np.mean(widths), color='r', linestyle='--', label=f'平均: {np.mean(widths):.4f}')
            ax1.axvline(np.median(widths), color='g', linestyle='--', label=f'中位数: {np.median(widths):.4f}')
        else:
            ax1.bar([widths[0]], [1], alpha=0.7)
            ax1.axvline(widths[0], color='r', linestyle='--', label=f'值: {widths[0]:.4f}')

        ax1.set_title(f'{dist_name} - 可行F宽度分布')
        ax1.set_xlabel('F宽度')
        ax1.set_ylabel('频数')
        ax1.legend()

        # 中心点分布
        if len(centers) > 1:
            ax2.hist(centers, bins=min(10, len(centers)), alpha=0.7)
            ax2.axvline(np.mean(centers), color='r', linestyle='--', label=f'平均: {np.mean(centers):.4f}')
            ax2.axvline(np.median(centers), color='g', linestyle='--', label=f'中位数: {np.median(centers):.4f}')
        else:
            ax2.bar([centers[0]], [1], alpha=0.7)
            ax2.axvline(centers[0], color='r', linestyle='--', label=f'值: {centers[0]:.4f}')

        # 标记最优价格位置
        opt_price = result_dict["uniform_optimal_price"]
        ax2.axvline(opt_price, color='b', linestyle='-', label=f'最优价格: {opt_price:.4f}')

        ax2.set_title(f'{dist_name} - 可行F中心位置分布')
        ax2.set_xlabel('F中心位置')
        ax2.set_ylabel('频数')
        ax2.legend()

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            fig_name = f"{dist_name}_feasible_distribution.png"
            plt.savefig(save_path / fig_name, dpi=300)
            plt.close(fig)

    def _plot_feasibility_analysis(self, candidates, feasible, x_key, x_label, title, dist_name, save_dir=None):
        """绘制F的可行性分析图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 转换F列表到可分析的形式
        all_data = []
        for F in candidates:
            if x_key == "width":
                x_value = F[1] - F[0]
            elif x_key == "center":
                x_value = (F[0] + F[1]) / 2

            # 检查是否在可行集合中
            is_feasible = any(np.isclose(F[0], f[0]) and np.isclose(F[1], f[1]) for f in feasible)

            all_data.append({
                "F": F,
                "x_value": x_value,
                "is_feasible": is_feasible
            })

        # 分拣数据
        feasible_x = [d["x_value"] for d in all_data if d["is_feasible"]]
        feasible_y = [1] * len(feasible_x)  # y=1表示可行

        infeasible_x = [d["x_value"] for d in all_data if not d["is_feasible"]]
        infeasible_y = [0] * len(infeasible_x)  # y=0表示不可行

        # 绘制散点图
        ax.scatter(feasible_x, feasible_y, c='green', marker='o', s=100, label='可行的F')
        ax.scatter(infeasible_x, infeasible_y, c='red', marker='x', s=100, label='不可行的F')

        # 设置y轴为离散值
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['不可行', '可行'])

        # 添加辅助线和标签
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('可行性')
        ax.legend()
        ax.grid(alpha=0.3)

        # 保存图表
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            fig_name = f"{dist_name}_feasibility_{x_key}.png"
            plt.savefig(save_path / fig_name, dpi=300)
            plt.close(fig)

    def _plot_area_vs_parameter(self, results, param_name, title, x_label, dist_name, save_dir=None):
        """绘制三角形面积与参数的关系图"""
        if not results:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        x_values = [r[param_name] for r in results]
        passive_areas = [r["features"]["passive_intermediary"]["area"] for r in results]
        active_areas = [r["features"]["active_intermediary"]["area"] for r in results]

        # 所有点都是可行的
        ax.plot(x_values, passive_areas, 'ro-', label="被动中介", markersize=8)
        ax.plot(x_values, active_areas, 'bo-', label="主动中介", markersize=8)

        # 添加面积比率
        for i, (x, p_area, a_area) in enumerate(zip(x_values, passive_areas, active_areas)):
            ratio = p_area / a_area if a_area > 1e-6 else float('inf')
            if ratio != float('inf'):
                ax.annotate(f"{ratio:.2f}",
                            (x, max(p_area, a_area)),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("三角形面积")
        ax.legend()
        ax.grid(alpha=0.3)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            fig_name = f"{dist_name}_{param_name}_analysis.png"
            plt.savefig(save_path / fig_name, dpi=300)
            plt.close(fig)

    def _plot_triangle_matrix(self, result_dict, dist_name, save_dir=None):
        """绘制三角形矩阵图"""
        # 合并所有有意义的F结果
        meaningful_results = [r for r in result_dict.get("all_results", []) if r.get("meaningful", False)]

        # 如果没有有意义的结果，使用所有结果
        if not meaningful_results:
            meaningful_results = result_dict.get("all_results", [])

        # 如果仍然没有结果，直接返回
        if not meaningful_results:
            print(f"警告：{dist_name}分布没有找到任何有效的F值，跳过三角形矩阵绘图")
            return

        # 最多绘制16个三角形（4x4网格）
        max_triangles = 16
        results_to_show = meaningful_results[:max_triangles]
        rows = int(np.sqrt(len(results_to_show))) or 1
        cols = (len(results_to_show) + rows - 1) // rows or 1

        fig = plt.figure(figsize=(cols * 4, rows * 4))
        fig.suptitle(f"{dist_name} - 三角形可视化矩阵 (可行F)", fontsize=16)

        # 创建可视化器
        visualizer = TriangleVisualizer(np.array(result_dict["masses"]), np.array(result_dict["values"]))

        for i, r in enumerate(results_to_show):
            ax = fig.add_subplot(rows, cols, i + 1)
            F = r["F"]
            try:
                visualizer.draw_triangles(F, ax=ax, fixed_axes=True, use_exact_algorithm=True)
            except Exception as e:
                print(f"绘制F={F}的三角形时出错: {str(e)[:50]}，尝试使用近似方法")
                try:
                    visualizer.draw_triangles(F, ax=ax, fixed_axes=True, use_exact_algorithm=False)
                except Exception as e2:
                    print(f"绘制F={F}的三角形失败: {str(e2)[:50]}")
                    ax.text(0.5, 0.5, f"无法绘制\nF={F}", ha='center', va='center')
                    continue

            # 简化标题，只显示F值和是否为单点F
            if np.isclose(F[0], F[1]):
                ax.set_title(f"单点F={F}")
            else:
                ax.set_title(f"F={F}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            fig_name = f"{dist_name}_triangle_matrix.png"
            plt.savefig(save_path / fig_name, dpi=300)
            plt.close(fig)

    def _plot_triangle_relationship(self, result_dict, dist_name, save_dir=None):
        """绘制三角形面积关系图"""
        # 过滤有意义的结果
        meaningful_results = [r for r in result_dict.get("all_results", []) if r.get("meaningful", False)]

        # 如果没有有意义的结果，使用所有结果
        if not meaningful_results:
            meaningful_results = result_dict.get("all_results", [])

        # 如果仍然没有结果，直接返回
        if not meaningful_results:
            print(f"警告：{dist_name}分布没有找到任何有效的F值，跳过三角形关系绘图")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # 提取数据
        passive_areas = [r["features"]["passive_intermediary"]["area"] for r in meaningful_results]
        active_areas = [r["features"]["active_intermediary"]["area"] for r in meaningful_results]
        F_values = [r["F"] for r in meaningful_results]
        F_widths = [r["F_width"] for r in meaningful_results]

        # 如果只有一个点，特殊处理
        if len(F_widths) == 1:
            sizes = [100]
        else:
            # 根据F的宽度设置点的大小
            max_width = max(F_widths)
            if max_width > 0:
                sizes = [50 + 1000 * (width / max_width) for width in F_widths]
            else:
                sizes = [100] * len(F_widths)

        # 如果所有面积都为0，设置一个默认的最大值
        if max(passive_areas + active_areas, default=0) == 0:
            max_val = 1.0
        else:
            max_val = max(max(passive_areas), max(active_areas)) * 1.1

        # 对角线
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="等面积线")

        # 绘制点，使用宽度信息调整大小
        if len(F_widths) > 1 and any(w > 0 for w in F_widths):
            scatter = ax.scatter(passive_areas, active_areas, c=F_widths, cmap='viridis',
                                 s=sizes, alpha=0.6, label="可行F")

            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('F宽度')
        else:
            # 单点或所有宽度为0的简化情况
            ax.scatter(passive_areas, active_areas, c='blue', s=100, alpha=0.6, label="可行F")

        # 为点添加标签
        for i, (p_area, a_area, F) in enumerate(zip(passive_areas, active_areas, F_values)):
            if i < 10:  # 只为前10个点添加标签，避免过度拥挤
                ax.annotate(f"{F}",
                            (p_area, a_area),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8)

        ax.set_title(f"{dist_name} - 被动 vs 主动 三角形面积关系")
        ax.set_xlabel("被动中介三角形面积")
        ax.set_ylabel("主动中介三角形面积")
        ax.grid(alpha=0.3)
        ax.legend()

        # 设置相等的坐标轴比例
        ax.set_aspect('equal')

        # 确保原点可见
        ax.set_xlim(left=-0.01 * max_val)
        ax.set_ylim(bottom=-0.01 * max_val)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            fig_name = f"{dist_name}_area_relationship.png"
            plt.savefig(save_path / fig_name, dpi=300)
            plt.close(fig)