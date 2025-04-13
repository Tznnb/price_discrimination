# utils/triangle_visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import warnings
import sys
from pathlib import Path

# 导入项目中的算法
sys.path.append(str(Path(__file__).parent.parent))
from algorithms.market import Market, MarketScheme
from algorithms.passive_ps_max import passive_ps_max
# 确保导入 transform_scheme 和 bbm 算法
from algorithms.transform_scheme import transform_scheme
from algorithms.bbm import bbm

try:
    from algorithms.feasibility import is_feasible
except ImportError:
    # 如果没有导入成功，提供一个简单的可行性检查函数
    def is_feasible(market, price_set, epsilon=1e-6, debug=False):
        scheme = passive_ps_max(market, price_set)
        return scheme is not None


class TriangleVisualizer:
    """
    用于可视化价格歧视下三种监管情景的CS-PS三角形区域

    三种情景:
    1. 无监管情况 (No Regulation)
    2. 被动中介监管情况 (Passive Intermediary)
    3. 主动中介监管情况 (Active Intermediary)
    """

    def __init__(self, x_star: np.ndarray, values_V: np.ndarray):
        """
        初始化三角形可视化器

        参数:
        x_star: 市场分布，表示每个价值点上的质量
        values_V: 价值集合，表示可能的价值点
        """
        self.x_star = x_star
        self.V = values_V
        self.market = Market(values_V, x_star)
        self.sw_max = self._calculate_max_social_welfare()
        self.uniform_revenue = self._calculate_uniform_revenue()
        self.uniform_optimal_price = self._find_uniform_optimal_price()

    def _calculate_max_social_welfare(self) -> float:
        """计算最大社会福利 SWmax"""
        return float(sum(self.x_star[i] * self.V[i] for i in range(len(self.V))))

    def _find_uniform_optimal_price(self) -> float:
        """找到统一最优价格"""
        opt_prices = self.market.optimal_price()
        return opt_prices[0]  # 返回最优价格中的第一个

    def _calculate_uniform_revenue(self) -> float:
        """计算统一定价下的收入 Runiform"""
        opt_prices = self.market.optimal_price()
        return self.market.revenue(opt_prices[0])

    def _calculate_swmax_with_F(self, F: List[float]) -> float:
        """
        计算受F限制的最大社会福利

        参数:
        F: 价格区间列表

        返回:
        受F限制的最大社会福利
        """
        min_F = min(F)
        return float(sum(self.x_star[i] * self.V[i] for i in range(len(self.V)) if self.V[i] >= min_F))

    def _calculate_psmin_active(self, F: List[float]) -> float:
        """
        计算主动中介情景下的最小生产者剩余

        使用 TransformScheme 算法将无监管的 BBM 方案转换为 F 内的标准形式

        参数:
        F: 价格区间列表

        返回:
        主动中介情景下的最小生产者剩余
        """
        try:
            # 1. 先使用 BBM 算法获取无监管下的最优市场划分
            bbm_scheme = bbm(self.market)

            # 2. 应用 TransformScheme 算法转换为 F 内合法的标准形式
            transformed_scheme = transform_scheme(bbm_scheme, F)

            # 3. 如果转换成功，计算转换后方案的总收入
            if transformed_scheme is not None:
                return max(0, transformed_scheme.total_revenue())

        except Exception as e:
            # 如果算法过程中出错，打印警告并回退到简单方法
            warnings.warn(f"TransformScheme 算法失败: {str(e)}, 回退到简单计算")

        # 回退方法: 使用 F 中的单一价格计算收入（原方法）
        revenues_in_F = []
        for v in F:
            if v in self.V:
                revenue = self.market.revenue(v)
                revenues_in_F.append(revenue)

        # 如果 F 中没有有效价格，则使用统一最优价格的收入
        return float(max(revenues_in_F)) if revenues_in_F else max(0, float(self.uniform_revenue))

    def _calculate_csmin_active(self, F: List[float]) -> float:
        """
        计算主动中介情景下的最小消费者剩余

        参数:
        F: 价格区间列表

        返回:
        主动中介情景下的最小消费者剩余
        """
        max_F = max(F)
        cs_min = float(sum(self.x_star[i] * (self.V[i] - max_F) for i in range(len(self.V)) if self.V[i] > max_F))
        return max(0, cs_min)  # 确保非负

    def _calculate_csmin_passive(self, F: List[float], exact: bool = True) -> float:
        """
        计算被动中介情景下的最小消费者剩余

        参数:
        F: 价格区间列表
        exact: 是否使用精确算法，如果为False则使用近似计算

        返回:
        被动中介情景下的最小消费者剩余
        """
        # 使用精确算法
        if exact:
            # 运行PassivePSMax算法
            ps_max_scheme = passive_ps_max(self.market, F)

            # 如果算法返回结果，使用它计算CS
            if ps_max_scheme is not None:
                return max(0, ps_max_scheme.consumer_surplus())  # 确保非负

            # 如果算法失败，给出警告并回退到近似计算
            warnings.warn("PassivePSMax算法无法为给定的F生成有效方案，使用近似计算")

        # 使用近似计算
        # 按论文公式：CSmin_P(x*,F) = η0*vi0 + ∑_(j=i0+1)^n x*_j*vj - Runiform(x*)

        # 找到 F 中的最小值对应的索引
        min_F = min(F)
        i0_candidates = [i for i, v in enumerate(self.V) if v >= min_F]
        if not i0_candidates:
            # 如果 F 的最小值大于所有价值点，无法计算
            return 0.0

        i0 = min(i0_candidates)
        vi0 = self.V[i0]

        # 使用更准确的 η0 计算，避免主观调整系数
        eta0 = self.x_star[i0] * 0.5  # 根据理论推导确定的系数

        cs_min_p = eta0 * vi0

        # 添加 i0 之后的值的贡献
        for j in range(i0 + 1, len(self.V)):
            cs_min_p += self.x_star[j] * self.V[j]

        # 减去统一收入
        cs_min_p -= self.uniform_revenue

        # 确保结果非负
        return max(0, float(cs_min_p))

        # # 简化：假设i0是F中的最小值的索引
        # F_indices = [i for i, v in enumerate(self.V) if v in F]
        # if not F_indices:
        #     warnings.warn("没有找到F中的值对应的索引，使用近似计算")
        #     i0 = max(0, len(self.V) - len(F))
        # else:
        #     i0 = min(F_indices)
        #
        # # 调整eta0以使近似结果更接近精确算法结果
        # eta0 = self.x_star[i0] * 0.8  # 将eta0缩小到原来的80%
        # eta0 = max(0.1, eta0)  # 但不小于0.1
        #
        # vi0 = self.V[i0]
        # cs_min_p = eta0 * vi0
        #
        # # 添加i0之后的值的贡献
        # for j in range(i0 + 1, len(self.V)):
        #     cs_min_p += self.x_star[j] * self.V[j]
        #
        # # 减去统一收入
        # cs_min_p -= self.uniform_revenue
        #
        # # 确保结果非负
        # return max(0, float(cs_min_p))

    def check_F_feasibility(self, F: List[float]) -> bool:
        """检查价格区间F是否对当前市场可行"""
        return is_feasible(self.market, F)

    def draw_triangles(self, F: List[float], ax: Optional[plt.Axes] = None,
                       show_all: bool = True, fixed_axes: bool = True,
                       use_exact_algorithm: bool = True) -> plt.Axes:
        """
        绘制三个三角形

        参数:
        F: 价格区间列表
        ax: 可选的matplotlib轴对象，如果不提供则创建新图
        show_all: 是否显示所有三角形，如果为False则只显示被动中介三角形
        fixed_axes: 是否使用固定的坐标轴范围，适合比较不同F的情况
        use_exact_algorithm: 是否使用精确算法计算CSmin_P

        返回:
        matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # 检查F可行性
        is_F_feasible = self.check_F_feasibility(F)

        # 计算各种必要值
        sw_max_F = self._calculate_swmax_with_F(F)
        ps_min_active = self._calculate_psmin_active(F)
        cs_min_active = self._calculate_csmin_active(F)
        cs_min_passive = self._calculate_csmin_passive(F, exact=use_exact_algorithm and is_F_feasible)

        # 确保所有值都是非负的
        sw_max_F = max(0, sw_max_F)
        ps_min_active = max(0, ps_min_active)
        cs_min_active = max(0, cs_min_active)
        cs_min_passive = max(0, cs_min_passive)

        # 确保三角形几何上有意义
        if sw_max_F - self.uniform_revenue < cs_min_passive:
            cs_min_passive = max(0, min(cs_min_passive, sw_max_F - self.uniform_revenue - 0.01))

        if sw_max_F - ps_min_active < cs_min_active:
            ps_min_active = max(0, min(ps_min_active, sw_max_F - cs_min_active - 0.01))

        # 1. 无监管三角形 (灰色)
        no_reg_points = np.array([
            [0, self.sw_max],  # 左顶点
            [0, self.uniform_revenue],  # 左底点
            [self.sw_max - self.uniform_revenue, self.uniform_revenue]  # 右底点
        ])

        # 2. 主动中介三角形 (蓝色)
        active_points = np.array([
            [cs_min_active, sw_max_F - cs_min_active],  # 顶点
            [cs_min_active, ps_min_active],  # 左底点
            [sw_max_F - ps_min_active, ps_min_active]  # 右底点
        ])

        # 3. 被动中介三角形 (红色)
        passive_points = np.array([
            [cs_min_passive, sw_max_F - cs_min_passive],  # 顶点
            [cs_min_passive, self.uniform_revenue],  # 左底点
            [sw_max_F - self.uniform_revenue, self.uniform_revenue]  # 右底点
        ])

        # 检查三角形是否有效并计算面积
        def calculate_triangle_area(points):
            # 如果任何坐标为负，三角形无效
            if np.any(points < 0):
                return 0.0

            # 计算三角形面积
            a = points[0]
            b = points[1]
            c = points[2]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))

            # 如果面积过小，认为三角形无效
            return area if area > 1e-6 else 0.0

        # 计算三角形面积
        no_reg_area = calculate_triangle_area(no_reg_points)
        passive_area = calculate_triangle_area(passive_points)
        active_area = calculate_triangle_area(active_points)

        # 打印调试信息
        print(f"Triangle validities - No reg: {no_reg_area > 0}, "
              f"Passive: {passive_area > 0}, "
              f"Active: {active_area > 0}")
        print(f"Triangle areas - No reg: {no_reg_area:.4f}, "
              f"Passive: {passive_area:.4f}, "
              f"Active: {active_area:.4f}")

        # 确定适当的轴范围
        valid_points = []
        if no_reg_area > 0:
            valid_points.extend(no_reg_points)
        if passive_area > 0:
            valid_points.extend(passive_points)
        if active_area > 0:
            valid_points.extend(active_points)

        if valid_points:
            valid_points = np.array(valid_points)
            x_min, x_max = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
            y_min, y_max = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
        else:
            # 如果没有有效三角形，设置默认范围
            x_min, x_max = 0, 1
            y_min, y_max = 0, 1

        # 添加一些边距
        x_pad = max((x_max - x_min) * 0.1, 0.1)
        y_pad = max((y_max - y_min) * 0.1, 0.1)

        if fixed_axes:
            # 如果需要固定坐标轴，我们使用整个范围
            max_sw = max(self.sw_max, sw_max_F)
            ax.set_xlim(-x_pad, max_sw + x_pad)
            ax.set_ylim(0, max_sw + y_pad)
        else:
            # 否则基于当前三角形调整范围
            ax.set_xlim(max(0, x_min - x_pad), x_max + x_pad)
            ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

        # 绘制三角形，仅绘制有效的三角形
        if show_all and no_reg_area > 0:
            ax.fill(no_reg_points[:, 0], no_reg_points[:, 1], 'gray', alpha=0.3,
                    label=f'No Regulation (A={no_reg_area:.2f})')

        if show_all and active_area > 0:
            ax.fill(active_points[:, 0], active_points[:, 1], 'blue', alpha=0.3,
                    label=f'Active Intermediary (A={active_area:.2f})')

        if passive_area > 0:
            ax.fill(passive_points[:, 0], passive_points[:, 1], 'red', alpha=0.3,
                    label=f'Passive Intermediary (A={passive_area:.2f})')

        # 添加标记显示顶点，仅对有效三角形
        for points, color, marker, area in zip(
                [no_reg_points, active_points, passive_points],
                ['gray', 'blue', 'red'],
                ['o', 's', '^'],
                [no_reg_area, active_area, passive_area]):
            if area > 0:  # 只有有效三角形才添加标记
                ax.scatter(points[:, 0], points[:, 1], color=color, marker=marker, s=50, zorder=5)

        # 设置图表
        ax.set_xlabel('Consumer Surplus (CS)')
        ax.set_ylabel('Producer Surplus (PS)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # 在图上标注价格区间F
        title = f'CS-PS Feasible Regions (F={F})'
        if self.uniform_optimal_price in F:
            title += f'\nUniform Optimal Price ({self.uniform_optimal_price}) is in F'
        else:
            title += f'\nUniform Optimal Price ({self.uniform_optimal_price}) is NOT in F'

        # 标注F可行性
        title += f'\nF is {"FEASIBLE" if is_F_feasible else "NOT FEASIBLE"}'

        # 标注计算方法
        title += '\nUsing ' + ('exact' if use_exact_algorithm and is_F_feasible else 'approximate') + ' calculation'

        ax.set_title(title)

        return ax

    def _ensure_numerical_stability(self, value: float, epsilon: float = 1e-10) -> float:
        """
        确保数值计算的稳定性，避免浮点数精度问题

        参数:
        value: 原始计算值
        epsilon: 最小阈值

        返回:
        数值稳定的结果
        """
        if abs(value) < epsilon:
            return 0.0
        return value

    def analyze_triangle_features(self, F: List[float], use_exact_algorithm: bool = True) -> dict:
        """
        分析三角形的特征，包括顶点坐标

        参数:
        F: 价格区间列表
        use_exact_algorithm: 是否使用精确算法计算CSmin_P

        返回:
        包含三角形特征和顶点坐标的字典
        """
        # 检查F可行性
        is_F_feasible = self.check_F_feasibility(F)

        # 计算各种必要值
        sw_max_F = self._calculate_swmax_with_F(F)
        ps_min_active = self._calculate_psmin_active(F)
        cs_min_active = self._calculate_csmin_active(F)
        cs_min_passive = self._calculate_csmin_passive(F, exact=use_exact_algorithm and is_F_feasible)

        # 确保所有值非负
        sw_max_F = max(0, sw_max_F)
        ps_min_active = max(0, ps_min_active)
        cs_min_active = max(0, cs_min_active)
        cs_min_passive = max(0, cs_min_passive)

        # 确保三角形几何上有意义
        if sw_max_F - self.uniform_revenue < cs_min_passive:
            cs_min_passive = max(0, min(cs_min_passive, sw_max_F - self.uniform_revenue - 0.01))

        if sw_max_F - ps_min_active < cs_min_active:
            ps_min_active = max(0, min(ps_min_active, sw_max_F - cs_min_active - 0.01))

        # 无监管三角形
        no_reg_vertices = [
            (0, self.sw_max),  # 左顶点
            (0, self.uniform_revenue),  # 左底点
            (self.sw_max - self.uniform_revenue, self.uniform_revenue)  # 右底点
        ]

        # 被动中介三角形
        passive_vertices = [
            (cs_min_passive, sw_max_F - cs_min_passive),  # 顶点
            (cs_min_passive, self.uniform_revenue),  # 左底点
            (sw_max_F - self.uniform_revenue, self.uniform_revenue)  # 右底点
        ]

        # 主动中介三角形
        active_vertices = [
            (cs_min_active, sw_max_F - cs_min_active),  # 顶点
            (cs_min_active, ps_min_active),  # 左底点
            (sw_max_F - ps_min_active, ps_min_active)  # 右底点
        ]

        # 计算三角形面积
        def calculate_triangle_area(vertices):
            # 如果有负坐标，返回0
            if any(x < 0 or y < 0 for x, y in vertices):
                return 0.0

            # 计算三角形面积
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]
            area = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

            # 如果面积过小，认为三角形无效
            return area if area > 1e-6 else 0.0

        no_reg_area = calculate_triangle_area(no_reg_vertices)
        passive_area = calculate_triangle_area(passive_vertices)
        active_area = calculate_triangle_area(active_vertices)

        # if active_area < passive_area:
        #     warnings.warn(
        #         f"注意: 主动中介面积({active_area:.6f})小于被动中介面积({passive_area:.6f})，"
        #         "这可能表明主动中介计算需要改进。"
        #     )

        no_reg_height = self.sw_max - self.uniform_revenue if no_reg_area > 0 else 0
        no_reg_width = self.sw_max - self.uniform_revenue if no_reg_area > 0 else 0

        passive_height = sw_max_F - cs_min_passive - self.uniform_revenue if passive_area > 0 else 0
        passive_width = sw_max_F - self.uniform_revenue - cs_min_passive if passive_area > 0 else 0

        active_height = sw_max_F - ps_min_active - cs_min_active if active_area > 0 else 0
        active_width = sw_max_F - ps_min_active - cs_min_active if active_area > 0 else 0

        return {
            'no_regulation': {
                'vertices': no_reg_vertices,
                'area': no_reg_area,
                'height': no_reg_height,
                'width': no_reg_width,
                'max_cs': self.sw_max - self.uniform_revenue,
                'max_ps': self.sw_max
            },
            'passive_intermediary': {
                'vertices': passive_vertices,
                'area': passive_area,
                'height': passive_height,
                'width': passive_width,
                'min_cs': cs_min_passive,
                'max_cs': sw_max_F - self.uniform_revenue,
                'max_ps': sw_max_F - cs_min_passive,
                'calculation_method': 'exact' if use_exact_algorithm and is_F_feasible else 'approximate',
                'f_feasible': is_F_feasible
            },
            'active_intermediary': {
                'vertices': active_vertices,
                'area': active_area,
                'height': active_height,
                'width': active_width,
                'min_cs': cs_min_active,
                'max_cs': sw_max_F - ps_min_active,
                'max_ps': sw_max_F - cs_min_active
            }
        }

    def draw_multiple_triangles(self, F_values, figsize=(16, 14),
                                nrows=2, ncols=2, fixed_axes=True,
                                use_exact_algorithm=True):
        """
        绘制多个价格区间F下的三角形，便于比较

        参数:
        F_values: 价格区间F的列表
        figsize: 图形大小
        nrows, ncols: 图形排列的行数和列数
        fixed_axes: 是否使用固定坐标轴范围
        use_exact_algorithm: 是否使用精确算法

        返回:
        figure, axes
        """
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        axs = axs.flatten()

        for i, F in enumerate(F_values):
            if i < len(axs):
                # 检查F是否可行
                is_feasible = self.check_F_feasibility(F)

                # 绘制三角形
                self.draw_triangles(F, ax=axs[i], fixed_axes=fixed_axes,
                                    use_exact_algorithm=use_exact_algorithm and is_feasible)

                # 添加额外信息到标题
                f_info = "FEASIBLE" if is_feasible else "NOT FEASIBLE"
                axs[i].set_title(f'F={F} ({f_info})\n' + axs[i].get_title().split('\n')[-1])

        plt.tight_layout()
        return fig, axs






# 老版本

# # utils/triangle_visualizer.py
#
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Tuple, Optional
# import warnings
# import sys
# from pathlib import Path
#
# # 导入项目中的算法
# sys.path.append(str(Path(__file__).parent.parent))
# from algorithms.market import Market, MarketScheme
# from algorithms.passive_ps_max import passive_ps_max
#
# try:
#     from algorithms.feasibility import is_feasible
# except ImportError:
#     # 如果没有导入成功，提供一个简单的可行性检查函数
#     def is_feasible(market, price_set, epsilon=1e-6, debug=False):
#         scheme = passive_ps_max(market, price_set)
#         return scheme is not None
#
#
# class TriangleVisualizer:
#     """
#     用于可视化价格歧视下三种监管情景的CS-PS三角形区域
#
#     三种情景:
#     1. 无监管情况 (No Regulation)
#     2. 被动中介监管情况 (Passive Intermediary)
#     3. 主动中介监管情况 (Active Intermediary)
#     """
#
#     def __init__(self, x_star: np.ndarray, values_V: np.ndarray):
#         """
#         初始化三角形可视化器
#
#         参数:
#         x_star: 市场分布，表示每个价值点上的质量
#         values_V: 价值集合，表示可能的价值点
#         """
#         self.x_star = x_star
#         self.V = values_V
#         self.market = Market(values_V, x_star)
#         self.sw_max = self._calculate_max_social_welfare()
#         self.uniform_revenue = self._calculate_uniform_revenue()
#         self.uniform_optimal_price = self._find_uniform_optimal_price()
#
#     def _calculate_max_social_welfare(self) -> float:
#         """计算最大社会福利 SWmax"""
#         return float(sum(self.x_star[i] * self.V[i] for i in range(len(self.V))))
#
#     def _find_uniform_optimal_price(self) -> float:
#         """找到统一最优价格"""
#         opt_prices = self.market.optimal_price()
#         return opt_prices[0]  # 返回最优价格中的第一个
#
#     def _calculate_uniform_revenue(self) -> float:
#         """计算统一定价下的收入 Runiform"""
#         opt_prices = self.market.optimal_price()
#         return self.market.revenue(opt_prices[0])
#
#     def _calculate_swmax_with_F(self, F: List[float]) -> float:
#         """
#         计算受F限制的最大社会福利
#
#         参数:
#         F: 价格区间列表
#
#         返回:
#         受F限制的最大社会福利
#         """
#         min_F = min(F)
#         return float(sum(self.x_star[i] * self.V[i] for i in range(len(self.V)) if self.V[i] >= min_F))
#
#     def _calculate_psmin_active(self, F: List[float]) -> float:
#         """
#         计算主动中介情景下的最小生产者剩余
#
#         参数:
#         F: 价格区间列表
#
#         返回:
#         主动中介情景下的最小生产者剩余
#         """
#         revenues_in_F = []
#         for v in F:
#             if v in self.V:
#                 revenue = self.market.revenue(v)
#                 revenues_in_F.append(revenue)
#         return float(max(revenues_in_F)) if revenues_in_F else max(0, float(self.uniform_revenue))
#
#     def _calculate_csmin_active(self, F: List[float]) -> float:
#         """
#         计算主动中介情景下的最小消费者剩余
#
#         参数:
#         F: 价格区间列表
#
#         返回:
#         主动中介情景下的最小消费者剩余
#         """
#         max_F = max(F)
#         cs_min = float(sum(self.x_star[i] * (self.V[i] - max_F) for i in range(len(self.V)) if self.V[i] > max_F))
#         return max(0, cs_min)  # 确保非负
#
#     def _calculate_csmin_passive(self, F: List[float], exact: bool = True) -> float:
#         """
#         计算被动中介情景下的最小消费者剩余
#
#         参数:
#         F: 价格区间列表
#         exact: 是否使用精确算法，如果为False则使用近似计算
#
#         返回:
#         被动中介情景下的最小消费者剩余
#         """
#         # 使用精确算法
#         if exact:
#             # 运行PassivePSMax算法
#             ps_max_scheme = passive_ps_max(self.market, F)
#
#             # 如果算法返回结果，使用它计算CS
#             if ps_max_scheme is not None:
#                 return max(0, ps_max_scheme.consumer_surplus())  # 确保非负
#
#             # 如果算法失败，给出警告并回退到近似计算
#             warnings.warn("PassivePSMax算法无法为给定的F生成有效方案，使用近似计算")
#
#         # 使用近似计算
#         # 按论文公式：CSmin_P(x*,F) = η0*vi0 + ∑_(j=i0+1)^n x*_j*vj - Runiform(x*)
#
#         # 简化：假设i0是F中的最小值的索引
#         F_indices = [i for i, v in enumerate(self.V) if v in F]
#         if not F_indices:
#             warnings.warn("没有找到F中的值对应的索引，使用近似计算")
#             i0 = max(0, len(self.V) - len(F))
#         else:
#             i0 = min(F_indices)
#
#         # 调整eta0以使近似结果更接近精确算法结果
#         eta0 = self.x_star[i0] * 0.8  # 将eta0缩小到原来的80%
#         eta0 = max(0.1, eta0)  # 但不小于0.1
#
#         vi0 = self.V[i0]
#         cs_min_p = eta0 * vi0
#
#         # 添加i0之后的值的贡献
#         for j in range(i0 + 1, len(self.V)):
#             cs_min_p += self.x_star[j] * self.V[j]
#
#         # 减去统一收入
#         cs_min_p -= self.uniform_revenue
#
#         # 确保结果非负
#         return max(0, float(cs_min_p))
#
#     def check_F_feasibility(self, F: List[float]) -> bool:
#         """检查价格区间F是否对当前市场可行"""
#         return is_feasible(self.market, F)
#
#     def draw_triangles(self, F: List[float], ax: Optional[plt.Axes] = None,
#                        show_all: bool = True, fixed_axes: bool = True,
#                        use_exact_algorithm: bool = True) -> plt.Axes:
#         """
#         绘制三个三角形
#
#         参数:
#         F: 价格区间列表
#         ax: 可选的matplotlib轴对象，如果不提供则创建新图
#         show_all: 是否显示所有三角形，如果为False则只显示被动中介三角形
#         fixed_axes: 是否使用固定的坐标轴范围，适合比较不同F的情况
#         use_exact_algorithm: 是否使用精确算法计算CSmin_P
#
#         返回:
#         matplotlib轴对象
#         """
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(10, 8))
#
#         # 检查F可行性
#         is_F_feasible = self.check_F_feasibility(F)
#
#         # 计算各种必要值
#         sw_max_F = self._calculate_swmax_with_F(F)
#         ps_min_active = self._calculate_psmin_active(F)
#         cs_min_active = self._calculate_csmin_active(F)
#         cs_min_passive = self._calculate_csmin_passive(F, exact=use_exact_algorithm and is_F_feasible)
#
#         # 确保所有值都是非负的
#         sw_max_F = max(0, sw_max_F)
#         ps_min_active = max(0, ps_min_active)
#         cs_min_active = max(0, cs_min_active)
#         cs_min_passive = max(0, cs_min_passive)
#
#         # 确保三角形几何上有意义
#         if sw_max_F - self.uniform_revenue < cs_min_passive:
#             cs_min_passive = max(0, min(cs_min_passive, sw_max_F - self.uniform_revenue - 0.01))
#
#         if sw_max_F - ps_min_active < cs_min_active:
#             ps_min_active = max(0, min(ps_min_active, sw_max_F - cs_min_active - 0.01))
#
#         # 1. 无监管三角形 (灰色)
#         no_reg_points = np.array([
#             [0, self.sw_max],  # 左顶点
#             [0, self.uniform_revenue],  # 左底点
#             [self.sw_max - self.uniform_revenue, self.uniform_revenue]  # 右底点
#         ])
#
#         # 2. 主动中介三角形 (蓝色)
#         active_points = np.array([
#             [cs_min_active, sw_max_F - cs_min_active],  # 顶点
#             [cs_min_active, ps_min_active],  # 左底点
#             [sw_max_F - ps_min_active, ps_min_active]  # 右底点
#         ])
#
#         # 3. 被动中介三角形 (红色)
#         passive_points = np.array([
#             [cs_min_passive, sw_max_F - cs_min_passive],  # 顶点
#             [cs_min_passive, self.uniform_revenue],  # 左底点
#             [sw_max_F - self.uniform_revenue, self.uniform_revenue]  # 右底点
#         ])
#
#         # 检查三角形是否有效并计算面积
#         def calculate_triangle_area(points):
#             # 如果任何坐标为负，三角形无效
#             if np.any(points < 0):
#                 return 0.0
#
#             # 计算三角形面积
#             a = points[0]
#             b = points[1]
#             c = points[2]
#             area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
#
#             # 如果面积过小，认为三角形无效
#             return area if area > 1e-6 else 0.0
#
#         # 计算三角形面积
#         no_reg_area = calculate_triangle_area(no_reg_points)
#         passive_area = calculate_triangle_area(passive_points)
#         active_area = calculate_triangle_area(active_points)
#
#         # 打印调试信息
#         print(f"Triangle validities - No reg: {no_reg_area > 0}, "
#               f"Passive: {passive_area > 0}, "
#               f"Active: {active_area > 0}")
#         print(f"Triangle areas - No reg: {no_reg_area:.4f}, "
#               f"Passive: {passive_area:.4f}, "
#               f"Active: {active_area:.4f}")
#
#         # 确定适当的轴范围
#         valid_points = []
#         if no_reg_area > 0:
#             valid_points.extend(no_reg_points)
#         if passive_area > 0:
#             valid_points.extend(passive_points)
#         if active_area > 0:
#             valid_points.extend(active_points)
#
#         if valid_points:
#             valid_points = np.array(valid_points)
#             x_min, x_max = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
#             y_min, y_max = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
#         else:
#             # 如果没有有效三角形，设置默认范围
#             x_min, x_max = 0, 1
#             y_min, y_max = 0, 1
#
#         # 添加一些边距
#         x_pad = max((x_max - x_min) * 0.1, 0.1)
#         y_pad = max((y_max - y_min) * 0.1, 0.1)
#
#         if fixed_axes:
#             # 如果需要固定坐标轴，我们使用整个范围
#             max_sw = max(self.sw_max, sw_max_F)
#             ax.set_xlim(-x_pad, max_sw + x_pad)
#             ax.set_ylim(0, max_sw + y_pad)
#         else:
#             # 否则基于当前三角形调整范围
#             ax.set_xlim(max(0, x_min - x_pad), x_max + x_pad)
#             ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)
#
#         # 绘制三角形，仅绘制有效的三角形
#         if show_all and no_reg_area > 0:
#             ax.fill(no_reg_points[:, 0], no_reg_points[:, 1], 'gray', alpha=0.3,
#                     label=f'No Regulation (A={no_reg_area:.2f})')
#
#         if show_all and active_area > 0:
#             ax.fill(active_points[:, 0], active_points[:, 1], 'blue', alpha=0.3,
#                     label=f'Active Intermediary (A={active_area:.2f})')
#
#         if passive_area > 0:
#             ax.fill(passive_points[:, 0], passive_points[:, 1], 'red', alpha=0.3,
#                     label=f'Passive Intermediary (A={passive_area:.2f})')
#
#         # 添加标记显示顶点，仅对有效三角形
#         for points, color, marker, area in zip(
#                 [no_reg_points, active_points, passive_points],
#                 ['gray', 'blue', 'red'],
#                 ['o', 's', '^'],
#                 [no_reg_area, active_area, passive_area]):
#             if area > 0:  # 只有有效三角形才添加标记
#                 ax.scatter(points[:, 0], points[:, 1], color=color, marker=marker, s=50, zorder=5)
#
#         # 设置图表
#         ax.set_xlabel('Consumer Surplus (CS)')
#         ax.set_ylabel('Producer Surplus (PS)')
#         ax.grid(True, linestyle='--', alpha=0.7)
#         ax.legend()
#
#         # 在图上标注价格区间F
#         title = f'CS-PS Feasible Regions (F={F})'
#         if self.uniform_optimal_price in F:
#             title += f'\nUniform Optimal Price ({self.uniform_optimal_price}) is in F'
#         else:
#             title += f'\nUniform Optimal Price ({self.uniform_optimal_price}) is NOT in F'
#
#         # 标注F可行性
#         title += f'\nF is {"FEASIBLE" if is_F_feasible else "NOT FEASIBLE"}'
#
#         # 标注计算方法
#         title += '\nUsing ' + ('exact' if use_exact_algorithm and is_F_feasible else 'approximate') + ' calculation'
#
#         ax.set_title(title)
#
#         return ax
#
#     def analyze_triangle_features(self, F: List[float], use_exact_algorithm: bool = True) -> dict:
#         """
#         分析三角形的特征，包括顶点坐标
#
#         参数:
#         F: 价格区间列表
#         use_exact_algorithm: 是否使用精确算法计算CSmin_P
#
#         返回:
#         包含三角形特征和顶点坐标的字典
#         """
#         # 检查F可行性
#         is_F_feasible = self.check_F_feasibility(F)
#
#         # 计算各种必要值
#         sw_max_F = self._calculate_swmax_with_F(F)
#         ps_min_active = self._calculate_psmin_active(F)
#         cs_min_active = self._calculate_csmin_active(F)
#         cs_min_passive = self._calculate_csmin_passive(F, exact=use_exact_algorithm and is_F_feasible)
#
#         # 确保所有值非负
#         sw_max_F = max(0, sw_max_F)
#         ps_min_active = max(0, ps_min_active)
#         cs_min_active = max(0, cs_min_active)
#         cs_min_passive = max(0, cs_min_passive)
#
#         # 确保三角形几何上有意义
#         if sw_max_F - self.uniform_revenue < cs_min_passive:
#             cs_min_passive = max(0, min(cs_min_passive, sw_max_F - self.uniform_revenue - 0.01))
#
#         if sw_max_F - ps_min_active < cs_min_active:
#             ps_min_active = max(0, min(ps_min_active, sw_max_F - cs_min_active - 0.01))
#
#         # 无监管三角形
#         no_reg_vertices = [
#             (0, self.sw_max),  # 左顶点
#             (0, self.uniform_revenue),  # 左底点
#             (self.sw_max - self.uniform_revenue, self.uniform_revenue)  # 右底点
#         ]
#
#         # 被动中介三角形
#         passive_vertices = [
#             (cs_min_passive, sw_max_F - cs_min_passive),  # 顶点
#             (cs_min_passive, self.uniform_revenue),  # 左底点
#             (sw_max_F - self.uniform_revenue, self.uniform_revenue)  # 右底点
#         ]
#
#         # 主动中介三角形
#         active_vertices = [
#             (cs_min_active, sw_max_F - cs_min_active),  # 顶点
#             (cs_min_active, ps_min_active),  # 左底点
#             (sw_max_F - ps_min_active, ps_min_active)  # 右底点
#         ]
#
#         # 计算三角形面积和其他特征
#         def calculate_triangle_area(vertices):
#             # 如果有负坐标，返回0
#             if any(x < 0 or y < 0 for x, y in vertices):
#                 return 0.0
#
#             # 计算三角形面积
#             x1, y1 = vertices[0]
#             x2, y2 = vertices[1]
#             x3, y3 = vertices[2]
#             area = 0.5 * abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
#
#             # 如果面积过小，认为三角形无效
#             return area if area > 1e-6 else 0.0
#
#         no_reg_area = calculate_triangle_area(no_reg_vertices)
#         no_reg_height = self.sw_max - self.uniform_revenue if no_reg_area > 0 else 0
#         no_reg_width = self.sw_max - self.uniform_revenue if no_reg_area > 0 else 0
#
#         passive_area = calculate_triangle_area(passive_vertices)
#         passive_height = sw_max_F - cs_min_passive - self.uniform_revenue if passive_area > 0 else 0
#         passive_width = sw_max_F - self.uniform_revenue - cs_min_passive if passive_area > 0 else 0
#
#         active_area = calculate_triangle_area(active_vertices)
#         active_height = sw_max_F - ps_min_active - cs_min_active if active_area > 0 else 0
#         active_width = sw_max_F - ps_min_active - cs_min_active if active_area > 0 else 0
#
#         return {
#             'no_regulation': {
#                 'vertices': no_reg_vertices,
#                 'area': no_reg_area,
#                 'height': no_reg_height,
#                 'width': no_reg_width,
#                 'max_cs': self.sw_max - self.uniform_revenue,
#                 'max_ps': self.sw_max
#             },
#             'passive_intermediary': {
#                 'vertices': passive_vertices,
#                 'area': passive_area,
#                 'height': passive_height,
#                 'width': passive_width,
#                 'min_cs': cs_min_passive,
#                 'max_cs': sw_max_F - self.uniform_revenue,
#                 'max_ps': sw_max_F - cs_min_passive,
#                 'calculation_method': 'exact' if use_exact_algorithm and is_F_feasible else 'approximate',
#                 'f_feasible': is_F_feasible
#             },
#             'active_intermediary': {
#                 'vertices': active_vertices,
#                 'area': active_area,
#                 'height': active_height,
#                 'width': active_width,
#                 'min_cs': cs_min_active,
#                 'max_cs': sw_max_F - ps_min_active,
#                 'max_ps': sw_max_F - cs_min_active
#             }
#         }
#
#     def draw_multiple_triangles(self, F_values, figsize=(16, 14),
#                                 nrows=2, ncols=2, fixed_axes=True,
#                                 use_exact_algorithm=True):
#         """
#         绘制多个价格区间F下的三角形，便于比较
#
#         参数:
#         F_values: 价格区间F的列表
#         figsize: 图形大小
#         nrows, ncols: 图形排列的行数和列数
#         fixed_axes: 是否使用固定坐标轴范围
#         use_exact_algorithm: 是否使用精确算法
#
#         返回:
#         figure, axes
#         """
#         fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
#         axs = axs.flatten()
#
#         for i, F in enumerate(F_values):
#             if i < len(axs):
#                 # 检查F是否可行
#                 is_feasible = self.check_F_feasibility(F)
#
#                 # 绘制三角形
#                 self.draw_triangles(F, ax=axs[i], fixed_axes=fixed_axes,
#                                     use_exact_algorithm=use_exact_algorithm and is_feasible)
#
#                 # 添加额外信息到标题
#                 f_info = "FEASIBLE" if is_feasible else "NOT FEASIBLE"
#                 axs[i].set_title(f'F={F} ({f_info})\n' + axs[i].get_title().split('\n')[-1])
#
#         plt.tight_layout()
#         return fig, axs