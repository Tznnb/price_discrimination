# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def plot_market(market, title=None, save_path=None):
    """
    可视化市场分布

    参数:
    market: Market对象
    title: 标题
    save_path: 保存路径，如果为None则显示图形
    """
    plt.figure(figsize=(10, 6))
    plt.bar(market.values, market.masses, width=0.4)
    plt.xlabel('Value')
    plt.ylabel('Mass')
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_surplus_triangle(market, scheme, price_set, title=None, save_path=None):
    """
    可视化剩余三角形

    参数:
    market: Market对象
    scheme: MarketScheme对象
    price_set: 价格区间F
    title: 标题
    save_path: 保存路径，如果为None则显示图形
    """
    # 计算统一定价收入
    opt_prices = market.optimal_price()
    uniform_revenue = market.revenue(opt_prices[0])

    # 计算最大社会福利
    min_f = min(price_set)
    max_welfare = sum(v * m for v, m in zip(market.values, market.masses) if v >= min_f)

    # 计算方案的消费者和生产者剩余
    cs = scheme.consumer_surplus()
    ps = scheme.producer_surplus()

    # 绘制三角形
    plt.figure(figsize=(8, 8))

    # 定义三角形顶点
    points = [
        (0, max_welfare),  # 消费者最优点
        (max_welfare, 0),  # 生产者最优点
        (0, uniform_revenue)  # 最小点
    ]

    # 绘制三角形
    triangle = plt.Polygon(points, fill=False, edgecolor='blue', linestyle='--')
    plt.gca().add_patch(triangle)

    # 标记当前方案的位置
    plt.scatter([cs], [ps], color='red', s=100, zorder=5)
    plt.annotate(f'Current Scheme', (cs, ps), xytext=(cs + 0.2, ps + 0.2))

    # 标记三角形顶点
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='blue')
    plt.annotate('Consumer Optimal', points[0], xytext=(points[0][0] + 0.2, points[0][1] - 0.2))
    plt.annotate('Producer Optimal', points[1], xytext=(points[1][0] - 0.2, points[1][1] + 0.2))
    plt.annotate('Min Welfare', points[2], xytext=(points[2][0] + 0.2, points[2][1] - 0.2))

    # 设置轴和标题
    plt.xlabel('Consumer Surplus')
    plt.ylabel('Producer Surplus')
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title)
    else:
        plt.title('Surplus Triangle')

    # 设置轴范围
    max_val = max(max_welfare, uniform_revenue) * 1.1
    plt.xlim(-0.05 * max_val, max_val)
    plt.ylim(-0.05 * max_val, max_val)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_feasibility_heatmap(values, results, title=None, save_path=None, ax=None):
    """
    绘制清晰的二值可行性热图（红色=可行，白色=不可行）
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cmap = sns.color_palette(["white", "red"])

    xticks = [str(v) for v in values]
    yticks = [str(v) for v in values]

    sns.heatmap(results, cmap=cmap, cbar=False,
                xticklabels=xticks, yticklabels=yticks,
                linewidths=0.5, linecolor='gray', square=True, ax=ax)

    ax.invert_yaxis()
    ax.set_xlabel('Upper Bound of F')
    ax.set_ylabel('Lower Bound of F')

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Feasibility Heatmap")

    if save_path:
        plt.savefig(save_path)
        plt.close()

# def plot_feasibility_heatmap(values, results, title=None, save_path=None, ax=None):
#     """
#     绘制高清矢量格式的二值可行性热图（红色=可行，白色=不可行）
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 8))  # 更大的图更清晰
#
#     cmap = sns.color_palette(["white", "red"])
#     xticks = [str(v) for v in values]
#     yticks = [str(v) for v in values]
#
#     sns.heatmap(results, cmap=cmap, cbar=False,
#                 xticklabels=xticks, yticklabels=yticks,
#                 linewidths=0.5, linecolor='gray', square=True, ax=ax)
#
#     ax.invert_yaxis()
#     ax.set_xlabel('Upper Bound of F', fontsize=14)
#     ax.set_ylabel('Lower Bound of F', fontsize=14)
#     ax.tick_params(labelsize=12)
#
#     ax.set_title(title or "Feasibility Heatmap", fontsize=16)
#
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         ext = os.path.splitext(save_path)[1].lower()
#         if ext in [".pdf", ".svg"]:
#             plt.savefig(save_path, bbox_inches='tight')
#         else:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#
#         plt.show()  # 👈 显示图片
#     else:
#         plt.show()


def plot_feasibility_trend(param_grid, results, title="Feasibility Trend", save_path=None):
    """
    可视化参数扫描的趋势结果，输出高清矢量图
    """
    counts = [r["count"] for r in results]
    min_widths = [r["min_width"] if r["min_width"] is not None else 0 for r in results]
    avg_widths = [r["avg_width"] if r["avg_width"] is not None else 0 for r in results]
    conn_ratios = [r["row_connected_ratio"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_grid, counts, label="Feasible Count", marker='o')
    ax.plot(param_grid, min_widths, label="Min Width", marker='x')
    ax.plot(param_grid, avg_widths, label="Avg Width", marker='s')
    ax.plot(param_grid, conn_ratios, label="Row Conn Ratio", marker='^')

    ax.set_xlabel("Parameter", fontsize=14)
    ax.set_ylabel("Metric", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(labelsize=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".pdf", ".svg"]:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()  # 👈 显示图片
    else:
        plt.show()


def plot_surplus_triangle_from_vertices(vertices, label="Passive", color="red", show=True):
    """
    使用三角形顶点绘图（轻量级）
    """
    x = [v[0] for v in vertices] + [vertices[0][0]]
    y = [v[1] for v in vertices] + [vertices[0][1]]

    plt.plot(x, y, label=label, color=color, linewidth=2)
    plt.fill(x, y, alpha=0.3, color=color)

    if show:
        plt.xlabel("Consumer Surplus")
        plt.ylabel("Producer Surplus")
        plt.legend()
        plt.grid(True)
        plt.title("Surplus Triangle Visualization")
        plt.show()


def plot_theoretical_passive_triangle(vertices, label="Passive Triangle", color="red"):
    """
    绘制基于三顶点（原点+极值点）的 passive triangle
    """
    x = [v[0] for v in vertices] + [vertices[0][0]]
    y = [v[1] for v in vertices] + [vertices[0][1]]

    plt.plot(x, y, label=label, color=color, linewidth=2)
    plt.fill(x, y, alpha=0.3, color=color)
    plt.xlabel("Consumer Surplus")
    plt.ylabel("Producer Surplus")
    plt.grid(True)
    plt.legend()
    plt.title("Passive Intermediary Surplus Triangle")
    plt.show()


def plot_passive_surplus_triangle_theory(market, F, show_scheme=False, scheme=None):
    """
    可视化论文中的 Passive Regulation 三角形（理论三角形）

    顶点：
    A: (0, SW_max)
    B: (0, R_uniform)
    C: (CS_uniform, R_uniform)
    """
    min_f = min(F)

    # 顶点 A：最大社会福利
    SW_max = sum(v * m for v, m in zip(market.values, market.masses) if v >= min_f)

    # 统一价格（可用 market.optimal_price()[0] 替代）
    uniform_price = market.optimal_price()[0]

    # 顶点 B：统一定价下的生产者收入
    R_uniform = uniform_price * sum(m for v, m in zip(market.values, market.masses) if v >= uniform_price)

    # 顶点 C：消费者剩余 = 社会福利 - 收入
    CS_uniform = SW_max - R_uniform

    # 三个顶点坐标
    A = (0, SW_max)
    B = (0, R_uniform)
    C = (CS_uniform, R_uniform)

    print("三角形顶点：")
    print(f"A = {A}  (最大社会福利)")
    print(f"B = {B}  (统一价格收入)")
    print(f"C = {C}  (CS = 福利 - 收入)")

    # 可视化绘图
    x = [A[0], B[0], C[0], A[0]]
    y = [A[1], B[1], C[1], A[1]]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="Passive Regulation Area", color='red', linewidth=2)
    plt.fill(x, y, alpha=0.3, color='red')

    if show_scheme and scheme:
        cs = scheme.consumer_surplus()
        ps = scheme.producer_surplus()
        plt.scatter([cs], [ps], color='black', label='Current Scheme', zorder=5)
        plt.annotate('Current', (cs, ps), xytext=(cs + 0.1, ps + 0.1))

    plt.xlabel("Consumer Surplus")
    plt.ylabel("Producer Surplus")
    plt.title("Theoretical Passive Intermediary Region")
    plt.legend()
    plt.grid(True)
    plt.show()
