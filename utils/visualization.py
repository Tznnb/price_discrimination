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


def plot_feasibility_heatmap(values, results, title=None, save_path=None):
    """
    绘制清晰的二值可行性热图（红色=可行，白色=不可行）
    """
    plt.figure(figsize=(8, 6))
    cmap = sns.color_palette(["white", "red"])

    xticks = [str(v) for v in values]
    yticks = [str(v) for v in values]

    sns.heatmap(results, cmap=cmap, cbar=False,
                xticklabels=xticks, yticklabels=yticks,
                linewidths=0.5, linecolor='gray', square=True)

    plt.gca().invert_yaxis()
    plt.xlabel('Upper Bound of F')
    plt.ylabel('Lower Bound of F')

    if title:
        plt.title(title)
    else:
        plt.title("Feasibility Heatmap")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_feasibility_trend(param_grid, results, title="Feasibility Trend", save_path=None):
    """
    可视化参数扫描的趋势结果

    参数:
        param_grid: 参数取值列表
        results: 每个参数对应的指标字典
        title: 图标题
        save_path: 若指定则保存图像
    """
    counts = [r["count"] for r in results]
    min_widths = [r["min_width"] if r["min_width"] is not None else 0 for r in results]
    avg_widths = [r["avg_width"] if r["avg_width"] is not None else 0 for r in results]
    conn_ratios = [r["row_connected_ratio"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(param_grid, counts, label="Feasible Count", marker='o')
    plt.plot(param_grid, min_widths, label="Min Width", marker='x')
    plt.plot(param_grid, avg_widths, label="Avg Width", marker='s')
    plt.plot(param_grid, conn_ratios, label="Row Conn Ratio", marker='^')
    plt.xlabel("Parameter")
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
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


def plot_passive_surplus_triangle_theory(market, price_set, show_scheme=False, scheme=None, save_path=None):
    """
    绘制 Passive Intermediary 理论三角形（论文图中红色部分）

    参数:
    - market: Market 对象
    - price_set: F 集合（区间价格）
    - show_scheme: 是否叠加当前机制点
    - scheme: 当前方案（可选）
    - save_path: 如果指定则保存图像
    """

    values = market.values
    masses = market.masses
    min_f = min(price_set)
    max_f = max(price_set)

    # 1. 最大社会福利 SW_max
    SW_max = sum(v * m for v, m in zip(values, masses) if v >= min_f)

    # 2. 均匀定价 Runiform
    uniform_revs = [p * sum(m for v, m in zip(values, masses) if v >= p) for p in values]
    R_uniform = max(uniform_revs)

    # 3. CS_uniform = SW_max - R_uniform
    CS_uniform = SW_max - R_uniform

    # 4. SW_min_P: 所有价格都 >= max_f 时的 welfare
    SW_min = sum(v * m for v, m in zip(values, masses) if v >= max_f)

    # 三个点：左下、右下、上
    A = (0, SW_min)
    B = (0, R_uniform)
    C = (CS_uniform, R_uniform)

    triangle_pts = [A, C, B]  # 逆时针顺序绘图更漂亮


    # --- 绘图 ---
    plt.figure(figsize=(7, 7))
    x = [pt[0] for pt in triangle_pts] + [triangle_pts[0][0]]
    y = [pt[1] for pt in triangle_pts] + [triangle_pts[0][1]]

    plt.plot(x, y, color='red', linewidth=2, label="Passive Regulation Area")
    plt.fill(x, y, color='red', alpha=0.3)

    if show_scheme and scheme is not None:
        cs = scheme.consumer_surplus()
        ps = scheme.producer_surplus()
        plt.scatter([cs], [ps], color='black', label="Current Scheme", zorder=5)
        plt.annotate("Current", (cs + 0.1, ps), fontsize=10)

    plt.xlabel("Consumer Surplus")
    plt.ylabel("Producer Surplus")
    plt.grid(True)
    plt.legend()
    plt.title("Theoretical Passive Intermediary Region")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
