
import numpy as np

def compute_surplus_triangle(market, price_set, scheme):
    """
    计算消费者-生产者剩余三角形的三个顶点
    
    参数:
        market: Market 对象
        price_set: 可行价格区间 F
        scheme: 已生成的 MarketScheme（假设已验证为 F-valid）

    返回:
        vertices: 三角形三个顶点坐标 [(x1, y1), (x2, y2), (x3, y3)]
        area: 三角形面积
    """
    # 假设 scheme 是分段结构，price[i], value[i], gamma[i]
    xs = []
    ys = []

    cumulative_mass = 0.0
    for segment, price in zip(scheme.segments, scheme.prices):
        total_mass = np.sum(segment.masses)
        surplus = np.sum(segment.values * segment.masses) - price * total_mass
        xs.append(cumulative_mass)
        ys.append(surplus)
        cumulative_mass += total_mass

    # 强制压成三角形三个点（只取前3）
    if len(xs) >= 3:
        vertices = [(xs[i], ys[i]) for i in range(3)]
    else:
        vertices = list(zip(xs, ys))
        while len(vertices) < 3:
            vertices.append(vertices[-1])  # 重复补满

    area = triangle_area(vertices)
    return vertices, area


def triangle_area(pts):
    """
    计算三角形面积（任意三个点）
    """
    (x1, y1), (x2, y2), (x3, y3) = pts
    return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2)


def triangle_features(pts):
    """
    返回边长、面积、形状类型（是否等腰/直角等简单判断）
    """
    from math import dist, isclose

    def d(a, b):
        return dist(a, b)

    a, b, c = pts
    l1 = d(a, b)
    l2 = d(b, c)
    l3 = d(c, a)
    lengths = [l1, l2, l3]

    # 粗略判断是否等边/等腰
    is_isoceles = (
        isclose(l1, l2, rel_tol=1e-2)
        or isclose(l2, l3, rel_tol=1e-2)
        or isclose(l1, l3, rel_tol=1e-2)
    )

    return {
        "edge_lengths": lengths,
        "is_isoceles": is_isoceles,
        "area": triangle_area(pts),
    }


def compute_total_surplus(market, scheme):
    """
    计算一个完整机制的总消费者剩余（CS）和生产者剩余（PS）
    """
    CS = 0.0
    PS = 0.0
    for segment, price in zip(scheme.segments, scheme.prices):
        total_mass = np.sum(segment.masses)
        segment_value = np.sum(segment.values * segment.masses)
        CS += segment_value - price * total_mass
        PS += price * total_mass
    return CS, PS


def construct_passive_surplus_triangle(market, price_set):
    """
    尝试根据 PassivePSMax 构建的方案，提取极值点以构成三角形
    """
    from algorithms.passive_ps_max import passive_ps_max

    cs_list = []
    ps_list = []
    vertex_set = []

    # 遍历所有 price_set 的子集 F'，找到可行的方案并记录其 surplus 点
    from itertools import combinations
    for r in range(1, len(price_set) + 1):
        for sub_F in combinations(price_set, r):
            scheme = passive_ps_max(market, list(sub_F), debug=False)
            if scheme:
                cs, ps = compute_total_surplus(market, scheme)
                # 过滤掉不合理的负值点
                if cs >= 0 and ps >= 0:
                    vertex_set.append((cs, ps))
                    cs_list.append(cs)
                    ps_list.append(ps)

    if not vertex_set:
        return None, None

    # 极值点 + 原点组成三角形
    cs_min, ps_max = min(vertex_set, key=lambda x: x[0]), max(vertex_set, key=lambda x: x[1])
    origin = (0.0, 0.0)
    triangle_vertices = [origin, cs_min, ps_max]

    # 面积计算
    def triangle_area(a, b, c):
        return abs((a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2)

    area = triangle_area(*triangle_vertices)
    return triangle_vertices, area
