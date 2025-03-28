
import numpy as np

def analyze_feasibility_matrix(values, matrix, verbose=True):
    """
    分析可行性矩阵结构，输出特征：
    - 可行区间数量
    - 可行区间最小 / 平均宽度
    - 是否存在包含 uniform 最优价格的可行区间
    - 行连通性（每行是否为连续可行段）
    """
    n = len(values)
    feasible_coords = [(i, j) for i in range(n) for j in range(i, n) if matrix[i][j]]
    count = len(feasible_coords)

    if count == 0:
        if verbose:
            print("❌ 无可行区间")
        return {
            "count": 0,
            "min_width": None,
            "avg_width": None,
            "contains_uniform_opt": False,
            "row_connected_ratio": 0.0
        }

    widths = [values[j] - values[i] for i, j in feasible_coords]
    min_width = min(widths)
    avg_width = sum(widths) / len(widths)

    # 是否包含 uniform optimal price
    uniform_price = compute_uniform_opt_price(values)
    contains_uniform = any(values[i] <= uniform_price <= values[j] for i, j in feasible_coords)

    # 连通性分析
    row_connected = 0
    for i in range(n):
        row = matrix[i, i:]
        if np.sum(row) > 0:
            start = np.argmax(row)
            end = len(row) - 1 - np.argmax(row[::-1])
            if np.all(row[start:end + 1]):
                row_connected += 1

    ratio = row_connected / n

    if verbose:
        print(f"✅ 可行区间数量: {count}")
        print(f"📏 最小宽度: {min_width:.3f}, 平均宽度: {avg_width:.3f}")
        print(f"🎯 是否包含统一最优价格: {'是' if contains_uniform else '否'}")
        print(f"🔗 行连通性比例: {ratio:.2f}")

    return {
        "count": count,
        "min_width": min_width,
        "avg_width": avg_width,
        "contains_uniform_opt": contains_uniform,
        "row_connected_ratio": ratio
    }

def compute_uniform_opt_price(values):
    """
    计算原始市场下的统一最优价格（即 values 中值）
    """
    return sorted(values)[len(values) // 2]
