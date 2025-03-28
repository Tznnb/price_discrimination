# algorithms/transform_scheme.py
import numpy as np
from algorithms.market import Market, MarketScheme
from algorithms.bbm import bbm


def transform_scheme(scheme, price_set):
    """
    实现TransformScheme算法

    参数:
    scheme: MarketScheme对象
    price_set: 价格区间F

    返回:
    标准形式的MarketScheme对象
    """
    transformed_scheme = MarketScheme()

    for segment, price in zip(scheme.segments, scheme.prices):
        # 应用BBM分解
        decomposed_scheme = bbm(segment)

        for sub_segment, sub_price in zip(decomposed_scheme.segments, decomposed_scheme.prices):
            # 找出子段中F内的值
            f_values = [v for v in sub_segment.values if v in price_set]

            if not f_values:
                continue

            max_f_val = max(f_values)

            # 处理F中除最大值外的值
            for f_val in [v for v in f_values if v != max_f_val]:
                f_idx = np.where(sub_segment.values == f_val)[0][0]
                if sub_segment.masses[f_idx] > 0:
                    # 创建单值市场
                    single_masses = np.zeros_like(sub_segment.masses)
                    single_masses[f_idx] = sub_segment.masses[f_idx]
                    single_market = Market(sub_segment.values, single_masses)

                    # 添加到方案
                    transformed_scheme.add_segment(single_market, f_val)

                    # 更新子段
                    sub_segment.masses[f_idx] = 0

            # 添加剩余子段
            if np.sum(sub_segment.masses) > 0:
                transformed_scheme.add_segment(sub_segment, max_f_val)

    return transformed_scheme