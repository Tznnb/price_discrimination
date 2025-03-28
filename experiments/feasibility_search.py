import numpy as np
import time
import multiprocessing as mp
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import os

from algorithms.market import Market
from algorithms.feasibility import is_feasible
from utils.market_generator import MarketGenerator


# 将check_pair函数移到全局作用域
def check_pair(args):
    """
    检查单个价格对的可行性

    参数:
    args: (pair, market, values) 元组，包含价格对、市场对象和所有价值点

    返回:
    价格对的可行性结果字典
    """
    pair, market, values = args
    start_price, end_price = pair

    # 获取区间内的所有价格
    price_set = [v for v in values if start_price <= v <= end_price]

    # 检查可行性
    feasible = is_feasible(market, price_set)

    return {
        'start_price': start_price,
        'end_price': end_price,
        'width': end_price - start_price,
        'num_prices': len(price_set),
        'prices': price_set,
        'feasible': feasible
    }


def feasibility_grid_search(market, grid_size=20, min_width=1, save_path=None, parallel=True, debug=False):
    """
    对市场进行价格区间可行性的网格搜索

    参数:
    market: Market对象
    grid_size: 网格大小
    min_width: 最小区间宽度
    save_path: 保存结果的路径
    parallel: 是否使用并行计算
    debug: 是否显示调试信息

    返回:
    可行性结果的DataFrame
    """
    values = market.values
    n = len(values)

    if debug:
        print(f"Market: {market}")
        print(f"Optimal uniform price: {market.optimal_price()}")
        print(f"Starting grid search with grid_size={grid_size}, min_width={min_width}")

    # 准备存储结果
    results = []

    # 考虑所有可能的价格对(start, end)，其中end >= start
    price_pairs = []
    for i in range(n):
        for j in range(i, n):  # j >= i，确保end >= start
            start_price = values[i]
            end_price = values[j]

            # 确保区间宽度至少为min_width
            if end_price - start_price >= min_width:
                price_pairs.append((start_price, end_price))

    if debug:
        print(f"Total price pairs to check: {len(price_pairs)}")

    # 执行网格搜索
    if parallel and len(price_pairs) > 1:
        # 使用多进程并行计算
        num_cpus = mp.cpu_count()

        # 准备参数 - 为每个价格对添加market和values
        args = [(pair, market, values) for pair in price_pairs]

        with mp.Pool(processes=num_cpus) as pool:
            results = list(tqdm(pool.imap(check_pair, args), total=len(price_pairs)))
    else:
        # 串行计算
        results = []
        for pair in tqdm(price_pairs):
            results.append(check_pair((pair, market, values)))

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        if debug:
            print(f"Results saved to {save_path}")

    return df


def analyze_feasibility_results(df, market=None):
    """
    分析可行性搜索结果

    参数:
    df: 搜索结果DataFrame
    market: 相关的Market对象，用于额外分析

    返回:
    analysis: 分析结果字典
    """
    analysis = {}

    # 基本统计
    total_cases = len(df)
    feasible_cases = df[df['feasible']].shape[0]
    feasible_ratio = feasible_cases / total_cases if total_cases > 0 else 0

    analysis['total_cases'] = total_cases
    analysis['feasible_cases'] = feasible_cases
    analysis['feasible_ratio'] = feasible_ratio

    # 按区间宽度分析
    width_analysis = df.groupby('width').agg({
        'feasible': ['count', 'sum', 'mean']
    })
    width_analysis.columns = ['count', 'feasible_count', 'feasible_ratio']
    width_analysis = width_analysis.reset_index()

    analysis['width_analysis'] = width_analysis.to_dict('records')

    # 按价格点数量分析
    price_count_analysis = df.groupby('num_prices').agg({
        'feasible': ['count', 'sum', 'mean']
    })
    price_count_analysis.columns = ['count', 'feasible_count', 'feasible_ratio']
    price_count_analysis = price_count_analysis.reset_index()

    analysis['price_count_analysis'] = price_count_analysis.to_dict('records')

    # 最小可行宽度
    if feasible_cases > 0:
        min_feasible_width = df[df['feasible']]['width'].min()
        analysis['min_feasible_width'] = min_feasible_width

    # 检查包含统一最优价格的区间是否更可能可行
    if market:
        opt_price = market.optimal_price()[0]
        contains_opt = df.apply(lambda row: opt_price in row['prices'], axis=1)

        if contains_opt.sum() > 0:
            with_opt_feasible_ratio = df[contains_opt]['feasible'].mean()
            without_opt_feasible_ratio = df[~contains_opt]['feasible'].mean()

            analysis['with_opt_price_count'] = contains_opt.sum()
            analysis['with_opt_price_feasible_ratio'] = with_opt_feasible_ratio
            analysis['without_opt_price_feasible_ratio'] = without_opt_feasible_ratio

    return analysis


def run_distribution_comparison(distribution_types, params_list, grid_size=20, min_width=1, save_dir=None, debug=False):
    """
    比较不同分布下的价格区间可行性

    参数:
    distribution_types: 分布类型列表
    params_list: 每种分布的参数列表
    grid_size: 网格大小
    min_width: 最小区间宽度
    save_dir: 保存结果的目录
    debug: 是否显示调试信息

    返回:
    比较结果的DataFrame
    """
    results = []

    assert len(distribution_types) == len(params_list), "分布类型和参数列表长度必须相同"

    for dist_type, params in zip(distribution_types, params_list):
        # 创建市场
        if dist_type == 'uniform':
            market = MarketGenerator.uniform_market(**params)
        elif dist_type == 'binomial':
            market = MarketGenerator.binomial_market(**params)
        elif dist_type == 'geometric':
            market = MarketGenerator.geometric_market(**params)
        elif dist_type == 'normal':
            market = MarketGenerator.normal_market(**params)
        elif dist_type == 'power_law':
            market = MarketGenerator.power_law_market(**params)
        elif dist_type == 'bimodal':
            market = MarketGenerator.bimodal_market(**params)
        else:
            raise ValueError(f"未知的分布类型: {dist_type}")

        if debug:
            print(f"\nAnalyzing {dist_type} distribution with params: {params}")

        # 保存路径
        save_path = None
        if save_dir:
            # 将参数转换为字符串用于文件名
            params_str = '_'.join([f"{k}{v}" for k, v in params.items()])
            save_path = os.path.join(save_dir, f"{dist_type}_{params_str}.csv")

        # 执行网格搜索
        df = feasibility_grid_search(market, grid_size, min_width, save_path, debug=debug)

        # 分析结果
        analysis = analyze_feasibility_results(df, market)

        # 添加分布信息
        analysis['distribution'] = dist_type
        analysis['params'] = params
        analysis['values'] = market.values.tolist()
        analysis['masses'] = market.masses.tolist()
        analysis['optimal_price'] = market.optimal_price()[0]

        results.append(analysis)

        if debug:
            print(f"可行率: {analysis['feasible_ratio']:.2%}")
            print(f"最小可行宽度: {analysis.get('min_feasible_width', 'N/A')}")

    # 转换为DataFrame
    comparison_df = pd.DataFrame(results)

    # 保存比较结果
    if save_dir:
        comparison_path = os.path.join(save_dir, "distribution_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        if debug:
            print(f"\n比较结果已保存到 {comparison_path}")

    return comparison_df