import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.market_generator import MarketGenerator
from experiments.feasibility_search import feasibility_grid_search, analyze_feasibility_results, \
    run_distribution_comparison
from utils.visualization import plot_market_distribution, plot_feasibility_heatmap, plot_distribution_comparison, \
    visualize_width_analysis


def run_example_1_2_feasibility():
    """对Example 1.2进行详细的可行性分析"""
    from algorithms.market import Market

    # 创建Example 1.2市场
    values = [1, 4, 5, 10]
    masses = [0.3, 0.2, 0.2, 0.3]
    market = Market(values, masses)

    print("分析Example 1.2的可行性...")

    # 创建结果目录
    results_dir = os.path.join('data', 'results', 'phase2', 'example_1_2')
    os.makedirs(results_dir, exist_ok=True)

    # 运行网格搜索
    df = feasibility_grid_search(
        market,
        grid_size=20,
        min_width=0,  # 允许宽度为0（单点区间）
        save_path=os.path.join(results_dir, 'feasibility_grid.csv'),
        parallel=True,
        debug=True
    )

    # 分析结果
    analysis = analyze_feasibility_results(df, market)
    print("\n分析结果:")
    print(f"- 总测试区间: {analysis['total_cases']}")
    print(f"- 可行区间: {analysis['feasible_cases']}")
    print(f"- 可行率: {analysis['feasible_ratio']:.2%}")
    if 'min_feasible_width' in analysis:
        print(f"- 最小可行宽度: {analysis['min_feasible_width']}")

    # 可视化市场分布
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_market_distribution(market, ax=ax, title="Example 1.2市场分布")
    fig.savefig(os.path.join(results_dir, 'market_distribution.png'), dpi=300)

    # 可视化可行性热图
    plot_feasibility_heatmap(
        df,
        market=market,
        title="Example 1.2价格区间可行性热图",
        save_path=os.path.join(results_dir, 'feasibility_heatmap.png')
    )

    # 可视化宽度分析
    visualize_width_analysis(
        df,
        title="Example 1.2区间宽度对可行性的影响",
        save_path=os.path.join(results_dir, 'width_analysis.png')
    )

    print(f"结果已保存到 {results_dir}")


def run_distribution_feasibility_study():
    """进行不同分布类型的可行性研究"""
    print("进行不同分布的可行性研究...")

    # 创建结果目录
    results_dir = os.path.join('data', 'results', 'phase2', 'distributions')
    os.makedirs(results_dir, exist_ok=True)

    # 定义要比较的分布
    distribution_types = [
        'uniform',
        'binomial',
        'geometric',
        'normal',
        'power_law',
        'bimodal'
    ]

    # 定义每种分布的参数
    params_list = [
        {'min_val': 1, 'max_val': 10, 'num_points': 5},  # 均匀分布
        {'low_val': 1, 'high_val': 10, 'p': 0.5},  # 二项分布
        {'base_val': 1, 'ratio': 2, 'num_points': 4},  # 几何分布
        {'mean': 5, 'std': 2, 'num_points': 5},  # 正态分布
        {'min_val': 1, 'max_val': 10, 'alpha': 1.5, 'num_points': 5},  # 幂律分布
        {'low_mean': 3, 'high_mean': 8, 'num_points': 6}  # 双峰分布
    ]

    # 运行分布比较
    comparison_df = run_distribution_comparison(
        distribution_types,
        params_list,
        grid_size=20,
        min_width=0,
        save_dir=results_dir,
        debug=True
    )

    # 可视化比较结果
    plot_distribution_comparison(
        comparison_df,
        metric='feasible_ratio',
        title="不同分布下的可行率比较",
        save_path=os.path.join(results_dir, 'feasible_ratio_comparison.png')
    )

    if 'min_feasible_width' in comparison_df.columns:
        plot_distribution_comparison(
            comparison_df,
            metric='min_feasible_width',
            title="不同分布下的最小可行宽度比较",
            save_path=os.path.join(results_dir, 'min_width_comparison.png')
        )

    # 可视化每种分布的市场
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
        else:
            market = MarketGenerator.bimodal_market(**params)

        # 可视化市场分布
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_market_distribution(market, ax=ax, title=f"{dist_type}市场分布")
        fig.savefig(os.path.join(results_dir, f'{dist_type}_distribution.png'), dpi=300)
        plt.close(fig)

    print(f"结果已保存到 {results_dir}")


def run_uniform_parameters_study():
    """研究均匀分布的参数对可行性的影响"""
    print("研究均匀分布参数对可行性的影响...")

    # 创建结果目录
    results_dir = os.path.join('data', 'results', 'phase2', 'uniform_params')
    os.makedirs(results_dir, exist_ok=True)

    # 变化点数
    num_points_values = [3, 4, 5, 6, 7]

    results = []

    for num_points in num_points_values:
        print(f"\n测试均匀分布，点数={num_points}")

        # 创建市场
        market = MarketGenerator.uniform_market(min_val=1, max_val=10, num_points=num_points)

        # 保存路径
        save_path = os.path.join(results_dir, f'uniform_n{num_points}.csv')

        # 运行网格搜索
        df = feasibility_grid_search(
            market,
            grid_size=20,
            min_width=0,
            save_path=save_path,
            parallel=True,
            debug=True
        )

        # 分析结果
        analysis = analyze_feasibility_results(df, market)
        analysis['num_points'] = num_points

        results.append(analysis)

        # 可视化市场分布
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_market_distribution(market, ax=ax, title=f"均匀分布市场 (n={num_points})")
        fig.savefig(os.path.join(results_dir, f'uniform_n{num_points}_distribution.png'), dpi=300)
        plt.close(fig)

        # 可视化可行性热图
        plot_feasibility_heatmap(
            df,
            market=market,
            title=f"均匀分布市场可行性热图 (n={num_points})",
            save_path=os.path.join(results_dir, f'uniform_n{num_points}_heatmap.png')
        )

    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'uniform_params_results.csv'), index=False)

    # 可视化结果
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['num_points'], results_df['feasible_ratio'], 'o-', label='可行率')

    ax.set_xlabel('价值点数量')
    ax.set_ylabel('可行率')
    ax.set_title('均匀分布价值点数量对可行性的影响')
    ax.grid(alpha=0.3)

    # 添加数值标签
    for x, y in zip(results_df['num_points'], results_df['feasible_ratio']):
        ax.text(x, y + 0.02, f"{y:.2f}", ha='center')

    fig.savefig(os.path.join(results_dir, 'uniform_params_analysis.png'), dpi=300)
    plt.close(fig)

    print(f"结果已保存到 {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='阶段二：价格区间可行性研究')
    parser.add_argument('--study', type=str, default='all',
                        choices=['example_1_2', 'distributions', 'uniform_params', 'all'],
                        help='要运行的研究类型')

    args = parser.parse_args()

    # 记录开始时间
    start_time = datetime.now()
    print(f"开始时间: {start_time}")

    if args.study == 'example_1_2' or args.study == 'all':
        run_example_1_2_feasibility()

    if args.study == 'distributions' or args.study == 'all':
        run_distribution_feasibility_study()

    if args.study == 'uniform_params' or args.study == 'all':
        run_uniform_parameters_study()

    # 记录结束时间
    end_time = datetime.now()
    print(f"结束时间: {end_time}")
    print(f"总耗时: {end_time - start_time}")


if __name__ == '__main__':
    main()