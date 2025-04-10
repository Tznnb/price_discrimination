# run_airline_analysis.py

from pathlib import Path
import traceback
from airline_pricing_analysis import AirlinePricingAnalysis

# 创建结果目录
result_dir = Path("results/flight_price")
result_dir.mkdir(exist_ok=True)

# 初始化分析器
analysis = AirlinePricingAnalysis()

try:
    # 加载数据
    data_path = "data/flight_price/Clean_Dataset.csv"
    analysis.load_data(data_path)

    # 数据预处理
    processed_data = analysis.preprocess_data()

    # 分析价格歧视行为
    price_stats = analysis.analyze_price_discrimination()

    # 构建价格预测模型
    try:
        model = analysis.build_price_prediction_model()

        # 分析特征重要性
        try:
            importance = analysis.analyze_feature_importance(save_dir=result_dir)
        except Exception as e:
            print(f"特征重要性分析出错: {e}")
            traceback.print_exc()

        # 模拟不同监管策略的效果
        try:
            cap_result, _ = analysis.simulate_regulation_impact('price_cap')
            floor_result, _ = analysis.simulate_regulation_impact('price_floor')
            range_result, _ = analysis.simulate_regulation_impact('price_range')

            # 比较不同监管策略
            comparison = analysis.compare_regulation_strategies()
        except Exception as e:
            print(f"监管模拟分析出错: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"模型构建或分析过程中出错: {e}")
        traceback.print_exc()

    print("\n分析完成！结果保存在results目录")

except Exception as e:
    print(f"程序运行出错: {e}")
    traceback.print_exc()