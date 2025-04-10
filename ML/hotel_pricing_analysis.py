# hotel_pricing_analysis.py (修复版)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
from pathlib import Path
import datetime as dt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 确保中文显示正确
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HotelPricingAnalysis:
    """酒店价格歧视分析与监管效果评估类"""

    def __init__(self):
        self.data = None
        self.processed_data = None
        self.model = None
        self.features = None
        self.results_dir = Path("results/hotel_pricing")
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def load_data(self, file_path):
        """加载酒店预订数据"""
        print(f"\n加载数据: {file_path}")
        try:
            self.data = pd.read_csv(file_path)

            print(f"数据集大小: {self.data.shape}")
            print("\n数据样本:")
            print(self.data.head())

            # 显示关键列的基本统计
            print("\n房价(ADR)统计:")
            print(self.data['adr'].describe())

            return self.data
        except FileNotFoundError:
            print(f"错误: 找不到文件 {file_path}")
            print("请确保路径正确或从 Kaggle 下载数据集")
            raise

    def preprocess_data(self):
        """预处理酒店数据"""
        if self.data is None:
            raise ValueError("请先加载数据")

        print("\n开始数据预处理...")
        df = self.data.copy()

        # 1. 处理缺失值
        print(f"处理前行数: {len(df)}")
        missing_before = df.isnull().sum().sum()

        # 处理ADR缺失值和异常值
        df = df.dropna(subset=['adr'])  # 删除价格缺失的行
        df = df[df['adr'] > 0]  # 删除价格为零或负数的记录

        missing_after = df.isnull().sum().sum()
        print(f"删除的缺失值行数: {missing_before - missing_after}")
        print(f"处理后行数: {len(df)}")

        # 2. 特征工程
        # 转换category类型以防止问题
        if df['arrival_date_month'].dtype != 'category':
            df['arrival_date_month'] = pd.Categorical(
                df['arrival_date_month'],
                categories=['January', 'February', 'March', 'April', 'May', 'June',
                            'July', 'August', 'September', 'October', 'November', 'December'],
                ordered=True
            )

        # 将月份转为数字
        month_map = {month: i for i, month in enumerate(['January', 'February', 'March', 'April', 'May', 'June',
                                                         'July', 'August', 'September', 'October', 'November',
                                                         'December'], 1)}
        df['arrival_month'] = df['arrival_date_month'].map(month_map)

        # 创建季节特征
        df['season'] = pd.cut(df['arrival_month'],
                              bins=[0, 3, 6, 9, 12],
                              labels=['Winter', 'Spring', 'Summer', 'Fall'],
                              include_lowest=True)

        # 创建工作日/周末特征
        df['is_weekend'] = df['arrival_date_day_of_month'].apply(lambda x: 1 if x % 7 in [0, 6] else 0)

        # 创建提前预订类别
        df['lead_time_category'] = pd.cut(df['lead_time'],
                                          bins=[-1, 7, 30, 90, 365, float('inf')],
                                          labels=['Last Minute', 'Short Notice', 'Medium', 'Long Term', 'Very Long'])

        # 创建入住时长特征
        df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

        # 创建每晚价格
        df['price_per_night'] = df['adr']  # ADR已经是每晚平均价格

        # 创建人均价格 (避免除零)
        df['price_per_person'] = df['adr'] / (df['adults'] + df['children'] + df['babies']).replace(0, 1)

        # 3. 编码分类变量
        cat_features = ['hotel', 'meal', 'market_segment', 'distribution_channel',
                        'reserved_room_type', 'assigned_room_type', 'customer_type']

        for feature in cat_features:
            if feature in df.columns:
                df[feature] = df[feature].astype('category').cat.codes

        self.processed_data = df
        print("数据预处理完成")

        # 打印预处理后的数据类型
        print("\n预处理后的数据类型:")
        print(df.dtypes.head(10))

        return df

    def analyze_price_discrimination(self):
        """分析价格歧视模式"""
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        df = self.processed_data

        print("\n开始价格歧视分析...")

        # 1. 按客户类型比较价格 - 添加ylim限制
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='customer_type', y='adr', data=self.data)
        plt.title('不同客户类型的价格分布')
        plt.xlabel('客户类型')
        plt.ylabel('平均每日价格(ADR)')
        plt.ylim(0, 600)  # 限制y轴范围为0-1000
        plt.grid(alpha=0.3)
        plt.savefig(self.results_dir / 'customer_type_price.png', dpi=300)
        plt.close()

        # 计算不同客户类型的平均价格
        customer_price = self.data.groupby('customer_type')['adr'].agg(['mean', 'median', 'std', 'count']).reset_index()
        print("\n不同客户类型的价格统计:")
        print(customer_price)

        # 2. 按预订提前期分析价格 - 添加ylim限制
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='lead_time_category', y='adr', data=df)
            plt.title('不同预订提前期的价格分布')
            plt.xlabel('预订提前期')
            plt.ylabel('平均每日价格(ADR)')
            plt.ylim(0, 600)  # 限制y轴范围为0-1000
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'lead_time_price.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"绘制预订提前期图表时出错: {e}")

        # 3. 按季节分析价格 - 添加ylim限制
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='arrival_date_month', y='adr', data=self.data)
        plt.title('不同月份的价格分布')
        plt.xlabel('月份')
        plt.ylabel('平均每日价格(ADR)')
        plt.ylim(0, 600)  # 限制y轴范围为0-1000
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'month_price.png', dpi=300)
        plt.close()

        # 4. 按酒店类型和客户类型的交叉分析 - 添加ylim限制
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='customer_type', y='adr', hue='hotel', data=self.data)
        plt.title('不同酒店和客户类型的价格分布')
        plt.xlabel('客户类型')
        plt.ylabel('平均每日价格(ADR)')
        plt.ylim(0, 600)  # 限制y轴范围为0-1000
        plt.grid(alpha=0.3)
        plt.legend(title='酒店类型')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'hotel_customer_price.png', dpi=300)
        plt.close()

        # 5. 价格歧视指数计算
        hotel_types = self.data['hotel'].unique()
        customer_types = self.data['customer_type'].unique()

        price_discrimination_index = {}

        for hotel in hotel_types:
            hotel_data = self.data[self.data['hotel'] == hotel]
            avg_prices = {}

            for customer in customer_types:
                customer_data = hotel_data[hotel_data['customer_type'] == customer]
                if len(customer_data) > 0:
                    avg_prices[customer] = customer_data['adr'].mean()

            if avg_prices:
                max_price = max(avg_prices.values())
                min_price = min(avg_prices.values())

                # 价格歧视指数 = 最高价格与最低价格之比
                if min_price > 0:
                    price_discrimination_index[hotel] = max_price / min_price
                else:
                    price_discrimination_index[hotel] = np.nan

        print("\n价格歧视指数(最高价格/最低价格):")
        for hotel, index in price_discrimination_index.items():
            print(f"{hotel}: {index:.4f}")

        # 创建热图显示价格歧视矩阵 - 修复部分
        try:
            # 创建一个新的DataFrame来保存价格数据
            price_data = []

            for hotel in hotel_types:
                for customer in customer_types:
                    hotel_customer_data = self.data[(self.data['hotel'] == hotel) &
                                                    (self.data['customer_type'] == customer)]

                    if len(hotel_customer_data) > 0:
                        avg_price = hotel_customer_data['adr'].mean()
                        price_data.append({
                            'hotel': hotel,
                            'customer_type': customer,
                            'avg_price': avg_price
                        })

            # 转换为宽格式DataFrame
            if price_data:
                price_df = pd.DataFrame(price_data)
                price_matrix = price_df.pivot(index='hotel', columns='customer_type', values='avg_price')

                # 绘制热图
                plt.figure(figsize=(10, 6))
                sns.heatmap(price_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
                plt.title('酒店-客户类型价格矩阵')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'price_discrimination_matrix.png', dpi=300)
                plt.close()
            else:
                print("无法创建价格矩阵：没有足够的数据")

        except Exception as e:
            print(f"生成价格矩阵热图时出错: {e}")

        return {
            'price_discrimination_index': price_discrimination_index,
            'customer_price_stats': customer_price
        }

    def build_price_prediction_model(self):
        """构建价格预测模型，用于理解价格影响因素"""
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        df = self.processed_data
        print("\n建立价格预测模型...")

        # 选择特征和目标变量
        potential_features = ['hotel', 'lead_time', 'arrival_month', 'stays_in_weekend_nights',
                              'stays_in_week_nights', 'adults', 'children', 'meal',
                              'market_segment', 'distribution_channel', 'is_repeated_guest',
                              'reserved_room_type', 'assigned_room_type', 'customer_type',
                              'required_car_parking_spaces', 'total_of_special_requests']

        # 确认所有特征都在数据集中
        features = [f for f in potential_features if f in df.columns]
        self.features = features

        # 检查是否有足够的特征
        if len(features) < 3:
            print("警告: 特征数量不足，可能影响模型质量")

        X = df[features]
        y = df['adr']

        # 检查数据是否有NaN值
        if X.isnull().any().any():
            print("警告: 特征中存在NaN值，进行简单填充")
            X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        try:
            # 训练XGBoost模型
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

            model.fit(X_train, y_train)

            # 评估模型
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"模型性能: MSE = {mse:.2f}, R² = {r2:.4f}")

            self.model = model
            self.X_test = X_test
            self.y_test = y_test

            return model
        except Exception as e:
            print(f"模型训练失败: {e}")
            return None

    def analyze_feature_importance(self):
        """分析影响价格的因素重要性"""
        if self.model is None:
            model = self.build_price_prediction_model()
            if model is None:
                print("无法分析特征重要性：模型构建失败")
                return {}

        try:
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title('酒店定价的特征重要性')
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [self.features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'feature_importance.png', dpi=300)
            plt.close()

            # 打印特征重要性
            print("\n特征重要性排名:")
            for i, idx in enumerate(indices):
                print(f"{i + 1}. {self.features[idx]}: {importance[idx]:.4f}")

            # 生成SHAP值解释
            try:
                if len(self.X_test) > 0:
                    explainer = shap.Explainer(self.model)
                    sample_size = min(100, len(self.X_test))
                    shap_values = explainer(self.X_test.iloc[:sample_size])

                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(shap_values, self.X_test.iloc[:sample_size], plot_type="bar", show=False)
                    plt.title("SHAP特征重要性")
                    plt.tight_layout()
                    plt.savefig(self.results_dir / 'shap_importance.png', dpi=300)
                    plt.close()
            except Exception as e:
                print(f"生成SHAP图时出错: {e}")

            return {
                'feature_names': self.features,
                'importance': importance,
                'indices': indices
            }
        except Exception as e:
            print(f"分析特征重要性时出错: {e}")
            return {}

    def simulate_regulation_impact(self, regulation_type='price_cap', params=None):
        """模拟不同监管策略对价格和消费者福利的影响"""
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        df = self.processed_data.copy()
        print(f"\n模拟{regulation_type}监管效果...")

        if params is None:
            params = {}

        original_prices = df['adr'].copy()
        original_mean = original_prices.mean()
        original_std = original_prices.std()

        # 使用论文中明确定义的区间值
        if regulation_type == 'price_range':
            # 固定使用论文中定义的区间 [56.81, 123.88]
            price_floor = 56.81
            price_cap = 123.88
            df['regulated_price'] = df['adr'].clip(lower=price_floor, upper=price_cap)

        elif regulation_type == 'price_cap':
            # 仅上限监管，使用论文中的上限值
            price_cap = 123.88
            df['regulated_price'] = df['adr'].clip(upper=price_cap)

        else:
            # 其他监管类型保持原有默认逻辑
            price_floor = params.get('floor', original_mean * 0.5)
            price_cap = params.get('cap', original_mean * 1.5)
            if regulation_type == 'price_floor':
                df['regulated_price'] = df['adr'].clip(lower=price_floor)
            else:
                df['regulated_price'] = df['adr'].clip(lower=price_floor, upper=price_cap)

        # 计算监管效果
        df['price_change'] = df['regulated_price'] - df['adr']

        # 计算消费者剩余变化 (假设需求弹性为-1)
        elasticity = params.get('elasticity', -1.0)
        df['demand_change_pct'] = elasticity * (df['regulated_price'] / df['adr'] - 1)
        df['original_demand'] = 1  # 标准化需求为1
        df['new_demand'] = df['original_demand'] * (1 + df['demand_change_pct'])

        # 消费者剩余变化 = 新需求×旧价格 - 新需求×新价格
        df['consumer_surplus_change'] = df['new_demand'] * (df['adr'] - df['regulated_price'])

        # 生产者剩余变化 = 新需求×新价格 - 旧需求×旧价格
        df['producer_surplus_change'] = df['new_demand'] * df['regulated_price'] - df['original_demand'] * df['adr']

        # 社会福利变化
        df['welfare_change'] = df['consumer_surplus_change'] + df['producer_surplus_change']

        # 结果汇总
        results = {
            'regulation_type': regulation_type,
            'params': params,
            'avg_original_price': original_mean,
            'avg_regulated_price': df['regulated_price'].mean(),
            'price_reduction_pct': (df['regulated_price'].mean() / original_mean - 1) * 100,
            'price_variation_before': original_std / original_mean,
            'price_variation_after': df['regulated_price'].std() / df['regulated_price'].mean(),
            'avg_consumer_surplus_change': df['consumer_surplus_change'].mean(),
            'avg_producer_surplus_change': df['producer_surplus_change'].mean(),
            'avg_welfare_change': df['welfare_change'].mean(),
            'pct_prices_affected': (df['adr'] != df['regulated_price']).mean() * 100
        }

        print(f"\n{regulation_type.upper()}监管效果:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

            # 创建更清晰的价格分布可视化
            plt.figure(figsize=(14, 8))

            # 添加直方图以清晰显示截断效果
            plt.hist(df['adr'], bins=50, alpha=0.3, color='blue', density=True, label='原始价格分布')
            plt.hist(df['regulated_price'], bins=50, alpha=0.3, color='orange', density=True, label='监管后价格分布')

            # 添加密度曲线
            sns.kdeplot(df['adr'], color='blue', label='_nolegend_')
            sns.kdeplot(df['regulated_price'], color='orange', label='_nolegend_')

            # 添加平均线和监管限制线
            plt.axvline(original_mean, color='blue', linestyle='--', label=f'原始平均价格: {original_mean:.2f}')
            plt.axvline(df['regulated_price'].mean(), color='orange', linestyle='--',
                        label=f'监管后平均价格: {df["regulated_price"].mean():.2f}')

            if regulation_type in ['price_cap', 'price_range']:
                plt.axvline(price_cap, color='red', linestyle='-',
                            label=f'价格上限: {price_cap:.2f}')

            if regulation_type in ['price_floor', 'price_range']:
                plt.axvline(price_floor, color='green', linestyle='-',
                            label=f'价格下限: {price_floor:.2f}')

            plt.title(f'{regulation_type.title()}监管前后的价格分布')
            plt.xlabel('价格')
            plt.ylabel('密度')
            plt.xlim(0, 300)
            plt.ylim(0, 0.03)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.results_dir / f'{regulation_type}_impact.png', dpi=300)
            plt.close()

        try:
            # 分析不同客户类型受监管影响的差异
            customer_impact = df.groupby('customer_type').agg(
                avg_price_before=('adr', 'mean'),
                avg_price_after=('regulated_price', 'mean'),
                avg_cs_change=('consumer_surplus_change', 'mean'),
                pct_affected=('price_change', lambda x: (x != 0).mean() * 100)
            ).reset_index()

            print("\n不同客户类型受监管影响:")
            print(customer_impact)

            # 可视化不同客户类型的福利变化
            plt.figure(figsize=(12, 6))
            sns.barplot(x='customer_type', y='avg_cs_change', data=customer_impact)
            plt.title(f'{regulation_type.title()}监管下不同客户类型的消费者剩余变化')
            plt.xlabel('客户类型')
            plt.ylabel('平均消费者剩余变化')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / f'{regulation_type}_customer_impact.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"分析客户类型影响时出错: {e}")

        return results, df

    def compare_regulation_strategies(self):
        """比较不同监管策略的效果"""
        print("\n比较不同监管策略...")

        # 1. 不同价格上限
        cap_results = []
        for cap_pct in [1.2, 1.5, 2.0]:
            try:
                result, _ = self.simulate_regulation_impact(
                    'price_cap', {'cap': self.processed_data['adr'].mean() * cap_pct})
                result['cap_pct'] = cap_pct
                cap_results.append(result)
            except Exception as e:
                print(f"价格上限{cap_pct}模拟失败: {e}")

        # 2. 不同价格区间
        range_results = []
        for width_pct in [0.3, 0.5, 0.7]:
            try:
                center = self.processed_data['adr'].mean()
                result, _ = self.simulate_regulation_impact(
                    'price_range', {
                        'floor': center * (1 - width_pct),
                        'cap': center * (1 + width_pct)
                    })
                result['width_pct'] = width_pct
                range_results.append(result)
            except Exception as e:
                print(f"价格区间{width_pct}模拟失败: {e}")

        # 检查是否有足够的结果进行比较
        if not cap_results or not range_results:
            print("警告: 没有足够的监管策略数据进行比较")
            return {'price_cap_results': cap_results, 'price_range_results': range_results}

        # 可视化不同策略的效果
        # 1. 消费者福利变化
        plt.figure(figsize=(10, 6))

        if cap_results:
            # 价格上限策略
            cap_pcts = [r['cap_pct'] for r in cap_results]
            cap_cs = [r['avg_consumer_surplus_change'] for r in cap_results]
            plt.plot(cap_pcts, cap_cs, 'o-', label='价格上限策略')

        if range_results:
            # 价格区间策略
            width_pcts = [r['width_pct'] for r in range_results]
            range_cs = [r['avg_consumer_surplus_change'] for r in range_results]
            plt.plot(width_pcts, range_cs, 's-', label='价格区间策略')

        plt.title('不同监管策略对消费者剩余的影响')
        plt.xlabel('监管参数 (上限倍数/区间宽度)')
        plt.ylabel('平均消费者剩余变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'regulation_cs_comparison.png', dpi=300)
        plt.close()

        # 2. 社会福利变化
        plt.figure(figsize=(10, 6))

        if cap_results:
            # 价格上限策略
            cap_welfare = [r['avg_welfare_change'] for r in cap_results]
            plt.plot(cap_pcts, cap_welfare, 'o-', label='价格上限策略')

        if range_results:
            # 价格区间策略
            range_welfare = [r['avg_welfare_change'] for r in range_results]
            plt.plot(width_pcts, range_welfare, 's-', label='价格区间策略')

        plt.title('不同监管策略对社会福利的影响')
        plt.xlabel('监管参数 (上限倍数/区间宽度)')
        plt.ylabel('平均社会福利变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'regulation_welfare_comparison.png', dpi=300)
        plt.close()

        # 3. 价格变异性影响
        if len(cap_results) > 0 and len(range_results) > 0:
            try:
                plt.figure(figsize=(10, 6))

                # 价格上限策略
                cap_variation = [(r['price_variation_after'] / r['price_variation_before'] - 1) * 100
                                 for r in cap_results]
                plt.plot(cap_pcts, cap_variation, 'o-', label='价格上限策略')

                # 价格区间策略
                range_variation = [(r['price_variation_after'] / r['price_variation_before'] - 1) * 100
                                   for r in range_results]
                plt.plot(width_pcts, range_variation, 's-', label='价格区间策略')

                plt.title('不同监管策略对价格变异性的影响')
                plt.xlabel('监管参数 (上限倍数/区间宽度)')
                plt.ylabel('价格变异系数变化百分比')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(self.results_dir / 'regulation_variation_comparison.png', dpi=300)
                plt.close()
            except Exception as e:
                print(f"绘制价格变异性比较图时出错: {e}")

        return {
            'price_cap_results': cap_results,
            'price_range_results': range_results
        }

    def map_to_triangle_model(self, sample_size=1000):
        """将酒店价格数据映射到三角形模型，与理论结果对比"""
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        print("\n映射数据到三角形模型...")

        # 按客户类型创建分布
        df = self.processed_data

        # 选择一个典型酒店类型进行分析
        hotel_types = df['hotel'].unique()
        if len(hotel_types) > 0:
            hotel_type = hotel_types[0]  # 选择第一个酒店类型
            hotel_data = df[df['hotel'] == hotel_type]

            try:
                # 按房型和客户类型分组，计算价格分布
                room_customer_prices = hotel_data.groupby(['reserved_room_type', 'customer_type'])['adr'].mean()

                # 简化为仅按价格分布
                prices = hotel_data['adr'].values
                prices = prices[~np.isnan(prices)]  # 移除NaN值

                if len(prices) == 0:
                    print("没有有效的价格数据进行三角形建模")
                    return {}

                # 创建直方图数据
                hist, bin_edges = np.histogram(prices, bins=10, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # 归一化
                values = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
                masses = hist / hist.sum()

                # 确保没有NaN值
                valid = ~np.isnan(values) & ~np.isnan(masses)
                values = values[valid]
                masses = masses[valid]

                if len(values) == 0 or len(masses) == 0:
                    print("归一化后没有有效数据进行三角形建模")
                    return {}

                # 从理论模型导入TriangleVisualizer（如果可用）
                try:
                    from utils.triangle_visualizer import TriangleVisualizer

                    # 创建可视化器
                    visualizer = TriangleVisualizer(np.array(masses), np.array(values))

                    # 测试不同的F值
                    F_values = [
                        [0.5, 0.5],  # 单点F
                        [0.3, 0.7],  # 中等区间F
                        [0.0, 1.0]  # 全范围F
                    ]

                    # 创建图表
                    fig, axes = plt.subplots(1, len(F_values), figsize=(15, 5))
                    if len(F_values) == 1:
                        axes = [axes]  # 转换为列表以便统一处理

                    results = []
                    for i, F in enumerate(F_values):
                        ax = axes[i]
                        visualizer.draw_triangles(F, ax=ax)

                        # 检查可行性并计算面积
                        is_feasible = visualizer.check_F_feasibility(F)
                        features = visualizer.analyze_triangle_features(F, use_exact_algorithm=True)

                        passive_area = features["passive_intermediary"]["area"]
                        active_area = features["active_intermediary"]["area"]

                        # 计算比率，避免除零
                        ratio = passive_area / active_area if active_area > 0 else float('inf')
                        ratio_str = f"{ratio:.4f}" if ratio != float('inf') else "Inf"

                        F_desc = "单点F" if F[0] == F[1] else f"区间F[{F[0]:.1f},{F[1]:.1f}]"
                        ax.set_title(f"{F_desc}\n可行: {is_feasible}, 比率: {ratio_str}")

                        results.append({
                            'F': F,
                            'is_feasible': is_feasible,
                            'passive_area': passive_area,
                            'active_area': active_area,
                            'ratio': ratio if ratio != float('inf') else 9999.99
                        })

                    plt.tight_layout()
                    plt.savefig(self.results_dir / 'triangle_model_mapping.png', dpi=300)
                    plt.close()

                    print("三角形模型映射完成")

                    return {
                        'values': values.tolist(),
                        'masses': masses.tolist(),
                        'F_results': results
                    }
                except ImportError as e:
                    print(f"无法导入TriangleVisualizer: {e}")
                except Exception as e:
                    print(f"三角形模型映射失败: {e}")

                # 创建基本散点图显示价格分布
                plt.figure(figsize=(10, 6))
                plt.scatter(bin_centers, hist)
                plt.title('酒店价格分布')
                plt.xlabel('价格')
                plt.ylabel('密度')
                plt.grid(alpha=0.3)
                plt.savefig(self.results_dir / 'price_distribution.png', dpi=300)
                plt.close()
            except Exception as e:
                print(f"映射到三角形模型时出错: {e}")

        return {
            'values': values.tolist() if 'values' in locals() else None,
            'masses': masses.tolist() if 'masses' in locals() else None
        }

    def run_complete_analysis(self, file_path):
        """运行完整的分析流程"""
        print("\n===== 开始酒店价格歧视分析 =====")

        try:
            # 1. 加载和预处理数据
            self.load_data(file_path)
            self.preprocess_data()

            # 2. 探索性分析
            discrimination_analysis = self.analyze_price_discrimination()

            # 3. 构建预测模型和特征重要性分析
            self.build_price_prediction_model()
            feature_importance = self.analyze_feature_importance()

            # 4. 监管效果模拟
            cap_results, _ = self.simulate_regulation_impact('price_cap')
            range_results, _ = self.simulate_regulation_impact('price_range')

            # 5. 比较不同监管策略
            regulation_comparison = self.compare_regulation_strategies()

            # 6. 三角形模型映射
            triangle_mapping = self.map_to_triangle_model()

            # 7. 生成总结报告
            self._generate_summary_report(
                discrimination_analysis,
                feature_importance,
                cap_results,
                range_results,
                regulation_comparison,
                triangle_mapping
            )

            print("\n分析完成! 结果保存在:", self.results_dir)
        except Exception as e:
            print(f"\n分析过程中发生错误: {e}")

    def _generate_summary_report(self, discrimination_analysis, feature_importance,
                                 cap_results, range_results, regulation_comparison,
                                 triangle_mapping):
        """生成分析总结报告"""

        report = [
            "# 酒店价格歧视分析与监管效果评估报告",
            "",
            "## 1. 价格歧视模式分析",
            ""
        ]

        # 添加价格歧视指数
        if discrimination_analysis and 'price_discrimination_index' in discrimination_analysis:
            report.append("### 1.1 价格歧视指数")
            report.append("")
            report.append("价格歧视指数是按客户类型计算的最高与最低平均价格之比：")
            report.append("")
            for hotel, index in discrimination_analysis['price_discrimination_index'].items():
                if not np.isnan(index):
                    report.append(f"- {hotel}: {index:.4f}")

        # 添加特征重要性
        report.extend([
            "",
            "## 2. 价格影响因素分析",
            "",
            "### 2.1 关键定价因素排名",
            ""
        ])

        if feature_importance and 'indices' in feature_importance:
            indices = feature_importance['indices']
            importance = feature_importance['importance']
            features = feature_importance['feature_names']

            for i in range(min(10, len(indices))):
                idx = indices[i]
                report.append(f"{i + 1}. {features[idx]}: {importance[idx]:.4f}")
        else:
            report.append("未能生成特征重要性分析")

        # 添加监管效果
        report.extend([
            "",
            "## 3. 监管效果分析",
            "",
            "### 3.1 价格上限监管",
            ""
        ])

        if cap_results:
            report.extend([
                f"- 平均原始价格: {cap_results['avg_original_price']:.2f}",
                f"- 平均监管后价格: {cap_results['avg_regulated_price']:.2f} ({cap_results['price_reduction_pct']:.2f}% 变化)",
                f"- 消费者剩余变化: {cap_results['avg_consumer_surplus_change']:.2f}",
                f"- 生产者剩余变化: {cap_results['avg_producer_surplus_change']:.2f}",
                f"- 净福利变化: {cap_results['avg_welfare_change']:.2f}",
                f"- 受影响价格比例: {cap_results['pct_prices_affected']:.2f}%"
            ])

        report.extend([
            "",
            "### 3.2 价格区间监管",
            ""
        ])

        if range_results:
            report.extend([
                f"- 平均原始价格: {range_results['avg_original_price']:.2f}",
                f"- 平均监管后价格: {range_results['avg_regulated_price']:.2f} ({range_results['price_reduction_pct']:.2f}% 变化)",
                f"- 消费者剩余变化: {range_results['avg_consumer_surplus_change']:.2f}",
                f"- 生产者剩余变化: {range_results['avg_producer_surplus_change']:.2f}",
                f"- 净福利变化: {range_results['avg_welfare_change']:.2f}",
                f"- 受影响价格比例: {range_results['pct_prices_affected']:.2f}%"
            ])

        # 添加不同监管策略比较
        cap_results_list = regulation_comparison.get('price_cap_results', [])
        range_results_list = regulation_comparison.get('price_range_results', [])

        if cap_results_list:
            report.extend([
                "",
                "### 3.3 监管参数敏感性分析",
                "",
                "#### 价格上限策略",
                "",
                "| 上限倍数 | 消费者剩余变化 | 生产者剩余变化 | 净福利变化 | 受影响价格比例 |",
                "| -------- | -------------- | -------------- | ---------- | -------------- |"
            ])

            for r in cap_results_list:
                report.append(f"| {r['cap_pct']:.2f} | {r['avg_consumer_surplus_change']:.2f} | "
                              f"{r['avg_producer_surplus_change']:.2f} | {r['avg_welfare_change']:.2f} | "
                              f"{r['pct_prices_affected']:.2f}% |")

        if range_results_list:
            report.extend([
                "",
                "#### 价格区间策略",
                "",
                "| 区间宽度 | 消费者剩余变化 | 生产者剩余变化 | 净福利变化 | 受影响价格比例 |",
                "| -------- | -------------- | -------------- | ---------- | -------------- |"
            ])

            for r in range_results_list:
                report.append(f"| {r['width_pct']:.2f} | {r['avg_consumer_surplus_change']:.2f} | "
                              f"{r['avg_producer_surplus_change']:.2f} | {r['avg_welfare_change']:.2f} | "
                              f"{r['pct_prices_affected']:.2f}% |")

        # 添加三角形模型映射结果
        if triangle_mapping and 'F_results' in triangle_mapping:
            report.extend([
                "",
                "## 4. 三角形理论模型映射",
                "",
                "| F值 | 可行性 | 被动中介面积 | 主动中介面积 | 面积比率 |",
                "| --- | ------ | ------------ | ------------ | -------- |"
            ])

            for r in triangle_mapping['F_results']:
                F = r['F']
                is_feasible = r['is_feasible']
                passive_area = r['passive_area']
                active_area = r['active_area']
                ratio = r['ratio']
                ratio_str = f"{ratio:.4f}" if ratio < 9999 else "无穷"

                report.append(
                    f"| {F} | {'是' if is_feasible else '否'} | {passive_area:.4f} | {active_area:.4f} | {ratio_str} |")

        # 添加结论
        report.extend([
            "",
            "## 5. 结论与政策建议",
            "",
            "### 5.1 主要发现",
            ""
        ])

        # 根据实际结果生成结论
        if discrimination_analysis and 'price_discrimination_index' in discrimination_analysis:
            avg_index = np.nanmean(list(discrimination_analysis['price_discrimination_index'].values()))
            report.append(f"- 酒店业存在显著的价格歧视现象，平均价格歧视指数为{avg_index:.4f}")

        if cap_results and range_results:
            cap_welfare = cap_results['avg_welfare_change']
            range_welfare = range_results['avg_welfare_change']
            better_strategy = "价格上限" if cap_welfare > range_welfare else "价格区间"
            report.append(f"- {better_strategy}监管在社会福利提升方面表现更好")

        if feature_importance and 'indices' in feature_importance and len(feature_importance['indices']) >= 3:
            top_features = [feature_importance['feature_names'][i] for i in feature_importance['indices'][:3]]
            report.append(f"- 影响酒店价格的三大关键因素是{', '.join(top_features)}")

        report.extend([
            "",
            "### 5.2 政策建议",
            "",
            "1. 根据分析结果，建议采用适度的价格区间监管，平衡效率与公平",
            "2. 监管参数应当基于市场特性定制，考虑季节性和客户类型差异",
            "3. 监管设计应当避免过度限制市场定价自由，保留合理的价格歧视空间",
            "4. 进一步研究应当关注不同类型酒店和不同区域的监管效果差异"
        ])

        # 保存报告
        try:
            with open(self.results_dir / "analysis_report.md", "w", encoding="utf-8") as f:
                f.write("\n".join(report))
            print(f"分析报告已保存至 {self.results_dir / 'analysis_report.md'}")
        except Exception as e:
            print(f"保存报告时出错: {e}")


if __name__ == "__main__":
    # 运行完整分析
    analyzer = HotelPricingAnalysis()

    # 设置文件路径
    data_path = "data/hotel_bookings.csv"  # 替换为实际路径

    try:
        analyzer.run_complete_analysis(data_path)
    except FileNotFoundError:
        print(f"错误: 未找到文件 {data_path}")