import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
from pathlib import Path

class AirlinePricingAnalysis:
    def __init__(self):
        self.data = None
        self.model = None
        self.features = None

    def load_data(self, file_path):
        print(f"\nLoading data from: {file_path}")
        self.data = pd.read_csv(file_path)

        print(f"Dataset shape: {self.data.shape}")
        print("\nSample data:")
        print(self.data.head())
        return self.data

    def preprocess_data(self):
        if self.data is None:
            raise ValueError("Please load data first.")

        df = self.data.copy()

        # Drop unnecessary columns
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')

        # Create route feature
        df['route'] = df['source_city'] + '-' + df['destination_city']

        # Encode categorical variables
        cat_cols = ['airline', 'flight', 'source_city', 'departure_time',
                    'arrival_time', 'destination_city', 'class', 'stops', 'route']  # 添加route到列表
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes

        df = df.dropna()
        self.processed_data = df

        # 打印数据类型以便调试
        print("\nData types after preprocessing:")
        print(df.dtypes.head())

        print(f"\nProcessed data shape: {df.shape}")
        return df

    def analyze_price_discrimination(self):
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        df = self.processed_data

        Path("results").mkdir(exist_ok=True)

        # Price distribution by airline
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='airline', y='price', data=df)
        plt.title('Price Distribution by Airline')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/airline_price_comparison.svg')
        plt.close()

        # Price vs. days_left
        if 'days_left' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='days_left', y='price', data=df)
            plt.title('Price vs Days to Flight')
            plt.tight_layout()
            plt.savefig('results/price_vs_days.svg')
            plt.close()

        # Price distribution by route
        top_routes = df['route'].value_counts().head(10).index
        route_df = df[df['route'].isin(top_routes)]

        plt.figure(figsize=(14, 7))
        sns.boxplot(x='route', y='price', data=route_df)
        plt.title('Price Distribution by Popular Routes')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('results/route_price_comparison.svg')
        plt.close()

        # Price discrimination metrics
        price_stats = df.groupby('route')['price'].agg(['min', 'max', 'mean', 'std', 'count'])
        price_stats['price_range'] = price_stats['max'] - price_stats['min']
        price_stats['price_variation'] = price_stats['std'] / price_stats['mean']
        price_stats = price_stats[price_stats['count'] >= 10]

        print("\nTop routes with highest price variation:")
        print(price_stats.sort_values('price_variation', ascending=False).head(10))

        return price_stats

    def build_price_prediction_model(self):
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        df = self.processed_data

        # 检查所有列的数据类型
        for col in df.columns:
            if df[col].dtype.name not in ['int64', 'float64', 'bool']:
                print(f"Column {col} with type {df[col].dtype} is already encoded as numbers")

        y = df['price']
        X = df.drop(['price'], axis=1)
        self.features = X.columns.tolist()

        print("\nFeatures used in the model:")
        for feature in self.features:
            print(f"{feature}: {X[feature].dtype}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 不使用enable_categorical参数
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
            # 移除 enable_categorical=True
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance: MSE = {mse:.2f}, R² = {r2:.4f}")

        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        return model

    def analyze_feature_importance(self, save_dir=None):
        if self.model is None:
            self.build_price_prediction_model()

        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Airline Pricing')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.features[i] for i in indices], rotation=90)
        plt.tight_layout()

        if save_dir:
            plt.savefig(Path(save_dir) / 'feature_importance.svg', dpi=300)
        plt.show()

        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.X_test.iloc[:100])

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, self.X_test.iloc[:100], plot_type="bar")

        if save_dir:
            plt.savefig(Path(save_dir) / 'shap_importance.svg', dpi=300)
        plt.show()

        return {
            'feature_names': self.features,
            'importance': importance,
            'indices': indices
        }

    def simulate_regulation_impact(self, regulation_type='price_cap', params=None):
        """
        模拟不同监管策略对价格和消费者福利的影响

        regulation_type: 监管类型 ('price_cap', 'price_floor', 'price_range')
        params: 监管参数，如价格上限、下限等
        """
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()

        df = self.processed_data.copy()

        if params is None:
            params = {}

        original_prices = df['price'].copy()
        original_mean = original_prices.mean()
        original_std = original_prices.std()

        # 设置默认监管参数
        if regulation_type == 'price_cap':
            # 默认上限为平均价格的1.5倍
            price_cap = params.get('cap', original_mean * 1.5)
            df['regulated_price'] = df['price'].clip(upper=price_cap)

        elif regulation_type == 'price_floor':
            # 默认下限为平均价格的0.5倍
            price_floor = params.get('floor', original_mean * 0.5)
            df['regulated_price'] = df['price'].clip(lower=price_floor)

        elif regulation_type == 'price_range':
            # 默认范围为平均价格的0.5-1.5倍
            price_floor = params.get('floor', original_mean * 0.5)
            price_cap = params.get('cap', original_mean * 1.5)
            df['regulated_price'] = df['price'].clip(lower=price_floor, upper=price_cap)

        # 计算监管效果
        df['price_change'] = df['regulated_price'] - df['price']

        # 计算消费者剩余变化 (假设需求弹性为-1)
        elasticity = params.get('elasticity', -1.0)
        df['demand_change_pct'] = elasticity * (df['regulated_price'] / df['price'] - 1)
        df['original_demand'] = 1  # 标准化需求为1
        df['new_demand'] = df['original_demand'] * (1 + df['demand_change_pct'])

        # 消费者剩余变化 = 新需求×旧价格 - 新需求×新价格
        df['consumer_surplus_change'] = df['new_demand'] * (df['price'] - df['regulated_price'])

        # 生产者剩余变化 = 新需求×新价格 - 旧需求×旧价格
        df['producer_surplus_change'] = df['new_demand'] * df['regulated_price'] - df['original_demand'] * df['price']

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
            'pct_prices_affected': (df['price'] != df['regulated_price']).mean() * 100
        }

        print(f"\n{regulation_type.upper()} Regulation Effects:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

        # 可视化价格分布变化
        plt.figure(figsize=(12, 6))
        sns.kdeplot(df['price'], label='Original Price')
        sns.kdeplot(df['regulated_price'], label='Regulated Price')
        plt.axvline(original_mean, color='blue', linestyle='--', label='Original Mean Price')
        plt.axvline(df['regulated_price'].mean(), color='orange', linestyle='--', label='Regulated Mean Price')

        if regulation_type in ['price_cap', 'price_range']:
            plt.axvline(params.get('cap', original_mean * 1.5), color='red',
                        linestyle='-', label='Price Cap')

        if regulation_type in ['price_floor', 'price_range']:
            plt.axvline(params.get('floor', original_mean * 0.5), color='green',
                        linestyle='-', label='Price Floor')

        plt.title(f'Price Distribution Before and After {regulation_type.title()} Regulation')
        plt.xlabel('Price')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/{regulation_type}_impact.svg')
        plt.close()

        return results, df

    def compare_regulation_strategies(self):
        """比较不同监管策略的效果"""
        # 1. 不同价格上限
        cap_results = []
        for cap_pct in [1.2, 1.5, 2.0]:
            result, _ = self.simulate_regulation_impact(
                'price_cap', {'cap': self.processed_data['price'].mean() * cap_pct})
            result['cap_pct'] = cap_pct
            cap_results.append(result)

        # 2. 不同价格区间
        range_results = []
        for width_pct in [0.3, 0.5, 0.7]:
            center = self.processed_data['price'].mean()
            result, _ = self.simulate_regulation_impact(
                'price_range', {
                    'floor': center * (1 - width_pct),
                    'cap': center * (1 + width_pct)
                })
            result['width_pct'] = width_pct
            range_results.append(result)

        # 可视化不同策略的效果
        # 1. 消费者福利变化
        plt.figure(figsize=(10, 6))

        # 价格上限策略
        cap_pcts = [r['cap_pct'] for r in cap_results]
        cap_cs = [r['avg_consumer_surplus_change'] for r in cap_results]
        plt.plot(cap_pcts, cap_cs, 'o-', label='Price Cap Strategy')

        # 价格区间策略
        width_pcts = [r['width_pct'] for r in range_results]
        range_cs = [r['avg_consumer_surplus_change'] for r in range_results]
        plt.plot(width_pcts, range_cs, 's-', label='Price Range Strategy')

        plt.title('Impact of Different Regulation Strategies on Consumer Surplus')
        plt.xlabel('Regulation Parameter (Cap Multiple / Range Width)')
        plt.ylabel('Average Consumer Surplus Change')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/regulation_cs_comparison.svg')
        plt.close()

        # 2. 社会福利变化
        plt.figure(figsize=(10, 6))

        # 价格上限策略
        cap_welfare = [r['avg_welfare_change'] for r in cap_results]
        plt.plot(cap_pcts, cap_welfare, 'o-', label='Price Cap Strategy')

        # 价格区间策略
        range_welfare = [r['avg_welfare_change'] for r in range_results]
        plt.plot(width_pcts, range_welfare, 's-', label='Price Range Strategy')

        plt.title('Impact of Different Regulation Strategies on Social Welfare')
        plt.xlabel('Regulation Parameter (Cap Multiple / Range Width)')
        plt.ylabel('Average Welfare Change')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/regulation_welfare_comparison.svg')
        plt.close()

        return {
            'price_cap_results': cap_results,
            'price_range_results': range_results
        }
