"""
# system.py: 主系统（工业级量化流水线）

功能：
    实现 Alpha158 工业级因子计算 + 行业中性化 + 模型预测 + TopK组合 + 回测/实盘选股

特点：
    - 自动增量因子更新，无需手动调用 update_factors()
    - 回测/实盘统一流水线
    - 涨跌停、停牌、T+1、交易费用真实模拟
    - 支持行业中性化

五步流水线：
1. 因子更新（自动）
2. 行业中性化（可选）
3. 模型预测
4. 组合构建（TopK）
5. 回测/实盘选股
"""

from factor_store import FactorStore
from factor_engine import FactorEngine
from neutralize import industry_neutralize
from signal_engine import SignalEngine
from portfolio import PortfolioEngine
from backtest import BacktestEngine


class QuantSystem:
    """
    QuantSystem 工业级量化系统

    自动管理因子存储、增量更新、行业中性化、模型预测、组合构建和回测/实盘流程
    """

    def __init__(self, topk=30, neutralize=True):
        # 因子存储
        self.store = FactorStore()

        # 因子计算引擎
        self.factor_engine = FactorEngine()

        # 模型预测
        self.signal = SignalEngine("models/model.pkl")

        # 组合构建
        self.portfolio = PortfolioEngine(topk=topk)

        # 回测引擎
        self.backtest = BacktestEngine(topk=topk,
                                       neutralize_func=industry_neutralize if neutralize else None)

        self.neutralize = neutralize

    # ----------------------------------------
    # 因子检查 & 增量更新
    # ----------------------------------------
    def ensure_factors(self, start, end):
        """
        检查因子是否存在，缺失则自动增量更新
        """
        df = self.store.load_range(start, end)
        if df.empty:
            print("[*] 因子缺失，自动计算因子")
            self.factor_engine.update(start, end)
        else:
            # 检查最后日期是否覆盖
            last_date = df.index.get_level_values(0).max()
            if pd.Timestamp(last_date) < pd.Timestamp(end):
                print(f"[*] 增量更新因子: {last_date} -> {end}")
                self.factor_engine.update(last_date, end)

    # ----------------------------------------
    # 回测
    # ----------------------------------------
    def run_backtest(self, start, end):
        """
        回测主函数

        参数：
            start, end: 回测日期区间
        """
        # 确保因子覆盖
        self.ensure_factors(start, end)

        # 读取因子
        df = self.store.load_range(start, end)

        # 预测
        df = self.signal.predict(df)

        # 回测
        result = self.backtest.run(df, self.portfolio)
        self.backtest.plot(result)
        return result

    # ----------------------------------------
    # 实盘选股
    # ----------------------------------------
    def recommend(self, date):
        """
        返回指定交易日的目标组合
        用 date 日的因子和收盘行情计算每只股票的预测分数（score），然后生成目标组合。
        因为 A 股遵循 T+1 交易规则，所以当天收盘后选出的组合，实际交易在下一交易日执行。
        """
        # 确保因子覆盖
        self.ensure_factors(date, date)

        # 读取因子
        df = self.store.load_range(date, date)

        # 预测
        df = self.signal.predict(df)

        # 提取当天分数
        scores = df.loc[date]["score"]

        # 构建目标组合
        portfolio = self.portfolio.build_target_portfolio(scores)
        return portfolio


# --------------------------------------------
# Usage 示例
# --------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    system = QuantSystem(topk=30, neutralize=True)

    # 回测 2022-01-01 ~ 2024-12-31
    system.run_backtest("2022-01-01", "2024-12-31")

    # 当日选股 2025-01-10
    # 用2025-01-10的因子和收盘行情计算每只股票的预测分数，然后生成目标组合。
    # 因为 A 股遵循 T+1 交易规则，所以当天收盘后选出的组合，实际交易会在 下一交易日（2025-01-11） 执行。
    portfolio = system.recommend("2025-01-10")
    print(portfolio)
