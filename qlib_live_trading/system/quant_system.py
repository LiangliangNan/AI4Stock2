"""
# quant_system.py: 主系统（工业级量化流水线）

功能：
    实现 Alpha158 工业级因子计算 + 模型预测 + TopK组合 + 回测/实盘选股

特点：
    - 自动增量因子更新，无需手动调用 update_factors()
    - 回测/实盘统一流水线
    - 涨跌停、停牌、T+1、交易费用真实模拟

四步流水线：
1. 因子更新（自动）
2. 模型预测
3. 组合构建（TopK）
4. 回测/实盘选股
"""

import pandas as pd
from factor_store import FactorStore
from factor_engine import FactorEngine
from model import GenericModel
from signal_engine import SignalEngine
from portfolio import PortfolioEngine
from backtest import BacktestEngine

class QuantSystem:
    """
    QuantSystem 工业级量化系统

    自动管理因子存储、增量更新、模型预测、
    组合构建和回测/实盘流程
    """

    def __init__(self, topk=30):
        # -----------------------------
        # 因子存储
        # -----------------------------
        self.store = FactorStore()

        # -----------------------------
        # 因子计算引擎
        # -----------------------------
        self.factor_engine = FactorEngine()

        # -----------------------------
        # 模型路径
        # -----------------------------
        model_path = "models/lightgbm.pkl"

        # -----------------------------
        # SignalEngine 初始化
        # -----------------------------
        import os
        if os.path.exists(model_path):
            print(f"[*] 加载已有模型: {model_path}")
            self.signal = SignalEngine(model_path)

        else:
            print("[*] 未发现训练好的模型，开始训练 LightGBM")

            # 创建模型
            self.signal = SignalEngine(GenericModel("lightgbm"))

            # 训练并保存模型
            self._train_and_save_model(model_path)

        # -----------------------------
        # 组合构建
        # -----------------------------
        self.portfolio = PortfolioEngine(topk=topk)

        # -----------------------------
        # 回测引擎
        # -----------------------------
        self.backtest = BacktestEngine(topk=topk)


    def _train_and_save_model(self, model_path):
        """
        训练模型并保存
        """

        print("[*] 准备训练数据")

        # 训练区间
        train_start = "1996-01-01"
        train_end = "2023-12-31"
        print(f"[*] 模型训练用数据起止日期: {train_start} -> {train_end}")
        print(f"!!!TODO：以后任何回测和预测，判断日期不能与模型训练日期有重叠。调用QuantSystem.check_data_leakage()-----")

        # 确保因子存在
        self.ensure_factors(train_start, train_end)

        # 读取因子
        df = self.store.load_range(train_start, train_end)

        if df.empty:
            raise RuntimeError("训练数据为空")

        # 特征列
        exclude_cols = {"LABEL0", "year", "score"}
        features = [c for c in df.columns if c not in exclude_cols]

        X = df[features]
        y = df["LABEL0"]

        print(f"[*] 训练样本: {len(X)}")
        print(f"[*] 因子数量: {len(features)}")

        # 训练
        # print(f"[*] [DEBUG] system 特征列预览: {list(X.columns[:5])} ...")
        self.signal.fit(X, y)

        print("[*] 模型训练完成")

        # 保存模型
        import os
        os.makedirs("models", exist_ok=True)
        self.signal.save_model(model_path)


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
                # 使用 :%Y-%m-%d 格式化 Timestamp 对象
                print(f"[*] 增量更新因子: {last_date:%Y-%m-%d} -> {end}")
                self.factor_engine.update(last_date, end)

    def check_data_leakage(train_start, train_end, test_start, test_end):
        """
        检查训练集和测试集是否存在日期重叠（数据泄露）
        """
        t_start, t_end = pd.Timestamp(train_start), pd.Timestamp(train_end)
        v_start, v_end = pd.Timestamp(test_start), pd.Timestamp(test_end)

        # 判断区间是否有交集
        if not (t_end < v_start or v_end < t_start):
            print("\n" + "!" * 60)
            print("严重警告: 发现数据泄露 (Data Leakage)！")
            print(f"训练区间: {t_start.date()} 至 {t_end.date()}")
            print(f"测试区间: {v_start.date()} 至 {v_end.date()}")
            print("原因: 测试日期与训练日期存在重叠。回测结果将严重虚高，无法代表实盘表现！")
            print("!" * 60 + "\n")
            return True
        return False

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

        # 计算有成本的各项指标
        metrics = self.backtest.analyze_performance(result["equity_cost"])
        print("\t" + "=" * 40)
        print(f"\t回测报告 [{start} 至 {end}]")
        print("\t" + "-" * 40)
        print(f"\t\t年化收益: {metrics['annual_return']:>10.2%}")
        print(f"\t\t最大回撤: {metrics['max_drawdown']:>10.2%}")
        print(f"\t\t夏普比率: {metrics['sharpe']:>10.3f}")
        print("\t" + "=" * 40 + "\n")

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


"""
# --------------------------------------------
# Usage 示例
# --------------------------------------------
if __name__ == "__main__":
    # -----------------------------
    # 初始化 Qlib
    # -----------------------------
    qlib.init(
        provider_uri="data/qlib_data_cn",
        region="cn"
    )

    # -----------------------------
    # 1. 初始化量化系统
    # -----------------------------
    # topk: 每日选取前 topk 支持组合构建
    system = QuantSystem(topk=30)

    # -----------------------------
    # 2. 回测历史策略
    # -----------------------------
    # 回测时间段：2024-01-01 ~ 2025-12-31
    # 使用历史因子 + Qlib 行情数据生成策略净值曲线
    # 注意：回测期间无需重新抓取数据，默认使用 data/qlib_data_cn
    system.run_backtest("2024-01-01", "2025-12-31")

    # -----------------------------
    # 3. 当日选股（预测下一交易日组合）
    # -----------------------------
    # 以 2025-01-10 的因子和收盘行情计算每只股票预测分数
    # 因为 A 股遵循 T+1 交易规则，当日收盘生成组合，实际交易在下一交易日（2025-01-11）执行
    portfolio = system.recommend("2025-01-10")

    # 输出当日选股组合
    # 可能包含字段：['symbol', 'score', 'rank', 'weight']
    print(portfolio)

    # -----------------------------
    # 4. 后续操作建议
    # -----------------------------
    # 1) 可将 portfolio 保存为 CSV，供实盘交易系统使用
    # 2) 可循环多日生成连续组合
    # 3) 回测结果与实盘使用同一套接口，保证策略一致性
"""