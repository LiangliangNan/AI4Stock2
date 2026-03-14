# system_backtest.py
import pandas as pd
import pickle
from qlib.backtest import backtest as qlbt
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data.dataset import DatasetH
from pathlib import Path

class QuantSystem:
    def __init__(self, model, handler):
        self.model = model
        self.handler = handler

    def build_dataset(self, start, end):
        """构建 DatasetH 对象，用于回测或预测"""
        segments = {"test": (start, end)}
        dataset = DatasetH(handler=self.handler, segments=segments, col_set="feature")
        return dataset

    def run_backtest(self, start="2025-01-01", end=None, topk=20, n_drop=0):
        """回测整个区间"""
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        print(f"[*] 回测时间区间: {start} ~ {end}")

        dataset = self.build_dataset(start, end)

        # 1️⃣ 模型预测
        print("[*] 正在生成预测信号...")
        preds = self.model.predict(dataset)
        if preds.empty:
            raise ValueError("[!] 错误：预测结果为空")

        dataset.df["score"] = preds.iloc[:, 0]  # 预测信号列

        # 2️⃣ 配置策略
        strategy = TopkDropoutStrategy(topk=topk, n_drop=n_drop, predict_score="score")

        # 3️⃣ 配置执行器
        executor = SimulatorExecutor()
        print("[*] 正在执行回测...")
        account_df, trades_df = qlbt(
            dataset.df,
            strategy=strategy,
            executor=executor
        )

        print("[*] 回测完成")
        return account_df, trades_df

    def recommend_today(self, topk=20, n_drop=0):
        """收盘后生成今日推荐股票"""
        latest_day = pd.Timestamp.today().strftime("%Y-%m-%d")
        lookback_days = 100
        calendar = self.handler.calendar
        inference_start_day = calendar[-lookback_days].strftime("%Y-%m-%d")

        dataset = self.build_dataset(start=inference_start_day, end=latest_day)

        print(f"[*] 生成 {latest_day} 收盘后推荐信号...")
        preds = self.model.predict(dataset)
        if preds.empty:
            print("[!] 错误：预测结果为空")
            return pd.DataFrame()

        # 取最新一天预测
        actual_latest_date = preds.index.get_level_values(0).max()
        latest_preds = preds.loc[actual_latest_date].copy()
        latest_preds["score"] = latest_preds.iloc[:, 0]

        # 排序 topk
        recommended = latest_preds.sort_values("score", ascending=False).head(topk)
        print(f"[✅] 今日推荐前 {topk} 股票：")
        print(recommended[["score"]])
        return recommended

# ===== Usage =====
if __name__ == "__main__":
    # 1️⃣ 加载模型与 handler
    model_path = Path("results/lgbm/model.pkl")
    handler_path = Path("results/lgbm/handler.pkl")
    model = pickle.load(open(model_path, "rb"))
    handler = pickle.load(open(handler_path, "rb"))

    qs = QuantSystem(model, handler)

    # 2️⃣ 回测示例
    account_df, trades_df = qs.run_backtest(
        start="2025-01-01",
        end="2025-03-10",
        topk=30,
        n_drop=5
    )
    print("[回测账户余额示例]")
    print(account_df.head())
    print("[回测交易记录示例]")
    print(trades_df.head())

    # 3️⃣ 收盘后今日推荐
    today_recommendation = qs.recommend_today(topk=20, n_drop=0)
    print(today_recommendation)
