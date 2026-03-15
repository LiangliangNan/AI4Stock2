"""
backtest.py: 工业级 A 股回测引擎

功能说明：
- T+1 交易
- 涨停/跌停/停牌判断
- 买入手续费，卖出手续费 + 印花税
- 行业中性化因子支持
- 权重组合收益计算
- 输出两条 equity curve: 有/无交易成本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qlib.data import D


class BacktestEngine:
    """
    工业级 A 股回测引擎

    特性：
    - T+1 持仓
    - 涨停/跌停/停牌判断
    - 买入手续费，卖出手续费 + 印花税
    - 顺延 TopK 股票
    - 权重组合收益计算
    - 输出两条 equity curve: 有/无交易成本
    """

    def __init__(
        self,
        topk=30,
        commission=0.0003,
        stamp_tax=0.001,
        slippage=0.0005,
        neutralize_func=None
    ):
        self.topk = topk
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.neutralize_func = neutralize_func

    # ------------------------
    # 行业中性化
    # ------------------------
    def apply_neutralize(self, df):
        if self.neutralize_func is not None:
            df = self.neutralize_func(df)
        return df

    # ------------------------
    # 市场状态
    # ------------------------
    def get_market_info(self, stocks, date):
        df = D.features(
            stocks,
            ["$close", "Ref($close,1)", "$high_limit", "$low_limit", "$volume"],
            start_time=date,
            end_time=date
        )
        return df

    def is_limit_up(self, close, limit_up):
        return close >= limit_up

    def is_limit_down(self, close, limit_down):
        return close <= limit_down

    def is_suspended(self, volume):
        return volume == 0

    # ------------------------
    # 执行交易（T+1 + 顺延 TopK）
    # ------------------------
    def execute_trades(self, target_scores, holdings, market_df):
        """
        target_scores: pd.Series, index=stock, value=score
        holdings: set 当前持仓
        market_df: 当日市场数据
        """
        new_holdings = set()
        cost = 0.0

        # 排序 TopK，顺延处理
        ranked_stocks = target_scores.sort_values(ascending=False)
        selected_count = 0
        for stock, score in ranked_stocks.items():
            if selected_count >= self.topk:
                break
            if stock not in market_df.index:
                continue
            row = market_df.loc[stock]
            close = row["$close"]
            vol = row["$volume"]
            high_limit = row["$high_limit"]
            low_limit = row["$low_limit"]

            # 判断能否买入
            if self.is_suspended(vol) or self.is_limit_up(close, high_limit):
                continue

            new_holdings.add(stock)
            selected_count += 1

            # 买入成本
            if stock not in holdings:
                cost += self.commission

        # 卖出不在新持仓的股票
        for stock in holdings:
            if stock not in new_holdings:
                if stock not in market_df.index:
                    continue
                row = market_df.loc[stock]
                close = row["$close"]
                vol = row["$volume"]
                low_limit = row["$low_limit"]
                if self.is_suspended(vol) or self.is_limit_down(close, low_limit):
                    # 涨停或停牌无法卖，留在仓内
                    new_holdings.add(stock)
                    continue
                cost += self.commission + self.stamp_tax

        return new_holdings, cost

    # ------------------------
    # 组合收益
    # ------------------------
    def compute_return(self, holdings, date):
        if len(holdings) == 0:
            return 0.0
        df = D.features(list(holdings), ["$close/Ref($close,1)-1"], start_time=date, end_time=date)
        if df.empty:
            return 0.0
        return df.mean().values[0]

    # ------------------------
    # 回测主函数
    # ------------------------
    def run(self, signal_df, portfolio_engine):
        signal_df = self.apply_neutralize(signal_df)
        dates = sorted(signal_df.index.get_level_values(0).unique())

        holdings = set()
        equity_no_cost = [1.0]
        equity_cost = [1.0]
        turnover = []

        # T+1 持仓
        next_holdings = holdings.copy()

        for date in dates:
            day_df = signal_df.loc[date]
            scores = day_df["score"]

            market_df = self.get_market_info(list(scores.index), date)

            # 执行上一日 signal -> 今日持仓
            new_holdings, cost = self.execute_trades(scores, next_holdings, market_df)
            turnover.append(len(new_holdings.symmetric_difference(next_holdings)))
            next_holdings = new_holdings.copy()

            # 当日收益
            ret = self.compute_return(holdings, date)
            equity_no_cost.append(equity_no_cost[-1] * (1 + ret))
            equity_cost.append(equity_cost[-1] * (1 + ret - cost - self.slippage))

            holdings = next_holdings.copy()

        result = {
            "equity_no_cost": np.array(equity_no_cost),
            "equity_cost": np.array(equity_cost),
            "turnover": np.array(turnover)
        }
        return result

    # ------------------------
    # 绘图
    # ------------------------
    def plot(self, result):
        plt.figure(figsize=(12, 6))
        plt.plot(result["equity_no_cost"], label="无交易费用")
        plt.plot(result["equity_cost"], label="考虑交易费用")
        plt.title("回测权益曲线")
        plt.xlabel("交易日")
        plt.ylabel("组合净值")
        plt.grid(True)
        plt.legend()
        plt.show()
