"""
backtest.py: 工业级 A 股回测引擎

功能说明：
- T+1 交易
- 涨停/跌停/停牌判断
- 买入手续费，卖出手续费 + 印花税
- 顺延 TopK 股票
- 权重组合收益计算
- 输出两条 equity curve: 有/无交易成本
- 支持基础回测指标计算：年化收益、最大回撤、夏普比率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qlib.data import D
from tqdm import tqdm
import matplotlib

# 设置中文字体，防止中文显示为方块
plt.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['axes.unicode_minus'] = False


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
        slippage=0.0005
    ):
        """
        初始化回测引擎

        Args:
            topk (int): 每日持仓数量
            commission (float): 买入/卖出佣金比例
            stamp_tax (float): 卖出印花税
            slippage (float): 滑点
        """
        self.topk = topk
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.market_cache = None  # 缓存全量市场数据，提升性能

    # ------------------------
    # 市场状态判断函数
    # ------------------------
    def get_market_info(self, stocks, date):
        """
        从缓存中获取指定日期和股票的市场数据
        """
        try:
            day_df = self.market_cache.loc[date]
            return day_df.loc[stocks]
        except KeyError:
            # 如果该日期市场无数据（节假日或缺失），返回空 DataFrame
            return pd.DataFrame()

    def is_limit_up(self, close, limit_up):
        """判断是否涨停"""
        return close >= limit_up

    def is_limit_down(self, close, limit_down):
        """判断是否跌停"""
        return close <= limit_down

    def is_suspended(self, volume):
        """判断是否停牌"""
        return volume == 0

    # ------------------------
    # 执行交易（T+1 + 顺延 TopK）
    # ------------------------
    def execute_trades(self, target_scores, holdings, market_df):
        """
        执行交易逻辑

        Args:
            target_scores (pd.Series): 股票打分，index=stock, value=score
            holdings (set): 当前持仓
            market_df (pd.DataFrame): 当日市场数据，索引为 stock
        Returns:
            new_holdings (set): 更新后的持仓
            cost (float): 当日交易成本
        """
        new_holdings = set()
        cost = 0.0

        # 排序 TopK
        ranked_stocks = target_scores.sort_values(ascending=False)
        selected_count = 0

        # -------------------- 买入 --------------------
        for stock, score in ranked_stocks.items():
            if selected_count >= self.topk:
                break
            if stock not in market_df.index:
                continue

            row = market_df.loc[stock].iloc[0]
            close = row["$close"]
            vol = row["$volume"]
            high_limit = row["$high_limit"]

            # 买入限制判断
            if self.is_suspended(vol) or self.is_limit_up(close, high_limit):
                continue

            new_holdings.add(stock)
            selected_count += 1

            # 买入成本
            if stock not in holdings:
                cost += self.commission

        # -------------------- 卖出 --------------------
        for stock in holdings:
            if stock in new_holdings:
                continue
            if stock not in market_df.index:
                # 当日缺失数据，无法卖出，保留
                new_holdings.add(stock)
                continue

            row = market_df.loc[stock].iloc[0]
            close = row["$close"]
            vol = row["$volume"]
            low_limit = row["$low_limit"]

            # 卖出限制判断
            if self.is_suspended(vol) or self.is_limit_down(close, low_limit):
                new_holdings.add(stock)
                continue

            # 卖出成本
            cost += self.commission + self.stamp_tax

        return new_holdings, cost

    # ------------------------
    # 计算组合收益
    # ------------------------
    def compute_return(self, holdings, date):
        """
        计算当日组合收益率

        Args:
            holdings (set): 当前持仓
            date (str or pd.Timestamp): 日期
        Returns:
            ret (float): 当日组合收益率
        """
        if len(holdings) == 0:
            return 0.0
        df = D.features(list(holdings), ["$close/Ref($close,1)-1"], start_time=date, end_time=date)
        if df.empty:
            return 0.0
        return df.mean().values[0]

    # ------------------------
    # 回测主函数
    # ------------------------
    def run(self, signal_df, portfolio_engine=None):
        """
        回测主逻辑

        Args:
            signal_df (pd.DataFrame): 信号数据，MultiIndex(date, stock) + score列
            portfolio_engine: 可选组合权重计算引擎
        Returns:
            result (dict): 包含 equity_no_cost, equity_cost, turnover
        """
        dates = sorted(signal_df.index.get_level_values(0).unique())
        all_stocks = signal_df.index.get_level_values(1).unique()

        # --------------------
        # 一次性加载全量市场数据，提升性能
        # --------------------
        self.market_cache = D.features(
            all_stocks,
            ["$close", "Ref($close,1)", "$high_limit", "$low_limit", "$volume"],
            start_time=dates[0],
            end_time=dates[-1]
        )

        holdings = set()
        equity_no_cost = [1.0]
        equity_cost = [1.0]
        turnover = []

        # T+1 持仓
        next_holdings = holdings.copy()

        # --------------------
        # 循环每日信号，执行回测
        # --------------------
        for date in tqdm(dates, desc="回测进度"):
            day_df = signal_df.loc[date]
            scores = day_df["score"]

            # 从缓存中切片当日市场数据
            market_df = self.get_market_info(list(scores.index), date)

            # 执行交易
            new_holdings, cost = self.execute_trades(scores, next_holdings, market_df)
            turnover.append(len(new_holdings.symmetric_difference(next_holdings)))
            next_holdings = new_holdings.copy()

            # 当日收益
            ret = self.compute_return(holdings, date)
            equity_no_cost.append(equity_no_cost[-1] * (1 + ret))
            equity_cost.append(equity_cost[-1] * (1 + ret - cost - self.slippage))

            holdings = next_holdings.copy()

        # 返回结果
        result = {
            "equity_no_cost": np.array(equity_no_cost),
            "equity_cost": np.array(equity_cost),
            "turnover": np.array(turnover)
        }
        return result

    # ------------------------
    # 绘制权益曲线
    # ------------------------
    def plot(self, result):
        """
        绘制回测结果图

        Args:
            result (dict): run() 返回的结果
        """
        plt.figure(figsize=(12, 6))
        plt.plot(result["equity_no_cost"], label="无交易费用")
        plt.plot(result["equity_cost"], label="考虑交易费用")
        plt.title("回测权益曲线")
        plt.xlabel("交易日")
        plt.ylabel("组合净值")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ------------------------
    # 计算基础回测指标
    # ------------------------
    @staticmethod
    def analyze_performance(equity_curve, periods_per_year=252):
        """
        计算基础回测指标：年化收益、最大回撤、夏普比率

        Args:
            equity_curve (np.array): 权益曲线
            periods_per_year (int): 每年交易日
        Returns:
            dict: 包含 'annual_return', 'max_drawdown', 'sharpe'
        """
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = equity_curve[-1] / equity_curve[0] - 1
        annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

        # 最大回撤
        cum_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cum_max) / cum_max
        max_drawdown = drawdown.min()

        # 夏普比率（无风险利率为0）
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan

        return {
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe
        }
