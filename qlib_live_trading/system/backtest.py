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
import matplotlib

class BacktestEngine:
    """
    工业级 A 股回测引擎
    """

    def __init__(self, topk=30, commission=0.0003, stamp_tax=0.001, slippage=0.0005):
        """
        初始化回测引擎
        """
        self.topk = topk
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.market_cache = None  # 缓存全量市场数据

    def _clean_symbols(self, symbols):
        """统一代码格式"""
        return [str(s).replace('SZ', '').replace('SH', '').split('.')[0].zfill(6) for s in symbols]

    def get_market_info(self, stocks, date):
        """从缓存中获取指定日期和股票的市场数据"""
        if self.market_cache is None or self.market_cache.empty:
            return pd.DataFrame()
        try:
            target_date = pd.Timestamp(date)
            # 索引对齐：处理 Qlib 默认的 instrument-datetime 索引
            day_df = self.market_cache.loc[target_date]

            clean_stocks = self._clean_symbols(stocks)
            # 确保 index 类型一致（字符串）
            day_df.index = day_df.index.astype(str)
            valid_index = day_df.index.intersection(clean_stocks)
            return day_df.loc[valid_index]
        except (KeyError, ValueError):
            return pd.DataFrame()

    def is_limit_up(self, close, limit_up):
        return False if limit_up is None or np.isnan(limit_up) else close >= limit_up

    def is_limit_down(self, close, limit_down):
        return False if limit_down is None or np.isnan(limit_down) else close <= limit_down

    def is_suspended(self, volume):
        return volume == 0 or np.isnan(volume)

    def execute_trades(self, target_scores, holdings, market_df):
        """执行交易逻辑（T+1 + 顺延 TopK）"""
        new_holdings = set()
        cost = 0.0
        if market_df.empty:
            return holdings, 0.0

        ranked_stocks = target_scores.sort_values(ascending=False)
        selected_count = 0

        # 买入/持有逻辑
        for stock, score in ranked_stocks.items():
            if selected_count >= self.topk: break
            clean_s = self._clean_symbols([stock])[0]
            if clean_s not in market_df.index: continue

            row = market_df.loc[clean_s]
            close = row["$close"]
            vol = row.get("$volume", 0)
            high_limit = row.get("$high_limit", np.nan)

            # 停牌或涨停无法买入（除非已经在持仓中）
            if self.is_suspended(vol) or self.is_limit_up(close, high_limit):
                if stock in holdings:
                    new_holdings.add(stock)
                    selected_count += 1
                continue

            new_holdings.add(stock)
            selected_count += 1
            if stock not in holdings:
                cost += self.commission + self.slippage

        # 卖出逻辑
        for stock in holdings:
            if stock in new_holdings: continue
            clean_s = self._clean_symbols([stock])[0]
            if clean_s not in market_df.index:
                new_holdings.add(stock) # 缺失数据被迫保留
                continue

            row = market_df.loc[clean_s]
            close, vol = row["$close"] , row.get("$volume", 0)
            low_limit = row.get("$low_limit", np.nan)

            # 停牌或跌停无法卖出
            if self.is_suspended(vol) or self.is_limit_down(close, low_limit):
                new_holdings.add(stock)
                continue

            cost += self.commission + self.stamp_tax + self.slippage

        return new_holdings, cost / self.topk

    def compute_return(self, holdings, date):
        """使用缓存计算组合收益"""
        if not holdings or self.market_cache is None: return 0.0
        try:
            day_df = self.market_cache.loc[pd.Timestamp(date)]
            clean_holdings = self._clean_symbols(holdings)
            day_df.index = day_df.index.astype(str)
            valid_stocks = day_df.index.intersection(clean_holdings)
            if len(valid_stocks) == 0: return 0.0

            prices = day_df.loc[valid_stocks]
            return (prices['$close'] / prices['Ref($close,1)'] - 1).mean()
        except:
            return 0.0

    def run(self, signal_df, portfolio_engine=None):
        """回测主函数，保留原参数接口"""
        # 预清洗信号数据的股票代码
        clean_stock_levels = self._clean_symbols(signal_df.index.levels[1])
        signal_df.index = signal_df.index.set_levels(clean_stock_levels, level=1)

        dates = sorted(signal_df.index.get_level_values(0).unique())
        all_stocks = signal_df.index.get_level_values(1).unique()

        # 加载全量市场数据缓存
        print("[*] 正在加载市场数据缓存...")
        fields = ["$close", "$volume", "Ref($close,1)"]
        try:
            cache = D.features(list(all_stocks), fields + ["$high_limit", "$low_limit"],
                               start_time=dates[0], end_time=dates[-1])
        except:
            cache = D.features(list(all_stocks), fields, start_time=dates[0], end_time=dates[-1])

        # 核心：转换为 datetime 为一级索引
        self.market_cache = cache.swaplevel().sort_index()

        holdings = set()
        equity_no_cost = [1.0]
        equity_cost = [1.0]
        daily_rets = []

        for date in dates:
            # A. 计算收益
            ret = self.compute_return(holdings, date)
            daily_rets.append(ret)

            # B. 调仓
            day_df = signal_df.loc[date]
            market_df = self.get_market_info(list(day_df.index), date)
            new_holdings, cost = self.execute_trades(day_df["score"], holdings, market_df)

            # C. 更新权益
            equity_no_cost.append(equity_no_cost[-1] * (1 + ret))
            equity_cost.append(equity_cost[-1] * (1 + ret - cost))
            holdings = new_holdings

        positive_rets = [r for r in daily_rets if r > 0]
        negative_rets = [r for r in daily_rets if r < 0]
        print(f"[*] 回测完成: 总天数={len(daily_rets)}, 正收益天数={len(positive_rets)}, 负收益天数={len(negative_rets)}")

        return {
            "equity_no_cost": np.array(equity_no_cost),
            "equity_cost": np.array(equity_cost),
            "dates": [dates[0]] + list(dates) # 保留日期序列用于绘图
        }

    def plot(self, result):
        """
        绘制精简版回测曲线
        - 保留日期横轴
        - 添加 1.0 盈亏平衡参考线
        - 性能指标与图例统一放置在左上角顶部
        """
        dates = result["dates"]
        equity_cost = result["equity_cost"]
        equity_no_cost = result["equity_no_cost"]

        plt.figure(figsize=(12, 7))

        # 1. 绘制 1.0 盈亏基准横线
        plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.8)

        # 2. 绘制净值线
        plt.plot(dates, equity_no_cost, label="No fee", color="#3498db", alpha=0.6, linewidth=1)
        plt.plot(dates, equity_cost, label="With fee", color="#e74c3c", linewidth=2)

        # 3. 性能指标标注 (放置在左上角最顶部)
        perf = self.analyze_performance(equity_cost)
        info_text = (f"Annual return: {perf['annual_return']:.2%}\n"
                     f"Max drawdown: {perf['max_drawdown']:.2%}\n"
                     f"Sharp rate: {perf['sharpe']:.2f}")

        # 使用 transform=plt.gca().transAxes 相对坐标
        # x=0.02 (左偏), y=0.96 (极靠顶), verticalalignment='top'
        t = plt.text(0.02, 0.96, info_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     fontsize=10,
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.8),
                     zorder=5)

        # 4. 图例放置 (紧挨在指标框下面)
        # loc='upper left'，并通过 bbox_to_anchor 微调位置，y 坐标设在 0.78 左右避开文字
        plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.78), frameon=True, fontsize=10)

        # 5. 图表美化
        plt.title(f"Backtest accumulated value ({dates[0].date()} ~ {dates[-1].date()})", fontsize=14, pad=20)
        plt.xlabel("Trade date")
        plt.ylabel("Accumulated value")
        plt.grid(True, axis='y', linestyle=':', alpha=0.5)

        # 自动旋转日期标记
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.show()
        plt.savefig("backtest_result.png")

    @staticmethod
    def analyze_performance(equity_curve, periods_per_year=252):
        if len(equity_curve) < 2: return {"annual_return": 0, "max_drawdown": 0, "sharpe": 0}
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = equity_curve[-1] / equity_curve[0] - 1
        days = len(returns)
        annual_return = (1 + total_return) ** (periods_per_year / days) - 1 if days > 0 else 0
        cum_max = np.maximum.accumulate(equity_curve)
        max_drawdown = ((equity_curve - cum_max) / cum_max).min()
        std = returns.std()
        sharpe = (returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0
        return {"annual_return": annual_return, "max_drawdown": max_drawdown, "sharpe": sharpe}