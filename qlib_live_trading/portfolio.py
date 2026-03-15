"""
portfolio.py: 目标组合生成模块（不涉及交易执行）

功能：
    根据模型预测分数生成目标持仓组合。
    支持：
        - TopK 股票选取
        - 等权分配（可后续拓展加权）
        - 可结合行业中性化处理的分数
"""

import pandas as pd


class PortfolioEngine:
    """
    组合构建引擎

    根据模型分数生成目标组合，返回 {股票代码: 权重}。
    与回测和实盘完全解耦，不涉及交易执行。
    """

    def __init__(self, topk=30):
        """
        初始化组合引擎

        参数
        ----
        topk : int
            每日选股数量上限
        """
        self.topk = topk

    def build_target_portfolio(self, scores: pd.Series):
        """
        构建目标组合
        参数
        ----
        scores : pd.Series
            index = 股票代码
            value = 模型预测分数（越高优先选入组合）
        返回
        ----
        dict
            {股票代码: 权重}，默认等权分配
        """

        # 按分数排序，分数高的优先选
        ranked = scores.sort_values(ascending=False)
        # 取 TopK
        selected = ranked.head(self.topk)
        # 等权分配
        weight = 1.0 / len(selected)
        portfolio = {s: weight for s in selected.index}
        return portfolio
