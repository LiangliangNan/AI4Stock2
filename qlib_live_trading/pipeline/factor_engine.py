"""
factor_engine.py : Alpha158 增量因子计算引擎

功能：
    计算 Alpha158 因子，并增量更新本地 Parquet 存储。
    避免全量重算，提高效率。

运行示例：

    # 全量计算或指定区间
    engine = FactorEngine()
    engine.update(start="2018-01-01", end="2026-01-01")

    # 每日增量更新
    engine.update(last_day, today)

性能：
    全量计算（全市场）       5-20 分钟
    增量更新（每日收盘）     1-2 秒
"""


import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from factor_store import FactorStore


class FactorEngine:

    """
    Alpha158 incremental factor engine
    """

    def __init__(self, instruments="main_board"):
        self.store = FactorStore()
        self.instruments = instruments

    def compute_range(self, start, end):
        handler = Alpha158(
            instruments=self.instruments,
            start_time=start,
            end_time=end,
            fit_start_time=start,
            fit_end_time=end
        )
        dataset = DatasetH(
            handler=handler,
            segments={"test": (start, end)},
            col_set="feature"
        )
        df = dataset.prepare("test")
        return df

    def update(self, start, end):
        """
        增量更新 Alpha158 因子
        """
        try:
            old = self.store.load_range(start, end)
            last = old.index.get_level_values(0).max()
            start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[*] 增量计算开始日期: {start}")
        except Exception:
            print(f"[*] 全量计算区间: {start} 到 {end}")

        df = self.compute_range(start, end)
        self.store.save(df)
        print(f"[+] 因子数据更新完成，保存至本地存储")

