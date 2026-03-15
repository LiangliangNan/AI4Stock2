"""
factor_store.py: Parquet 因子仓库（Alpha158 工业级缓存）

功能：
    - 将 Alpha158 因子按年存储在 Parquet 文件中
    - 提供按日期区间读取功能
    - 支持增量更新，避免重复写入
    - IO 高效，适合全市场回测与实时推荐
"""


import pandas as pd
from pathlib import Path


class FactorStore:
    """
    Industrial grade factor storage.

    Stores factors in parquet partitions by year.
    """

    def __init__(self, root="data/factors/alpha158"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def file(self, year):
        return self.root / f"{year}.parquet"

    def load_range(self, start, end):
        start_year = pd.Timestamp(start).year
        end_year = pd.Timestamp(end).year

        dfs = []

        for y in range(start_year, end_year + 1):
            f = self.file(y)
            if f.exists():
                dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()  # 如果没有数据返回空 DataFrame

        df = pd.concat(dfs, ignore_index=False)
        df = df.loc[start:end]
        return df

    def save(self, df):
        df = df.copy()
        df["year"] = df.index.get_level_values(0).year

        for y, g in df.groupby("year"):
            f = self.file(y)
            if f.exists():
                old = pd.read_parquet(f)
                g = pd.concat([old, g], ignore_index=False)
                g = g[~g.index.duplicated(keep='last')]
            g.to_parquet(f)
            print(f"[+] 因子数据已保存: {f} (共 {len(g)} 行)")

