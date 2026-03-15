"""
neutralize.py: 行业中性化工具

说明：
- 对因子进行行业中性化处理，使得因子在行业间可比
- 经典方法：
    factor_industry_neutral = factor - industry_mean
- 可选扩展：市值中性化
    回归模型：factor ~ industry + log(market_cap)
    使用回归残差作为中性化因子

用法：
    import neutralize
    df_neutral = neutralize.industry_neutralize(df, industry_col="industry")
"""

import pandas as pd


def industry_neutralize(df: pd.DataFrame, industry_col: str = "industry") -> pd.DataFrame:
    """
    对因子数据进行行业中性化处理

    参数：
    -----------
    df : pd.DataFrame
        MultiIndex 或普通 DataFrame，列包含因子，如 alpha1 ... alpha158
    industry_col : str
        行业列名，用于分组计算行业均值

    返回：
    -----------
    df_neutral : pd.DataFrame
        行业中性化后的因子 DataFrame，原有列被覆盖
    """

    # 选择所有 alpha 因子列（假设列名中包含 'alpha'）
    fac_cols = [c for c in df.columns if "alpha" in c]

    # 按行业分组
    grouped = df.groupby(industry_col)

    # 对每一列因子做行业中性化
    for col in fac_cols:
        df[col] = df[col] - grouped[col].transform("mean")

    return df


if __name__ == "__main__":
    # 测试示例
    data = {
        "industry": ["A", "A", "B", "B"],
        "alpha1": [0.5, 0.7, 0.2, 0.4],
        "alpha2": [1.0, 1.2, 0.8, 0.6],
    }
    df = pd.DataFrame(data)
    print("原始因子：")
    print(df)

    df_neutral = industry_neutralize(df, industry_col="industry")
    print("\n行业中性化后因子：")
    print(df_neutral)
