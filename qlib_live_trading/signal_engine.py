"""
signal_engine.py: 模型预测模块

功能：
    - 加载训练好的模型
    - 根据输入因子数据计算每只股票的预测分数
    - 输出带有 'score' 列的 DataFrame，供组合构建和回测使用
"""

import pickle


class SignalEngine:
    """
    模型信号引擎

    用于生成每日股票预测分数 (score)
    """

    def __init__(self, model_path):
        """
        初始化模型

        参数
        ----
        model_path : str
            训练好的模型文件路径（pickle 格式）
        """
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, df):
        """
        对因子数据进行预测，生成 score

        参数
        ----
        df : pd.DataFrame
            MultiIndex: (date, stock)
            包含所有因子列，列名中包含 "alpha"

        返回
        ----
        pd.DataFrame
            在原 DataFrame 上增加 'score' 列
        """
        # 自动选取因子列
        features = [c for c in df.columns if "alpha" in c]

        # 模型预测
        pred = self.model.predict(df[features])

        # 写入 score 列
        df["score"] = pred

        return df
