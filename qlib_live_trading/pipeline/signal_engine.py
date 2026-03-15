"""
signal_engine.py: 模型预测模块（支持 GenericModel 或 pickle 文件）

功能：
    - 接收训练好的 GenericModel 或 pickle 文件
    - 根据输入因子数据计算每只股票的预测分数
    - 输出带有 'score' 列的 DataFrame，供组合构建和回测使用
"""

import pickle
import pandas as pd
from typing import Union
from model import GenericModel


class SignalEngine:
    """
    模型信号引擎

    用于生成每日股票预测分数 (score)
    """

    def __init__(self, model_or_path: Union[str, GenericModel]):
        """
        初始化 SignalEngine

        参数
        ----
        model_or_path : str 或 GenericModel
            - str: pickle 文件路径，加载已训练好的模型
            - GenericModel: 直接传入训练好的 GenericModel 对象
        """
        if isinstance(model_or_path, str):
            # 从 pickle 文件加载
            with open(model_or_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"[*] 已加载 pickle 模型: {model_or_path}")
        elif isinstance(model_or_path, GenericModel):
            # 直接使用传入的 GenericModel
            self.model = model_or_path
            print(f"[*] 已使用传入的 GenericModel 实例")
        else:
            raise TypeError("model_or_path 必须是 str (pickle 文件路径) 或 GenericModel 对象")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
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

        if not features:
            raise ValueError("输入 DataFrame 中未找到 alpha 因子列")

        # 调用 GenericModel 的 predict 方法
        pred = self.model.predict(df[features])

        # 写入 score 列
        df["score"] = pred

        return df

    def save_model(self, path: str):
        """
        将当前模型保存为 pickle 文件，用于实盘或回测

        参数
        ----
        path : str
            保存路径
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[*] 模型已保存到 {path}")
