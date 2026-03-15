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
        self.feature_names = None  # 保存训练时列顺序

        if isinstance(model_or_path, str):
            # 从 pickle 文件加载
            with open(model_or_path, "rb") as f:
                self.model = pickle.load(f)
            # --- [增强点] 如果加载的是 GenericModel 实例且它存过特征名，可以在这里恢复 ---
            # if hasattr(self.model, 'feature_names'): self.feature_names = self.model.feature_names
            print(f"[*] 已加载 pickle 模型: {model_or_path}")
        elif isinstance(model_or_path, GenericModel):
            # 直接使用传入的 GenericModel
            self.model = model_or_path
            print(f"[*] 新创建 GenericModel 实例")
        else:
            raise TypeError("model_or_path 必须是 str (pickle 文件路径) 或 GenericModel 对象")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        训练模型并保存训练时特征列顺序

        参数
        ----
        X : pd.DataFrame
            因子特征矩阵
        y : pd.Series
            标签
        """
        # 保存训练列顺序
        self.feature_names = X.columns.tolist()

        # 直接调用模型 fit，保留 DataFrame 以传递特征名
        self.model.fit(X, y)
        print("[*] 模型训练完成，已保存训练列顺序")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对因子数据进行预测，生成 score
        """
        # 自动选取因子列
        exclude_cols = {"LABEL0", "year", "score"}

        # --- [优化逻辑] ---
        if self.feature_names is not None:
            # 如果训练过，严格按照训练时的列和顺序提取
            # 如果 df 缺列，reindex 会补 NaN 而不是直接报错，保证程序不崩
            features = self.feature_names
            X = df.reindex(columns=features)
        else:
            # 如果是直接 load 进来的模型没经过 fit，走原始排除逻辑
            features = [c for c in df.columns if c not in exclude_cols]
            X = df[features]

        if X.shape[1] == 0:
            raise ValueError("未找到可用因子列")

        print(f"[*] 使用 {X.shape[1]} 个因子进行预测")
        # ============== DEBUG INFO ==============
        # print(f"=== DEBUG INPUT TO {getattr(self.model, 'model_type', 'PICKLE').upper()} ===")
        # print(type(X), X.shape, X.columns if isinstance(X, pd.DataFrame) else None)
        # ========================================
        # 核心：直接传入 DataFrame X，不带 .values
        df["score"] = self.model.predict(X)
        return df

    def save_model(self, path: str):
        """
        将当前模型保存为 pickle 文件
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[*] 模型已保存到 {path}")