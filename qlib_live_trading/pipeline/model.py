"""
model.py: 通用预测模型模块

功能：
    - 提供多种可选模型：LightGBM、XGBoost、PyTorch MLP、LSTM、Transformer
    - 提供统一接口 fit/predict，便于 SignalEngine 使用
    - 支持通过参数灵活切换模型类型
"""

from typing import Optional, Union
import numpy as np
import pandas as pd

# 机器学习库
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# PyTorch 相关
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None

# -----------------------------
# PyTorch 模型定义
# -----------------------------
class MLPModel(nn.Module):
    """简单多层感知机 (MLP)"""
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 返回一维预测值

class LSTMModel(nn.Module):
    """单层 LSTM"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze(-1)

class TransformerModel(nn.Module):
    """简单 Transformer Encoder + 回归头"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len=1, feature)
        x = self.linear_in(x)
        x = self.transformer(x)  # 输出 (batch, seq_len, d_model)
        out = self.fc_out(x[:, -1, :])  # 取最后时间步
        return out.squeeze(-1)

# -----------------------------
# 通用模型包装器
# -----------------------------
class GenericModel:
    """
    通用模型接口

    参数
    ----
    model_type : str
        可选 'lightgbm', 'xgboost', 'mlp', 'lstm', 'transformer'
    kwargs : dict
        传入具体模型的参数，如 hidden_dims、d_model 等
    """
    def __init__(self, model_type: str = "lightgbm", **kwargs):
        self.model_type = model_type.lower()
        self.model = None
        self.kwargs = kwargs

        if self.model_type == "lightgbm":
            if lgb is None:
                raise ImportError("请先安装 lightgbm")
            self.model = lgb.LGBMRegressor(**kwargs)
        elif self.model_type == "xgboost":
            if xgb is None:
                raise ImportError("请先安装 xgboost")
            self.model = xgb.XGBRegressor(**kwargs)
        elif self.model_type in ["mlp", "lstm", "transformer"]:
            if torch is None:
                raise ImportError("请先安装 torch")
            self.model = None  # PyTorch 模型将在 fit 时初始化
        else:
            raise ValueError(f"未知 model_type: {model_type}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
        """
        训练模型

        参数
        ----
        X : pd.DataFrame 或 np.ndarray
            特征矩阵
        y : pd.Series 或 np.ndarray
            目标变量
        epochs : int
            PyTorch 模型训练轮数
        batch_size : int
            PyTorch 批量大小
        lr : float
            PyTorch 学习率
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y

        if self.model_type in ["lightgbm", "xgboost"]:
            self.model.fit(X_np, y_np)
        else:
            # PyTorch 模型
            input_dim = X_np.shape[1]
            if self.model_type == "mlp":
                self.model = MLPModel(input_dim, **self.kwargs)
            elif self.model_type == "lstm":
                self.model = LSTMModel(input_dim, **self.kwargs)
                X_np = X_np[:, np.newaxis, :]  # (batch, seq_len=1, feature)
            elif self.model_type == "transformer":
                self.model = TransformerModel(input_dim, **self.kwargs)
                X_np = X_np[:, np.newaxis, :]  # (batch, seq_len=1, feature)

            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            dataset = TensorDataset(torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                epoch_loss = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                print(f"[{self.model_type}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.6f}")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测分数

        参数
        ----
        X : pd.DataFrame 或 np.ndarray
            特征矩阵

        返回
        ----
        np.ndarray
            一维预测分数
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        if self.model_type in ["lightgbm", "xgboost"]:
            return self.model.predict(X_np)
        else:
            if self.model_type in ["lstm", "transformer"]:
                X_np = X_np[:, np.newaxis, :]  # (batch, seq_len=1, feature)
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.tensor(X_np, dtype=torch.float32)).numpy()



"""
GenericModel 使用示例

`GenericModel` 支持以下模型：
    - LightGBM (`"lightgbm"`)
    - XGBoost (`"xgboost"`)
    - PyTorch MLP (`"mlp"`)
    - PyTorch LSTM (`"lstm"`)
    - PyTorch Transformer (`"transformer"`)

所有模型都支持统一接口：
    model.fit(X_train, y_train, **kwargs)
    y_pred = model.predict(X_test)

    对于 PyTorch 模型，可以通过 epochs、batch_size、lr 等参数调整训练策略。
    LightGBM/XGBoost 是直接表格输入 (samples, features)，PyTorch LSTM/Transformer 需要 (samples, seq_len, features)。

1. LightGBM 示例
    from model import GenericModel
    # 初始化 LightGBM 模型
    model = GenericModel("lightgbm", n_estimators=100, max_depth=5, learning_rate=0.05)
    # 拟合训练数据
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)

2. XGBoost 示例
    from model import GenericModel
    # 初始化 XGBoost 模型
    model = GenericModel("xgboost", n_estimators=200, max_depth=6, learning_rate=0.1)
    # 拟合训练数据
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)

3. PyTorch MLP 示例
    from model import GenericModel
    # 初始化 MLP 模型
    model = GenericModel("mlp", input_dim=X_train.shape[1], hidden_dims=[64, 32], output_dim=1)
    # 训练
    model.fit(X_train, y_train, epochs=20, batch_size=128, lr=1e-3)
    # 预测
    y_pred = model.predict(X_test)
    
4. PyTorch LSTM 示例
    from model import GenericModel
    # 初始化 LSTM 模型
    model = GenericModel("lstm", input_dim=X_train.shape[2], hidden_dim=64, num_layers=2, output_dim=1, seq_len=X_train.shape[1])
    # 训练
    model.fit(X_train, y_train, epochs=30, batch_size=64, lr=1e-3)
    # 预测
    y_pred = model.predict(X_test)
  注意: 对 LSTM 输入数据要求是三维 (samples, seq_len, features)。

5. PyTorch Transformer 示例
    from model import GenericModel
    # 初始化 Transformer 模型
    model = GenericModel("transformer", input_dim=X_train.shape[2], d_model=64, nhead=4, num_layers=2, seq_len=X_train.shape[1], output_dim=1)
    # 训练
    model.fit(X_train, y_train, epochs=25, batch_size=64, lr=1e-3)
    # 预测
    y_pred = model.predict(X_test)
  注意: Transformer 输入数据同样为三维 (samples, seq_len, features)，并且通常需要归一化或标准化。
  
"""