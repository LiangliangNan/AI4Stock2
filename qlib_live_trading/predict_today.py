import pickle
import pandas as pd
import qlib

from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH

import config


############################################
# 设置预测日期
############################################

today = "2026-03-14"


############################################
# 初始化 Qlib
############################################

qlib.init(
    provider_uri=config.QLIB_PROVIDER_URI,
    region=REG_CN
)

############################################
# 加载模型
############################################

with open(config.MODEL_PATH, "rb") as f:
    model = pickle.load(f)

############################################
# 加载 handler
############################################

with open(config.HANDLER_PATH, "rb") as f:
    handler = pickle.load(f)

############################################
# 扩展 handler 数据范围
############################################

handler.config(
    start_time=config.TRAIN_START,
    end_time=today
)

############################################
# Dataset (仅 inference)
############################################

dataset = DatasetH(
    handler=handler,
    segments={
        "infer": (today, today)
    }
)

############################################
# 只取 feature
############################################

feature = dataset.prepare(
    "infer",
    col_set="feature"
)

feature = feature.dropna()

############################################
# 预测
############################################

pred = model.predict(feature)

pred = pd.DataFrame(
    pred,
    index=feature.index,
    columns=["score"]
)

############################################
# 选 TopK
############################################

topk = pred.sort_values(
    "score",
    ascending=False
).head(20)

print("\nTomorrow Buy List")
print(topk)

topk.to_csv("tomorrow_buy_list.csv")
