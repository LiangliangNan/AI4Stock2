import pickle
import qlib

from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel

import config


############################################
# 初始化 Qlib
############################################

qlib.init(
    provider_uri=config.QLIB_PROVIDER_URI,
    region=REG_CN
)

############################################
# 构建 handler
############################################

handler = Alpha158(
    instruments=config.MARKET,
    start_time=config.TRAIN_START,
    end_time=config.TRAIN_END,
    fit_start_time=config.FIT_START,
    fit_end_time=config.FIT_END
)

############################################
# Dataset
############################################

dataset = DatasetH(
    handler=handler,
    segments={
        "train": (config.TRAIN_START, config.FIT_END),
        "valid": (config.FIT_END, config.TRAIN_END)
    }
)

############################################
# 模型
############################################

model = LGBModel(
    loss="mse",
    learning_rate=0.05,
    num_leaves=210,
    n_estimators=300,
    subsample=0.9,
    colsample_bytree=0.9
)

print("Training model...")
model.fit(dataset)

############################################
# 保存模型
############################################

with open(config.MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(config.HANDLER_PATH, "wb") as f:
    pickle.dump(handler, f)

print("Model and handler saved")
