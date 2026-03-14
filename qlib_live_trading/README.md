完整的 Qlib 实盘（live inference）选股框架，结构接近真实量化系统。
    - 训练与预测使用同一个 handler
    - processor 的 fit 参数不会改变
    - 预测只使用 feature，不使用 label
    - 每日可以对未来一天进行预测
核心软件：Microsoft Qlib

一、项目结构
qlib_live_trading/
    data/
    models/
    train_model.py    训练模型（只偶尔运行。滚动训练：每月或每周重新训练）
    predict_today.py  每日预测
    update_data.py    更新数据
    config.py         公共配置

二、每日运行流程
每天收盘后：
    - Step1 更新行情
    - Step2 运行预测脚本
    - Step3 输出第二天买入列表

执行：

python predict_today.py


输出：

Tomorrow Buy List

SH600519
SZ000858
SH600036
...