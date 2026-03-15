# QuantSystem 工业级 A股量化交易系统

## 概述
本系统实现了一个完整的 A 股量化交易流水线，基于 **Alpha158 因子**，支持 **增量因子更新、行业中性化、模型预测、TopK组合构建、机构级回测/实盘选股**。  
设计目标是工业级可靠性，兼顾速度和准确性，适合科研、策略开发和模拟实盘验证。

---

## 系统模块及功能说明
```text
project/
├─ data/             数据层
│  ├─ raw/              抓取的原始数据
│  │  ├─ daily/            AkShare 抓取的 HFQ 日线行情 parquet
│  │  └─ valuation/        AkShare 抓取的估值 parquet（PE、PB、流通市值等）
│  ├─ processed/           daily + valuation + industry 列合并后的 parquet
│  └─ qlib_data_cn/        Qlib 二进制数据（dump_bin 生成，用于模型训练/回测/推荐）
├─ pipeline/         信号计算和因子处理的核心逻辑
│  ├─ factor_engine.py     因子计算引擎（Alpha158），支持全量计算和增量更新，避免每天重算全量因子
│  ├─ factor_store.py      管理因子缓存，避免重复计算。按年分区存储 Alpha158 因子，支持按日期区间读取和增量写入
│  ├─ portfolio.py         生成组合（Top-K）。支持 TopK 股票选取和等权分配，返回 `{stock: weight}` 字典 
│  ├─ get_data.py          提供干净的数据源（完成抓取、增量更新、合并、转换到 Qlib 的整个过程）
│  ├─ model.py             预测模型模块。提供多种可选模型：LightGBM、XGBoost、PyTorch MLP、LSTM、Transformer。提供统一接口 fit/predict
│  └─ signal_engine.py     模型预测，引入训练好的模型对因子生成每日预测分数 (`score`) 
├─ system/           策略执行和回测，与数据源分离
│  ├─ backtest.py          工业级回测引擎，支持涨跌停、停牌、T+1、交易费用、滑点、行业中性化、生成净值曲线、指标报告
│  └─ quant_system.py      主系统，整合以上模块，实现完整流水线及回测/实盘统一接口
└─ main.py
```

---

## 五步完整流水线

1. **因子更新（Factor Update）**
    - 使用 `FactorEngine` + `FactorStore`
    - 第一次运行会生成全量历史因子（按年分区 Parquet）
        ```
        data/factors/alpha158/
            2018.parquet
            2019.parquet
            ...
        ```
    - 日常收盘增量更新，只计算缺失日期的因子，避免全量重算
    - 系统自动管理，无需手动调用

2. **模型预测（Signal Prediction）**
    - 使用 `SignalEngine` 对因子数据生成每只股票预测分数 `score`
    - 输出 DataFrame 增加 `score` 列
    - 可直接作为组合构建和回测输入

3. **组合构建（Portfolio Construction）**
    - 使用 `PortfolioEngine` 按 TopK 分数构建目标组合
    - 支持行业约束，可等权分配
    - 输出 `{stock: weight}` 字典

4. **回测 / 实盘选股（Backtest / Live Recommendation）**
    - 回测使用 `BacktestEngine`，考虑涨跌停、停牌、T+1、交易费用、滑点等真实交易条件
    - 支持绘制收益曲线（有交易费用 vs 无交易费用）
    - 实盘选股直接调用 `recommend(date)` 返回当天目标组合

---

## 安装与依赖

```bash
pip install qlib pandas matplotlib pyarrow scikit-learn
```

---

## Usage 示例

```python
import pandas as pd
from system import QuantSystem

# 初始化系统
system = QuantSystem(topk=30, neutralize=True)

# 回测
# 系统会自动检查因子数据，并进行增量计算
result = system.run_backtest(start="2022-01-01", end="2024-12-31")

# 当日选股
# 获取指定交易日目标组合
portfolio = system.recommend("2025-01-10")
print(portfolio)
```

---

## 性能参考（真实量化系统水平）

| 任务                  | 耗时      |
|----------------------|----------|
| Alpha158 全市场计算    | 8-20 min |
| 因子增量更新           | 1-3 sec  |
| 全市场预测             | 0.2 sec  |
| 回测 5 年              | 3-10 sec |

---

## 注意事项

1. 系统默认使用 **CSI300 股票池**，可根据需要修改 `FactorEngine` 中的 `instruments`。
2. 行业中性化可选，如果不需要设置 `QuantSystem(neutralize=False)`。
3. 回测与实盘使用同一套函数接口，确保策略一致性。
4. 回测输出两条曲线：
   - **无交易费用**：理想净值增长
   - **考虑交易费用**：真实换手成本净值
5. 建议在 **Python 3.12 + 最新 Qlib** 环境下运行，确保 **NumPy 2.x** 兼容。
