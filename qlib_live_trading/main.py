"""
main.py - 简化量化系统入口示例
--------------------------------
功能：
1. 回测历史策略表现
2. 当日选股预测组合（T+1 执行）

依赖：
- pipeline/prepare_data.py 已完成数据抓取、融合和 Qlib 转换
- system/quant_system.py 已实现 QuantSystem 类

目录约定：
- data/qlib_data_cn/ : Qlib 二进制行情数据目录
"""
import qlib

import sys
from pathlib import Path
# 把相关目录加入 Python 模块搜索路径
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "pipeline"))
sys.path.append(str(Path(__file__).resolve().parent / "system"))

from system.quant_system import QuantSystem
import pandas as pd


if __name__ == "__main__":
    # -----------------------------
    # 初始化 Qlib
    # -----------------------------
    qlib.init(
        provider_uri="data/qlib_data_cn",
        region="cn"
    )

    # -----------------------------
    # 1. 初始化量化系统
    # -----------------------------
    # topk: 每日选取前 topk 支持组合构建
    system = QuantSystem(topk=10)

    # -----------------------------
    # 2. 回测历史策略
    # -----------------------------
    # 回测时间段：2024-01-01 ~ 2025-12-31
    # 使用历史因子 + Qlib 行情数据生成策略净值曲线
    # 注意：回测期间无需重新抓取数据，默认使用 data/qlib_data_cn
    system.run_backtest("2024-01-01", "2025-12-31")

    # -----------------------------
    # 3. 当日选股（预测下一交易日组合）
    # -----------------------------
    # 以 2025-01-10 的因子和收盘行情计算每只股票预测分数
    # 因为 A 股遵循 T+1 交易规则，当日收盘生成组合，实际交易在下一交易日（2025-01-11）执行
    portfolio = system.recommend("2025-01-10")

    # 输出当日选股组合
    # 可能包含字段：['symbol', 'score', 'rank', 'weight']
    print(portfolio)

    # -----------------------------
    # 4. 后续操作建议
    # -----------------------------
    # 1) 可将 portfolio 保存为 CSV，供实盘交易系统使用
    # 2) 可循环多日生成连续组合
    # 3) 回测结果与实盘使用同一套接口，保证策略一致性
