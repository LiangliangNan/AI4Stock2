"""
=========================================================================================
AI4Stock2 - 极简版每日个股推荐脚本 (Daily Top-K Stock Recommender)
=========================================================================================

【系统核心逻辑与架构说明】

1. 特征与数据处理 (Feature & Processors)：
   - 基础特征：使用 Qlib 内置的 Alpha158 量价特征。
   - 扩展特征：融合了 8 个基本面估值因子（PE, PB, PS, PCF, PEG, 总市值, 流通市值, 换手率）。
   - 数据清洗：在推断（Inference）阶段，模型会对近期的特征数据执行 `RobustZScoreNorm`（去极值+标准化）和 `Fillna`（缺失值填充）。

2. 交易逻辑与标签设计 (Label & Trading Logic)：
   - 预测目标 (Label)：`Ref($open, -2)/Ref($open, -1) - 1`。
     这代表 T 日预测，T+1 日早盘集合竞价（开盘价）买入，T+2 日开盘价卖出的真实收益率。
   - 执行时间：本脚本应当在 T 日（交易日）的收盘后运行，获取推荐列表，然后用于 T+1 日的交易。

3. 时序数据切片 (Time-Series Dataset)：
   - 深度学习模型（如 LSTM, Transformer）不是只看当天的截面数据，而是需要回顾过去一段时间（Lookback，通常为 60 天）的数据序列。
   - 因此，即便是为了预测今天这一天，脚本也会自动向前提拉过去 100 个交易日的数据，以拼凑出完整的时序窗口供模型推断。

【使用方法 (Usage)】
请在项目根目录下运行以下命令：
    # 默认使用 LSTM 模型，推荐 Top 5
    python recommend.py

    # 指定使用 Transformer 模型，推荐 Top 10
    python recommend.py --model transformer --top_k 10

【前置要求 (Prerequisites)】
运行此脚本前，必须确保 Qlib 数据库已更新到最新交易日。
你可以通过运行 `python main.py --download-only` 来完成数据更新。
=========================================================================================
"""

import argparse
import pickle
import yaml
import pandas as pd
from pathlib import Path

# 导入 Qlib 的核心组件
from qlib.data import D
from src.data_setup import init_qlib
from src.features import build_alpha158_handler
from src.dataset import build_ts_dataset, build_tabular_dataset


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    # -------------------------------------------------------------------------
    # 第一步：解析命令行参数
    # 让用户可以在不改动代码的情况下，通过终端灵活调整模型类型和推荐数量。
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Generate Daily Top K Stock Recommendations")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--model", default="lgbm", help="使用的模型名称：lstm / transformer / lgbm")
    parser.add_argument("--top_k", type=int, default=10, help="需要推荐的个股数量")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 第二步：初始化 Qlib 和加载模型
    # -------------------------------------------------------------------------
    cfg = load_config(args.config)

    # 启动 Qlib 环境，指向你的数据目录（通常是 ~/.qlib/qlib_data/cn_data）
    init_qlib(provider_uri=cfg["qlib"]["provider_uri"], region=cfg["qlib"]["region"])

    # 定位到你之前训练好的模型文件 (model.pkl)
    model_path = Path("results") / args.model / "model.pkl"
    if not model_path.exists():
        # 如果模型不存在，抛出明确的错误提示，避免程序崩溃得不明不白
        raise FileNotFoundError(f"[错误] 在 {model_path} 没有找到模型。请先运行 main.py 训练模型。")

    # 使用 pickle 反序列化加载模型到内存中
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[*] 成功加载模型: {args.model.upper()} (路径: {model_path})")

    # -------------------------------------------------------------------------
    # 第三步：确定数据提取的时间窗口
    # -------------------------------------------------------------------------
    # 获取 Qlib 数据库中存在的所有交易日历
    calendar = D.calendar()

    # 找到数据库里最新的一天（通常是你刚刚下载完数据的今天）
    latest_day = calendar[-1].strftime("%Y-%m-%d")

    # 为什么要往前推 100 天？
    # 因为 LSTM 等时序模型需要过去 step_len（例如 60 天）的数据作为输入特征。
    # 我们提取 100 天，是为了给特征处理器（比如去极值、标准化）提供足够的近期样本来计算均值和方差。
    lookback_days = 100
    start_day = calendar[-lookback_days].strftime("%Y-%m-%d")

    print(f"[*] 正在从数据库提取数据，时间区间：{start_day} 到 {latest_day}...")

    # -------------------------------------------------------------------------
    # 第四步：构造数据处理器 (Data Handler)
    # -------------------------------------------------------------------------
    # Handler 的作用是从底层的 Bin 文件中把日线数据（开高低收、因子）提取出来，并做清洗。
    # 注意：这里的 fit_start_time 和 fit_end_time 设为了近期的 100 天。
    # 这意味着它会用最近 100 天的数据来计算 Z-Score 的均值和标准差，相当于做了一个 Rolling Z-Score，
    # 这样可以防止很久以前的数据分布影响当前的预测。
    handler = build_alpha158_handler(
        instruments=cfg["universe"],  # 股票池，比如 csi300 或全市场
        start_time=start_day,  # 提取数据的起点
        end_time=latest_day,  # 提取数据的终点
        fit_start_time=start_day,  # 用于特征标准化的起点
        fit_end_time=latest_day,  # 用于特征标准化的终点
        use_valuation=cfg["features"].get("use_valuation", True)  # 是否包含扩展的 8 个估值因子
    )

    # =========================================================================
    # ======================= 【核心修改位置：尽早检查日期】 =======================
    # 从 Handler 中直接提取已经加载的数据的索引，获取最新日期
    try:
        # fetch(col_set="label") 只是为了拿索引，速度极快
        all_data_index = handler.fetch(col_set="label").index
        data_latest_date = all_data_index.get_level_values(0).max()

        import datetime
        today = datetime.datetime.now()
        days_diff = (today - data_latest_date).days

        print("\n" + "-" * 40)
        if days_diff > 4:
            print(f"❌ 【严重警告】：本地数据库极度过时！")
            print(f"    当前数据最新日期: {data_latest_date.strftime('%Y-%m-%d')}")
            print(f"    系统今日日期:    {today.strftime('%Y-%m-%d')}")
            print(f"    相差天数:       {days_diff} 天")
            print(f"    请先运行 'python main.py --download-only' 更新数据，否则预测结果无效！")
            print("-" * 40 + "\n")
            # 如果你希望强制停止，可以解除下面注释
            # exit(1)
        else:
            status = "✅ 数据新鲜" if days_diff <= 1 else "⚠️ 数据略有滞后(周末/节假日)"
            print(f"{status}: 最新交易日为 {data_latest_date.strftime('%Y-%m-%d')}")
            print(f"   可先运行 'python main.py --download-only' 更新数据！")
            print("-" * 40 + "\n")
    except Exception as e:
        print(f"⚠️ 无法检查数据时效性: {e}")
    # =========================================================================

    # -------------------------------------------------------------------------
    # 第五步：构造数据集 (Dataset)
    # -------------------------------------------------------------------------
    # Dataset 负责把 Handler 提取出来的一维表格数据，转换成模型需要的 3D 张量 (Tensor)。
    # 我们只需要预测最新的这一天，所以测试集 (test segment) 的起点和终点都是 latest_day。
    segments = {"test": (latest_day, latest_day)}

    if args.model == "lgbm":
        # 树模型不需要时序维度，直接构造表格型数据集
        dataset = build_tabular_dataset(handler=handler, segments=segments)
    else:
        # 深度学习模型需要时间步长 (step_len) 维度
        dataset = build_ts_dataset(
            handler=handler,
            segments=segments,
            step_len=cfg["features"]["lookback"],  # 通常在 config 中设为 60 左右
        )

    # -------------------------------------------------------------------------
    # 第六步：执行模型预测 (Inference)
    # -------------------------------------------------------------------------
    print(f"[*] 正在生成 {latest_day} 的全市场预测信号...")
    # 传入组装好的 Dataset，模型吐出每个股票的预测得分
    preds = model.predict(dataset)

    # 容错处理：如果因为停牌或数据缺失导致没预测出结果，及时终止
    if preds.empty:
        print("[!] 警告：没有生成任何预测结果。请检查今日的数据是否已正确下载并入库。")
        return

    # -------------------------------------------------------------------------
    # 第七步：提取预测结果并排序
    # -------------------------------------------------------------------------
    # Qlib 返回的结果是一个 MultiIndex (多重索引) 的 Pandas Series。
    # 索引级别 0 是日期 (datetime)，级别 1 是股票代码 (instrument)。
    # 我们获取预测结果中日期最大的一天（确保拿到的是最新数据）。
    actual_latest_date = preds.index.get_level_values(0).max()

    # 用 .loc 提取出最新这一天所有股票的打分（分数越高，模型认为未来收益越好）
    latest_preds = preds.loc[actual_latest_date]

    # -------------------------------------------------------------------------
    # --- 新增：板块过滤逻辑 ---
    def is_main_board(code):
        # 仅允许 00 或 60 开头的代码
        return code.startswith(('00', '60'))
    filtered_preds = latest_preds[latest_preds.index.map(is_main_board)]
    top_k_stocks = filtered_preds.nlargest(args.top_k)
    # 在过滤后的结果中提取得分最高的 Top K
    # .nlargest 函数会自动帮我们找出分数最高的 Top K 只股票
    top_k_stocks = filtered_preds.nlargest(args.top_k)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 第八步：终端展示
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"🚀 AI量化系统 - 交易日 {actual_latest_date.strftime('%Y-%m-%d')} 收盘后分析报告")
    print(f"   [执行策略] 明日 (T+1) 早盘集合竞价市价买入以下 TOP {args.top_k} 股票")
    print("=" * 60)

    # 遍历推荐列表并格式化打印
    for rank, (stock_id, score) in enumerate(top_k_stocks.items(), 1):
        # 打印排名、股票代码和预测得分（保留 4 位小数并带正负号）
        print(f"  🏆 推荐排名 {rank}: {stock_id:10s} | 模型预测得分 (Expected Return): {score:+.4f}")
    print("=" * 60)
    print("[*] 提示: 投资有风险，量化模型输出仅供参考，请结合大盘环境谨慎操作。\n")


if __name__ == "__main__":
    main()