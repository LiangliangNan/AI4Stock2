"""
=========================================================================================
AI4Stock2 - 极简版每日个股推荐脚本 (Daily Top-K Stock Recommender)
=========================================================================================

------------------------- 【 第一部分：实战操作指南 】 -------------------------

1. 使用场景：
   - 每日 15:00 收盘后，运行数据更新脚本获取当日最新行情。
   - 运行本脚本，获得 T+1 日（明日）开盘时的调仓指令（买入/卖出/持有）。

2. 运行命令：
   python recommend.py  # 默认读取 config.yaml 配置中的模型和 topk 参数

3. 前置要求：
   - 必须先运行 main.py 完成模型训练，生成 model.pkl 和 handler_state.pkl。

------------------------- 【 第二部分：设计逻辑与技术细节 】 -------------------------

1. 特征工程与状态复用：
   - 脚本通过载入训练时保存的 handler_state.pkl，直接继承了训练集的标准化标尺（均值/方差）。
   - 极速模式：仅提取最近 100 天数据即可满足 Alpha158 算子的滑动窗口需求，实现秒级推断。

2. 调仓机制 (TopK Dropout 算法)：
   - 核心逻辑：在维持组合质量与控制换手率之间取得平衡。
   - 只有当持仓股排名跌出 TopK 且属于“表现最差”的 n_drop 名额内时，才触发强制卖出。
   - 若持仓股虽跌出 TopK 但未进入末尾淘汰名额，则采取“留校察看”策略，减少无效调仓。

3. 交易撮合一致性：
   - 标签定义为 T+1 开盘买入至 T+2 开盘卖出的收益，输出结果与回测引擎配置严格对齐。

=========================================================================================
"""

import pickle
import yaml
import pandas as pd
from pathlib import Path
import datetime


from qlib.data import D
from src.data_setup import init_qlib
from src.dataset import build_ts_dataset, build_tabular_dataset


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载项目的 YAML 配置文件。"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_rebalancing_advice(latest_preds: pd.Series, current_holdings: list, topk: int, n_drop: int):
    """
    基于 TopK Dropout 逻辑计算调仓数据。
    返回：持仓信息、全场 Top 列表、卖出候选、买入候选、保留股票列表。
    """
    # 0. 容错：自动补全代码前缀 (SH/SZ)
    formatted_holdings = []
    for s in current_holdings:
        s = str(s).strip()
        if s.startswith("SH") or s.startswith("SZ"):
            formatted_holdings.append(s)
        else:
            prefix = "SH" if s.startswith("6") else "SZ"
            formatted_holdings.append(f"{prefix}{s}")
    current_holdings = formatted_holdings

    # 1. 对全市场预测分进行排名
    sorted_preds = latest_preds.sort_values(ascending=False)
    rank_map = {stock: rank for rank, stock in enumerate(sorted_preds.index, 1)}

    # 2. 评估当前持仓现状
    holding_info = []
    for stock in current_holdings:
        if stock in sorted_preds.index:
            score = sorted_preds[stock]
            rank = rank_map[stock]
        else:
            # 停牌或缺失数据的股票处理
            score = -float('inf')
            rank = float('inf')
        holding_info.append({"stock": stock, "score": score, "rank": rank})

    # 3. 识别跌出 TopK 的持仓（潜在卖出目标）
    out_of_topk = [info for info in holding_info if info["rank"] > topk]
    # 按分数从低到高排序（分数越低越该卖）
    out_of_topk_sorted = sorted(out_of_topk, key=lambda x: x["score"])

    # 4. 应用 Dropout 限制（每日最大换仓数）
    sell_candidates = out_of_topk_sorted[:n_drop]
    sell_stocks = [x["stock"] for x in sell_candidates]

    # 5. 确定继续持有的股票并计算买入缺口
    keep_stocks = [s for s in current_holdings if s not in sell_stocks]
    num_to_buy = topk - len(keep_stocks)

    # 6. 从全市场顶端挑选新标的（跳过已持有的）
    buy_candidates = []
    for stock, score in sorted_preds.items():
        if len(buy_candidates) >= num_to_buy:
            break
        if stock not in keep_stocks:
            buy_candidates.append({"stock": stock, "score": score, "rank": rank_map[stock]})

    return {
        "holding_info": holding_info,
        "market_top_list": sorted_preds.iloc[:topk * 2],
        "sell_list": sell_candidates,
        "buy_list": buy_candidates,
        "topk": topk
    }


def print_report(report_data: dict, model_name: str, trade_date: datetime.datetime):
    """
    [表现层] 统一决策报告输出：将复杂的策略逻辑转化为直观的两段式执行清单。

    该函数将计算结果分为“现有持仓执行清单”与“市场顶尖标的参考”两个模块，
    并在列表中通过“实战建议”列直接给出操作指令，实现逻辑计算与实战操作的无缝对接。

    Args:
        report_data (dict): 包含以下键值的字典:
            - holding_info (list): 当前持仓的详细排名与得分信息。
            - market_top_list (pd.Series): 全市场预测得分最高的 Top-K*2 标的。
            - sell_list (list): 触发 TopK-Dropout 淘汰逻辑的卖出候选。
            - buy_list (list): 填补仓位空缺的建议买入候选。
            - topk (int): 策略设定的目标持仓数量。
        model_name (str): 当前运行的模型名称（如 LGBM, Transformer 等）。
        trade_date (datetime.datetime): 预测对应的交易日期。
    """
    date_str = trade_date.strftime('%Y-%m-%d')
    line_width = 88
    top_k = report_data["topk"]

    # 建立快捷索引集合，提升匹配效率
    sell_set = {s['stock'] for s in report_data["sell_list"]}
    buy_set = {s['stock'] for s in report_data["buy_list"]}
    holding_set = {info['stock'] for info in report_data["holding_info"]}

    print("\n" + "█" * line_width)
    print(f"  AI4Stock2 每日量化决策报告 | 交易日期: {date_str} | 核心模型: {model_name.upper()}")
    print("█" * line_width)

    # -------------------------------------------------------------------------
    # 模块一：当前持仓执行清单 (Portfolio Action List)
    # 逻辑说明：直接映射 T+1 日开盘的确定性操作，包含 卖出(❌) 与 继续持有(✅/⚠️)
    # -------------------------------------------------------------------------
    print(f"\n[ 1. 当前持仓执行清单 (Portfolio Execution) ]")
    print("-" * line_width)
    print(f"{'代码':<10} | {'预测得分':>10} | {'全场排名':>8} | {'实战操作指令'}")
    print("-" * line_width)

    for info in report_data["holding_info"]:
        if info['stock'] in sell_set:
            advice = "❌ 建议卖出 (排名跌出核心区且触发末尾淘汰)"
        elif info['rank'] > top_k:
            advice = "⚠️ 继续持有 (虽出TopK但受换手率保护/留校察看)"
        else:
            advice = "✅ 继续持有 (模型预测表现稳健)"
        print(f"{info['stock']:<10} | {info['score']:>10.4f} | {info['rank']:>8} | {advice}")

    # -------------------------------------------------------------------------
    # 模块二：市场全景参考池 (Market Alpha Leaders - Top K*2)
    # 逻辑说明：展示全市场最强标的排名，并标注其相对于当前组合的状态（买入/持有/关注）
    # -------------------------------------------------------------------------
    print(f"\n[ 2. 市场顶尖候选池 (Alpha Ranking - Top {top_k * 2}) ]")
    print("-" * line_width)
    print(f"{'排名':<6} | {'代码':<10} | {'预测得分':>10} | {'组合关联状态'}")
    print("-" * line_width)

    top_list = report_data["market_top_list"]
    for rank, (stock, score) in enumerate(top_list.items(), 1):
        # 确定与当前组合的关联逻辑
        if stock in buy_set:
            status = "🛒 建议买入 (补足仓位空缺的新晋强势标的)"
        elif stock in sell_set:
            status = "❌ 触发卖出 (虽在名单但已触发淘汰逻辑)"
        elif stock in holding_set:
            status = "💎 核心持仓 (当前组合内的顶尖标的)"
        else:
            # 根据是否在 TopK 范围内区分关注级别
            status = "👀 重点关注 (处于策略核心池)" if rank <= top_k else "--- (备选观察)"

        print(f"#{rank:<5} | {stock:<10} | {score:>+10.4f} | {status}")

    # -------------------------------------------------------------------------
    # 报告注脚：风险提示与操作规范
    # -------------------------------------------------------------------------
    print("\n" + "█" * line_width)
    print("  [*] 决策备注：")
    print("      1. '建议买入' 基于模型评分排名与组合缺口自动计算，建议开盘集合竞价执行。")
    print("      2. 若持仓状态显示 '留校察看'，代表该股评分虽有所下降，但未达强制换仓标准。")
    print("      3. 量化信号不考虑个股基本面突发利空，买入前请务必确认无重大负面公告。")
    print("█" * line_width + "\n")


def recommend(current_holdings):
    """
    执行完整的推荐与决策生成流程。
    """
    # 1. 环境初始化
    cfg = load_config()
    init_qlib(provider_uri=cfg["qlib"]["provider_uri"], region=cfg["qlib"]["region"])

    top_k = cfg["strategy"]["topk"]
    n_drop = cfg["strategy"]["n_drop"]
    model_name = cfg["model"]["name"]

    # 2. 载入模型与特征处理器状态
    model_path = Path("results") / model_name / "model.pkl"
    state_path = Path("results") / model_name / "handler.pkl"

    if (not model_path.exists()):
        raise FileNotFoundError(f"[错误] 在 results/{model_name} 下未找到“model.pkl”。请先运行训练")
    if (not state_path.exists()):
        raise FileNotFoundError(f"[错误] 在 results/{model_name} 下未找到“handler.pkl”。请先运行训练")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(state_path, "rb") as f:
        handler = pickle.load(f)

    print(f"[*] 成功加载模型: {model_name.upper()}")

    # 3. 确定数据时间窗口 (极速模式)
    calendar = D.calendar()
    # latest_day = calendar[-1].strftime("%Y-%m-%d")
    latest_day = pd.Timestamp(calendar[-1])
    print(f"[DEBUG] latest day in calendar: {latest_day}")

    # 手动重设 Handler 时间窗口，避免加载全量历史数据
    lookback_days = 100
    inference_start_day = calendar[-lookback_days].strftime("%Y-%m-%d")
    print(f"[DEBUG] inference start day: {inference_start_day}")
    #---------------------------------------------------------------
    # Liangliang: 比较以下两种处理方式：
    # 方式一：直接修改handler的start_time和end_time属性
    # handler.start_time = pd.Timestamp(inference_start_day)
    # handler.end_time = pd.Timestamp(latest_day)
    # 这样做不会触发任何内部逻辑。在某些情况下可能出现：
    #   1. 缓存未刷新
    #   2. processor 的状态未更新，不会重新 setup
    #---------------------------------------------------------------
    # 方式二：调用handler.config()函数。该函数会做三件事：
    #   1 更新 handler.start_time / end_time
    #   2 标记 handler 数据状态失效
    #   3 触发 setup_data()
    #   旧缓存会被明确清除，feature pipeline会重新初始化
    handler.config(
        start_time=pd.Timestamp(inference_start_day),
        end_time=pd.Timestamp(latest_day)
    )
    #---------------------------------------------------------------

    # 4. 数据时效性安全检查
    try:
        all_data_index = handler.fetch(col_set="feature").index
        data_latest_date = all_data_index.get_level_values(0).max()
        days_diff = (datetime.datetime.now() - data_latest_date).days
        print(f"[DEBUG] data latest date: {data_latest_date}")

        print("-" * 45)
        if days_diff > 4:
            print(f"❌ 【警告】：本地数据已过期 ({days_diff}天前)。请更新数据库！")
        else:
            print(f"✅ 【时效】：最新数据日期为 {data_latest_date.strftime('%Y-%m-%d')}")
        print("-" * 45)
    except Exception as e:
        print(f"⚠️ 无法校验数据时效: {e}")

    # 5. 构造 Dataset 并执行预测
    segments = {"test": (latest_day, latest_day)}
    if model_name == "lgbm":
        dataset = build_tabular_dataset(handler, segments)
    else:
        dataset = build_ts_dataset(handler, segments, step_len=cfg["features"]["lookback"])

    print(f"[*] 正在计算全市场信号...")
    preds = model.predict(dataset)
    if preds.empty:
        print("[!] 错误：预测结果为空，请检查今日行情是否完整。")
        return

    # 6. 提取最新预测并生成决策数据
    actual_latest_date = preds.index.get_level_values(0).max()
    latest_preds = preds.loc[actual_latest_date]
    print(f"[DEBUG] actual latest date (target date for prediction): {actual_latest_date}")

    advice_data = generate_rebalancing_advice(
        latest_preds=latest_preds,
        current_holdings=current_holdings,
        topk=top_k,
        n_drop=n_drop
    )

    # 6. 打印报告
    print_report(advice_data, model_name, actual_latest_date)


if __name__ == "__main__":
    # 示例持仓代码：支持 6 位数字或带前缀的格式
    current_holdings = [
        "600000",
        "000001",
        "000858",
        "600519"
    ]
    recommend(current_holdings)