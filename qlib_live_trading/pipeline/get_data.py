# prepare_data.py
"""
=========================================================
Prepare Data for Qlib Pipeline (A-share 主板)
---------------------------------------------------------
功能：
1. 使用 AkShare 抓取 A 股主板股票日线行情（HFQ）和估值数据。
2. 合并日线数据与估值数据，并生成 industry 占位列。
3. 支持增量更新（只抓取缺失日期的数据）。
4. 转换合并后的 parquet 数据为 Qlib 二进制格式。
5. 生成 Qlib instruments 索引文件（main_board.txt）。
---------------------------------------------------------
目录结构要求：
data/
├─ raw/
│  ├─ daily/       # 存放原始 daily parquet
│  └─ valuation/   # 存放原始 valuation parquet
├─ processed/      # 合并后的 parquet
└─ qlib_data_cn/   # Qlib 二进制目录
=========================================================
"""

import os
import pandas as pd
from pathlib import Path
import akshare as ak
import pyarrow.parquet as pq
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import urllib.request

# ----------------------------
# 目录定义
# ----------------------------
RAW_DAILY_DIR = Path("../data/raw/daily")
RAW_VAL_DIR = Path("../data/raw/valuation")
PROCESSED_DIR = Path("../data/processed")
QLIB_DIR = Path("../data/qlib_data_cn")
QLIB_CSV_DIR = Path("../data/qlib_csv_temp")

for d in [RAW_DAILY_DIR, RAW_VAL_DIR, PROCESSED_DIR, QLIB_DIR, QLIB_CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Qlib 字段定义
# ----------------------------
QLIB_FIELDS = ['open','high','low','close','volume','amount','turnover',
               'total_mv','circ_mv','total_share','circ_share',
               'pe_ttm','pe_static','pb','peg','pcf','ps',
               'factor','industry']

# 估值字段重命名映射
VAL_RENAME_MAP = {
    '数据日期': 'date', '当日收盘价': 'v_close', '总市值': 'total_mv',
    '流通市值': 'circ_mv','总股本': 'total_share','流通股本': 'circ_share',
    'PE(TTM)': 'pe_ttm','PE(静)': 'pe_static','市净率': 'pb','PEG值': 'peg',
    '市现率': 'pcf','市销率': 'ps'
}


# ----------------------------
# 股票列表获取
# ----------------------------
#------------------------------------------------------------------------------------------
# def fetch_stock_list():
#     """
#     获取 A 股主板股票列表（00/60 开头）
#     返回列表：['600000', '600004', ...]
#     """
#     try:
#         print("[*] 获取主板股票列表...")
#         df = ak.stock_zh_a_spot_em()
#         df = df[df["代码"].str.match(r"^(00|60)")]
#         return df["代码"].tolist()
#     except Exception as e:
#         print(f"[!] Error fetching stock list: {e}")
#         exit(1)
#------------------------------------------------------------------------------------------
# 假如IP被封，从本地记录提取A股个股代码 （建议每隔一段时间跑一次在线抓取）
def fetch_stock_list(file=f"{QLIB_DIR}/instruments/main_board.txt"):
    """
    从本地 main_board.txt 读取主板股票列表。
    文件格式：
        code    list_date    last_update
    示例：
        000001  1991-04-03   2026-03-13
    返回
    -------
    list[str]
        股票代码列表
    """
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"{file} not found")
    symbols = []
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            code = parts[0]
            symbols.append(code)
    print(f"[*] Loaded {len(symbols)} stocks from {file}")
    return symbols
#------------------------------------------------------------------------------------------
# ----------------------------
# 数据抓取函数
# ----------------------------
def fetch_daily(symbol, start_date=None):
    """
    抓取单只股票 HFQ 日线行情
    start_date: YYYYMMDD，可用于增量更新
    返回 DataFrame
    """
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="hfq",
                                start_date=start_date, end_date="20251231")
        if df is None or df.empty:
            return None
        df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close",
                                "最高":"high","最低":"low",
                                "成交量":"volume","成交额":"amount",
                                "换手率":"turnover"})
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"[!] {symbol} 日线抓取失败: {e}")
        return None

def fetch_valuation(symbol):
    """
    抓取单只股票估值数据
    返回 DataFrame
    """
    try:
        df = ak.stock_value_em(symbol)
        if df is None or df.empty:
            return None
        df = df.rename(columns=VAL_RENAME_MAP)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"[!] {symbol} 估值抓取失败: {e}")
        return None

# ----------------------------
# 数据融合与增量更新
# ----------------------------
def merge_and_save(symbol):
    """
    合并日线和估值数据，补充 industry 列，保存到 processed/
    支持增量更新：已有 parquet 文件时只抓取缺失日期
    """
    # --- 1. 确定增量起始日期 ---
    daily_file = RAW_DAILY_DIR / f"{symbol}.parquet"
    start_daily = "19900101"
    if daily_file.exists():
        try:
            df_exist = pd.read_parquet(daily_file)
            last_date = pd.to_datetime(df_exist['date']).max()
            start_daily = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        except: pass

    df_daily = fetch_daily(symbol, start_date=start_daily)
    if df_daily is None:
        print(f"[!] {symbol} 日线无新数据")
        return False

    # 合并已有数据
    if daily_file.exists():
        df_exist = pd.read_parquet(daily_file)
        df_daily = pd.concat([df_exist, df_daily], ignore_index=True).drop_duplicates('date').sort_values('date')
    df_daily.to_parquet(daily_file, index=False)

    # ----------------------------
    # 估值数据
    # ----------------------------
    val_file = RAW_VAL_DIR / f"{symbol}.parquet"
    fetch_val = True
    if val_file.exists():
        try:
            meta = pq.read_metadata(str(val_file))
            rg = meta.row_group(meta.num_row_groups-1)
            max_date = pd.Timestamp(rg.column(meta.schema.names.index('数据日期')).statistics.max)
            if max_date >= pd.Timestamp("2025-12-31"):
                fetch_val = False
        except: pass

    if fetch_val:
        df_val = fetch_valuation(symbol)
        if df_val is not None:
            df_val.to_parquet(val_file, index=False)

    # ----------------------------
    # 融合 daily + valuation
    # ----------------------------
    df_val = pd.read_parquet(val_file).rename(columns=VAL_RENAME_MAP)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_val['date'] = pd.to_datetime(df_val['date'])
    df = pd.merge(df_daily, df_val, on='date', how='outer')
    df['close'] = df['close'].fillna(df.get('v_close', pd.NA))
    df = df.drop(columns=['v_close'], errors='ignore')

    # 补充 industry 占位
    df['industry'] = "Unknown"

    processed_file = PROCESSED_DIR / f"{symbol}.parquet"
    df.to_parquet(processed_file, index=False)
    return True

def collect_all(symbols=None, max_workers=4):
    """
    多线程抓取/更新全市场数据
    """
    if symbols is None:
        symbols = fetch_stock_list()
    print(f"[*] 开始抓取 {len(symbols)} 只股票...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(merge_and_save, s): s for s in symbols}
        for f in tqdm(as_completed(futures), total=len(symbols)):
            if f.result():
                success_count += 1
    print(f"[+] 完成抓取: 成功 {success_count}/{len(symbols)}")

# ----------------------------
# 转换为 Qlib 二进制
# ----------------------------
def convert_to_qlib():
    """
    将 processed/ 下的 parquet 数据转换为 Qlib 二进制
    """
    QLIB_CSV_DIR.mkdir(parents=True, exist_ok=True)
    files = list(PROCESSED_DIR.glob("*.parquet"))

    print("[*] 临时生成 CSV 用于 Qlib dump_bin...")
    for f in tqdm(files, desc="CSV Export"):
        df = pd.read_parquet(f)
        df['factor'] = 1.0  # 占位
        for c in QLIB_FIELDS:
            if c not in df.columns:
                df[c] = float('nan')
        df[['date'] + QLIB_FIELDS].to_csv(QLIB_CSV_DIR / f"{f.stem}.csv", index=False)

    # 获取 dump_bin.py
    dump_bin_path = Path("dump_bin.py")
    if not dump_bin_path.exists():
        url = "https://raw.githubusercontent.com/microsoft/qlib/main/scripts/dump_bin.py"
        urllib.request.urlretrieve(url, dump_bin_path)

    cmd = ["python", str(dump_bin_path), "dump_all",
           "--data_path", str(QLIB_CSV_DIR.resolve()),
           "--qlib_dir", str(QLIB_DIR.resolve()),
           "--include_fields", ",".join(QLIB_FIELDS),
           "--date_field_name", "date"]
    print(f"[*] 执行 Qlib 转换: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 清理临时 CSV
    shutil.rmtree(QLIB_CSV_DIR)
    print("[+] Qlib 数据生成完成！")

# ----------------------------
# 生成主板索引文件 main_board.txt
# ----------------------------
def generate_main_board_index():
    """
    扫描 processed/ 文件夹，生成 Qlib instruments 索引
    """
    inst_dir = QLIB_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    files = list(PROCESSED_DIR.glob("*.parquet"))

    lines = []
    for f in files:
        code = f.stem.upper()
        if not code.startswith(("00","60")):
            continue
        df = pd.read_parquet(f, columns=['date'])
        if df.empty: continue
        start_dt = df['date'].min().strftime('%Y-%m-%d')
        end_dt = df['date'].max().strftime('%Y-%m-%d')
        lines.append(f"{code}\t{start_dt}\t{end_dt}")

    out_file = inst_dir / "main_board.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(sorted(lines)) + "\n")
    print(f"[+] 主板索引已生成: {out_file} (共 {len(lines)} 只股票)")

# ----------------------------
# CLI 执行入口
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="抓取全市场主板股票")
    parser.add_argument("--symbols", type=str, help="逗号分隔股票代码")
    parser.add_argument("--update", action="store_true", help="增量更新已有数据")
    parser.add_argument("--qlib", action="store_true", help="转换为 Qlib 二进制")
    args = parser.parse_args()
    # 全量抓取 + Qlib 转换
    #       python prepare_data.py --all --qlib
    # 增量更新 + Qlib 转换
    #       python prepare_data.py --update --qlib
    # 指定单只股票
    #       python prepare_data.py --symbols 600000,600004 --qlib

    # ===== DEBUG MODE (for PyCharm) =====
    args.all = True
    args.qlib = True
    # ====================================

    symbols_to_fetch = None
    if args.symbols:
        symbols_to_fetch = args.symbols.split(",")
    elif args.all or args.update:
        symbols_to_fetch = fetch_stock_list()

    if symbols_to_fetch:
        collect_all(symbols_to_fetch)

    if args.qlib:
        convert_to_qlib()
        generate_main_board_index()
