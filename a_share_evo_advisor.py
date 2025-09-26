# -*- coding: utf-8 -*-
"""
A股“进化式”投研助手（商业化优化版，单文件 GUI）
- 数据源：baostock
- 增量下载：仅拉取新增交易日
- 量化特征：SMA/EMA/MACD/RSI/ATR/BOLL/KDJ/CCI/WR/OBV/MFI/CMF/BBP/Range/Gap/动量等
- 模型训练：
    * 岭回归（时间序列CV自动选 λ，目标可选 IC 或 MSE）
    * 遗传算法（特征子集+正则λ，目标最大化验证Rank IC）
- 市场扫描：
    * 支持 HS300/ZZ500/SZ50/ALL/CUSTOM
    * 无权重时自动使用“基线打分”也可扫描（不依赖训练）
    * 高级筛选：趋势/接近55日高/波动Z阈值/RSI区间/流动性阈值
- 扩展建议：结合预测、动量、波动、ATR给出仓位/止损/止盈/数量
- 回测：TopN选股 + H日持有 + 交易费率，输出年化/回撤/Sharpe/胜率，并绘制净值曲线
- 绩效评估：按日期/类型筛选、导出CSV、分位统计、收益分布图
- 用户操作：记录每日操作（买/卖/调仓），为连续化进化提供数据基础
- 持久化：SQLite（价格、名称、建议、权重、元信息、用户操作）
- GUI：tkinter + matplotlib + Notebook 分页 + 异步线程 + 进度条 + 滚动条 + 菜单栏 + 主题切换
- 可中止大任务（Stop），参数自动持久化，日志写文件，权重导出/导入

依赖:
    pip install baostock pandas numpy matplotlib

免责声明:
    本程序仅用于教育与研究目的，不构成任何投资建议。
"""

import datetime as dt
import json
import math
import os
import re
import sqlite3
import threading
import time
from typing import List, Tuple, Dict, Any, Optional, Callable

import matplotlib
import numpy as np
import pandas as pd

# 使用 TkAgg 后端用于嵌入式窗口
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 中文字体兼容
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import baostock as bs
except ImportError:
    bs = None

# =========================== 全局常量与状态 ===========================
APP_NAME = "A股进化式投研助手"
APP_VERSION = "1.3.0"
DB_PATH = "advisor.db"
DATE_FMT = "%Y-%m-%d"
CONFIG_PATH = "advisor_config.json"
LOG_PATH = "advisor.log"

app_state: Dict[str, Any] = {"horizon": 10}


# =========================== 通用工具函数 ===========================
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def rank_spearman_ic(pred: pd.Series, real: pd.Series) -> float:
    """
    计算横截面Rank IC（Spearman相关）。要求索引对齐且同长度。
    """
    if pred.empty or real.empty:
        return np.nan
    s1 = pd.Series(pred.values).rank(method="average")
    s2 = pd.Series(real.values).rank(method="average")
    v1 = s1.values
    v2 = s2.values
    v1 = (v1 - v1.mean())
    v2 = (v2 - v2.mean())
    denom = (np.sqrt((v1 ** 2).sum()) * np.sqrt((v2 ** 2).sum()))
    if denom <= 1e-12:
        return np.nan
    return float((v1 @ v2) / denom)


def groupby_date_ic(pred: pd.Series, real: pd.Series, dates: pd.Series) -> Tuple[float, float, int]:
    """
    按交易日计算横截面IC，输出 (IC均值, IC标准差, 有效样本天数)
    """
    df = pd.DataFrame({"pred": pred.values, "real": real.values, "date": dates.values})
    ics = []
    for d, g in df.groupby("date"):
        if len(g) >= 5:
            ic = rank_spearman_ic(g["pred"], g["real"])
            if not np.isnan(ic):
                ics.append(ic)
    if not ics:
        return np.nan, np.nan, 0
    arr = np.array(ics, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), len(arr)


def ts_cv_splits(dates: np.ndarray, n_folds: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    时间序列CV：根据样本对应的日期数组分割，保持时间顺序。返回[(train_idx, val_idx), ...]
    """
    unique_dates = np.unique(dates)
    if len(unique_dates) < n_folds + 1:
        n_folds = max(2, min(3, len(unique_dates) - 1))
    folds = []
    N = len(unique_dates)
    if N < 3:
        idx_all = np.arange(len(dates))
        return [(idx_all, idx_all[:0])]
    segs = np.array_split(unique_dates, n_folds)
    for i in range(len(segs) - 1):
        train_dates = np.concatenate(segs[: i + 1])
        val_dates = segs[i + 1]
        train_mask = np.isin(dates, train_dates)
        val_mask = np.isin(dates, val_dates)
        tr_idx = np.where(train_mask)[0]
        va_idx = np.where(val_mask)[0]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            folds.append((tr_idx, va_idx))
    if not folds:
        idx_all = np.arange(len(dates))
        folds.append((idx_all[:-1], idx_all[-1:]))
    return folds


# =========================== 数据库层 ===========================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
              CREATE TABLE IF NOT EXISTS prices
              (
                  code   TEXT NOT NULL,
                  date   TEXT NOT NULL,
                  open   REAL,
                  high   REAL,
                  low    REAL,
                  close  REAL,
                  volume REAL,
                  amount REAL,
                  PRIMARY KEY (code, date)
              )
              """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_code ON prices(code)")
    c.execute("""
              CREATE TABLE IF NOT EXISTS advice
              (
                  id        INTEGER PRIMARY KEY AUTOINCREMENT,
                  date      TEXT NOT NULL,
                  code      TEXT NOT NULL,
                  score     REAL,
                  advice    TEXT,
                  reasoning TEXT,
                  horizon   INTEGER
              )
              """)
    c.execute("""
              CREATE TABLE IF NOT EXISTS weights
              (
                  timestamp TEXT PRIMARY KEY,
                  horizon   INTEGER,
                  features  TEXT,
                  weights   TEXT,
                  intercept REAL,
                  mu        TEXT,
                  sigma     TEXT,
                  lambda    REAL,
                  notes     TEXT
              )
              """)
    c.execute("""
              CREATE TABLE IF NOT EXISTS meta
              (
                  key TEXT PRIMARY KEY,
                  value TEXT
              )
              """)
    c.execute("""
              CREATE TABLE IF NOT EXISTS stock_names
              (
                  code TEXT PRIMARY KEY,
                  name TEXT
              )
              """)
    c.execute("""
              CREATE TABLE IF NOT EXISTS user_ops
              (
                  id     INTEGER PRIMARY KEY AUTOINCREMENT,
                  date   TEXT NOT NULL,
                  code   TEXT NOT NULL,
                  action TEXT NOT NULL, -- BUY/SELL/ADJ  
                  price  REAL,
                  qty    INTEGER,
                  note   TEXT
              )
              """)
    conn.commit()
    conn.close()


def upsert_prices(code: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    conn = get_conn()
    c = conn.cursor()
    rows = []
    for _, r in df.iterrows():
        rows.append((
            code, str(r["date"]), safe_float(r.get("open")), safe_float(r.get("high")),
            safe_float(r.get("low")), safe_float(r.get("close")),
            safe_float(r.get("volume")), safe_float(r.get("amount"))
        ))
    c.executemany("""  
        INSERT OR REPLACE INTO prices (code, date, open, high, low, close, volume, amount)  
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)  
    """, rows)
    conn.commit()
    conn.close()


def get_latest_date_in_db(code: str) -> Optional[str]:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT MAX(date) FROM prices WHERE code=?", (code,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def load_price_df(code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    conn = get_conn()
    c = conn.cursor()
    if start_date and end_date:
        c.execute("""
                  SELECT date, open, high, low, close, volume, amount
                  FROM prices
                  WHERE code = ?
                    AND date >= ?
                    AND date <= ?
                  ORDER BY date ASC
                  """, (code, start_date, end_date))
    elif start_date:
        c.execute("""
                  SELECT date, open, high, low, close, volume, amount
                  FROM prices
                  WHERE code = ?
                    AND date >= ?
                  ORDER BY date ASC
                  """, (code, start_date))
    else:
        c.execute("""
                  SELECT date, open, high, low, close, volume, amount
                  FROM prices
                  WHERE code = ?
                  ORDER BY date ASC
                  """, (code,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def insert_advice(records: List[Tuple[str, str, float, str, str, int]]):
    if not records:
        return
    conn = get_conn()
    c = conn.cursor()
    c.executemany("""
                  INSERT INTO advice (date, code, score, advice, reasoning, horizon)
                  VALUES (?, ?, ?, ?, ?, ?)
                  """, records)
    conn.commit()
    conn.close()


def query_advice(code: Optional[str] = None, limit: int = 200,
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 type_filter: Optional[str] = None) -> List[Tuple]:
    """
    新增日期与类型筛选（type_filter: None/B/BUY/S/SELL）
    """
    conn = get_conn()
    c = conn.cursor()
    sql = """
          SELECT date, code, score, advice, reasoning, horizon
          FROM advice
          WHERE 1 = 1 \
          """
    params: List[Any] = []
    if code:
        sql += " AND code=?"
        params.append(code)
    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)
    if type_filter:
        if type_filter.upper() in ("B", "BUY"):
            sql += " AND advice LIKE '%买入%'"
        elif type_filter.upper() in ("S", "SELL"):
            sql += " AND advice LIKE '%卖出%'"
    sql += " ORDER BY date DESC LIMIT ?"
    params.append(limit)
    c.execute(sql, tuple(params))
    rows = c.fetchall()
    conn.close()
    return rows


def save_weights_record(horizon: int, features: List[str], weights: np.ndarray, intercept: float,
                        mu: np.ndarray, sigma: np.ndarray, lam: float, notes: str = "") -> str:
    conn = get_conn()
    c = conn.cursor()
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""  
        INSERT OR REPLACE INTO weights (timestamp, horizon, features, weights, intercept, mu, sigma, lambda, notes)  
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)  
    """, (
        ts, int(horizon), json.dumps(features, ensure_ascii=False), json.dumps(list(map(float, weights))),
        float(intercept), json.dumps(list(map(float, mu))), json.dumps(list(map(float, sigma))),
        float(lam), notes or ""
    ))
    c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('latest_weights_ts', ?)", (ts,))
    conn.commit()
    conn.close()
    return ts


def load_latest_weights() -> Optional[Dict[str, Any]]:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT value FROM meta WHERE key='latest_weights_ts'")
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    ts = row[0]
    c.execute("""
              SELECT timestamp,
                     horizon,
                     features,
                     weights,
                     intercept,
                     mu,
                     sigma,
                     lambda,
                     notes
              FROM weights
              WHERE timestamp = ?
              """, (ts,))
    w = c.fetchone()
    conn.close()
    if not w:
        return None
    return {
        "timestamp": w[0],
        "horizon": int(w[1]),
        "features": json.loads(w[2]),
        "weights": np.array(json.loads(w[3]), dtype=float),
        "intercept": float(w[4]),
        "mu": np.array(json.loads(w[5]), dtype=float),
        "sigma": np.array(json.loads(w[6]), dtype=float),
        "lambda": float(w[7]),
        "notes": w[8] or ""
    }


def set_stock_name(code: str, name: Optional[str]):
    if not code:
        return
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO stock_names(code, name) VALUES(?,?)", (code, name or ""))
    conn.commit()
    conn.close()


def get_stock_name(code: str) -> str:
    if not code:
        return ""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name FROM stock_names WHERE code=?", (code,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else ""


def get_name_map(codes: List[str]) -> Dict[str, str]:
    if not codes:
        return {}
    conn = get_conn()
    c = conn.cursor()
    qmarks = ",".join(["?"] * len(codes))
    c.execute(f"SELECT code, name FROM stock_names WHERE code IN ({qmarks})", tuple(codes))
    rows = c.fetchall()
    conn.close()
    mp = {r[0]: (r[1] or "") for r in rows}
    return mp


# =========================== baostock 接入 ===========================
def bs_safe_login():
    if bs is None:
        raise RuntimeError("未安装 baostock，请先执行：pip install baostock")
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")


def bs_safe_logout():
    if bs is None:
        return
    try:
        bs.logout()
    except Exception:
        pass


# =========================== 数据获取 ===========================
def fetch_k_data_incremental(code: str, start_date: str, end_date: Optional[str] = None,
                             adjustflag: str = "2", retry: int = 3, sleep_sec: float = 0.4) -> pd.DataFrame:
    """
    从 baostock 拉取K线数据，自动增量（根据DB最新日期 + 1）
    adjustflag: 1-后复权, 2-前复权, 3-不复权
    """
    if bs is None:
        raise RuntimeError("未安装 baostock，无法拉取数据")
    if end_date is None:
        end_date = dt.datetime.now().strftime(DATE_FMT)
    latest = get_latest_date_in_db(code)
    real_start = start_date
    if latest and latest >= start_date:
        d = dt.datetime.strptime(latest, DATE_FMT) + dt.timedelta(days=1)
        real_start = d.strftime(DATE_FMT)
    if real_start > end_date:
        return pd.DataFrame()  # 无新增
    fields = "date,open,high,low,close,volume,amount"
    for i in range(retry):
        try:
            rs = bs.query_history_k_data_plus(
                code, fields, start_date=real_start, end_date=end_date,
                frequency="d", adjustflag=adjustflag
            )
            if rs.error_code != '0':
                if i < retry - 1:
                    bs_safe_logout()
                    time.sleep(0.2)
                    bs_safe_login()
                    continue
                raise RuntimeError(f"baostock错误: {rs.error_msg}")
            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return pd.DataFrame()
            df = pd.DataFrame(data_list, columns=fields.split(","))
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            if i == retry - 1:
                raise
            time.sleep(sleep_sec)
    return pd.DataFrame()


def fetch_and_cache_name(code: str):
    """
    通过 baostock 获取并缓存股票名称
    """
    try:
        if bs is None:
            return
        rs = bs.query_stock_basic(code=code)
        if rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            # row: [code, code_name, ipoDate, outDate, type, status]
            name = row[1]
            if name:
                set_stock_name(code, name)
    except Exception:
        pass


# =========================== 特征工程与信号 ===========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标与特征：
    - 均线/EMA/MACD/RSI
    - ATR/Bollinger/KDJ/CCI/WilliamsR
    - OBV/MFI/CMF
    - 位置关系/放量/波动/动量/多周期收益/Range/Gap
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 基础均线
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma60"] = df["close"].rolling(60).mean()

    # EMA & MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = df["ema12"] - df["ema26"]
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["dif"] - df["dea"])

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ATR(14)
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # Bollinger(20,2)
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_mid"] = mid
    df["bb_up"] = mid + 2 * std
    df["bb_low"] = mid - 2 * std
    df["bbp"] = (df["close"] - df["bb_low"]) / (df["bb_up"] - df["bb_low"] + 1e-9)

    # KDJ(9,3,3)
    low_min = df["low"].rolling(9).min()
    high_max = df["high"].rolling(9).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100
    df["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # CCI(14)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = tp.rolling(14).mean()
    md = (tp - ma_tp).abs().rolling(14).mean()
    df["cci14"] = (tp - ma_tp) / (0.015 * (md + 1e-9))

    # Williams %R(14)
    hh = df["high"].rolling(14).max()
    ll = df["low"].rolling(14).min()
    df["wr14"] = -100 * (hh - df["close"]) / (hh - ll + 1e-9)

    # OBV
    obv = (np.sign(df["close"].diff().fillna(0)) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ema"] = obv.ewm(span=20, adjust=False).mean()

    # MFI(14)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_money = typical * df["volume"]
    pmf = raw_money.where(typical.diff() > 0, 0.0)
    nmf = raw_money.where(typical.diff() < 0, 0.0)
    pmf14 = pmf.rolling(14).sum()
    nmf14 = nmf.rolling(14).sum().abs()
    mfr = pmf14 / (nmf14 + 1e-9)
    df["mfi14"] = 100 - (100 / (1 + mfr))

    # CMF(20)
    mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-9)
    mf_vol = mf_mult * df["volume"]
    df["cmf20"] = mf_vol.rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-9)

    # 波动率（年化近20日）
    df["ret1"] = df["close"].pct_change()
    df["vol20"] = df["ret1"].rolling(20).std() * np.sqrt(252)

    # 动量与多周期收益
    df["mom10"] = df["close"] / df["close"].shift(10) - 1.0
    df["ret5"] = df["close"].pct_change(5)
    df["ret20"] = df["close"].pct_change(20)
    df["ret60"] = df["close"].pct_change(60)

    # 成交量均线与放量
    df["v_ma5"] = df["volume"].rolling(5).mean()
    df["v_surge"] = df["volume"] / (df["v_ma5"] + 1e-9)

    # 位置关系
    df["above_sma20"] = (df["close"] > df["sma20"]).astype(float)
    df["dif_pos"] = (df["dif"] > 0).astype(float)
    df["macd_up"] = (df["macd"] > 0).astype(float)

    # 日内振幅、缺口
    df["range_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["gap_pct"] = df["open"] / (df["close"].shift(1) + 1e-9) - 1.0

    # 平均成交额（近20日）
    df["amt20"] = df["amount"].rolling(20).mean()

    # 55日高与回撤
    df["high55"] = df["high"].rolling(55).max()
    df["near_high55"] = df["close"] / (df["high55"] + 1e-9)
    df["max60"] = df["close"].rolling(60).max()
    df["dd60"] = df["close"] / (df["max60"] + 1e-9) - 1.0

    # sma20斜率（5日差）
    df["sma20_slope5"] = df["sma20"] - df["sma20"].shift(5)

    return df


def feature_columns() -> List[str]:
    return [
        # 趋势/均线
        "sma5", "sma10", "sma20", "sma60",
        "ema12", "ema26",
        # MACD族
        "dif", "dea", "macd",
        # 震荡类
        "rsi14", "kdj_k", "kdj_d", "kdj_j", "cci14", "wr14",
        # 布林带
        "bb_mid", "bbp",
        # 成交/资金
        "v_surge", "obv_ema", "mfi14", "cmf20",
        # 波动与动量
        "vol20", "mom10", "ret5", "ret20", "range_pct", "gap_pct",
        # 位置特征
        "above_sma20", "dif_pos", "macd_up",
        # 风险
        "atr14"
    ]


def build_training_data(codes: List[str], horizon: int, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    构造训练集：X=特征，y=未来horizon日收益（close_{t+H}/close_t - 1）
    使用 MultiIndex(code, date) 严格对齐。
    """
    feats = feature_columns()
    X_list = []
    y_list = []
    for code in codes:
        df = load_price_df(code, start_date, end_date)
        if df.empty or len(df) < max(80, horizon + 40):
            continue
        df = compute_indicators(df)
        df["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0
        usable = df.iloc[:-horizon].copy() if len(df) > horizon else df.iloc[:0].copy()
        usable = usable.dropna(subset=feats + ["fwd_ret"])
        if usable.empty:
            continue
        mi = pd.MultiIndex.from_arrays(
            [np.repeat(code, len(usable)), usable["date"].astype(str).values],
            names=["code", "date"]
        )
        xdf = pd.DataFrame(usable[feats].values, index=mi, columns=feats)
        ysr = pd.Series(usable["fwd_ret"].values, index=mi, name="fwd_ret")
        X_list.append(xdf)
        y_list.append(ysr)

    if not X_list:
        return pd.DataFrame(columns=feats), pd.Series(dtype=float)

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    # 目标收益剪裁
    y = y.clip(lower=-0.25, upper=0.25)

    # 极值剪裁
    X = X.copy()
    clip_cols = ["rsi14", "v_surge", "vol20", "mom10", "macd", "dif", "dea",
                 "bbp", "kdj_j", "cci14", "wr14", "range_pct", "gap_pct", "mfi14", "cmf20"]
    for c in clip_cols:
        if c in X.columns and X[c].notna().any():
            lo = np.nanpercentile(X[c], 1)
            hi = np.nanpercentile(X[c], 99)
            X[c] = X[c].clip(lower=lo, upper=hi)

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def standardize_fit(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feats = list(X.columns)
    mu = np.array([X[c].mean() for c in feats], dtype=float)
    sigma_raw = np.array([X[c].std(ddof=0) for c in feats], dtype=float)
    sigma = np.where(sigma_raw > 1e-9, sigma_raw, 1.0)
    Xs = (X.values - mu) / sigma
    return Xs, mu, sigma, feats


def standardize_apply(x_row: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x_row - mu) / sigma


def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> Tuple[np.ndarray, float]:
    """
    带截距的Ridge回归：y ≈ X w + b （b不正则）
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    X_ext = np.hstack([X, np.ones((n, 1))])
    I = np.eye(d + 1)
    I[-1, -1] = 0.0
    try:
        XtX = X_ext.T @ X_ext + lam * I
        Xty = X_ext.T @ y
        wb = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        wb = np.linalg.pinv(X_ext.T @ X_ext + lam * I) @ (X_ext.T @ y)
    w = wb[:-1]
    b = wb[-1]
    return w, b


def score_to_advice(score: float, pos_thr: float, neg_thr: float) -> str:
    if score >= pos_thr * 1.5:
        return "强烈买入"
    elif score >= pos_thr:
        return "买入"
    elif score <= neg_thr * 1.5:
        return "强烈卖出"
    elif score <= neg_thr:
        return "卖出"
    else:
        return "观望"


def reasoning_from_signals(row: pd.Series) -> str:
    parts = []
    if row.get("above_sma20", 0) > 0.5:
        parts.append("股价站上SMA20")
    if row.get("macd", 0) > 0:
        parts.append("MACD红柱")
    if row.get("dif_pos", 0) > 0.5:
        parts.append("DIF>0")
    rsi = row.get("rsi14", np.nan)
    if pd.notna(rsi):
        if rsi < 30:
            parts.append("RSI超卖")
        elif rsi > 70:
            parts.append("RSI超买")
    mom = row.get("mom10", np.nan)
    if pd.notna(mom):
        parts.append(f"动量10日: {mom:.1%}")
    vs = row.get("v_surge", np.nan)
    if pd.notna(vs) and vs > 1.5:
        parts.append("放量")
    if not parts:
        return "常规信号不显著"
    return "；".join(parts)


# =========================== 训练与进化 ===========================
def evaluate_ic_on_Xy(Xdf: pd.DataFrame, y: pd.Series, mu: np.ndarray, sigma: np.ndarray,
                      feats: List[str], w: np.ndarray, b: float) -> Dict[str, Any]:
    Xv = Xdf.values
    pred_std = (Xv - mu) / sigma
    pred = pred_std @ w + b
    dates = Xdf.index.get_level_values("date")
    ic_mean, ic_std, n_days = groupby_date_ic(pd.Series(pred), y, pd.Series(dates))
    mse = float(np.mean((pred - y.values) ** 2))
    return {"ic_mean": ic_mean, "ic_std": ic_std, "icir": (ic_mean / (ic_std + 1e-9)) if ic_std == ic_std else np.nan,
            "mse": mse, "n_days": n_days}


def train_ridge_with_cv(Xdf: pd.DataFrame, y: pd.Series, target_metric: str = "IC",
                        lam_grid: Optional[List[float]] = None, n_folds: int = 4) -> Dict[str, Any]:
    if lam_grid is None:
        lam_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    Xs, mu, sigma, feats = standardize_fit(Xdf)
    dates = Xdf.index.get_level_values("date").values
    folds = ts_cv_splits(dates, n_folds=n_folds)

    best_lam = lam_grid[0]
    best_score = -1e18 if target_metric.upper() == "IC" else 1e18

    for lam in lam_grid:
        fold_scores = []
        for tr_idx, va_idx in folds:
            Xtr = Xs[tr_idx]
            ytr = y.values[tr_idx]
            Xva = Xs[va_idx]
            yva = y.values[va_idx]
            w, b = ridge_regression(Xtr, ytr, lam=lam)
            pred = Xva @ w + b
            d_va = dates[va_idx]
            if target_metric.upper() == "IC":
                ic_mean, _, _ = groupby_date_ic(pd.Series(pred), pd.Series(yva), pd.Series(d_va))
                val = ic_mean if ic_mean == ic_mean else -1e9
                fold_scores.append(val)
            else:
                mse = float(np.mean((pred - yva) ** 2))
                fold_scores.append(-mse)
        score = float(np.mean(fold_scores)) if fold_scores else (-1e9)
        if score > best_score:
            best_score = score
            best_lam = lam

    w, b = ridge_regression(Xs, y.values, lam=best_lam)
    eval_res = evaluate_ic_on_Xy(Xdf, y, mu, sigma, feats, w, b)
    notes = {
        "method": "ridge_cv",
        "target": target_metric,
        "lam": best_lam,
        "cv_score": best_score,
        "eval": eval_res
    }
    ts = save_weights_record(horizon=app_state.get("horizon", 10), features=feats, weights=w, intercept=b,
                             mu=mu, sigma=sigma, lam=best_lam, notes=json.dumps(notes, ensure_ascii=False))
    pack = {
        "timestamp": ts, "horizon": app_state.get("horizon", 10), "features": feats, "weights": w,
        "intercept": b, "mu": mu, "sigma": sigma, "lambda": best_lam, "notes": notes
    }
    return pack


def evolve_ga_feature_selection(Xdf: pd.DataFrame, y: pd.Series, n_pop: int = 24, n_gen: int = 8,
                                min_feat: int = 6, max_feat: Optional[int] = None,
                                lam_grid: Optional[List[float]] = None, n_folds: int = 3,
                                random_state: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(random_state)
    feats_all = list(Xdf.columns)
    D = len(feats_all)
    if max_feat is None:
        max_feat = min(D, 20)
    if lam_grid is None:
        lam_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    def random_mask():
        k = rng.randint(min_feat, max_feat + 1)
        idx = rng.choice(D, size=k, replace=False)
        mask = np.zeros(D, dtype=bool)
        mask[idx] = True
        return mask

    def mutate(mask: np.ndarray, p_flip: float = 0.08) -> np.ndarray:
        newm = mask.copy()
        for i in range(D):
            if rng.rand() < p_flip:
                newm[i] = not newm[i]
        if newm.sum() < min_feat:
            idx0 = np.where(~newm)[0]
            add = rng.choice(idx0, size=min_feat - newm.sum(), replace=False)
            newm[add] = True
        if newm.sum() > max_feat:
            idx1 = np.where(newm)[0]
            drop = rng.choice(idx1, size=newm.sum() - max_feat, replace=False)
            newm[drop] = False
        return newm

    def crossover(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        cut = rng.randint(1, D - 1)
        child = np.concatenate([m1[:cut], m2[cut:]])
        if child.sum() < min_feat:
            idx0 = np.where(~child)[0]
            add = rng.choice(idx0, size=min_feat - child.sum(), replace=False)
            child[add] = True
        if child.sum() > max_feat:
            idx1 = np.where(child)[0]
            drop = rng.choice(idx1, size=child.sum() - max_feat, replace=False)
            child[drop] = False
        return child

    def fitness(mask: np.ndarray, lam: float) -> float:
        sub_feats = [f for f, m in zip(feats_all, mask) if m]
        if len(sub_feats) < min_feat:
            return -1e9
        Xsub = Xdf[sub_feats]
        Xs, mu, sigma, feats = standardize_fit(Xsub)
        dates = Xsub.index.get_level_values("date").values
        folds = ts_cv_splits(dates, n_folds=n_folds)
        scores = []
        for tr_idx, va_idx in folds:
            Xtr = Xs[tr_idx];
            ytr = y.values[tr_idx]
            Xva = Xs[va_idx];
            yva = y.values[va_idx]
            w, b = ridge_regression(Xtr, ytr, lam=lam)
            pred = Xva @ w + b
            ic_mean, _, _ = groupby_date_ic(pd.Series(pred), pd.Series(yva), pd.Series(dates[va_idx]))
            scores.append(ic_mean if ic_mean == ic_mean else -1e9)
        return float(np.mean(scores)) if scores else -1e9

    pop = []
    for _ in range(n_pop):
        m = random_mask()
        lam = rng.choice(lam_grid)
        pop.append((m, lam, None))

    best = None
    for _ in range(n_gen):
        evaluated = []
        for m, lam, fit in pop:
            if fit is None:
                fit = fitness(m, lam)
            evaluated.append((m, lam, fit))
        evaluated.sort(key=lambda x: x[2], reverse=True)
        if best is None or evaluated[0][2] > best[2]:
            best = evaluated[0]

        elites = evaluated[: max(2, len(evaluated) // 5)]

        def select():
            k = 3
            cand_idx = np.random.choice(len(evaluated), size=min(k, len(evaluated)), replace=False)
            cand = [evaluated[i] for i in cand_idx]
            cand = sorted(cand, key=lambda x: x[2], reverse=True)
            return cand[0]

        next_pop = elites.copy()
        while len(next_pop) < len(pop):
            p1 = select()
            p2 = select()
            child_mask = crossover(p1[0], p2[0])
            child_mask = mutate(child_mask, p_flip=0.08)
            child_lam = p1[1] if np.random.rand() < 0.5 else p2[1]
            if np.random.rand() < 0.3:
                child_lam = np.random.choice(lam_grid)
            next_pop.append((child_mask, child_lam, None))
        pop = next_pop

    best_mask, best_lam, best_fit = best
    best_feats = [f for f, m in zip(feats_all, best_mask) if m]
    Xsub = Xdf[best_feats]
    Xs, mu, sigma, feats = standardize_fit(Xsub)
    w, b = ridge_regression(Xs, y.values, lam=best_lam)
    eval_res = evaluate_ic_on_Xy(Xsub, y, mu, sigma, feats, w, b)
    notes = {
        "method": "ga_ridge",
        "lam": best_lam,
        "ga": {"n_pop": len(pop), "fitness_ic": best_fit, "n_features": len(best_feats)},
        "eval": eval_res
    }
    ts = save_weights_record(horizon=app_state.get("horizon", 10), features=best_feats, weights=w, intercept=b,
                             mu=mu, sigma=sigma, lam=best_lam, notes=json.dumps(notes, ensure_ascii=False))
    pack = {
        "timestamp": ts, "horizon": app_state.get("horizon", 10), "features": best_feats, "weights": w,
        "intercept": b, "mu": mu, "sigma": sigma, "lambda": best_lam, "notes": notes
    }
    return pack


def train_and_save_weights(codes: List[str], horizon: int, mode: str = "ridge_cv",
                           lam: float = 1e-2, target_metric: str = "IC",
                           ga_conf: Optional[Dict[str, int]] = None, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    Xdf, y = build_training_data(codes, horizon, start_date=start_date, end_date=end_date)
    if Xdf.empty or y.empty:
        return None
    common_idx = Xdf.index.intersection(y.index)
    Xdf = Xdf.loc[common_idx]
    y = y.loc[common_idx]
    app_state["horizon"] = horizon

    if mode == "ridge_cv":
        pack = train_ridge_with_cv(Xdf, y, target_metric=target_metric)
        return pack
    else:
        if ga_conf is None:
            ga_conf = {"n_pop": 24, "n_gen": 8, "min_feat": 6, "max_feat": min(18, len(Xdf.columns))}
        pack = evolve_ga_feature_selection(
            Xdf, y,
            n_pop=ga_conf.get("n_pop", 24),
            n_gen=ga_conf.get("n_gen", 8),
            min_feat=ga_conf.get("min_feat", 6),
            max_feat=ga_conf.get("max_feat", min(18, len(Xdf.columns))),
            n_folds=ga_conf.get("n_folds", 3),
            random_state=ga_conf.get("random_state", 42)
        )
        return pack


def score_latest_for_codes(codes: List[str], weights_pack: Dict[str, Any],
                           start_date: Optional[str] = None) -> List[Tuple[str, str, float, str, str]]:
    feats = weights_pack["features"]
    w = weights_pack["weights"]
    b = weights_pack["intercept"]
    mu = weights_pack["mu"]
    sigma = weights_pack["sigma"]
    pos_thr = 0.02
    neg_thr = -0.02

    out = []
    for code in codes:
        df = load_price_df(code, start_date=start_date)
        if df.empty:
            continue
        df = compute_indicators(df)
        if df.empty or any(f not in df.columns for f in feats):
            continue
        last = df.iloc[-1]
        x = last[feats].astype(float).values
        xz = standardize_apply(x, mu, sigma)
        s = float(np.dot(xz, w) + b)
        adv = score_to_advice(s, pos_thr, neg_thr)
        reason = reasoning_from_signals(last)
        out.append((last["date"], code, s, adv, reason))
    return out


# =========================== 市场扫描与优质股筛选 ===========================
def get_market_codes(index_flag: str, on_date: str) -> List[str]:
    index_flag = (index_flag or "").upper()
    if bs is None:
        raise RuntimeError("未安装 baostock，无法获取指数成分")
    codes = []
    names = []
    if index_flag == "HS300":
        rs = bs.query_hs300_stocks(date=on_date)
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1]
            name = row[2] if len(row) > 2 else ""
            if code and (code.startswith("sh.") or code.startswith("sz.")):
                codes.append(code);
                names.append(name)
    elif index_flag == "ZZ500":
        rs = bs.query_zz500_stocks(date=on_date)
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1]
            name = row[2] if len(row) > 2 else ""
            if code and (code.startswith("sh.") or code.startswith("sz.")):
                codes.append(code);
                names.append(name)
    elif index_flag == "SZ50":
        rs = bs.query_sz50_stocks(date=on_date)
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1]
            name = row[2] if len(row) > 2 else ""
            if code and (code.startswith("sh.") or code.startswith("sz.")):
                codes.append(code);
                names.append(name)
    else:
        rs = bs.query_all_stock(day=on_date)
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[0]
            if code and (code.startswith("sh.") or code.startswith("sz.")):
                codes.append(code)
    # 去重&排序
    codes = sorted(list(set(codes)))
    # 缓存名称（指数场景可批量缓存）
    if names:
        for code, name in zip(codes, names):
            if name:
                set_stock_name(code, name)
    return codes


def update_data_for_codes(codes: List[str], start_date: str, end_date: str, adjustflag: str,
                          logger: Optional[Callable[[str], None]] = None,
                          stop_cb: Optional[Callable[[], bool]] = None):
    if bs is None:
        if logger:
            logger("未安装baostock，跳过远端拉取，仅使用本地数据")
        return
    for i, code in enumerate(codes):
        if stop_cb and stop_cb():
            if logger:
                logger("任务已被用户中止")
            break
        try:
            if logger:
                logger(f"更新 {code} ({i + 1}/{len(codes)}) ...")
            df_new = fetch_k_data_incremental(code, start_date, end_date, adjustflag=adjustflag)
            if df_new is not None and not df_new.empty:
                upsert_prices(code, df_new)
                if logger:
                    logger(f"{code} 新增 {len(df_new)} 行")
            else:
                if logger:
                    logger(f"{code} 无新增")
            # 缓存名称
            fetch_and_cache_name(code)
            time.sleep(0.06)
        except Exception as e:
            if logger:
                logger(f"{code} 更新失败: {e}")
            time.sleep(0.08)


def _trend_flag(last: pd.Series) -> float:
    return float(
        (last.get("close", np.nan) > last.get("sma20", np.inf)) and
        (last.get("sma20", -np.inf) > last.get("sma60", np.inf)) and
        (last.get("macd", -1e9) > 0) and
        (last.get("sma20_slope5", 0.0) > 0)
    )


def _breakout_flag(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    last = df.iloc[-1]
    near = float(last.get("near_high55") or 0.0)  # close / high55
    return float(near >= 0.98)


def quality_score_advanced(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    综合评分：qscore = 0.5*pred_z + 0.25*mom_z - 0.25*vol_z + 0.12*trend + 0.08*breakout + 0.05*dd60
    说明：dd60为[-1,0]，为负，直接相加相当于惩罚深回撤
    """
    out = df_res.copy()
    for col in ["score", "mom10", "vol20"]:
        if col not in out.columns:
            out[col] = np.nan
    # z分
    for col, new in [("score", "score_z"), ("mom10", "mom_z"), ("vol20", "vol_z")]:
        mu = float(out[col].mean())
        sd = float(out[col].std(ddof=0)) if out[col].std(ddof=0) > 1e-9 else 1.0
        out[new] = (out[col] - mu) / sd
    # 组合
    out["qscore"] = 0.5 * out["score_z"] + 0.25 * out["mom_z"] - 0.25 * out["vol_z"] \
                    + 0.12 * out.get("trend", 0.0) + 0.08 * out.get("breakout55", 0.0) \
                    + 0.05 * out.get("dd60", 0.0)
    return out


def baseline_score_no_model(last: pd.Series) -> float:
    """
    无模型时的基线打分：鼓励中短期动量、低波动、接近55日高、趋势成立
    """
    s = 0.0
    mom = float(last.get("mom10") or 0.0)
    vol = float(last.get("vol20") or 0.0)
    trend = float(last.get("close", 0) > last.get("sma20", 1e9)) + float(last.get("sma20", 0) > last.get("sma60", 1e9))
    near = float(last.get("near_high55") or 0.0)
    macd = float(last.get("macd") or 0.0)
    s += 0.8 * mom
    s -= 0.15 * vol
    s += 0.2 * (near - 0.9)
    s += 0.05 * np.tanh(macd)
    s += 0.1 * trend
    return s


def scan_market_and_rank(index_flag: str, start_date: str, end_date: str, adjustflag: str,
                         weights_pack: Optional[Dict[str, Any]], min_amt20: float = 2e8, topN: int = 30,
                         logger=None,
                         codes_override: Optional[List[str]] = None,
                         skip_update: bool = False,
                         batch_size: int = 300,
                         sleep_ms_between_batches: int = 200,
                         require_trend: bool = False,
                         require_breakout: bool = False,
                         max_vol_z: Optional[float] = None,
                         rsi_min: Optional[float] = None,
                         rsi_max: Optional[float] = None,
                         stop_cb: Optional[Callable[[], bool]] = None) -> pd.DataFrame:
    """
    强化版扫描（无权重也可运行）：
    - codes_override: 使用自定义股票池
    - skip_update: 跳过全量增量更新（快）
    - batch_size + sleep: ALL 扫描时控制接口压力
    - 高级筛选：趋势/突破/最大波动z阈值/RSI区间
    - stop_cb: 可选的中止回调
    """
    scan_date = end_date
    try:
        if codes_override is not None:
            codes = list(sorted(set([c.strip() for c in codes_override if c.strip()])))
        else:
            codes = get_market_codes(index_flag, scan_date)
    except Exception as e:
        if logger:
            logger(f"获取指数成分失败: {e}")
        return pd.DataFrame()

    if not codes:
        if logger:
            logger("指数成分/自定义股票池为空")
        return pd.DataFrame()

    if not skip_update and bs is not None:
        # 批量更新
        if len(codes) > batch_size and index_flag.upper() in ("ALL",):
            n = len(codes)
            for i in range(0, n, batch_size):
                if stop_cb and stop_cb():
                    if logger:
                        logger("任务已被用户中止")
                    break
                part = codes[i:i + batch_size]
                if logger:
                    logger(f"批次更新 {i // batch_size + 1}/{math.ceil(n / batch_size)}，数量={len(part)}")
                update_data_for_codes(part, start_date, end_date, adjustflag, logger=logger, stop_cb=stop_cb)
                time.sleep(max(0, sleep_ms_between_batches) / 1000.0)
        else:
            update_data_for_codes(codes, start_date, end_date, adjustflag, logger=logger, stop_cb=stop_cb)
    else:
        if logger:
            logger("跳过行情增量更新（使用本地DB数据）")

    # 模型参数（可能为空）
    feats = weights_pack["features"] if weights_pack else None
    w = weights_pack["weights"] if weights_pack else None
    b = weights_pack["intercept"] if weights_pack else 0.0
    mu = weights_pack["mu"] if weights_pack else None
    sigma = weights_pack["sigma"] if weights_pack else None

    rows = []
    for idx, code in enumerate(codes):
        if stop_cb and stop_cb():
            if logger:
                logger("任务已被用户中止")
            break
        try:
            df = load_price_df(code, start_date=start_date, end_date=end_date)
            if df.empty or len(df) < 80:
                continue
            df = compute_indicators(df)
            last = df.iloc[-1]
            amt20 = float(last.get("amt20") or 0.0)
            if np.isnan(amt20) or amt20 < float(min_amt20):
                continue

            # 预测分：若无模型，使用基线分；若模型特征缺失，跳过预测分但仍可进入质量打分
            s = np.nan
            if feats and w is not None and sigma is not None:
                if all(f in df.columns for f in feats):
                    x = last[feats].astype(float).values
                    xz = standardize_apply(x, mu, sigma)
                    s = float(np.dot(xz, w) + b)
            if not np.isfinite(s):
                s = baseline_score_no_model(last)

            # 质量信号
            trend = _trend_flag(last)
            breakout55 = _breakout_flag(df)
            dd60 = float(last.get("dd60") or 0.0)  # 负值为回撤
            mom10 = float(last.get("mom10") or np.nan)
            vol20 = float(last.get("vol20") or np.nan)
            atr14 = float(last.get("atr14") or np.nan)
            rsi14 = float(last.get("rsi14") or np.nan)

            rows.append({
                "code": code, "date": str(last["date"]), "close": float(last["close"]), "score": s,
                "mom10": mom10, "vol20": vol20, "atr14": atr14, "amt20": amt20,
                "trend": trend, "breakout55": float(breakout55), "dd60": dd60, "rsi14": rsi14
            })
        except Exception as e:
            if logger:
                logger(f"{code} 扫描失败: {e}")

    if not rows:
        return pd.DataFrame()

    df_res = pd.DataFrame(rows)

    # 计算综合质量分
    df_res = quality_score_advanced(df_res)

    # 波动z阈值过滤（可选，需先计算z分）
    if max_vol_z is not None:
        df_res = df_res[df_res["vol_z"] <= float(max_vol_z)].copy()

    # RSI区间过滤（可选）
    if rsi_min is not None:
        df_res = df_res[df_res["rsi14"] >= float(rsi_min)].copy()
    if rsi_max is not None:
        df_res = df_res[df_res["rsi14"] <= float(rsi_max)].copy()

    # 趋势/突破过滤（可选）
    if require_trend:
        df_res = df_res[df_res["trend"] >= 0.5].copy()
    if require_breakout:
        df_res = df_res[df_res["breakout55"] >= 0.5].copy()

    # TopN
    if df_res.empty:
        return df_res
    df_res = df_res.sort_values(["qscore", "score", "mom10"], ascending=[False, False, False]).head(topN).reset_index(
        drop=True)
    return df_res


def gen_extended_advice(row: pd.Series, risk_pct: float, capital: float) -> Tuple[str, str, float, float, float, int]:
    """
    扩展建议：仓位/止损/止盈/数量
    - 止损：close - 1.5 * ATR14（至少-5%）
    - 止盈：close + max(2.5*ATR, (close-stop)*1.5)
    - 仓位：pos_pct = min(30%, risk_pct / 跌幅%)
    - 数量：100股一手
    """
    close = float(row.get("close"))
    atr14 = float(row.get("atr14") or 0.0)
    if np.isnan(atr14) or atr14 <= 0:
        stop = close * 0.95
    else:
        stop = min(close * 0.95, close - 1.5 * atr14)
    target = close + max(2.5 * atr14, (close - stop) * 1.5)
    drop_pct = max(1e-4, (close - stop) / close)
    pos_pct = min(0.3, risk_pct / drop_pct)

    score_val = float(row.get("score", 0.0))
    mom10 = float(row.get("mom10") or 0.0)
    adv_type = "买入" if score_val > 0 and mom10 >= -0.02 else "观望"

    pos_amount = pos_pct * capital
    qty = int(max(0, math.floor(pos_amount / close / 100.0) * 100))

    advice = f"建议：{adv_type}；建议仓位≈{pos_pct * 100:.1f}%；止损≈{stop:.2f}；止盈≈{target:.2f}；建议数量≈{qty}股"
    reason = f"扩展：预测/基线分={score_val:.3f}；动量10日={mom10:.1%}；波动20日={float(row.get('vol20') or 0):.2f}；ATR14={atr14:.2f}"
    return advice, reason, pos_pct, stop, target, qty


# =========================== 历史建议绩效评估 ===========================
def _find_future_return(code: str, date_str: str, horizon: int) -> Optional[float]:
    df = load_price_df(code)
    if df.empty:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    idx_list = df.index[df["date"] >= date_str].tolist()
    if not idx_list:
        return None
    idx = idx_list[0]
    if idx + horizon >= len(df):
        return None
    p0 = float(df.loc[idx, "close"])
    p1 = float(df.loc[idx + horizon, "close"])
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0


def _parse_basic_advice(text: str) -> str:
    if not text:
        return "观望"
    if "强烈买入" in text:
        return "强烈买入"
    if "强烈卖出" in text:
        return "强烈卖出"
    buy = "买入" in text
    sell = "卖出" in text
    if buy and not sell:
        return "买入"
    if sell and not buy:
        return "卖出"
    return "观望"


def evaluate_history_performance(code: Optional[str] = None, limit: int = 500,
                                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                                 type_filter: Optional[str] = None, horizon: Optional[int] = None) -> pd.DataFrame:
    rows = query_advice(code=code, limit=limit, start_date=start_date, end_date=end_date, type_filter=type_filter)
    if not rows:
        return pd.DataFrame(columns=["type", "n", "avg_ret", "win_rate"])
    eval_rows = []
    for date, code_, score, advice, reasoning, hz in rows:
        hz_use = int(horizon) if horizon is not None else int(hz)
        ret = _find_future_return(code_, date, hz_use)
        if ret is None:
            continue
        typ = _parse_basic_advice(advice or "")
        signed_ret = ret
        if typ in ("卖出", "强烈卖出"):
            signed_ret = -ret
        eval_rows.append({"type": typ, "ret": signed_ret, "score": float(score)})
    if not eval_rows:
        return pd.DataFrame(columns=["type", "n", "avg_ret", "win_rate"])
    df = pd.DataFrame(eval_rows)
    g = df.groupby("type")
    out = g["ret"].agg(n="count", avg_ret="mean", win_rate=lambda x: (x > 0).mean()).reset_index()
    out = out.sort_values(["avg_ret", "win_rate", "n"], ascending=[False, False, False])
    return out


def evaluate_quantile_performance(code: Optional[str] = None, limit: int = 1000,
                                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  horizon: Optional[int] = None, q: int = 5) -> pd.DataFrame:
    """
    分位统计：按score分为q组，统计每组平均收益和胜率（买卖混合为签名收益）
    """
    rows = query_advice(code=code, limit=limit, start_date=start_date, end_date=end_date, type_filter=None)
    if not rows:
        return pd.DataFrame(columns=["quantile", "n", "avg_ret", "win_rate"])
    recs = []
    for date, code_, score, advice, reasoning, hz in rows:
        hz_use = int(horizon) if horizon is not None else int(hz)
        ret = _find_future_return(code_, date, hz_use)
        if ret is None:
            continue
        typ = _parse_basic_advice(advice or "")
        signed_ret = ret if typ in ("买入", "强烈买入") else (-ret if typ in ("卖出", "强烈卖出") else 0.0)
        recs.append({"date": date, "code": code_, "score": float(score), "ret": float(signed_ret)})
    if not recs:
        return pd.DataFrame(columns=["quantile", "n", "avg_ret", "win_rate"])
    df = pd.DataFrame(recs).sort_values("score")
    df["q"] = pd.qcut(df["score"], q=q, labels=False, duplicates="drop")
    g = df.groupby("q")
    out = g["ret"].agg(n="count", avg_ret="mean", win_rate=lambda x: (x > 0).mean()).reset_index()
    out["quantile"] = out["q"].astype(int) + 1
    out = out.drop(columns=["q"]).sort_values("quantile")
    return out


# =========================== 代码格式化工具 ===========================
def normalize_code(code: str) -> Optional[str]:
    """
    归一化代码：
    - 支持: sh.600000 / sz.000001
    - 600000 或 000001 -> 按首位推断交易所 6->sh, 0/3->sz
    - 600000.SH / 000001.SZ
    - 过滤非法
    """
    s = code.strip()
    if not s:
        return None
    s = s.replace("_", ".").replace("-", ".")
    s = s.lower()
    if s.startswith("sh.") or s.startswith("sz."):
        return s
    m = re.match(r"^(\d{6})\.(sh|sz)$", s)
    if m:
        six, ex = m.group(1), m.group(2)
        return f"{ex}.{six}"
    m = re.match(r"^(\d{6})$", s)
    if m:
        six = m.group(1)
        if six.startswith("6"):
            return f"sh.{six}"
        elif six.startswith(("0", "3")):
            return f"sz.{six}"
    return None


def parse_codes_from_text(text: str) -> List[str]:
    toks = re.split(r"[,\s\r\n;]+", text.strip())
    out = []
    for t in toks:
        nc = normalize_code(t)
        if nc:
            out.append(nc)
    return sorted(list(set(out)))


# =========================== 回测模块 ===========================
def compute_metrics_from_curve(nav: pd.Series, period_days: int) -> Dict[str, float]:
    if nav.empty:
        return {"CAGR": np.nan, "MaxDD": np.nan, "Sharpe": np.nan, "WinRate": np.nan}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"CAGR": np.nan, "MaxDD": np.nan, "Sharpe": np.nan, "WinRate": np.nan}
    total_days = len(nav)
    cagr = float(nav.iloc[-1]) ** (252.0 / (total_days or 1)) - 1.0
    peak = nav.cummax()
    dd = (nav / peak - 1.0)
    maxdd = float(dd.min())
    mean_r = float(rets.mean()) * (252.0 / period_days)
    std_r = float(rets.std(ddof=0)) * math.sqrt(252.0 / period_days)
    sharpe = mean_r / (std_r + 1e-9)
    win = float((rets > 0).mean())
    return {"CAGR": cagr, "MaxDD": maxdd, "Sharpe": sharpe, "WinRate": win}


def backtest_topN(weights_pack: Optional[Dict[str, Any]],
                  codes: List[str], start_date: str, end_date: str,
                  topN: int, hold_days: int, min_amt20: float = 2e8,
                  fee_bps: float = 1.0, logger=None,
                  stop_cb: Optional[Callable[[], bool]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    简易回测：每 hold_days 调仓一次，按当日打分选 TopN 等权持有 hold_days
    - 支持无模型（使用基线打分）
    - 交易费率：单边 bps（如1=万分之一）
    返回：交易期收益表 period_returns（date, ret），净值序列 nav（按调仓期）
    - stop_cb: 可选，支持中止
    """
    if not codes:
        return pd.DataFrame(), pd.Series(dtype=float)

    # 预加载数据
    feats = weights_pack["features"] if weights_pack else None
    w = weights_pack["weights"] if weights_pack else None
    b = weights_pack["intercept"] if weights_pack else 0.0
    mu = weights_pack["mu"] if weights_pack else None
    sigma = weights_pack["sigma"] if weights_pack else None

    data_map: Dict[str, pd.DataFrame] = {}
    for code in codes:
        if stop_cb and stop_cb():
            if logger:
                logger("任务已被用户中止")
            break
        df = load_price_df(code, start_date=start_date, end_date=end_date)
        if df.empty or len(df) < max(60, hold_days + 10):
            continue
        df = compute_indicators(df)
        data_map[code] = df

    if not data_map:
        return pd.DataFrame(), pd.Series(dtype=float)

    # 所有交易日（用并集，然后每期以可交易样本为准）
    all_dates = sorted(list(set(np.concatenate([df["date"].values for df in data_map.values()]))))
    if len(all_dates) < hold_days + 2:
        return pd.DataFrame(), pd.Series(dtype=float)

    # 生成调仓日期序列（每 hold_days 一个）
    rebalance_dates = all_dates[::hold_days]
    # 确保有未来持有期
    rebalance_dates = [d for d in rebalance_dates if
                       (d in all_dates) and (all_dates.index(d) + hold_days < len(all_dates))]
    if not rebalance_dates:
        return pd.DataFrame(), pd.Series(dtype=float)

    period_rets = []
    for d in rebalance_dates:
        if stop_cb and stop_cb():
            if logger:
                logger("任务已被用户中止")
            break
        # 这期的可选股票与打分
        basket = []
        for code, df in data_map.items():
            if d not in list(df["date"]):
                continue
            row = df[df["date"] == d].iloc[0]
            # 流动性过滤
            amt20 = float(row.get("amt20") or 0.0)
            if not np.isfinite(amt20) or amt20 < min_amt20:
                continue
            # 模型/基线打分
            s = np.nan
            if feats and w is not None and sigma is not None and all(f in df.columns for f in feats):
                x = row[feats].astype(float).values
                xz = standardize_apply(x, mu, sigma)
                s = float(np.dot(xz, w) + b)
            if not np.isfinite(s):
                s = baseline_score_no_model(row)
            basket.append((code, s))

        if not basket:
            continue
        basket.sort(key=lambda x: x[1], reverse=True)
        picks = [code for code, _ in basket[:max(1, topN)]]

        # 期收益：未来 hold_days 的简单收盘收益，等权
        rets = []
        end_idx = all_dates.index(d) + hold_days
        d1 = all_dates[end_idx]
        for code in picks:
            df = data_map[code]
            # 找d和d1在该股的可用行
            if d not in list(df["date"]) or d1 not in list(df["date"]):
                continue
            p0 = float(df[df["date"] == d]["close"].iloc[0])
            p1 = float(df[df["date"] == d1]["close"].iloc[0])
            r = p1 / p0 - 1.0
            # 费用：进+出
            fee = (fee_bps / 10000.0) * 2.0
            r = r - fee
            rets.append(r)
        if not rets:
            continue
        period_rets.append((d1, float(np.mean(rets))))

    if not period_rets:
        return pd.DataFrame(), pd.Series(dtype=float)

    pr = pd.DataFrame(period_rets, columns=["date", "ret"])
    pr = pr.sort_values("date").reset_index(drop=True)
    nav = (1.0 + pr["ret"]).cumprod()
    nav.index = pd.to_datetime(pr["date"])
    return pr, nav


# =========================== GUI 应用 ===========================
class EvoAdvisorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION} - Baostock")
        self.geometry("1380x960")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # 取消/停止控制
        self.cancel_event = threading.Event()

        # 变量（去掉预设股票）
        self.codes_var = tk.StringVar(value="")
        self.start_var = tk.StringVar(value="2024-09-24")
        self.end_var = tk.StringVar(value=dt.datetime.now().strftime(DATE_FMT))
        self.horizon_var = tk.IntVar(value=10)
        self.adjustflag_var = tk.StringVar(value="2")

        # 训练配置
        self.train_mode_var = tk.StringVar(value="ridge_cv")  # ridge_cv / ga
        self.target_metric_var = tk.StringVar(value="IC")
        self.cv_folds_var = tk.IntVar(value=4)
        self.ga_pop_var = tk.IntVar(value=24)
        self.ga_gen_var = tk.IntVar(value=8)

        # 扫描/资金与风险参数
        self.index_flag_var = tk.StringVar(value="HS300")
        self.min_amt20_var = tk.DoubleVar(value=2e8)
        self.topN_var = tk.IntVar(value=30)
        self.capital_var = tk.DoubleVar(value=100000.0)
        self.risk_pct_var = tk.DoubleVar(value=0.01)
        self.skip_update_before_scan_var = tk.BooleanVar(value=False)  # ALL时建议勾选
        self.batch_size_var = tk.IntVar(value=300)
        self.sleep_ms_var = tk.IntVar(value=200)

        # 高级筛选
        self.require_trend_var = tk.BooleanVar(value=False)
        self.require_breakout_var = tk.BooleanVar(value=False)
        self.max_vol_z_var = tk.DoubleVar(value=2.5)
        self.filter_volz_enable_var = tk.BooleanVar(value=True)
        self.rsi_min_var = tk.DoubleVar(value=20.0)
        self.rsi_max_var = tk.DoubleVar(value=85.0)
        self.filter_rsi_enable_var = tk.BooleanVar(value=False)

        # 自定义股票池
        self.custom_codes: List[str] = []
        self.custom_codes_path: Optional[str] = None
        self.custom_count_var = tk.StringVar(value="未载入")
        self.use_custom_for_train_var = tk.BooleanVar(value=False)
        self.use_custom_for_scan_var = tk.BooleanVar(value=False)

        # 回测参数
        self.bt_topN_var = tk.IntVar(value=20)
        self.bt_hold_var = tk.IntVar(value=10)
        self.bt_fee_bps_var = tk.DoubleVar(value=1.0)  # 单边bp
        self.bt_min_amt20_var = tk.DoubleVar(value=2e8)

        # 主题
        self.theme_var = tk.StringVar(value="light")  # light / dark

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")

        # UI 初始化
        self._init_style()
        self._build_menu()
        self._build_ui()
        self._load_config()  # 恢复配置
        self._apply_theme(self.theme_var.get())

        # baostock 登录
        try:
            bs_safe_login()
            self.log("baostock 登录成功")
            self.set_status("baostock 登录成功")
        except Exception as e:
            self.log(f"[错误] {e}", error=True)
            self.set_status("未登录 baostock（离线模式）")

        # 启动时刷新用户操作
        self.refresh_ops()

    # ---------- 样式与主题 ----------
    def _init_style(self):
        self.style = ttk.Style()
        # 兼容平台主题
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

    def _apply_theme(self, theme: str):
        if theme == "dark":
            bg = "#20252b";
            fg = "#e6edf3";
            acc = "#2b90d9"
            self.configure(bg=bg)
            # 配置常见控件
            for element in ["TFrame", "TLabelframe", "TLabelframe.Label", "TLabel", "TButton", "TNotebook",
                            "TNotebook.Tab", "TEntry", "TCombobox", "Treeview"]:
                try:
                    self.style.configure(element, background=bg, foreground=fg, fieldbackground="#2a3036")
                except Exception:
                    pass
            try:
                self.style.map("TButton", foreground=[("active", fg)], background=[("active", acc)])
                self.style.configure("Treeview", background="#262c33", fieldbackground="#262c33", foreground=fg,
                                     rowheight=22)
            except Exception:
                pass
        else:
            # light
            try:
                self.style.theme_use("clam")
            except Exception:
                pass

    def _build_menu(self):
        menubar = tk.Menu(self)

        # 文件
        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(label="导出扫描结果CSV", command=self.export_scan_csv)
        menu_file.add_separator()
        menu_file.add_command(label="导出最新权重...", command=self.on_export_weights)
        menu_file.add_command(label="导入权重...", command=self.on_import_weights)
        menu_file.add_separator()
        menu_file.add_command(label="退出", command=self.on_close)
        menubar.add_cascade(label="文件", menu=menu_file)

        # 视图
        menu_view = tk.Menu(menubar, tearoff=0)
        menu_view.add_radiobutton(label="浅色主题", variable=self.theme_var, value="light",
                                  command=lambda: self._apply_theme("light"))
        menu_view.add_radiobutton(label="深色主题", variable=self.theme_var, value="dark",
                                  command=lambda: self._apply_theme("dark"))
        menubar.add_cascade(label="视图", menu=menu_view)

        # 帮助
        menu_help = tk.Menu(menubar, tearoff=0)
        menu_help.add_command(label="关于", command=self._on_about)
        menubar.add_cascade(label="帮助", menu=menu_help)

        self.config(menu=menubar)

    def _on_about(self):
        messagebox.showinfo("关于",
                            f"{APP_NAME}\n版本：v{APP_VERSION}\n\n开源组件：tkinter/pandas/numpy/matplotlib/baostock\n仅用于研究学习，不构成投资建议。")

    # ---------- UI ----------
    def _build_ui(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # 页：数据
        self.page_data = ttk.Frame(self.nb)
        self.nb.add(self.page_data, text="数据 / 股票池")
        self._build_page_data(self.page_data)

        # 页：训练/进化
        self.page_train = ttk.Frame(self.nb)
        self.nb.add(self.page_train, text="训练 / 进化")
        self._build_page_train(self.page_train)

        # 页：自选建议
        self.page_adv = ttk.Frame(self.nb)
        self.nb.add(self.page_adv, text="自选建议")
        self._build_page_adv(self.page_adv)

        # 页：市场扫描
        self.page_scan = ttk.Frame(self.nb)
        self.nb.add(self.page_scan, text="市场扫描")
        self._build_page_scan(self.page_scan)

        # 页：回测
        self.page_bt = ttk.Frame(self.nb)
        self.nb.add(self.page_bt, text="回测")
        self._build_page_backtest(self.page_bt)

        # 页：绩效/历史
        self.page_eval = ttk.Frame(self.nb)
        self.nb.add(self.page_eval, text="绩效 / 历史")
        self._build_page_eval(self.page_eval)

        # 页：图表
        self.page_plot = ttk.Frame(self.nb)
        self.nb.add(self.page_plot, text="图表")
        self._build_page_plot(self.page_plot)

        # 页：用户操作
        self.page_ops = ttk.Frame(self.nb)
        self.nb.add(self.page_ops, text="用户操作")
        self._build_page_ops(self.page_ops)

        # 页：操作说明
        self.page_help = ttk.Frame(self.nb)
        self.nb.add(self.page_help, text="使用说明")
        self._build_page_help(self.page_help)

        # 页：日志
        self.page_log = ttk.Frame(self.nb)
        self.nb.add(self.page_log, text="日志")
        self._build_page_log(self.page_log)

        # 状态栏
        status_bar = ttk.Frame(self, padding=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status_var, foreground="green").pack(side=tk.LEFT)
        # 停止按钮
        self.btn_stop = ttk.Button(status_bar, text="停止任务", command=self.on_stop_tasks)
        self.btn_stop.pack(side=tk.RIGHT, padx=6)

    def _add_tree_with_scroll(self, parent, columns, height=12, widths: Optional[Dict[str, int]] = None):
        """
        创建带垂直滚动条的Treeview
        """
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        # 设置列
        for c in columns:
            tree.heading(c, text=c)
            if widths:
                tree.column(c, width=widths.get(c, 100), anchor=tk.W)
        return frame, tree

    def _build_page_data(self, parent):
        frm = ttk.LabelFrame(parent, text="股票池与数据更新", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="股票代码(逗号/换行分隔)").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.codes_var, width=80).grid(row=0, column=1, columnspan=4, sticky="we", padx=6)
        ttk.Button(frm, text="导入股票列表文件", command=self.on_import_codes).grid(row=0, column=5, padx=6, sticky="w")
        ttk.Button(frm, text="从指数填充（HS300）", command=self.fill_codes_from_index_hs300).grid(row=0, column=6,
                                                                                                 padx=6, sticky="w")
        ttk.Label(frm, text="自定义池状态:").grid(row=0, column=7, sticky="e")
        ttk.Label(frm, textvariable=self.custom_count_var, foreground="blue").grid(row=0, column=8, sticky="w")

        ttk.Checkbutton(frm, text="训练用自定义池", variable=self.use_custom_for_train_var).grid(row=1, column=1,
                                                                                                 sticky="w")
        ttk.Checkbutton(frm, text="扫描用自定义池", variable=self.use_custom_for_scan_var).grid(row=1, column=2,
                                                                                                sticky="w")

        ttk.Label(frm, text="起始日期").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.start_var, width=12).grid(row=2, column=1, sticky="w")
        ttk.Label(frm, text="结束日期").grid(row=2, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.end_var, width=12).grid(row=2, column=3, sticky="w")
        ttk.Label(frm, text="复权").grid(row=2, column=4, sticky="e")
        ttk.Combobox(frm, textvariable=self.adjustflag_var, values=["1", "2", "3"], width=6, state="readonly").grid(
            row=2, column=5, sticky="w")

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        self.btn_update = ttk.Button(btns, text="增量更新数据（按输入框/自定义池）", command=self.on_update_data_async)
        self.btn_update.pack(side=tk.LEFT, padx=6)
        ttk.Label(btns, text="说明：增量拉取baostock数据并写入SQLite；离线模式将仅使用本地数据").pack(side=tk.LEFT,
                                                                                                    padx=8)

    def _build_page_train(self, parent):
        frm = ttk.LabelFrame(parent, text="训练配置", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(frm, text="预测视野H(日)").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.horizon_var, width=8).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="训练模式").grid(row=0, column=2, sticky="e")
        ttk.Combobox(frm, textvariable=self.train_mode_var, values=["ridge_cv", "ga"], width=12, state="readonly").grid(
            row=0, column=3, sticky="w", padx=4)
        ttk.Label(frm, text="目标指标").grid(row=0, column=4, sticky="e")
        ttk.Combobox(frm, textvariable=self.target_metric_var, values=["IC", "MSE"], width=8, state="readonly").grid(
            row=0, column=5, sticky="w", padx=4)
        ttk.Label(frm, text="CV折数").grid(row=0, column=6, sticky="e")
        ttk.Entry(frm, textvariable=self.cv_folds_var, width=6).grid(row=0, column=7, sticky="w", padx=4)

        frm2 = ttk.LabelFrame(parent, text="进化(遗传算法)参数", padding=8)
        frm2.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(frm2, text="种群大小").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm2, textvariable=self.ga_pop_var, width=8).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frm2, text="迭代代数").grid(row=0, column=2, sticky="e")
        ttk.Entry(frm2, textvariable=self.ga_gen_var, width=8).grid(row=0, column=3, sticky="w", padx=4)

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        self.btn_train = ttk.Button(btns, text="开始训练/进化", command=self.on_train_async)
        self.btn_train.pack(side=tk.LEFT, padx=6)
        self.pb = ttk.Progressbar(btns, mode="indeterminate", length=240)
        self.pb.pack(side=tk.LEFT, padx=8)

        # 权重导入/导出
        self.btn_export_w = ttk.Button(btns, text="导出最新权重", command=self.on_export_weights)
        self.btn_export_w.pack(side=tk.LEFT, padx=6)
        self.btn_import_w = ttk.Button(btns, text="导入权重", command=self.on_import_weights)
        self.btn_import_w.pack(side=tk.LEFT, padx=6)

        # 结果表
        cols = ("timestamp", "horizon", "method", "lambda", "ic", "icir", "mse", "n_days")
        frame, self.train_tree = self._add_tree_with_scroll(parent, columns=cols, height=8,
                                                            widths={"timestamp": 160, "horizon": 70, "method": 110,
                                                                    "lambda": 90, "ic": 80, "icir": 80, "mse": 120,
                                                                    "n_days": 80})
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def _build_page_adv(self, parent):
        top = ttk.LabelFrame(parent, text="参数", padding=8)
        top.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(top, text="自选股票(逗号/换行分隔)").grid(row=0, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.codes_var, width=80).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Button(top, text="从自定义池填充", command=self.fill_codes_from_custom).grid(row=0, column=2, padx=6)
        ttk.Label(top, text="资金(元)").grid(row=0, column=3, sticky="e")
        ttk.Entry(top, textvariable=self.capital_var, width=12).grid(row=0, column=4, sticky="w", padx=4)
        ttk.Label(top, text="单笔风险%").grid(row=0, column=5, sticky="e")
        ttk.Entry(top, textvariable=self.risk_pct_var, width=8).grid(row=0, column=6, sticky="w", padx=4)
        self.btn_adv = ttk.Button(top, text="生成扩展建议并保存", command=self.on_advise_async)
        self.btn_adv.grid(row=0, column=7, padx=8)

        cols = ("date", "code", "name", "score", "advice", "reasoning", "horizon")
        frame, self.adv_tree = self._add_tree_with_scroll(parent, columns=cols, height=12,
                                                          widths={"date": 100, "code": 90, "name": 120, "score": 80,
                                                                  "advice": 240, "reasoning": 840, "horizon": 80})
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def _build_page_scan(self, parent):
        frm = ttk.LabelFrame(parent, text="扫描参数", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="指数/池").grid(row=0, column=0, sticky="e")
        ttk.Combobox(frm, textvariable=self.index_flag_var, values=["HS300", "ZZ500", "SZ50", "ALL", "CUSTOM"],
                     width=10, state="readonly").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Checkbutton(frm, text="使用自定义池", variable=self.use_custom_for_scan_var).grid(row=0, column=2,
                                                                                              sticky="w")
        ttk.Button(frm, text="导入股票列表文件", command=self.on_import_codes).grid(row=0, column=3, padx=6)
        ttk.Label(frm, text="状态:").grid(row=0, column=4, sticky="e")
        ttk.Label(frm, textvariable=self.custom_count_var, foreground="blue").grid(row=0, column=5, sticky="w")

        ttk.Label(frm, text="TopN").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.topN_var, width=8).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="近20日均额≥").grid(row=1, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.min_amt20_var, width=14).grid(row=1, column=3, sticky="w", padx=4)
        ttk.Label(frm, text="资金(元)").grid(row=1, column=4, sticky="e")
        ttk.Entry(frm, textvariable=self.capital_var, width=12).grid(row=1, column=5, sticky="w", padx=4)
        ttk.Label(frm, text="单笔风险%").grid(row=1, column=6, sticky="e")
        ttk.Entry(frm, textvariable=self.risk_pct_var, width=8).grid(row=1, column=7, sticky="w", padx=4)

        frm2 = ttk.LabelFrame(parent, text="ALL扫描优化 / 高级筛选", padding=8)
        frm2.pack(fill=tk.X, padx=8, pady=6)
        ttk.Checkbutton(frm2, text="扫描前跳过增量更新（快）", variable=self.skip_update_before_scan_var).grid(row=0,
                                                                                                             column=0,
                                                                                                             sticky="w")
        ttk.Label(frm2, text="批大小").grid(row=0, column=1, sticky="e")
        ttk.Entry(frm2, textvariable=self.batch_size_var, width=8).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(frm2, text="批间隔(ms)").grid(row=0, column=3, sticky="e")
        ttk.Entry(frm2, textvariable=self.sleep_ms_var, width=8).grid(row=0, column=4, sticky="w", padx=4)

        ttk.Checkbutton(frm2, text="启用波动Z阈值", variable=self.filter_volz_enable_var).grid(row=1, column=0,
                                                                                               sticky="w")
        ttk.Label(frm2, text="max vol_z").grid(row=1, column=1, sticky="e")
        ttk.Entry(frm2, textvariable=self.max_vol_z_var, width=8).grid(row=1, column=2, sticky="w", padx=4)
        ttk.Checkbutton(frm2, text="启用RSI区间", variable=self.filter_rsi_enable_var).grid(row=1, column=3, sticky="w")
        ttk.Label(frm2, text="RSI[min,max]").grid(row=1, column=4, sticky="e")
        ttk.Entry(frm2, textvariable=self.rsi_min_var, width=6).grid(row=1, column=5, sticky="w", padx=2)
        ttk.Entry(frm2, textvariable=self.rsi_max_var, width=6).grid(row=1, column=6, sticky="w", padx=2)
        ttk.Checkbutton(frm2, text="要求趋势(价>SMA20>SMA60, MACD>0, SMA20上行)", variable=self.require_trend_var).grid(
            row=2, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(frm2, text="要求接近55日高(≥98%)", variable=self.require_breakout_var).grid(row=2, column=4,
                                                                                                    columnspan=3,
                                                                                                    sticky="w")

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        self.btn_scan = ttk.Button(btns, text="开始市场扫描（无权重也可用）", command=self.on_scan_market_async)
        self.btn_scan.pack(side=tk.LEFT, padx=6)
        self.btn_export_scan = ttk.Button(btns, text="导出结果CSV", command=self.export_scan_csv, state=tk.DISABLED)
        self.btn_export_scan.pack(side=tk.LEFT, padx=6)

        cols = ["date", "code", "name", "close", "score", "qscore", "mom10", "vol20", "atr14", "amt20",
                "trend", "breakout55", "dd60",
                "pos_pct", "stop", "target", "qty", "advice", "reasoning"]
        widths = {
            "date": 90, "code": 90, "name": 120, "close": 80, "score": 70, "qscore": 70, "mom10": 80, "vol20": 80,
            "atr14": 80, "amt20": 120, "trend": 70, "breakout55": 90, "dd60": 80,
            "pos_pct": 80, "stop": 90, "target": 90, "qty": 80, "advice": 240, "reasoning": 520
        }
        frame, self.scan_tree = self._add_tree_with_scroll(parent, columns=cols, height=18, widths=widths)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.scan_df_cache: Optional[pd.DataFrame] = None

    def _build_page_backtest(self, parent):
        frm = ttk.LabelFrame(parent, text="回测参数", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="TopN").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.bt_topN_var, width=8).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="持有H(日)").grid(row=0, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.bt_hold_var, width=8).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(frm, text="单边费率(bps)").grid(row=0, column=4, sticky="e")
        ttk.Entry(frm, textvariable=self.bt_fee_bps_var, width=8).grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(frm, text="近20日均额≥").grid(row=0, column=6, sticky="e")
        ttk.Entry(frm, textvariable=self.bt_min_amt20_var, width=14).grid(row=0, column=7, sticky="w", padx=4)

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        self.btn_bt = ttk.Button(btns, text="开始回测（可无权重）", command=self.on_backtest_async)
        self.btn_bt.pack(side=tk.LEFT, padx=6)

        cols = ("date", "ret")
        frame, self.bt_tree = self._add_tree_with_scroll(parent, columns=cols, height=10,
                                                         widths={"date": 120, "ret": 120})
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        frm_m = ttk.Frame(parent, padding=6)
        frm_m.pack(fill=tk.X)
        self.bt_metrics_var = tk.StringVar(value="暂无结果")
        ttk.Label(frm_m, textvariable=self.bt_metrics_var, foreground="purple").pack(side=tk.LEFT)

    def _build_page_eval(self, parent):
        frm = ttk.LabelFrame(parent, text="筛选与操作", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="开始日期").grid(row=0, column=0, sticky="e")
        self.eval_start_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.eval_start_var, width=12).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="结束日期").grid(row=0, column=2, sticky="e")
        self.eval_end_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.eval_end_var, width=12).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(frm, text="类型").grid(row=0, column=4, sticky="e")
        self.eval_type_var = tk.StringVar(value="ALL")
        ttk.Combobox(frm, textvariable=self.eval_type_var, values=["ALL", "BUY", "SELL"], width=8,
                     state="readonly").grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(frm, text="H(日)").grid(row=0, column=6, sticky="e")
        self.eval_hz_var = tk.StringVar(value="")  # 可为空表示按当时的horizon
        ttk.Entry(frm, textvariable=self.eval_hz_var, width=6).grid(row=0, column=7, sticky="w", padx=4)

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="评估历史建议绩效（若自选表选中则仅该股）", command=self.on_evaluate).pack(side=tk.LEFT,
                                                                                                       padx=6)
        ttk.Button(btns, text="查看/导出建议历史（可按选中股）", command=self.on_view_history).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="分位统计（按score分组）", command=self.on_quant_stats).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="绘制收益分布图", command=self.on_plot_ret_hist).pack(side=tk.LEFT, padx=6)

        frame1, self.eval_tree = self._add_tree_with_scroll(parent, columns=("type", "n", "avg_ret", "win_rate"),
                                                            height=8,
                                                            widths={"type": 140, "n": 120, "avg_ret": 120,
                                                                    "win_rate": 120})
        frame1.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        frame2, self.quant_tree = self._add_tree_with_scroll(parent, columns=("quantile", "n", "avg_ret", "win_rate"),
                                                             height=6,
                                                             widths={"quantile": 120, "n": 120, "avg_ret": 120,
                                                                     "win_rate": 120})
        frame2.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def _build_page_plot(self, parent):
        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="绘图（若自选表有选中则绘该股，否则取输入框首个）", command=self.on_plot).pack(side=tk.LEFT,
                                                                                                          padx=6)

    def _build_page_ops(self, parent):
        frm = ttk.LabelFrame(parent, text="记录每日操作（买/卖/调仓）", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)
        self.op_code_var = tk.StringVar(value="")
        self.op_action_var = tk.StringVar(value="BUY")
        self.op_date_var = tk.StringVar(value=dt.datetime.now().strftime(DATE_FMT))
        self.op_price_var = tk.DoubleVar(value=0.0)
        self.op_qty_var = tk.IntVar(value=0)
        self.op_note_var = tk.StringVar(value="")
        ttk.Label(frm, text="代码").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.op_code_var, width=12).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="动作").grid(row=0, column=2, sticky="e")
        ttk.Combobox(frm, textvariable=self.op_action_var, values=["BUY", "SELL", "ADJ"], width=8,
                     state="readonly").grid(row=0, column=3, sticky="w")
        ttk.Label(frm, text="日期").grid(row=0, column=4, sticky="e")
        ttk.Entry(frm, textvariable=self.op_date_var, width=12).grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(frm, text="价格").grid(row=0, column=6, sticky="e")
        ttk.Entry(frm, textvariable=self.op_price_var, width=10).grid(row=0, column=7, sticky="w", padx=4)
        ttk.Label(frm, text="数量").grid(row=0, column=8, sticky="e")
        ttk.Entry(frm, textvariable=self.op_qty_var, width=10).grid(row=0, column=9, sticky="w", padx=4)
        ttk.Label(frm, text="备注").grid(row=0, column=10, sticky="e")
        ttk.Entry(frm, textvariable=self.op_note_var, width=24).grid(row=0, column=11, sticky="w", padx=4)
        ttk.Button(frm, text="保存操作", command=self.on_save_op).grid(row=0, column=12, padx=6)

        frame, self.op_tree = self._add_tree_with_scroll(parent,
                                                         columns=("date", "code", "name", "action", "price", "qty",
                                                                  "note"),
                                                         height=10,
                                                         widths={"date": 100, "code": 90, "name": 120, "action": 80,
                                                                 "price": 100, "qty": 100, "note": 480})
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        ttk.Button(parent, text="刷新操作记录", command=self.refresh_ops).pack(pady=4)

    def _build_page_help(self, parent):
        text = tk.Text(parent, wrap="word")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=vsb.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        vsb.pack(side=tk.RIGHT, fill=tk.Y, padx=4)
        help_msg = """  
使用说明（快速上手）：  
1. 股票池管理  
   - 在“数据/股票池”页输入股票代码（支持 600000 / sh.600000 / 600000.SH），逗号或换行分隔  
   - 可“导入股票列表文件”（.txt 每行一个；.csv 默认首列或名为 code 的列）  
   - 也可“从指数填充（HS300）”快速获取股票池；勾选“训练/扫描用自定义池”使用该集合  

2. 数据更新  
   - 填好“起始/结束日期”“复权方式”，点击“增量更新数据”  
   - 已有数据只从最新日期+1拉取增量，保存到本地SQLite（advisor.db）  
   - 若baostock未安装或未登录，也可离线使用本地数据  

3. 训练/进化  
   - 设置 horizon（H日）、训练模式（ridge_cv/ga）、目标指标（IC/MSE）  
   - ridge_cv：时间序列CV自动选λ；ga：进化搜索特征子集+λ（更强但更慢）  
   - 训练完成自动保存权重，并展示IC/ICIR/MSE等评估；自选建议/扫描/回测将自动使用最新权重  
   - 支持“导出/导入权重（JSON）”，便于迁移与交付  

4. 自选建议  
   - 在“自选建议”页粘贴/填充股票列表，设置资金与风险参数，点击“生成扩展建议并保存”  
   - 将写入历史 advice 表；建议包含仓位、止损、止盈、建议数量等；无权重时也能基于“基线打分”生成建议  

5. 市场扫描  
   - 指数支持 HS300/ZZ500/SZ50/ALL/CUSTOM；也可勾选“扫描用自定义池”  
   - 可勾选“跳过增量更新（快）”，在ALL大范围扫描时建议先离线用本地数据  
   - 高级筛选：趋势/接近55日高/波动Z阈值/RSI区间/流动性阈值  
   - 结果支持导出CSV；无权重时将使用“基线打分”得到排序结果  

6. 回测  
   - 设置 TopN、持有H日、费率、流动性阈值，点击“开始回测”  
   - 回测按每H日等权调仓TopN；输出期收益表、净值曲线与指标（年化、回撤、Sharpe、胜率）  
   - 若无权重则使用“基线打分”进行回测  
   - 可随时点击底部“停止任务”中止  

7. 绩效/历史  
   - 可按“日期范围、类型（ALL/BUY/SELL）、H(日)”筛选评估历史建议的表现  
   - 支持分位统计、收益分布图、导出历史  

8. 主题与配置  
   - 顶部菜单“视图”可切换浅色/深色主题  
   - 所有参数在退出时会自动保存，下次启动自动恢复  

常见问题：  
- baostock 未安装/登录失败：请先 pip install baostock，再重启程序  
- ALL 扫描很慢：建议勾选“跳过增量更新”，并适当调大“近20日均额阈值”  
- 代码格式：支持 600000 / sh.600000 / 600000.SH，自动归一化  
- 名称显示：自动从baostock缓存到SQLite；也可导入指数后缓存  
- 免责声明：仅用于研究，不构成投资建议  
"""
        text.insert(tk.END, help_msg)
        text.configure(state="disabled")

    def _build_page_log(self, parent):
        frm = ttk.Frame(parent, padding=6)
        frm.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(frm, height=16)
        vsb = ttk.Scrollbar(frm, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=vsb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------- 工具 ----------
    def parse_codes(self) -> List[str]:
        raw = self.codes_var.get().strip()
        codes = parse_codes_from_text(raw)
        return codes

    def log(self, msg: str, error: bool = False):
        prefix = "[ERROR] " if error else ""
        text = f"{dt.datetime.now().strftime('%H:%M:%S')} {prefix}{msg}\n"
        try:
            self.log_text.insert(tk.END, text)
            self.log_text.see(tk.END)
        except Exception:
            pass
        # 写入文件
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
        self.update_idletasks()

    def set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def beep(self):
        try:
            self.bell()
        except Exception:
            pass

    def on_stop_tasks(self):
        self.cancel_event.set()
        self.log("收到停止任务请求...")

    def _clear_cancel(self):
        self.cancel_event.clear()

    def _set_busy(self, busy: bool, doing: str = ""):
        try:
            if busy:
                self.pb.start(80)
                if doing:
                    self.set_status(doing + " ...")
            else:
                self.pb.stop()
                self.set_status("就绪")
        except Exception:
            pass
        for name in ["btn_update", "btn_train", "btn_adv", "btn_scan", "btn_export_scan", "btn_bt", "btn_export_w",
                     "btn_import_w"]:
            btn = getattr(self, name, None)
            if btn is None:
                continue
            try:
                btn.configure(state=tk.DISABLED if busy and name != "btn_export_scan" else tk.NORMAL)
            except Exception:
                pass
        self.update_idletasks()

    # ---------- 配置持久化 ----------
    def _load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # 恢复关键参数
            self.start_var.set(cfg.get("start_date", self.start_var.get()))
            self.end_var.set(cfg.get("end_date", self.end_var.get()))
            self.horizon_var.set(int(cfg.get("horizon", self.horizon_var.get())))
            self.adjustflag_var.set(cfg.get("adjustflag", self.adjustflag_var.get()))
            self.train_mode_var.set(cfg.get("train_mode", self.train_mode_var.get()))
            self.target_metric_var.set(cfg.get("target_metric", self.target_metric_var.get()))
            self.cv_folds_var.set(int(cfg.get("cv_folds", self.cv_folds_var.get())))
            self.ga_pop_var.set(int(cfg.get("ga_pop", self.ga_pop_var.get())))
            self.ga_gen_var.set(int(cfg.get("ga_gen", self.ga_gen_var.get())))
            self.index_flag_var.set(cfg.get("index_flag", self.index_flag_var.get()))
            self.min_amt20_var.set(float(cfg.get("min_amt20", self.min_amt20_var.get())))
            self.topN_var.set(int(cfg.get("topN", self.topN_var.get())))
            self.capital_var.set(float(cfg.get("capital", self.capital_var.get())))
            self.risk_pct_var.set(float(cfg.get("risk_pct", self.risk_pct_var.get())))
            self.skip_update_before_scan_var.set(bool(cfg.get("skip_update", self.skip_update_before_scan_var.get())))
            self.batch_size_var.set(int(cfg.get("batch_size", self.batch_size_var.get())))
            self.sleep_ms_var.set(int(cfg.get("sleep_ms", self.sleep_ms_var.get())))
            self.require_trend_var.set(bool(cfg.get("require_trend", self.require_trend_var.get())))
            self.require_breakout_var.set(bool(cfg.get("require_breakout", self.require_breakout_var.get())))
            self.max_vol_z_var.set(float(cfg.get("max_vol_z", self.max_vol_z_var.get())))
            self.filter_volz_enable_var.set(bool(cfg.get("filter_volz_enable", self.filter_volz_enable_var.get())))
            self.rsi_min_var.set(float(cfg.get("rsi_min", self.rsi_min_var.get())))
            self.rsi_max_var.set(float(cfg.get("rsi_max", self.rsi_max_var.get())))
            self.filter_rsi_enable_var.set(bool(cfg.get("filter_rsi_enable", self.filter_rsi_enable_var.get())))
            self.bt_topN_var.set(int(cfg.get("bt_topN", self.bt_topN_var.get())))
            self.bt_hold_var.set(int(cfg.get("bt_hold", self.bt_hold_var.get())))
            self.bt_fee_bps_var.set(float(cfg.get("bt_fee_bps", self.bt_fee_bps_var.get())))
            self.bt_min_amt20_var.set(float(cfg.get("bt_min_amt20", self.bt_min_amt20_var.get())))
            self.theme_var.set(cfg.get("theme", self.theme_var.get()))
            # 自定义池路径提示（不自动加载，避免路径变动报错）
            if cfg.get("custom_codes_path"):
                self.custom_codes_path = cfg.get("custom_codes_path")
                self.custom_count_var.set(f"路径已记录：{os.path.basename(self.custom_codes_path)}（需手动导入）")
        except Exception as e:
            self.log(f"读取配置失败：{e}", error=True)

    def _save_config(self):
        try:
            cfg = {
                "start_date": self.start_var.get().strip(),
                "end_date": self.end_var.get().strip(),
                "horizon": int(self.horizon_var.get()),
                "adjustflag": self.adjustflag_var.get().strip(),
                "train_mode": self.train_mode_var.get().strip(),
                "target_metric": self.target_metric_var.get().strip(),
                "cv_folds": int(self.cv_folds_var.get()),
                "ga_pop": int(self.ga_pop_var.get()),
                "ga_gen": int(self.ga_gen_var.get()),
                "index_flag": self.index_flag_var.get().strip(),
                "min_amt20": float(self.min_amt20_var.get()),
                "topN": int(self.topN_var.get()),
                "capital": float(self.capital_var.get()),
                "risk_pct": float(self.risk_pct_var.get()),
                "skip_update": bool(self.skip_update_before_scan_var.get()),
                "batch_size": int(self.batch_size_var.get()),
                "sleep_ms": int(self.sleep_ms_var.get()),
                "require_trend": bool(self.require_trend_var.get()),
                "require_breakout": bool(self.require_breakout_var.get()),
                "max_vol_z": float(self.max_vol_z_var.get()),
                "filter_volz_enable": bool(self.filter_volz_enable_var.get()),
                "rsi_min": float(self.rsi_min_var.get()),
                "rsi_max": float(self.rsi_max_var.get()),
                "filter_rsi_enable": bool(self.filter_rsi_enable_var.get()),
                "bt_topN": int(self.bt_topN_var.get()),
                "bt_hold": int(self.bt_hold_var.get()),
                "bt_fee_bps": float(self.bt_fee_bps_var.get()),
                "bt_min_amt20": float(self.bt_min_amt20_var.get()),
                "theme": self.theme_var.get(),
                "custom_codes_path": self.custom_codes_path or ""
            }
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"保存配置失败：{e}", error=True)

    # ---------- 文件/指数填充 ----------
    def on_import_codes(self):
        path = filedialog.askopenfilename(
            title="选择股票代码文件（.txt/.csv）",
            filetypes=[("文本/CSV", "*.txt *.csv"), ("文本", "*.txt"), ("CSV", "*.csv"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            codes: List[str] = []
            if path.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                codes = parse_codes_from_text(text)
            elif path.lower().endswith(".csv"):
                df = pd.read_csv(path, nrows=100000)
                col = None
                for c in df.columns:
                    if str(c).lower().strip() in ("code", "证券代码", "ts_code", "wind_code"):
                        col = c
                        break
                if col is None:
                    col = df.columns[0]
                vals = df[col].astype(str).tolist()
                codes = []
                for v in vals:
                    nc = normalize_code(str(v))
                    if nc:
                        codes.append(nc)
                codes = sorted(list(set(codes)))
            else:
                messagebox.showwarning("提示", "仅支持 .txt 或 .csv")
                return
            self.custom_codes = codes
            self.custom_codes_path = path
            self.custom_count_var.set(f"{len(codes)}个 | {os.path.basename(path)}")
            if codes:
                self.codes_var.set(", ".join(codes[:50]) + (" ..." if len(codes) > 50 else ""))
                self.log(f"已导入自定义股票池 {len(codes)} 个")
            else:
                self.log("导入文件未解析到有效股票代码", error=True)
        except Exception as e:
            self.log(f"导入失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def fill_codes_from_custom(self):
        if not self.custom_codes:
            messagebox.showinfo("提示", "尚未导入自定义股票池")
            return
        self.codes_var.set(", ".join(self.custom_codes[:200]) + (" ..." if len(self.custom_codes) > 200 else ""))

    def fill_codes_from_index_hs300(self):
        try:
            if bs is None:
                messagebox.showinfo("提示", "未安装baostock，无法从指数获取。")
                return
            codes = get_market_codes("HS300", self.end_var.get().strip())
            self.custom_codes = codes
            self.custom_count_var.set(f"{len(codes)}个 | HS300")
            self.codes_var.set(", ".join(codes[:200]) + (" ..." if len(codes) > 200 else ""))
            self.log(f"已从HS300填充股票池 {len(codes)} 个")
        except Exception as e:
            self.log(f"HS300填充失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    # ---------- 权重导出/导入 ----------
    def on_export_weights(self):
        try:
            w = load_latest_weights()
            if not w:
                messagebox.showinfo("提示", "尚无已保存的最新权重")
                return
            path = filedialog.asksaveasfilename(
                title="导出最新权重为JSON",
                defaultextension=".json",
                filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
            )
            if not path:
                return
            data = {
                "timestamp": w["timestamp"], "horizon": int(w["horizon"]),
                "features": w["features"], "weights": list(map(float, w["weights"].ravel())),
                "intercept": float(w["intercept"]), "mu": list(map(float, w["mu"].ravel())),
                "sigma": list(map(float, w["sigma"].ravel())), "lambda": float(w["lambda"]),
                "notes": w.get("notes", "")
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.log(f"权重已导出：{path}")
            messagebox.showinfo("完成", f"已导出：{path}")
        except Exception as e:
            self.log(f"导出权重失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def on_import_weights(self):
        try:
            path = filedialog.askopenfilename(
                title="导入权重（JSON）",
                filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
            )
            if not path:
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data["features"]
            weights = np.array(data["weights"], dtype=float)
            intercept = float(data["intercept"])
            mu = np.array(data["mu"], dtype=float)
            sigma = np.array(data["sigma"], dtype=float)
            lam = float(data.get("lambda", 1e-2))
            horizon = int(data.get("horizon", self.horizon_var.get()))
            notes = data.get("notes", "imported")
            # 保存至DB
            app_state["horizon"] = horizon
            ts = save_weights_record(horizon=horizon, features=features, weights=weights, intercept=intercept,
                                     mu=mu, sigma=sigma, lam=lam,
                                     notes=json.dumps({"method": "import", "eval": {}, "notes": notes},
                                                      ensure_ascii=False))
            self.log(f"权重已导入并保存，时间戳：{ts}")
            messagebox.showinfo("完成", f"已导入权重，时间戳：{ts}")
        except Exception as e:
            self.log(f"导入权重失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    # ---------- 事件（异步包装） ----------
    def on_update_data_async(self):
        t = threading.Thread(target=self._update_data_worker, daemon=True)
        t.start()

    def _update_data_worker(self):
        self._clear_cancel()
        self._set_busy(True, "更新数据")
        try:
            codes = self.custom_codes if self.use_custom_for_train_var.get() and self.custom_codes else self.parse_codes()
            if not codes:
                messagebox.showwarning("提示", "请输入或导入至少一个股票代码（如 sh.600000）")
                return
            start_date = self.start_var.get().strip()
            end_date = self.end_var.get().strip()
            adj = self.adjustflag_var.get().strip()
            self.log(f"开始增量更新，标的数={len(codes)}，区间={start_date}~{end_date}，复权={adj}")
            update_data_for_codes(codes, start_date, end_date, adj, logger=self.log,
                                  stop_cb=lambda: self.cancel_event.is_set())
            self.log("数据更新完成")
            self.beep()
        except Exception as e:
            self.log(f"更新失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    def on_train_async(self):
        t = threading.Thread(target=self._train_worker, daemon=True)
        t.start()

    def _train_worker(self):
        self._clear_cancel()
        self._set_busy(True, "训练模型")
        try:
            codes = self.custom_codes if self.use_custom_for_train_var.get() and self.custom_codes else self.parse_codes()
            if not codes:
                messagebox.showwarning("提示", "请输入/导入至少一个股票代码")
                return
            horizon = int(self.horizon_var.get())
            start_date = self.start_var.get().strip()
            end_date = self.end_var.get().strip()
            mode = self.train_mode_var.get()
            target_metric = self.target_metric_var.get()
            self.log(f"开始训练：模式={mode}, H={horizon}, 目标={target_metric}, 样本数~取决于股票池/区间")
            if mode == "ridge_cv":
                pack = train_and_save_weights(codes, horizon, mode=mode, target_metric=target_metric,
                                              start_date=start_date, end_date=end_date)
            else:
                ga_conf = {"n_pop": int(self.ga_pop_var.get()), "n_gen": int(self.ga_gen_var.get()),
                           "n_folds": int(self.cv_folds_var.get()), "min_feat": 6, "max_feat": 18}
                pack = train_and_save_weights(codes, horizon, mode="ga", ga_conf=ga_conf,
                                              start_date=start_date, end_date=end_date)
            if not pack:
                self.log("训练数据不足，权重未更新", error=True)
                messagebox.showwarning("提示", "训练数据不足（可能数据太少或特征为空）。")
                return
            self.log(f"训练完成，权重已保存，时间戳: {pack['timestamp']}")
            notes = pack.get("notes", {})
            if isinstance(notes, str):
                try:
                    notes = json.loads(notes)
                except Exception:
                    notes = {}
            eval_res = notes.get("eval", {})
            vals = (
                pack["timestamp"], int(pack["horizon"]),
                notes.get("method", ""),
                f"{pack['lambda']:.4g}",
                f"{safe_float(eval_res.get('ic_mean')):.4f}" if eval_res else "",
                f"{safe_float(eval_res.get('icir')):.4f}" if eval_res else "",
                f"{safe_float(eval_res.get('mse')):.6f}" if eval_res else "",
                int(eval_res.get("n_days", 0)) if eval_res else 0
            )
            self.train_tree.insert("", tk.END, values=vals)
            messagebox.showinfo("完成", f"训练完成，权重时间戳：{pack['timestamp']}")
            self.beep()
        except Exception as e:
            self.log(f"训练失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    def on_advise_async(self):
        t = threading.Thread(target=self._advise_worker, daemon=True)
        t.start()

    def _advise_worker(self):
        self._clear_cancel()
        self._set_busy(True, "生成建议")
        try:
            codes = self.parse_codes()
            if not codes:
                messagebox.showwarning("提示", "请输入股票代码（或在数据页导入/填充）")
                return
            w = load_latest_weights()  # 可空
            start_date = self.start_var.get().strip()
            capital = float(self.capital_var.get())
            risk_pct = float(self.risk_pct_var.get())

            for i in self.adv_tree.get_children():
                self.adv_tree.delete(i)

            adv_records = []
            feats = w["features"] if w else None
            ww = w  # 可能为 None

            today_str = dt.datetime.now().strftime(DATE_FMT)
            for code in codes:
                if self.cancel_event.is_set():
                    self.log("任务已被用户中止")
                    break
                try:
                    df = load_price_df(code, start_date=self.start_var.get().strip(),
                                       end_date=self.end_var.get().strip())
                    if df.empty:
                        self.log(f"{code} 无本地数据，请先在“数据/股票池”页执行增量更新", error=True)
                        continue
                    df = compute_indicators(df)
                    last = df.iloc[-1]
                    x_date = str(last["date"])
                    # 分数（优先模型，否则基线）
                    s = np.nan
                    if ww and feats and all(f in df.columns for f in feats):
                        x = last[feats].astype(float).values
                        xz = standardize_apply(x, ww["mu"], ww["sigma"])
                        s = float(np.dot(xz, ww["weights"]) + ww["intercept"])
                    if not np.isfinite(s):
                        s = baseline_score_no_model(last)

                    # 扩展建议
                    advice_text, reason, pos_pct, stop, target, qty = gen_extended_advice(
                        last, float(self.risk_pct_var.get()), float(self.capital_var.get())
                    )
                    name = get_stock_name(code)
                    if not name and bs is not None:
                        fetch_and_cache_name(code)
                        name = get_stock_name(code)
                    # 插入UI与历史
                    self.adv_tree.insert("", tk.END, values=(
                        x_date, code, name, f"{s:.4f}", advice_text, reason,
                        int(app_state.get("horizon", self.horizon_var.get()))
                    ))
                    adv_records.append((x_date, code, float(s), advice_text, reason,
                                        int(app_state.get("horizon", self.horizon_var.get()))))
                except Exception as e:
                    self.log(f"{code} 生成建议失败: {e}", error=True)

            if adv_records:
                insert_advice(adv_records)
                self.log(f"已保存 {len(adv_records)} 条建议记录")
                messagebox.showinfo("完成", f"已生成并保存 {len(adv_records)} 条建议")
                self.beep()
            else:
                messagebox.showwarning("提示", "未生成任何建议（可能无数据或解析失败）")
        except Exception as e:
            self.log(f"生成建议失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    # ---------- 市场扫描 ----------
    def on_scan_market_async(self):
        t = threading.Thread(target=self._scan_market_worker, daemon=True)
        t.start()

    def _scan_market_worker(self):
        self._clear_cancel()
        self._set_busy(True, "市场扫描")
        try:
            # 参数
            index_flag = self.index_flag_var.get().strip().upper()
            start_date = self.start_var.get().strip()
            end_date = self.end_var.get().strip()
            adjustflag = self.adjustflag_var.get().strip()
            min_amt20 = float(self.min_amt20_var.get())
            topN = int(self.topN_var.get())
            capital = float(self.capital_var.get())
            risk_pct = float(self.risk_pct_var.get())
            skip_update = bool(self.skip_update_before_scan_var.get())
            batch_size = int(self.batch_size_var.get())
            sleep_ms = int(self.sleep_ms_var.get())
            require_trend = bool(self.require_trend_var.get())
            require_breakout = bool(self.require_breakout_var.get())
            max_vol_z = float(self.max_vol_z_var.get()) if bool(self.filter_volz_enable_var.get()) else None
            rsi_min = float(self.rsi_min_var.get()) if bool(self.filter_rsi_enable_var.get()) else None
            rsi_max = float(self.rsi_max_var.get()) if bool(self.filter_rsi_enable_var.get()) else None

            # 权重（可为空）
            w = load_latest_weights()
            # 自定义池
            codes_override = None
            if index_flag == "CUSTOM" or self.use_custom_for_scan_var.get():
                codes_override = self.custom_codes if self.custom_codes else self.parse_codes()
                if not codes_override:
                    messagebox.showwarning("提示", "自定义池为空，请在“数据/股票池”页导入或填写")
                    return

            self.log(
                f"开始扫描：index={index_flag}{'（自定义池）' if codes_override else ''}, TopN={topN}, min_amt20={min_amt20:,.0f}, 跳过更新={skip_update}")
            df_res = scan_market_and_rank(
                index_flag=index_flag, start_date=start_date, end_date=end_date, adjustflag=adjustflag,
                weights_pack=w, min_amt20=min_amt20, topN=topN, logger=self.log,
                codes_override=codes_override, skip_update=skip_update, batch_size=batch_size,
                sleep_ms_between_batches=sleep_ms,
                require_trend=require_trend, require_breakout=require_breakout, max_vol_z=max_vol_z,
                rsi_min=rsi_min, rsi_max=rsi_max, stop_cb=lambda: self.cancel_event.is_set()
            )
            # 清空UI
            for i in self.scan_tree.get_children():
                self.scan_tree.delete(i)
            self.btn_export_scan.configure(state=tk.DISABLED)
            self.scan_df_cache = None

            if df_res is None or df_res.empty:
                self.log("扫描无结果")
                messagebox.showinfo("提示", "扫描无结果（可能过滤条件过严或无数据）")
                return

            # 扩展建议 + 名称
            rows_ui = []
            advice_records = []
            for _, row in df_res.iterrows():
                if self.cancel_event.is_set():
                    self.log("任务已被用户中止")
                    break
                code = row["code"]
                name = get_stock_name(code)
                if not name and bs is not None:
                    fetch_and_cache_name(code)
                    name = get_stock_name(code)
                # 构造一行最小行以生成建议
                pseudo_row = pd.Series({
                    "close": row.get("close", np.nan),
                    "atr14": row.get("atr14", np.nan),
                    "vol20": row.get("vol20", np.nan),
                    "mom10": row.get("mom10", np.nan)
                })
                advice_text, reason, pos_pct, stop, target, qty = gen_extended_advice(pseudo_row, risk_pct, capital)
                # UI行
                vals = (
                    str(row.get("date", "")), code, name,
                    f"{float(row.get('close', np.nan)):.2f}" if np.isfinite(row.get("close", np.nan)) else "",
                    f"{float(row.get('score', np.nan)):.4f}" if np.isfinite(row.get("score", np.nan)) else "",
                    f"{float(row.get('qscore', np.nan)):.4f}" if np.isfinite(row.get("qscore", np.nan)) else "",
                    f"{float(row.get('mom10', np.nan)):.2%}" if np.isfinite(row.get("mom10", np.nan)) else "",
                    f"{float(row.get('vol20', np.nan)):.2f}" if np.isfinite(row.get("vol20", np.nan)) else "",
                    f"{float(row.get('atr14', np.nan)):.2f}" if np.isfinite(row.get("atr14", np.nan)) else "",
                    f"{float(row.get('amt20', np.nan)) / 1e8:.2f}亿" if np.isfinite(
                        row.get("amt20", np.nan)) else "",
                    int(row.get("trend", 0)),
                    int(row.get("breakout55", 0)),
                    f"{float(row.get('dd60', np.nan)):.2%}" if np.isfinite(row.get("dd60", np.nan)) else "",
                    f"{pos_pct * 100:.1f}%",
                    f"{stop:.2f}",
                    f"{target:.2f}",
                    int(qty),
                    advice_text,
                    reason
                )
                self.scan_tree.insert("", tk.END, values=vals)
                rows_ui.append({
                    "date": row.get("date", ""),
                    "code": code,
                    "name": name,
                    "close": row.get("close", np.nan),
                    "score": row.get("score", np.nan),
                    "qscore": row.get("qscore", np.nan),
                    "mom10": row.get("mom10", np.nan),
                    "vol20": row.get("vol20", np.nan),
                    "atr14": row.get("atr14", np.nan),
                    "amt20": row.get("amt20", np.nan),
                    "trend": row.get("trend", np.nan),
                    "breakout55": row.get("breakout55", np.nan),
                    "dd60": row.get("dd60", np.nan),
                    "pos_pct": pos_pct,
                    "stop": stop,
                    "target": target,
                    "qty": qty,
                    "advice": advice_text,
                    "reasoning": reason
                })
                # 同时写 advice 历史（便于绩效评估）
                advice_records.append((
                    str(row.get("date", "")), code, float(row.get("score", 0.0)), advice_text, reason,
                    int(app_state.get("horizon", self.horizon_var.get()))
                ))

            if advice_records:
                insert_advice(advice_records)

            # 缓存导出
            self.scan_df_cache = pd.DataFrame(rows_ui)
            self.btn_export_scan.configure(state=tk.NORMAL)
            self.log(f"扫描完成，共 {len(rows_ui)} 条；建议记录已追加 {len(advice_records)} 条")
            self.beep()
        except Exception as e:
            self.log(f"扫描失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    def export_scan_csv(self):
        try:
            if self.scan_df_cache is None or self.scan_df_cache.empty:
                messagebox.showinfo("提示", "暂无扫描结果可导出")
                return
            path = filedialog.asksaveasfilename(
                title="导出扫描结果CSV",
                defaultextension=".csv",
                filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
            )
            if not path:
                return
            df = self.scan_df_cache.copy()
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self.log(f"扫描结果已导出：{path}")
            messagebox.showinfo("完成", f"已导出：{path}")
        except Exception as e:
            self.log(f"导出扫描结果失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    # ---------- 回测 ----------
    def on_backtest_async(self):
        t = threading.Thread(target=self._backtest_worker, daemon=True)
        t.start()

    def _backtest_worker(self):
        self._clear_cancel()
        self._set_busy(True, "回测")
        try:
            start_date = self.start_var.get().strip()
            end_date = self.end_var.get().strip()
            topN = int(self.bt_topN_var.get())
            hold_days = int(self.bt_hold_var.get())
            fee_bps = float(self.bt_fee_bps_var.get())
            min_amt20 = float(self.bt_min_amt20_var.get())

            # 股票池：优先自定义，否则取指数设置
            index_flag = self.index_flag_var.get().strip().upper()
            if self.use_custom_for_scan_var.get() or index_flag == "CUSTOM":
                codes = self.custom_codes if self.custom_codes else self.parse_codes()
            else:
                try:
                    codes = get_market_codes(index_flag, end_date)
                except Exception as e:
                    self.log(f"获取指数成分失败：{e}", error=True)
                    codes = self.parse_codes()
            if not codes:
                messagebox.showwarning("提示", "股票池为空，请导入或选择指数")
                return

            w = load_latest_weights()  # 可空
            self.log(
                f"开始回测：标的数={len(codes)}, TopN={topN}, H={hold_days}, 费率={fee_bps}bps, min_amt20={min_amt20:,.0f}")
            pr, nav = backtest_topN(
                weights_pack=w, codes=codes, start_date=start_date, end_date=end_date,
                topN=topN, hold_days=hold_days, min_amt20=min_amt20, fee_bps=fee_bps,
                logger=self.log, stop_cb=lambda: self.cancel_event.is_set()
            )
            # 清表
            for i in self.bt_tree.get_children():
                self.bt_tree.delete(i)

            if pr is None or pr.empty or nav is None or nav.empty:
                self.bt_metrics_var.set("回测结果为空")
                messagebox.showinfo("提示", "回测无有效期收益（样本/过滤/区间原因）")
                return

            for _, r in pr.iterrows():
                self.bt_tree.insert("", tk.END, values=(str(r["date"]), f"{float(r['ret']):.2%}"))

            metrics = compute_metrics_from_curve(nav, period_days=hold_days)
            msg = f"CAGR={metrics['CAGR']:.2%} | MaxDD={metrics['MaxDD']:.2%} | Sharpe={metrics['Sharpe']:.2f} | 胜率={metrics['WinRate']:.2%}"
            self.bt_metrics_var.set(msg)
            self.log("回测完成：" + msg)

            # 绘制净值
            try:
                plt.figure(figsize=(8.4, 4.2))
                nav.plot(title="回测净值曲线", grid=True)
                plt.xlabel("Date")
                plt.ylabel("NAV")
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

            self.beep()
        except Exception as e:
            self.log(f"回测失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    # ---------- 绩效/历史 ----------
    def _get_selected_code_from_adv(self) -> Optional[str]:
        try:
            sel = self.adv_tree.selection()
            if not sel:
                return None
            vals = self.adv_tree.item(sel[0], "values")
            code = vals[1] if len(vals) > 1 else None
            return code
        except Exception:
            return None

    def on_evaluate(self):
        try:
            start_date = self.eval_start_var.get().strip() or None
            end_date = self.eval_end_var.get().strip() or None
            typ = self.eval_type_var.get().strip().upper()
            typ = None if typ == "ALL" else typ
            hz_txt = self.eval_hz_var.get().strip()
            horizon = int(hz_txt) if hz_txt else None
            code = self._get_selected_code_from_adv()

            df = evaluate_history_performance(
                code=code, limit=1000, start_date=start_date, end_date=end_date, type_filter=typ, horizon=horizon
            )
            # 清表
            for i in self.eval_tree.get_children():
                self.eval_tree.delete(i)
            if df is None or df.empty:
                messagebox.showinfo("提示", "无可评估的历史建议")
                return
            for _, r in df.iterrows():
                self.eval_tree.insert("", tk.END, values=(r["type"], int(r["n"]), f"{float(r['avg_ret']):.2%}",
                                                          f"{float(r['win_rate']):.2%}"))
            self.log("历史建议绩效评估完成")
        except Exception as e:
            self.log(f"评估失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def on_view_history(self):
        try:
            start_date = self.eval_start_var.get().strip() or None
            end_date = self.eval_end_var.get().strip() or None
            typ = self.eval_type_var.get().strip().upper()
            typ = None if typ == "ALL" else typ
            code = self._get_selected_code_from_adv()

            rows = query_advice(code=code, limit=2000, start_date=start_date, end_date=end_date, type_filter=typ)
            if not rows:
                messagebox.showinfo("提示", "没有历史建议可导出")
                return
            df = pd.DataFrame(rows, columns=["date", "code", "score", "advice", "reasoning", "horizon"])
            # 名称
            df["name"] = df["code"].apply(get_stock_name)
            # 导出
            path = filedialog.asksaveasfilename(
                title="导出建议历史CSV",
                defaultextension=".csv",
                filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")]
            )
            if not path:
                return
            df.to_csv(path, index=False, encoding="utf-8-sig")
            self.log(f"已导出建议历史：{path}")
            messagebox.showinfo("完成", f"已导出：{path}")
        except Exception as e:
            self.log(f"导出历史失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def on_quant_stats(self):
        try:
            start_date = self.eval_start_var.get().strip() or None
            end_date = self.eval_end_var.get().strip() or None
            hz_txt = self.eval_hz_var.get().strip()
            horizon = int(hz_txt) if hz_txt else None

            df = evaluate_quantile_performance(code=None, limit=2000, start_date=start_date, end_date=end_date,
                                               horizon=horizon, q=5)
            for i in self.quant_tree.get_children():
                self.quant_tree.delete(i)
            if df is None or df.empty:
                messagebox.showinfo("提示", "无分位统计结果")
                return
            for _, r in df.iterrows():
                self.quant_tree.insert("", tk.END,
                                       values=(int(r["quantile"]), int(r["n"]), f"{float(r['avg_ret']):.2%}",
                                               f"{float(r['win_rate']):.2%}"))
            self.log("分位统计完成")
        except Exception as e:
            self.log(f"分位统计失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def on_plot_ret_hist(self):
        try:
            start_date = self.eval_start_var.get().strip() or None
            end_date = self.eval_end_var.get().strip() or None
            hz_txt = self.eval_hz_var.get().strip()
            horizon = int(hz_txt) if hz_txt else int(app_state.get("horizon", self.horizon_var.get()))

            rows = query_advice(code=None, limit=2000, start_date=start_date, end_date=end_date, type_filter=None)
            if not rows:
                messagebox.showinfo("提示", "无历史建议可绘图")
                return
            rets = []
            for date, code, score, advice, reasoning, hz in rows:
                hz_use = horizon if horizon is not None else int(hz)
                r = _find_future_return(code, date, hz_use)
                if r is not None:
                    rets.append(float(r))
            if not rets:
                messagebox.showinfo("提示", "无法计算任何收益（数据不足）")
                return
            plt.figure(figsize=(7, 4))
            plt.hist(rets, bins=40, alpha=0.8, color="#4e79a7")
            plt.title("建议后H日收益分布")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.log(f"绘图失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    # ---------- 绘图 ----------
    def on_plot(self):
        try:
            # 优先取“自选建议”表选中，否则取输入框首个
            code = self._get_selected_code_from_adv()
            if not code:
                codes = self.parse_codes()
                code = codes[0] if codes else None
            if not code:
                messagebox.showwarning("提示", "请先在“自选建议”页选中一只股票或在输入框填写代码")
                return

            df = load_price_df(code, start_date=self.start_var.get().strip(), end_date=self.end_var.get().strip())
            if df.empty:
                messagebox.showinfo("提示", f"{code} 无本地数据，请先更新")
                return
            df = compute_indicators(df).dropna(subset=["sma20", "sma60", "dif", "dea", "macd"], how="any")
            if df.empty:
                messagebox.showinfo("提示", "指标数据不足，无法绘图")
                return

            name = get_stock_name(code)
            dt_idx = pd.to_datetime(df["date"])
            close = df["close"]
            sma20 = df["sma20"]
            sma60 = df["sma60"]
            macd = df["macd"]
            dif = df["dif"]
            dea = df["dea"]
            bb_up = df.get("bb_up", pd.Series(index=df.index, data=np.nan))
            bb_low = df.get("bb_low", pd.Series(index=df.index, data=np.nan))

            fig, axs = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            axs[0].plot(dt_idx, close, label="Close", color="#4e79a7")
            axs[0].plot(dt_idx, sma20, label="SMA20", color="#e15759")
            axs[0].plot(dt_idx, sma60, label="SMA60", color="#59a14f")
            axs[0].plot(dt_idx, bb_up, label="BB_UP", color="#9c755f", alpha=0.6, linestyle="--")
            axs[0].plot(dt_idx, bb_low, label="BB_LOW", color="#9c755f", alpha=0.6, linestyle="--")
            axs[0].set_title(f"{code} {name}")
            axs[0].grid(True, alpha=0.3)
            axs[0].legend(loc="best")

            axs[1].bar(dt_idx, macd, label="MACD", color=np.where(macd >= 0, "#76b7b2", "#f28e2b"), alpha=0.8)
            axs[1].plot(dt_idx, dif, label="DIF", color="#e15759")
            axs[1].plot(dt_idx, dea, label="DEA", color="#59a14f")
            axs[1].grid(True, alpha=0.3)
            axs[1].legend(loc="best")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.log(f"绘图失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    # ---------- 用户操作 ----------
    def on_save_op(self):
        try:
            code_raw = self.op_code_var.get().strip()
            code = normalize_code(code_raw) if code_raw else None
            if not code:
                messagebox.showwarning("提示", "请输入有效的股票代码")
                return
            action = self.op_action_var.get().strip().upper()
            date = self.op_date_var.get().strip()
            price = float(self.op_price_var.get())
            qty = int(self.op_qty_var.get())
            note = self.op_note_var.get().strip()

            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                      INSERT INTO user_ops(date, code, action, price, qty, note)
                      VALUES (?, ?, ?, ?, ?, ?)
                      """, (date, code, action, price, qty, note))
            conn.commit()
            conn.close()
            self.log(f"已保存操作：{date} {code} {action} {price} x {qty}")
            self.refresh_ops()
        except Exception as e:
            self.log(f"保存操作失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def refresh_ops(self):
        try:
            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                      SELECT date, code, action, price, qty, note
                      FROM user_ops
                      ORDER BY date DESC, id DESC
                      LIMIT 300
                      """)
            rows = c.fetchall()
            conn.close()
            # 清表
            for i in self.op_tree.get_children():
                self.op_tree.delete(i)
            # 插入
            for r in rows:
                date, code, action, price, qty, note = r
                name = get_stock_name(code)
                self.op_tree.insert("", tk.END,
                                    values=(date, code, name, action, f"{float(price):.2f}", int(qty), note))
        except Exception as e:
            self.log(f"刷新操作记录失败: {e}", error=True)

    # ---------- 关闭 ----------
    def on_close(self):
        try:
            self._save_config()
        except Exception:
            pass
        try:
            bs_safe_logout()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass


# ============== 主入口 ==============
if __name__ == "__main__":
    try:
        init_db()
    except Exception as e:
        print(f"初始化数据库失败: {e}")
    app = EvoAdvisorApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            bs_safe_logout()
        except Exception:
            pass
