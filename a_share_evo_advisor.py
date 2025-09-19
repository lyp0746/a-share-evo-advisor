
# -*- coding: utf-8 -*-
"""
A股“进化式”投研助手（重构升级版，单文件 GUI）
- 数据源：baostock
- 增量下载：仅拉取新增交易日
- 量化特征：SMA/EMA/MACD/RSI/ATR/BOLL/KDJ/CCI/WR/OBV/MFI/CMF/BBP/Range/Gap/动量等
- 模型训练：
    * 岭回归（时间序列CV自动选 λ，目标可选 IC 或 MSE）
    * 遗传算法（特征子集+正则λ，目标最大化验证Rank IC）
- 市场扫描：HS300/ZZ500/SZ50/ALL/CUSTOM（文件导入）
    * 优质股评分：预测Z分/动量Z分/波动Z分/趋势/突破/回撤综合；高级筛选与阈值可调
    * 可选跳过ALL全量前置更新、批量扫描降低接口压力
- 扩展建议：结合预测、动量、波动、ATR给出仓位/止损/止盈/数量
- 绩效评估优化：按日期/类型筛选、导出CSV、分位统计、分布图
- 持久化：SQLite（价格、建议、权重、元信息）
- GUI：tkinter + matplotlib + Notebook 分页 + 异步线程

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
from typing import List, Tuple, Dict, Any, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import baostock as bs
except ImportError:
    bs = None

DB_PATH = "advisor.db"
DATE_FMT = "%Y-%m-%d"

# =========================== 全局状态 ===========================
app_state: Dict[str, Any] = {"horizon": 10}


# =========================== 工具函数 ===========================
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
        WHERE 1=1
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
        ts, int(horizon), json.dumps(features), json.dumps(list(map(float, weights))),
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
              SELECT timestamp, horizon, features, weights, intercept, mu, sigma, lambda, notes
              FROM weights WHERE timestamp = ?
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
    df["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1/3, adjust=False).mean()
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
            Xtr = Xs[tr_idx]; ytr = y.values[tr_idx]
            Xva = Xs[va_idx]; yva = y.values[va_idx]
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
    if index_flag == "HS300":
        rs = bs.query_hs300_stocks(date=on_date)
    elif index_flag == "ZZ500":
        rs = bs.query_zz500_stocks(date=on_date)
    elif index_flag == "SZ50":
        rs = bs.query_sz50_stocks(date=on_date)
    else:
        rs = bs.query_all_stock(day=on_date)
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        code = row[1] if index_flag in ("HS300", "ZZ500", "SZ50") else row[0]
        if code and (code.startswith("sh.") or code.startswith("sz.")):
            codes.append(code)
    codes = sorted(list(set(codes)))
    return codes


def update_data_for_codes(codes: List[str], start_date: str, end_date: str, adjustflag: str, logger=None):
    for i, code in enumerate(codes):
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


def scan_market_and_rank(index_flag: str, start_date: str, end_date: str, adjustflag: str,
                         weights_pack: Dict[str, Any], min_amt20: float = 2e8, topN: int = 30,
                         logger=None,
                         codes_override: Optional[List[str]] = None,
                         skip_update: bool = False,
                         batch_size: int = 300,
                         sleep_ms_between_batches: int = 200,
                         require_trend: bool = False,
                         require_breakout: bool = False,
                         max_vol_z: Optional[float] = None,
                         rsi_min: Optional[float] = None,
                         rsi_max: Optional[float] = None) -> pd.DataFrame:
    """
    强化版扫描，支持：
    - codes_override: 使用自定义股票池
    - skip_update: 跳过全量增量更新（快）
    - batch_size + sleep: ALL 扫描时控制接口压力
    - 高级筛选：趋势/突破/最大波动z阈值/RSI区间
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

    if not skip_update:
        # 批量更新
        if len(codes) > batch_size and index_flag.upper() in ("ALL",):
            n = len(codes)
            for i in range(0, n, batch_size):
                part = codes[i:i + batch_size]
                if logger:
                    logger(f"批次更新 {i//batch_size + 1}/{math.ceil(n/batch_size)}，数量={len(part)}")
                update_data_for_codes(part, start_date, end_date, adjustflag, logger=logger)
                time.sleep(max(0, sleep_ms_between_batches) / 1000.0)
        else:
            update_data_for_codes(codes, start_date, end_date, adjustflag, logger=logger)
    else:
        if logger:
            logger("跳过行情增量更新（使用本地DB数据）")

    feats = weights_pack["features"]
    w = weights_pack["weights"]
    b = weights_pack["intercept"]
    mu = weights_pack["mu"]
    sigma = weights_pack["sigma"]

    rows = []
    for idx, code in enumerate(codes):
        try:
            df = load_price_df(code, start_date=start_date, end_date=end_date)
            if df.empty or len(df) < 80:
                continue
            df = compute_indicators(df)
            last = df.iloc[-1]
            amt20 = float(last.get("amt20") or 0.0)
            if np.isnan(amt20) or amt20 < float(min_amt20):
                continue
            if any(f not in df.columns for f in feats):
                continue

            x = last[feats].astype(float).values
            xz = standardize_apply(x, mu, sigma)
            s = float(np.dot(xz, w) + b)

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
    df_res = df_res.sort_values(["qscore", "score", "mom10"], ascending=[False, False, False]).head(topN).reset_index(drop=True)
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
    reason = f"扩展：预测分={score_val:.3f}；动量10日={mom10:.1%}；波动20日={float(row.get('vol20') or 0):.2f}；ATR14={atr14:.2f}"
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


# =========================== GUI 应用 ===========================
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


class EvoAdvisorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A股进化式投研助手 - Baostock（重构升级版）")
        self.geometry("1360x900")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # 变量
        self.codes_var = tk.StringVar(value="sh.600000,sz.000001,sz.300750")
        self.start_var = tk.StringVar(value="2015-01-01")
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

        self._build_ui()

        # baostock 登录
        try:
            bs_safe_login()
            self.log("baostock 登录成功")
        except Exception as e:
            self.log(f"[错误] {e}", error=True)

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

        # 页：绩效/历史
        self.page_eval = ttk.Frame(self.nb)
        self.nb.add(self.page_eval, text="绩效 / 历史")
        self._build_page_eval(self.page_eval)

        # 页：图表
        self.page_plot = ttk.Frame(self.nb)
        self.nb.add(self.page_plot, text="图表")
        self._build_page_plot(self.page_plot)

        # 页：操作说明
        self.page_help = ttk.Frame(self.nb)
        self.nb.add(self.page_help, text="操作说明")
        self._build_page_help(self.page_help)

        # 页：日志
        self.page_log = ttk.Frame(self.nb)
        self.nb.add(self.page_log, text="日志")
        self._build_page_log(self.page_log)

    def _build_page_data(self, parent):
        frm = ttk.LabelFrame(parent, text="股票池与数据更新", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="股票代码(逗号/换行分隔)").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.codes_var, width=80).grid(row=0, column=1, columnspan=4, sticky="we", padx=6)
        ttk.Button(frm, text="导入股票列表文件", command=self.on_import_codes).grid(row=0, column=5, padx=6, sticky="w")
        ttk.Label(frm, text="自定义池状态:").grid(row=0, column=6, sticky="e")
        ttk.Label(frm, textvariable=self.custom_count_var, foreground="blue").grid(row=0, column=7, sticky="w")

        ttk.Checkbutton(frm, text="训练使用自定义股票池", variable=self.use_custom_for_train_var).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(frm, text="扫描使用自定义股票池", variable=self.use_custom_for_scan_var).grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="起始日期").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.start_var, width=12).grid(row=2, column=1, sticky="w")
        ttk.Label(frm, text="结束日期").grid(row=2, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.end_var, width=12).grid(row=2, column=3, sticky="w")
        ttk.Label(frm, text="复权").grid(row=2, column=4, sticky="e")
        ttk.Combobox(frm, textvariable=self.adjustflag_var, values=["1", "2", "3"], width=6, state="readonly").grid(row=2, column=5, sticky="w")

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        self.btn_update = ttk.Button(btns, text="增量更新数据（按输入框/自定义池）", command=self.on_update_data_async)
        self.btn_update.pack(side=tk.LEFT, padx=6)
        ttk.Label(btns, text="说明：增量拉取baostock数据并写入SQLite").pack(side=tk.LEFT, padx=8)

    def _build_page_train(self, parent):
        frm = ttk.LabelFrame(parent, text="训练配置", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(frm, text="预测视野H(日)").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.horizon_var, width=8).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frm, text="训练模式").grid(row=0, column=2, sticky="e")
        ttk.Combobox(frm, textvariable=self.train_mode_var, values=["ridge_cv", "ga"], width=12, state="readonly").grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(frm, text="目标指标").grid(row=0, column=4, sticky="e")
        ttk.Combobox(frm, textvariable=self.target_metric_var, values=["IC", "MSE"], width=8, state="readonly").grid(row=0, column=5, sticky="w", padx=4)
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

        # 结果表
        self.train_tree = ttk.Treeview(parent, columns=("timestamp","horizon","method","lambda","ic","icir","mse","n_days"), show="headings", height=8)
        for c, w in [("timestamp",160),("horizon",70),("method",110),("lambda",90),("ic",80),("icir",80),("mse",120),("n_days",80)]:
            self.train_tree.heading(c, text=c)
            self.train_tree.column(c, width=w, anchor=tk.W)
        self.train_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

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

        cols = ("date", "code", "score", "advice", "reasoning", "horizon")
        self.adv_tree = ttk.Treeview(parent, columns=cols, show="headings", height=12)
        for c in cols:
            self.adv_tree.heading(c, text=c)
            width = 120 if c in ("date", "code", "horizon") else (80 if c == "score" else 840 if c == "reasoning" else 140)
            self.adv_tree.column(c, width=width, anchor=tk.W)
        self.adv_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def _build_page_scan(self, parent):
        frm = ttk.LabelFrame(parent, text="扫描参数", padding=8)
        frm.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(frm, text="指数/池").grid(row=0, column=0, sticky="e")
        ttk.Combobox(frm, textvariable=self.index_flag_var, values=["HS300", "ZZ500", "SZ50", "ALL", "CUSTOM"], width=10, state="readonly").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Checkbutton(frm, text="使用自定义池", variable=self.use_custom_for_scan_var).grid(row=0, column=2, sticky="w")
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
        ttk.Checkbutton(frm2, text="扫描前跳过增量更新（快）", variable=self.skip_update_before_scan_var).grid(row=0, column=0, sticky="w")
        ttk.Label(frm2, text="批大小").grid(row=0, column=1, sticky="e")
        ttk.Entry(frm2, textvariable=self.batch_size_var, width=8).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(frm2, text="批间隔(ms)").grid(row=0, column=3, sticky="e")
        ttk.Entry(frm2, textvariable=self.sleep_ms_var, width=8).grid(row=0, column=4, sticky="w", padx=4)

        ttk.Checkbutton(frm2, text="启用波动Z阈值", variable=self.filter_volz_enable_var).grid(row=1, column=0, sticky="w")
        ttk.Label(frm2, text="max vol_z").grid(row=1, column=1, sticky="e")
        ttk.Entry(frm2, textvariable=self.max_vol_z_var, width=8).grid(row=1, column=2, sticky="w", padx=4)
        ttk.Checkbutton(frm2, text="启用RSI区间", variable=self.filter_rsi_enable_var).grid(row=1, column=3, sticky="w")
        ttk.Label(frm2, text="RSI[min,max]").grid(row=1, column=4, sticky="e")
        ttk.Entry(frm2, textvariable=self.rsi_min_var, width=6).grid(row=1, column=5, sticky="w", padx=2)
        ttk.Entry(frm2, textvariable=self.rsi_max_var, width=6).grid(row=1, column=6, sticky="w", padx=2)
        ttk.Checkbutton(frm2, text="要求趋势(价>SMA20>SMA60, MACD>0, SMA20上行)", variable=self.require_trend_var).grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(frm2, text="要求接近55日高(≥98%)", variable=self.require_breakout_var).grid(row=2, column=4, columnspan=3, sticky="w")

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        self.btn_scan = ttk.Button(btns, text="开始市场扫描", command=self.on_scan_market_async)
        self.btn_scan.pack(side=tk.LEFT, padx=6)
        self.btn_export_scan = ttk.Button(btns, text="导出结果CSV", command=self.export_scan_csv, state=tk.DISABLED)
        self.btn_export_scan.pack(side=tk.LEFT, padx=6)

        cols = ["date", "code", "close", "score", "qscore", "mom10", "vol20", "atr14", "amt20",
                "trend", "breakout55", "dd60",
                "pos_pct", "stop", "target", "qty", "advice", "reasoning"]
        self.scan_tree = ttk.Treeview(parent, columns=cols, show="headings", height=16)
        widths = {
            "date": 90, "code": 90, "close": 80, "score": 70, "qscore": 70, "mom10": 80, "vol20": 80,
            "atr14": 80, "amt20": 120, "trend": 70, "breakout55": 90, "dd60": 80,
            "pos_pct": 80, "stop": 90, "target": 90, "qty": 80, "advice": 240, "reasoning": 520
        }
        for c in cols:
            self.scan_tree.heading(c, text=c)
            self.scan_tree.column(c, width=widths.get(c, 100), anchor=tk.W)
        self.scan_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.scan_df_cache: Optional[pd.DataFrame] = None

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
        ttk.Combobox(frm, textvariable=self.eval_type_var, values=["ALL", "BUY", "SELL"], width=8, state="readonly").grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(frm, text="H(日)").grid(row=0, column=6, sticky="e")
        self.eval_hz_var = tk.StringVar(value="")  # 可为空表示按当时的horizon
        ttk.Entry(frm, textvariable=self.eval_hz_var, width=6).grid(row=0, column=7, sticky="w", padx=4)

        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="评估历史建议绩效（若自选表选中则仅该股）", command=self.on_evaluate).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="查看/导出建议历史（可按选中股）", command=self.on_view_history).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="分位统计（按score分组）", command=self.on_quant_stats).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="绘制收益分布图", command=self.on_plot_ret_hist).pack(side=tk.LEFT, padx=6)

        self.eval_tree = ttk.Treeview(parent, columns=("type","n","avg_ret","win_rate"), show="headings", height=8)
        for c in ("type", "n", "avg_ret", "win_rate"):
            self.eval_tree.heading(c, text=c)
            self.eval_tree.column(c, width=140 if c == "type" else 120, anchor=tk.CENTER)
        self.eval_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.quant_tree = ttk.Treeview(parent, columns=("quantile","n","avg_ret","win_rate"), show="headings", height=6)
        for c in ("quantile", "n", "avg_ret", "win_rate"):
            self.quant_tree.heading(c, text=c)
            self.quant_tree.column(c, width=120, anchor=tk.CENTER)
        self.quant_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def _build_page_plot(self, parent):
        btns = ttk.Frame(parent, padding=8)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="绘图（若自选表有选中则绘该股，否则取输入框首个）", command=self.on_plot).pack(side=tk.LEFT, padx=6)

    def _build_page_help(self, parent):
        text = tk.Text(parent, wrap="word")
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        help_msg = """
使用说明（要点）：
1. 股票池管理
   - 输入框支持逗号/换行分隔，如：sh.600000, sz.000001
   - “导入股票列表文件”支持 .txt/.csv：
       * .txt：每行一个代码，支持 600000 或 sh.600000 或 600000.SH
       * .csv：默认读取首列，如存在名为 code 的列优先
   - “训练/扫描使用自定义股票池”可勾选使用刚导入的池

2. 数据更新
   - 填好“起始/结束日期”“复权方式”，点击“增量更新数据”
   - 仅会从本地已有最新日期+1开始增量拉取，写入SQLite

3. 训练/进化
   - 选择 horizon（H日）、训练模式（ridge_cv/ga）、目标指标（IC/MSE）
   - ridge_cv：时间序列CV选择λ；ga：进化搜索特征子集+λ（更慢但更强）
   - 完成后会保存最新权重，并在表格显示IC/ICIR/MSE等评估

4. 自选建议
   - 在“自选建议”页填写/填充股票列表（可从自定义池填充）
   - 设置资金与风险参数，点击“生成扩展建议并保存”
   - 结果会写入历史 advice 表中

5. 市场扫描（优质股挑选）
   - 指数支持：HS300/ZZ500/SZ50/ALL/CUSTOM
   - 可勾选“扫描前跳过增量更新（快）”，适合ALL扫描时使用本地数据
   - 高级筛选：趋势/接近55日高/波动Z阈值/RSI区间
   - 组合质量分 qscore = 预测Z分/动量Z分/波动Z分/趋势/突破/回撤综合
   - 扫描结果可导出CSV

6. 绩效/历史
   - 支持按日期/类型筛选评估，并可导出历史
   - 分位统计：按score分组查看收益与胜率
   - 收益分布图：快速查看分布形态

常见问题：
- baostock 未安装/登录失败：请 pip install baostock，重启程序
- ALL 扫描很慢：建议勾选“跳过增量更新”，并适当调大“近20日均额阈值”
- 代码格式：支持 600000 / sh.600000 / 600000.SH，自动归一化
- 免责声明：仅用于研究，不构成投资建议
"""
        text.insert(tk.END, help_msg)
        text.configure(state="disabled")

    def _build_page_log(self, parent):
        frm = ttk.Frame(parent, padding=6)
        frm.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(frm, height=16)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ---------- 工具 ----------
    def parse_codes(self) -> List[str]:
        raw = self.codes_var.get().strip()
        codes = parse_codes_from_text(raw)
        return codes

    def log(self, msg: str, error: bool = False):
        prefix = "[ERROR] " if error else ""
        self.log_text.insert(tk.END, f"{dt.datetime.now().strftime('%H:%M:%S')} {prefix}{msg}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def _set_busy(self, busy: bool):
        try:
            if busy:
                self.pb.start(80)
            else:
                self.pb.stop()
        except Exception:
            pass
        for btn in [getattr(self, n, None) for n in ["btn_update","btn_train","btn_adv","btn_scan","btn_export_scan"]]:
            if btn is None:
                continue
            try:
                btn.configure(state=tk.DISABLED if busy and n != "btn_export_scan" else tk.NORMAL)
            except Exception:
                pass

    # ---------- 文件导入 ----------
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
                    if c.lower().strip() in ("code", "证券代码", "ts_code", "wind_code"):
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
                # 顺便填入输入框，便于立即使用
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

    # ---------- 事件（异步包装） ----------
    def on_update_data_async(self):
        t = threading.Thread(target=self._update_data_worker, daemon=True)
        t.start()

    def _update_data_worker(self):
        self._set_busy(True)
        try:
            codes = self.custom_codes if self.use_custom_for_train_var.get() and self.custom_codes else self.parse_codes()
            if not codes:
                messagebox.showwarning("提示", "请输入或导入至少一个股票代码（如 sh.600000）")
                return
            start_date = self.start_var.get().strip()
            end_date = self.end_var.get().strip()
            adj = self.adjustflag_var.get().strip()
            self.log(f"开始增量更新，标的数={len(codes)}，区间={start_date}~{end_date}，复权={adj}")
            update_data_for_codes(codes, start_date, end_date, adj, logger=self.log)
            self.log("数据更新完成")
        except Exception as e:
            self.log(f"更新失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    def on_train_async(self):
        t = threading.Thread(target=self._train_worker, daemon=True)
        t.start()

    def _train_worker(self):
        self._set_busy(True)
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
                notes.get("method",""),
                f"{pack['lambda']:.4g}",
                f"{safe_float(eval_res.get('ic_mean')):.4f}" if eval_res else "",
                f"{safe_float(eval_res.get('icir')):.4f}" if eval_res else "",
                f"{safe_float(eval_res.get('mse')):.6f}" if eval_res else "",
                int(eval_res.get("n_days", 0)) if eval_res else 0
            )
            self.train_tree.insert("", tk.END, values=vals)
            messagebox.showinfo("完成", f"训练完成，权重时间戳：{pack['timestamp']}")
        except Exception as e:
            self.log(f"训练失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    def on_advise_async(self):
        t = threading.Thread(target=self._advise_worker, daemon=True)
        t.start()

    def _advise_worker(self):
        self._set_busy(True)
        try:
            codes = self.parse_codes()
            w = load_latest_weights()
            if not w:
                messagebox.showwarning("提示", "尚未训练权重，请先训练。")
                return
            start_date = self.start_var.get().strip()
            capital = float(self.capital_var.get())
            risk_pct = float(self.risk_pct_var.get())

            for i in self.adv_tree.get_children():
                self.adv_tree.delete(i)

            adv_records = []
            feats = w["features"]; ww = w["weights"]; bb = w["intercept"]; mu = w["mu"]; sigma = w["sigma"]
            pos_thr = 0.02; neg_thr = -0.02
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
                s = float(np.dot(xz, ww) + bb)
                base_adv = score_to_advice(s, pos_thr, neg_thr)
                base_reason = reasoning_from_signals(last)
                row = pd.Series({
                    "close": float(last["close"]),
                    "mom10": float(last.get("mom10") or 0.0),
                    "vol20": float(last.get("vol20") or 0.0),
                    "atr14": float(last.get("atr14") or 0.0),
                    "score": s
                })
                ext_text, ext_reason, pos_pct, stop, target, qty = gen_extended_advice(row, risk_pct=risk_pct, capital=capital)
                advice_text = ext_text
                reasoning = f"[{base_adv}] {base_reason} | {ext_reason}"
                self.adv_tree.insert("", tk.END, values=(last["date"], code, f"{s:.4f}", advice_text, reasoning, w["horizon"]))
                adv_records.append((str(last["date"]), code, float(s), advice_text, reasoning, int(w["horizon"])))
            if adv_records:
                insert_advice(adv_records)
                self.log(f"生成建议 {len(adv_records)} 条，并已持久化")
            else:
                self.log("未生成任何建议（可能数据为空或特征缺失）")
        except Exception as e:
            self.log(f"生成建议失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    def on_scan_market_async(self):
        t = threading.Thread(target=self._scan_worker, daemon=True)
        t.start()

    def _scan_worker(self):
        self._set_busy(True)
        try:
            w = load_latest_weights()
            if not w:
                messagebox.showwarning("提示", "尚未训练权重，请先用你的股票池训练一次。")
                return
            index_flag = self.index_flag_var.get().strip()
            start_date = self.start_var.get().strip()
            end_date = self.end_var.get().strip()
            adj = self.adjustflag_var.get().strip()
            min_amt20 = float(self.min_amt20_var.get())
            topN = int(self.topN_var.get())
            capital = float(self.capital_var.get())
            risk_pct = float(self.risk_pct_var.get())
            skip_upd = bool(self.skip_update_before_scan_var.get())
            batch_size = int(self.batch_size_var.get())
            sleep_ms = int(self.sleep_ms_var.get())

            # 高级筛选
            require_trend = bool(self.require_trend_var.get())
            require_breakout = bool(self.require_breakout_var.get())
            max_vol_z = float(self.max_vol_z_var.get()) if self.filter_volz_enable_var.get() else None
            rsi_min = float(self.rsi_min_var.get()) if self.filter_rsi_enable_var.get() else None
            rsi_max = float(self.rsi_max_var.get()) if self.filter_rsi_enable_var.get() else None

            # 自定义池
            codes_override = None
            if index_flag.upper() == "CUSTOM" or self.use_custom_for_scan_var.get():
                if not self.custom_codes:
                    messagebox.showwarning("提示", "未加载自定义股票池")
                    return
                codes_override = self.custom_codes

            self.log(f"开始市场扫描：index={index_flag}, Top{topN}, 均额≥{min_amt20:.0f}, skip_update={skip_upd}")
            df = scan_market_and_rank(
                index_flag, start_date, end_date, adj, w,
                min_amt20=min_amt20, topN=topN, logger=self.log,
                codes_override=codes_override, skip_update=skip_upd,
                batch_size=batch_size, sleep_ms_between_batches=sleep_ms,
                require_trend=require_trend, require_breakout=require_breakout,
                max_vol_z=max_vol_z, rsi_min=rsi_min, rsi_max=rsi_max
            )
            for i in self.scan_tree.get_children():
                self.scan_tree.delete(i)
            self.btn_export_scan.configure(state=tk.DISABLED)
            self.scan_df_cache = None

            if df is None or df.empty:
                messagebox.showinfo("提示", "未找到符合条件的股票")
                return

            records = []
            ext_rows = []
            for _, r in df.iterrows():
                ext_text, ext_reason, pos_pct, stop, target, qty = gen_extended_advice(r, risk_pct=risk_pct, capital=capital)
                advice_text = ext_text
                reasoning = f"[质量分={r['qscore']:.3f}] {ext_reason}"
                records.append((r["date"], r["code"], float(r["score"]), advice_text, reasoning, int(w["horizon"])))
                ext_rows.append({
                    "date": r["date"], "code": r["code"], "close": r["close"], "score": r["score"], "qscore": r["qscore"],
                    "mom10": r["mom10"], "vol20": r["vol20"], "atr14": r["atr14"], "amt20": r["amt20"],
                    "trend": r.get("trend", 0.0), "breakout55": r.get("breakout55", 0.0), "dd60": r.get("dd60", 0.0),
                    "pos_pct": pos_pct, "stop": stop, "target": target, "qty": qty, "advice": advice_text, "reasoning": reasoning
                })
            insert_advice(records)
            self.log(f"市场扫描完成，生成建议 {len(records)} 条（已持久化）")

            df_ext = pd.DataFrame(ext_rows)
            self.scan_df_cache = df_ext.copy()
            self.btn_export_scan.configure(state=tk.NORMAL)

            for _, r in df_ext.iterrows():
                self.scan_tree.insert("", tk.END, values=(
                    r["date"], r["code"], f"{r['close']:.2f}",
                    f"{r['score']:.4f}", f"{r['qscore']:.4f}",
                    f"{r['mom10']:.2%}", f"{r['vol20']:.3f}",
                    f"{r['atr14']:.3f}", f"{r['amt20']:.0f}",
                    f"{r.get('trend',0):.0f}", f"{r.get('breakout55',0):.0f}", f"{r.get('dd60',0):.2%}",
                    f"{r['pos_pct'] * 100:.1f}%", f"{r['stop']:.2f}", f"{r['target']:.2f}", int(r["qty"]),
                    r["advice"], r["reasoning"]
                ))
        except Exception as e:
            self.log(f"市场扫描失败: {e}", error=True)
            messagebox.showerror("错误", str(e))
        finally:
            self._set_busy(False)

    def export_scan_csv(self):
        if self.scan_df_cache is None or self.scan_df_cache.empty:
            messagebox.showinfo("提示", "暂无扫描结果")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile=f"scan_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not path:
            return
        try:
            self.scan_df_cache.to_csv(path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("导出成功", f"已导出到：\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{e}")

    def on_evaluate(self):
        code = None
        sel = self.adv_tree.selection()
        if sel:
            vals = self.adv_tree.item(sel[0], "values")
            if len(vals) >= 2:
                code = vals[1]
        start_date = self.eval_start_var.get().strip() or None
        end_date = self.eval_end_var.get().strip() or None
        type_choice = self.eval_type_var.get().strip().upper()
        type_filter = None if type_choice == "ALL" else type_choice
        hz = self.eval_hz_var.get().strip()
        horizon = int(hz) if hz else None

        df = evaluate_history_performance(code=code, limit=1000, start_date=start_date, end_date=end_date,
                                          type_filter=type_filter, horizon=horizon)
        for i in self.eval_tree.get_children():
            self.eval_tree.delete(i)
        if df is None or df.empty:
            messagebox.showinfo("提示", "历史建议不足或无法计算收益")
            return
        for _, r in df.iterrows():
            self.eval_tree.insert("", tk.END, values=(r["type"], int(r["n"]), f"{float(r['avg_ret']):.2%}", f"{float(r['win_rate']):.1%}"))

    def on_view_history(self):
        code = None
        sel = self.adv_tree.selection()
        if sel:
            vals = self.adv_tree.item(sel[0], "values")
            if len(vals) >= 2:
                code = vals[1]
        start_date = self.eval_start_var.get().strip() or None
        end_date = self.eval_end_var.get().strip() or None
        type_choice = self.eval_type_var.get().strip().upper()
        type_filter = None if type_choice == "ALL" else type_choice

        rows = query_advice(code=code, limit=2000, start_date=start_date, end_date=end_date, type_filter=type_filter)
        if not rows:
            messagebox.showinfo("提示", "暂无历史建议")
            return
        win = tk.Toplevel(self)
        win.title("建议历史" + (f" - {code}" if code else ""))
        tree = ttk.Treeview(win, columns=("date", "code", "score", "advice", "reasoning", "horizon"), show="headings")
        for c in ("date", "code", "score", "advice", "reasoning", "horizon"):
            tree.heading(c, text=c)
            width = 120 if c in ("date", "code", "horizon") else (80 if c == "score" else 720 if c == "reasoning" else 140)
            tree.column(c, width=width, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)

        for r in rows:
            date, code_, score, adv, reason, hz = r
            tree.insert("", tk.END, values=(date, code_, f"{score:.4f}", adv, reason, hz))

        def do_export():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                                initialfile=f"advice_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            if not path:
                return
            try:
                df = pd.DataFrame(rows, columns=["date", "code", "score", "advice", "reasoning", "horizon"])
                df.to_csv(path, index=False, encoding="utf-8-sig")
                messagebox.showinfo("导出成功", f"已导出到：\n{path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：{e}")

        ttk.Button(win, text="导出CSV", command=do_export).pack(pady=6)

    def on_quant_stats(self):
        code = None
        sel = self.adv_tree.selection()
        if sel:
            vals = self.adv_tree.item(sel[0], "values")
            if len(vals) >= 2:
                code = vals[1]
        start_date = self.eval_start_var.get().strip() or None
        end_date = self.eval_end_var.get().strip() or None
        hz = self.eval_hz_var.get().strip()
        horizon = int(hz) if hz else None
        df = evaluate_quantile_performance(code=code, limit=2000, start_date=start_date, end_date=end_date, horizon=horizon, q=5)
        for i in self.quant_tree.get_children():
            self.quant_tree.delete(i)
        if df is None or df.empty:
            messagebox.showinfo("提示", "数据不足，无法分位统计")
            return
        for _, r in df.iterrows():
            self.quant_tree.insert("", tk.END, values=(int(r["quantile"]), int(r["n"]), f"{float(r['avg_ret']):.2%}", f"{float(r['win_rate']):.1%}"))

    def on_plot_ret_hist(self):
        # 绘制历史建议的收益分布图（按筛选条件）
        code = None
        sel = self.adv_tree.selection()
        if sel:
            vals = self.adv_tree.item(sel[0], "values")
            if len(vals) >= 2:
                code = vals[1]
        start_date = self.eval_start_var.get().strip() or None
        end_date = self.eval_end_var.get().strip() or None
        type_choice = self.eval_type_var.get().strip().upper()
        type_filter = None if type_choice == "ALL" else type_choice
        hz = self.eval_hz_var.get().strip()
        horizon = int(hz) if hz else None

        rows = query_advice(code=code, limit=2000, start_date=start_date, end_date=end_date, type_filter=type_filter)
        if not rows:
            messagebox.showinfo("提示", "暂无历史建议")
            return
        rets = []
        for date, code_, score, advice, reasoning, hz0 in rows:
            hz_use = horizon if horizon is not None else int(hz0)
            r = _find_future_return(code_, date, hz_use)
            if r is None:
                continue
            typ = _parse_basic_advice(advice or "")
            signed = r if typ in ("买入", "强烈买入") else (-r if typ in ("卖出", "强烈卖出") else 0.0)
            rets.append(signed)
        if not rets:
            messagebox.showinfo("提示", "无法计算收益分布")
            return
        plt.figure(figsize=(8, 5))
        plt.hist(rets, bins=40, color="steelblue", alpha=0.85)
        plt.title("历史建议收益分布（签名后）")
        plt.xlabel("收益")
        plt.ylabel("频数")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()

    def on_plot(self):
        code = None
        sel = self.adv_tree.selection()
        if sel:
            vals = self.adv_tree.item(sel[0], "values")
            if len(vals) >= 2:
                code = vals[1]
        if not code:
            codes = self.parse_codes()
            if not codes:
                messagebox.showwarning("提示", "未指定股票代码")
                return
            code = codes[0]
        df = load_price_df(code)
        if df.empty or len(df) < 30:
            messagebox.showwarning("提示", f"{code} 数据不足无法绘图")
            return
        df = compute_indicators(df)
        self._plot_matplotlib(code, df)

    def _plot_matplotlib(self, code: str, df: pd.DataFrame):
        fig, ax = plt.subplots(5, 1, figsize=(11.5, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1, 0.8]})
        t = pd.to_datetime(df["date"])
        # 价格+均线+布林带
        ax[0].plot(t, df["close"], label="Close", color="black", linewidth=1)
        ax[0].plot(t, df["sma20"], label="SMA20", color="blue", linewidth=1)
        ax[0].plot(t, df["sma60"], label="SMA60", color="orange", linewidth=1)
        ax[0].plot(t, df["bb_up"], label="BBup", color="gray", alpha=0.5, linewidth=0.8)
        ax[0].plot(t, df["bb_low"], label="BBlow", color="gray", alpha=0.5, linewidth=0.8)
        ax[0].set_title(f"{code} 收盘价/均线/布林带")
        ax[0].legend(loc="upper left")
        ax[0].grid(True, linestyle="--", alpha=0.3)
        # MACD
        macd_vals = df["macd"].fillna(0).values
        colors = ["red" if x > 0 else "green" for x in macd_vals]
        ax[1].bar(t, macd_vals, label="MACD", color=colors)
        ax[1].plot(t, df["dif"], label="DIF", color="purple")
        ax[1].plot(t, df["dea"], label="DEA", color="brown")
        ax[1].legend(loc="upper left")
        ax[1].grid(True, linestyle="--", alpha=0.3)
        # KDJ
        ax[2].plot(t, df["kdj_k"], label="K", color="blue")
        ax[2].plot(t, df["kdj_d"], label="D", color="orange")
        ax[2].plot(t, df["kdj_j"], label="J", color="green")
        ax[2].legend(loc="upper left")
        ax[2].grid(True, linestyle="--", alpha=0.3)
        # ATR
        ax[3].plot(t, df["atr14"], label="ATR14", color="teal")
        ax[3].legend(loc="upper left")
        ax[3].grid(True, linestyle="--", alpha=0.3)
        # 趋势/突破/回撤（辅助线）
        ax[4].plot(t, df["near_high55"], label="近55日高比", color="darkorange")
        ax[4].plot(t, df["dd60"], label="60日回撤", color="crimson")
        ax[4].legend(loc="upper left")
        ax[4].grid(True, linestyle="--", alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def on_close(self):
        try:
            bs_safe_logout()
        except Exception:
            pass
        self.destroy()


# =========================== 入口 ===========================
def main():
    init_db()
    app = EvoAdvisorApp()
    app.mainloop()


if __name__ == "__main__":
    main()