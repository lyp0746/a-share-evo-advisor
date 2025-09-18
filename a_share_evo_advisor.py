# -*- coding: utf-8 -*-
"""
A股“进化式”投研助手（单文件 GUI）
- 数据源：baostock
- 增量分析：仅拉取新增交易日
- 自我进化：基于历史特征与事后收益拟合权重（Ridge）
- 市场扫描：支持 HS300/ZZ500/SZ50/全部A股 扫描优质股
- 扩展建议：结合预测、动量、波动与ATR给出仓位/止损/止盈建议＋下单数量
- 绩效评估：根据历史建议与实际收益计算命中率与收益
- 持久化：SQLite（价格、建议、权重）
- GUI：tkinter + matplotlib

依赖:
    pip install baostock pandas numpy matplotlib

注意:
    本程序仅用于教育与研究目的，不构成任何投资建议。
"""

import datetime as dt
import json
import math
import sqlite3
import time
from typing import List, Tuple, Dict, Any, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# 添加中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import baostock as bs
except ImportError:
    bs = None

DB_PATH = "advisor.db"
DATE_FMT = "%Y-%m-%d"


# ----------------------------- 数据库层 ----------------------------- #

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
                  features  TEXT, -- JSON list[str]  
                  weights   TEXT, -- JSON list[float]  
                  intercept REAL,
                  mu        TEXT, -- JSON list[float], feature means  
                  sigma     TEXT, -- JSON list[float], feature stds  
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
            code, r["date"], float(r.get("open", np.nan)), float(r.get("high", np.nan)),
            float(r.get("low", np.nan)), float(r.get("close", np.nan)),
            float(r.get("volume", np.nan)), float(r.get("amount", np.nan))
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
    # records: (date, code, score, advice, reasoning, horizon)
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


def query_advice(code: Optional[str] = None, limit: int = 200) -> List[Tuple]:
    conn = get_conn()
    c = conn.cursor()
    if code:
        c.execute("""
                  SELECT date, code, score, advice, reasoning, horizon
                  FROM advice
                  WHERE code = ?
                  ORDER BY date DESC
                  LIMIT ?
                  """, (code, limit))
    else:
        c.execute("""
                  SELECT date, code, score, advice, reasoning, horizon
                  FROM advice
                  ORDER BY date DESC
                  LIMIT ?
                  """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


def save_weights_record(horizon: int, features: List[str], weights: np.ndarray, intercept: float,
                        mu: np.ndarray, sigma: np.ndarray, lam: float, notes: str = ""):
    conn = get_conn()
    c = conn.cursor()
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""  
        INSERT OR REPLACE INTO weights (timestamp, horizon, features, weights, intercept, mu, sigma, lambda, notes)  
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)  
    """, (
        ts, horizon, json.dumps(features), json.dumps(list(map(float, weights))),
        float(intercept), json.dumps(list(map(float, mu))), json.dumps(list(map(float, sigma))),
        float(lam), notes
    ))
    # set latest active
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


# ----------------------------- baostock 接入 ----------------------------- #

def bs_safe_login():
    if bs is None:
        raise RuntimeError("未安装 baostock，请先执行：pip install baostock")
    try:
        lg = bs.login()
        if lg.error_code != '0':
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    except Exception as e:
        raise RuntimeError(f"baostock 登录异常: {e}")


def bs_safe_logout():
    if bs is None:
        return
    try:
        bs.logout()
    except Exception:
        pass


# ----------------------------- 数据获取 ----------------------------- #

def fetch_k_data_incremental(code: str, start_date: str, end_date: Optional[str] = None,
                             adjustflag: str = "2", retry: int = 3, sleep_sec: float = 0.4) -> pd.DataFrame:
    """
    从 baostock 拉取K线数据，自动增量（根据DB最新日期 + 1）
    adjustflag: 1-后复权, 2-前复权, 3-不复权
    """
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
                code, fields,
                start_date=real_start, end_date=end_date,
                frequency="d", adjustflag=adjustflag
            )
            if rs.error_code != '0':
                # 可能会话过期，尝试重登一次
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
        except Exception as e:
            if i == retry - 1:
                raise
            time.sleep(sleep_sec)
    return pd.DataFrame()


# ----------------------------- 特征工程与信号 ----------------------------- #

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    基本技术指标：SMA/EMA、MACD、RSI、波动率、动量、成交量均线、ATR
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # SMA
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma60"] = df["close"].rolling(60).mean()

    # EMA
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["dif"] = df["ema12"] - df["ema26"]
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["dif"] - df["dea"])

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 波动率（年化近20日）
    df["ret1"] = df["close"].pct_change()
    df["vol20"] = df["ret1"].rolling(20).std() * np.sqrt(252)

    # 动量
    df["mom10"] = df["close"] / df["close"].shift(10) - 1.0

    # 成交量均线与放量
    df["v_ma5"] = df["volume"].rolling(5).mean()
    df["v_surge"] = df["volume"] / (df["v_ma5"] + 1e-9)

    # 位置关系
    df["above_sma20"] = (df["close"] > df["sma20"]).astype(float)
    df["dif_pos"] = (df["dif"] > 0).astype(float)
    df["macd_up"] = (df["macd"] > 0).astype(float)

    # ATR(14)
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # 平均成交额（近20日）
    df["amt20"] = df["amount"].rolling(20).mean()

    return df


def feature_columns() -> List[str]:
    return [
        "sma5", "sma10", "sma20", "sma60",
        "ema12", "ema26",
        "dif", "dea", "macd",
        "rsi14", "vol20", "mom10",
        "v_surge", "above_sma20", "dif_pos", "macd_up"
    ]


def build_training_data(codes: List[str], horizon: int, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    构造训练集：X=特征，y=未来horizon日收益（close_{t+H}/close_t - 1）
    使用 MultiIndex(code, date) 严格对齐，避免索引错位。
    """
    feats = feature_columns()
    X_list = []
    y_list = []
    for code in codes:
        df = load_price_df(code, start_date, end_date)
        if df.empty or len(df) < max(60, horizon + 30):
            continue
        df = compute_indicators(df)
        df["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0
        # 仅使用未来有收益的行
        usable = df.iloc[:-horizon].copy() if len(df) > horizon else df.iloc[:0].copy()
        usable = usable.dropna(subset=feats + ["fwd_ret"])
        if usable.empty:
            continue
        # MultiIndex(code, date)
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

    # 目标收益剪裁以提升稳健性
    y = y.clip(lower=-0.2, upper=0.2)

    # 简单对部分特征剪裁（防极端值）
    X = X.copy()
    for c in ["rsi14", "v_surge", "vol20", "mom10", "macd", "dif", "dea"]:
        if c in X.columns and X[c].notna().any():
            lo = np.nanpercentile(X[c], 1)
            hi = np.nanpercentile(X[c], 99)
            X[c] = X[c].clip(lower=lo, upper=hi)

    # 统一对齐去空值
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y


def standardize_fit(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feats = list(X.columns)
    mu = np.array([X[c].mean() for c in feats], dtype=float)
    sigma_raw = np.array([X[c].std() for c in feats], dtype=float)
    sigma = np.where(sigma_raw > 1e-9, sigma_raw, 1.0)
    Xs = (X.values - mu) / sigma
    return Xs, mu, sigma, feats


def standardize_apply(x_row: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x_row - mu) / sigma


def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> Tuple[np.ndarray, float]:
    """
    带截距的Ridge回归：y ≈ X w + b
    使用扩展一列1，并对 w 正则化，b 不正则。
    同时增加严格形状检查，避免 matmul 维度不一致。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X 必须为二维数组，当前形状: {X.shape}")
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError(f"样本数不一致：X有 {n} 行，y有 {y.shape[0]} 行")
    # 扩展截距列
    X_ext = np.hstack([X, np.ones((n, 1))])
    I = np.eye(d + 1)
    I[-1, -1] = 0.0  # 不对截距正则
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


# ----------------------------- 训练与打分 ----------------------------- #

def train_and_save_weights(codes: List[str], horizon: int, lam: float, notes: str = "") -> Optional[Dict[str, Any]]:
    Xdf, y = build_training_data(codes, horizon)
    if Xdf.empty or y.empty:
        return None
    # 训练前再次严格检查
    if len(Xdf) != len(y):
        # 按索引对齐
        common_idx = Xdf.index.intersection(y.index)
        Xdf = Xdf.loc[common_idx]
        y = y.loc[common_idx]
    Xs, mu, sigma, feats = standardize_fit(Xdf)
    # 形状断言
    if Xs.shape[0] != y.shape[0]:
        raise ValueError(f"训练数据维度不一致：X行={Xs.shape[0]}, y行={y.shape[0]}")
    w, b = ridge_regression(Xs, y.values, lam=lam)
    ts = save_weights_record(horizon, feats, w, b, mu, sigma, lam, notes=notes)
    return {
        "timestamp": ts,
        "horizon": horizon,
        "features": feats,
        "weights": w,
        "intercept": b,
        "mu": mu,
        "sigma": sigma,
        "lambda": lam
    }


def score_latest_for_codes(codes: List[str],
                           weights_pack: Dict[str, Any],
                           start_date: Optional[str] = None) -> List[Tuple[str, str, float, str, str]]:
    """
    返回 [(date, code, score, advice, reasoning)]
    """
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
        if df.empty:
            continue
        last = df.iloc[-1]
        if any(f not in df.columns for f in feats):
            continue
        x = last[feats].astype(float).values
        xz = standardize_apply(x, mu, sigma)
        s = float(np.dot(xz, w) + b)
        adv = score_to_advice(s, pos_thr, neg_thr)
        reason = reasoning_from_signals(last)
        out.append((last["date"], code, s, adv, reason))
    return out


# ----------------------------- 市场扫描与优质股筛选 ----------------------------- #

def get_market_codes(index_flag: str, on_date: str) -> List[str]:
    """
    根据指数选择返回代码列表：
    - HS300 / ZZ500 / SZ50 / ALL
    """
    index_flag = (index_flag or "").upper()
    codes = []
    if index_flag == "HS300":
        rs = bs.query_hs300_stocks(date=on_date)
        while rs.error_code == '0' and rs.next():
            codes.append(rs.get_row_data()[1])  # code字段
    elif index_flag == "ZZ500":
        rs = bs.query_zz500_stocks(date=on_date)
        while rs.error_code == '0' and rs.next():
            codes.append(rs.get_row_data()[1])
    elif index_flag == "SZ50":
        rs = bs.query_sz50_stocks(date=on_date)
        while rs.error_code == '0' and rs.next():
            codes.append(rs.get_row_data()[1])
    else:
        # ALL：全部A股
        rs = bs.query_all_stock(day=on_date)
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[0]
            # 仅保留沪深A股
            if code.startswith("sh.") or code.startswith("sz."):
                codes.append(code)
    # 去重
    codes = sorted(list({c for c in codes if c}))
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
            time.sleep(0.08)
        except Exception as e:
            if logger:
                logger(f"{code} 更新失败: {e}")
            time.sleep(0.1)


def quality_score(last_row: pd.Series, pred_score: float, vol_z: float) -> float:
    """
    组合评分，用于市场扫描排序
    - 0.7 * 预测分（模型输出）
    - 0.2 * 动量
    - -0.1 * 波动z分
    - 轻微加分：均线多头（close > sma20 > sma60）
    """
    mom = float(last_row.get("mom10", 0.0))
    trend_bonus = 0.02 if (
                last_row.get("close", np.nan) > last_row.get("sma20", np.inf) > last_row.get("sma60", np.inf)) else 0.0
    return 0.7 * pred_score + 0.2 * mom - 0.1 * vol_z + trend_bonus


def scan_market_and_rank(index_flag: str,
                         start_date: str,
                         end_date: str,
                         adjustflag: str,
                         weights_pack: Dict[str, Any],
                         min_amt20: float = 2e8,
                         topN: int = 30,
                         logger=None) -> pd.DataFrame:
    """
    从市场扫描优质股：
    - 获取指数/全市场成分
    - 增量更新本地数据
    - 计算指标与预测分
    - 过滤流动性与数据完整度
    - 计算质量评分并排序
    返回 DataFrame: [code, date, close, score, mom10, vol20, atr14, amt20, qscore]
    """
    scan_date = end_date
    try:
        codes = get_market_codes(index_flag, scan_date)
    except Exception as e:
        if logger:
            logger(f"获取指数成分失败: {e}，改用当前输入的代码列表")
        codes = []

    if not codes:
        if logger:
            logger("指数成分为空，请检查日期或指数选择")
        return pd.DataFrame()

    # 增量更新
    update_data_for_codes(codes, start_date, end_date, adjustflag, logger=logger)

    feats = weights_pack["features"]
    w = weights_pack["weights"]
    b = weights_pack["intercept"]
    mu = weights_pack["mu"]
    sigma = weights_pack["sigma"]

    rows = []
    vol_vals = []
    for code in codes:
        df = load_price_df(code, start_date=start_date, end_date=end_date)
        if df.empty or len(df) < 60:
            continue
        df = compute_indicators(df)
        last = df.iloc[-1]
        # 流动性过滤
        amt20 = float(last.get("amt20") or 0.0)
        if np.isnan(amt20) or amt20 < float(min_amt20):
            continue
        if any(f not in df.columns for f in feats):
            continue
        x = last[feats].astype(float).values
        xz = standardize_apply(x, mu, sigma)
        s = float(np.dot(xz, w) + b)
        mom10 = float(last.get("mom10") or np.nan)
        vol20 = float(last.get("vol20") or np.nan)
        atr14 = float(last.get("atr14") or np.nan)
        rows.append({
            "code": code,
            "date": str(last["date"]),
            "close": float(last["close"]),
            "score": s,
            "mom10": mom10,
            "vol20": vol20,
            "atr14": atr14,
            "amt20": amt20
        })
        if not np.isnan(vol20):
            vol_vals.append(vol20)

    if not rows:
        return pd.DataFrame()

    df_res = pd.DataFrame(rows)
    # 计算波动 z 分
    if vol_vals:
        vol_mean = np.nanmean(vol_vals)
        vol_std = np.nanstd(vol_vals)
        vol_std = vol_std if vol_std > 1e-9 else 1.0
        df_res["vol_z"] = (df_res["vol20"] - vol_mean) / vol_std
    else:
        df_res["vol_z"] = 0.0

    # 质量评分
    df_res["qscore"] = df_res.apply(lambda r: quality_score(r, r["score"], r["vol_z"]), axis=1)

    df_res = df_res.sort_values(["qscore", "score", "mom10"], ascending=[False, False, False]).head(topN).reset_index(
        drop=True)
    return df_res


def gen_extended_advice(row: pd.Series, risk_pct: float, capital: float) -> Tuple[str, str, float, float, float, int]:
    """
    扩展建议：结合ATR 给出仓位/止损/止盈，并估算下单数量（100股一手）
    - 止损：close - 1.5 * ATR14（至少-5%）
    - 止盈：close + 2.5 * ATR14（或 1.5:1 盈亏比）
    - 仓位：单笔风险 risk_pct；pos_pct = min(30%, risk_pct / 跌幅%)
    - 数量：floor( pos_pct*capital / close / 100 ) * 100
    返回: (advice_text, reasoning_text, pos_pct, stop, target, qty)
    """
    close = float(row.get("close"))
    atr14 = float(row.get("atr14") or 0.0)
    if np.isnan(atr14) or atr14 <= 0:
        # 兜底用5%止损
        stop = close * 0.95
    else:
        stop = min(close * 0.95, close - 1.5 * atr14)
    # 目标位：2.5 * ATR，至少1.5倍盈亏比
    target = close + max(2.5 * atr14, (close - stop) * 1.5)

    # 仓位建议（上限30%）
    drop_pct = max(1e-4, (close - stop) / close)  # 风险占比
    pos_pct = min(0.3, risk_pct / drop_pct)

    # 建议类别：如果预测分>0且动量为正倾向买入
    score_val = float(row.get("score", 0.0))
    mom10 = float(row.get("mom10") or 0.0)
    adv_type = "买入" if score_val > 0 and mom10 >= -0.02 else "观望"

    pos_amount = pos_pct * capital
    qty = int(max(0, math.floor(pos_amount / close / 100.0) * 100))

    advice = f"建议：{adv_type}；建议仓位≈{pos_pct * 100:.1f}%；止损≈{stop:.2f}；止盈≈{target:.2f}；建议数量≈{qty}股"
    reason = f"扩展：预测分={score_val:.3f}；动量10日={mom10:.1%}；波动20日={float(row.get('vol20') or 0):.2f}；ATR14={atr14:.2f}"
    return advice, reason, pos_pct, stop, target, qty


# ----------------------------- 历史建议绩效评估 ----------------------------- #

def _find_future_return(code: str, date_str: str, horizon: int) -> Optional[float]:
    """给定 code/date/horizon，返回未来 horizon 日收盘收益，若数据不足返回 None"""
    df = load_price_df(code)
    if df.empty:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    # 定位当日或下一交易日
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
    """从 advice 文本中提取基础类别：强烈买入/买入/观望/卖出/强烈卖出"""
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


def evaluate_history_performance(code: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
    """
    评估历史建议绩效：
    - 读取最近 limit 条建议（可按 code 过滤）
    - 计算实际 horizon 日收益；对于卖出建议，统计“反向收益”（-ret）
    - 输出各类别的样本数、平均收益、胜率（>0 占比）
    """
    rows = query_advice(code=code, limit=limit)
    if not rows:
        return pd.DataFrame(columns=["type", "n", "avg_ret", "win_rate"])

    eval_rows = []
    for date, code_, score, advice, reasoning, hz in rows:
        ret = _find_future_return(code_, date, int(hz))
        if ret is None:
            continue
        typ = _parse_basic_advice(advice or "")
        signed_ret = ret
        if typ in ("卖出", "强烈卖出"):
            signed_ret = -ret  # 卖出建议希望后续下跌
        eval_rows.append({"type": typ, "ret": signed_ret})

    if not eval_rows:
        return pd.DataFrame(columns=["type", "n", "avg_ret", "win_rate"])

    df = pd.DataFrame(eval_rows)
    g = df.groupby("type")
    out = g["ret"].agg(n="count", avg_ret="mean", win_rate=lambda x: (x > 0).mean()).reset_index()
    out = out.sort_values(["avg_ret", "win_rate", "n"], ascending=[False, False, False])
    return out


# ----------------------------- GUI ----------------------------- #

class EvoAdvisorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A股进化式投研助手 - baostock")
        self.geometry("1280x820")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.codes_var = tk.StringVar(value="sh.600000,sz.000001,sz.300750")
        self.start_var = tk.StringVar(value="2015-01-01")
        self.end_var = tk.StringVar(value=dt.datetime.now().strftime(DATE_FMT))
        self.horizon_var = tk.IntVar(value=10)
        self.lambda_var = tk.DoubleVar(value=1e-2)
        self.adjustflag_var = tk.StringVar(value="2")  # 1:后复权,2:前复权,3:不复权
        self.train_on_update_var = tk.BooleanVar(value=True)

        # 市场扫描/资金与风险参数
        self.index_flag_var = tk.StringVar(value="HS300")  # HS300/ZZ500/SZ50/ALL
        self.min_amt20_var = tk.DoubleVar(value=2e8)  # 近20日均额门槛
        self.topN_var = tk.IntVar(value=30)
        self.capital_var = tk.DoubleVar(value=100000.0)  # 资金规模（元）
        self.risk_pct_var = tk.DoubleVar(value=0.01)  # 单笔风险（资金占比）

        self._build_ui()

        # baostock登录
        try:
            bs_safe_login()
            self.log("baostock 登录成功")
        except Exception as e:
            self.log(f"[错误] {e}", error=True)

    def _build_ui(self):
        # Top panel: inputs
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="股票代码(逗号分隔, 如 sh.600000,sz.000001)：").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.codes_var, width=70).grid(row=0, column=1, columnspan=6, sticky="we", padx=5)

        ttk.Label(top, text="起始日期").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.start_var, width=12).grid(row=1, column=1, sticky="w")
        ttk.Label(top, text="结束日期").grid(row=1, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.end_var, width=12).grid(row=1, column=3, sticky="w")
        ttk.Label(top, text="预测视野H(日)").grid(row=1, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.horizon_var, width=6).grid(row=1, column=5, sticky="w", padx=3)
        ttk.Label(top, text="岭回归λ").grid(row=1, column=6, sticky="e")
        ttk.Entry(top, textvariable=self.lambda_var, width=8).grid(row=1, column=7, sticky="w", padx=3)
        ttk.Label(top, text="复权").grid(row=1, column=8, sticky="e")
        ttk.Combobox(top, textvariable=self.adjustflag_var, values=["1", "2", "3"], width=4, state="readonly").grid(
            row=1, column=9, sticky="w", padx=3)
        ttk.Checkbutton(top, text="更新后自动训练", variable=self.train_on_update_var).grid(row=1, column=10,
                                                                                            sticky="w", padx=6)

        # Market scan panel
        scan = ttk.LabelFrame(self, text="市场扫描 / 风险参数")
        scan.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(scan, text="指数").grid(row=0, column=0, sticky="e")
        ttk.Combobox(scan, textvariable=self.index_flag_var, values=["HS300", "ZZ500", "SZ50", "ALL"], width=8,
                     state="readonly").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(scan, text="TopN").grid(row=0, column=2, sticky="e")
        ttk.Entry(scan, textvariable=self.topN_var, width=6).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(scan, text="近20日均额≥").grid(row=0, column=4, sticky="e")
        ttk.Entry(scan, textvariable=self.min_amt20_var, width=14).grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(scan, text="资金(元)").grid(row=0, column=6, sticky="e")
        ttk.Entry(scan, textvariable=self.capital_var, width=12).grid(row=0, column=7, sticky="w", padx=4)
        ttk.Label(scan, text="单笔风险%").grid(row=0, column=8, sticky="e")
        ttk.Entry(scan, textvariable=self.risk_pct_var, width=6).grid(row=0, column=9, sticky="w", padx=4)

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8)
        ttk.Button(btns, text="1) 增量更新数据", command=self.on_update_data).pack(side=tk.LEFT, padx=5, pady=6)
        ttk.Button(btns, text="2) 训练/进化权重", command=self.on_train).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="3) 自选股生成建议(扩展)", command=self.on_advise).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="市场扫描优质股", command=self.on_scan_market).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="评估历史建议", command=self.on_evaluate).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="查看建议历史", command=self.on_view_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="绘图(所选股票)", command=self.on_plot).pack(side=tk.LEFT, padx=5)

        # Advice Table
        table_frame = ttk.Frame(self)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        cols = ("date", "code", "score", "advice", "reasoning", "horizon")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            width = 120 if c in ("date", "code", "horizon") else (
                80 if c == "score" else 720 if c == "reasoning" else 140)
            self.tree.column(c, width=width, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Log
        log_frame = ttk.LabelFrame(self, text="日志")
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=8, pady=6)
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def parse_codes(self) -> List[str]:
        raw = self.codes_var.get().strip()
        codes = [c.strip() for c in raw.split(",") if c.strip()]
        return codes

    def log(self, msg: str, error: bool = False):
        prefix = "[ERROR] " if error else ""
        self.log_text.insert(tk.END, f"{dt.datetime.now().strftime('%H:%M:%S')} {prefix}{msg}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def on_update_data(self):
        codes = self.parse_codes()
        if not codes:
            messagebox.showwarning("提示", "请输入至少一个股票代码（如 sh.600000）")
            return
        start_date = self.start_var.get().strip()
        end_date = self.end_var.get().strip()
        adj = self.adjustflag_var.get().strip()
        try:
            update_data_for_codes(codes, start_date, end_date, adj, logger=self.log)
            self.log("数据更新完成")
            if self.train_on_update_var.get():
                self.on_train()
        except Exception as e:
            self.log(f"更新失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def on_train(self):
        codes = self.parse_codes()
        horizon = self.horizon_var.get()
        lam = self.lambda_var.get()
        try:
            self.log(f"开始训练/进化：H={horizon}, λ={lam}")
            pack = train_and_save_weights(codes, horizon, lam, notes="auto-train")
            if not pack:
                self.log("训练数据不足，权重未更新", error=True)
                messagebox.showwarning("提示", "训练数据不足（可能数据太少或特征为空）。")
                return
            self.log(f"训练完成，权重已保存，时间戳: {pack['timestamp']}")
            messagebox.showinfo("完成", f"训练完成，权重时间戳：{pack['timestamp']}")
        except Exception as e:
            self.log(f"训练失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def on_advise(self):
        """自选股生成“扩展建议”（含仓位/止损/止盈/数量）并持久化"""
        codes = self.parse_codes()
        w = load_latest_weights()
        if not w:
            messagebox.showwarning("提示", "尚未训练权重，请先训练。")
            return
        start_date = self.start_var.get().strip()
        capital = float(self.capital_var.get())
        risk_pct = float(self.risk_pct_var.get())
        feats = w["features"];
        ww = w["weights"];
        bb = w["intercept"];
        mu = w["mu"];
        sigma = w["sigma"]
        pos_thr = 0.02;
        neg_thr = -0.02

        try:
            # 清空表格
            for i in self.tree.get_children():
                self.tree.delete(i)

            adv_records = []
            for code in codes:
                df = load_price_df(code, start_date=start_date)
                if df.empty:
                    continue
                df = compute_indicators(df)
                if df.empty:
                    continue
                last = df.iloc[-1]
                if any(f not in df.columns for f in feats):
                    continue
                x = last[feats].astype(float).values
                xz = standardize_apply(x, mu, sigma)
                s = float(np.dot(xz, ww) + bb)
                base_adv = score_to_advice(s, pos_thr, neg_thr)
                base_reason = reasoning_from_signals(last)

                # 组装扩展决策输入
                row = pd.Series({
                    "close": float(last["close"]),
                    "mom10": float(last.get("mom10") or 0.0),
                    "vol20": float(last.get("vol20") or 0.0),
                    "atr14": float(last.get("atr14") or 0.0),
                    "score": s
                })
                ext_text, ext_reason, pos_pct, stop, target, qty = gen_extended_advice(row, risk_pct=risk_pct,
                                                                                       capital=capital)

                advice_text = ext_text  # 扩展建议文本
                reasoning = f"[{base_adv}] {base_reason} | {ext_reason}"

                self.tree.insert("", tk.END,
                                 values=(last["date"], code, f"{s:.4f}", advice_text, reasoning, w["horizon"]))
                adv_records.append((str(last["date"]), code, float(s), advice_text, reasoning, int(w["horizon"])))

            if adv_records:
                insert_advice(adv_records)
                self.log(f"生成建议 {len(adv_records)} 条，并已持久化")
            else:
                self.log("未生成任何建议（可能数据为空）")
        except Exception as e:
            self.log(f"生成建议失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def on_scan_market(self):
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

        try:
            self.log(f"开始市场扫描：{index_flag}, Top{topN}, 近20日均额≥{min_amt20:.0f}")
            df = scan_market_and_rank(index_flag, start_date, end_date, adj, w, min_amt20=min_amt20, topN=topN,
                                      logger=self.log)
            if df is None or df.empty:
                messagebox.showinfo("提示", "未找到符合条件的股票")
                return
            # 扩展建议与持久化
            records = []
            ext_rows = []
            for _, r in df.iterrows():
                ext_text, ext_reason, pos_pct, stop, target, qty = gen_extended_advice(r, risk_pct=risk_pct,
                                                                                       capital=capital)
                advice_text = ext_text
                reasoning = f"[质量分={r['qscore']:.3f}] {ext_reason}"
                records.append((r["date"], r["code"], float(r["score"]), advice_text, reasoning, int(w["horizon"])))
                ext_rows.append({
                    "date": r["date"],
                    "code": r["code"],
                    "close": r["close"],
                    "score": r["score"],
                    "qscore": r["qscore"],
                    "mom10": r["mom10"],
                    "vol20": r["vol20"],
                    "atr14": r["atr14"],
                    "amt20": r["amt20"],
                    "pos_pct": pos_pct,
                    "stop": stop,
                    "target": target,
                    "qty": qty,
                    "advice": advice_text,
                    "reasoning": reasoning
                })
            insert_advice(records)
            self.log(f"市场扫描完成，生成建议 {len(records)} 条（已持久化）")
            self._popup_scan_results(pd.DataFrame(ext_rows))
        except Exception as e:
            self.log(f"市场扫描失败: {e}", error=True)
            messagebox.showerror("错误", str(e))

    def _popup_scan_results(self, df: pd.DataFrame):
        win = tk.Toplevel(self)
        win.title("市场扫描结果")
        cols = ["date", "code", "close", "score", "qscore", "mom10", "vol20", "atr14", "amt20", "pos_pct", "stop",
                "target", "qty", "advice", "reasoning"]
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            width = {
                "date": 90, "code": 90, "close": 80, "score": 70, "qscore": 70,
                "mom10": 80, "vol20": 80, "atr14": 80, "amt20": 120,
                "pos_pct": 80, "stop": 90, "target": 90, "qty": 80,
                "advice": 240, "reasoning": 520
            }.get(c, 100)
            tree.column(c, width=width, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)

        for _, r in df.iterrows():
            tree.insert("", tk.END, values=(
                r["date"], r["code"], f"{r['close']:.2f}",
                f"{r['score']:.4f}", f"{r['qscore']:.4f}",
                f"{r['mom10']:.2%}", f"{r['vol20']:.3f}",
                f"{r['atr14']:.3f}", f"{r['amt20']:.0f}",
                f"{r['pos_pct'] * 100:.1f}%", f"{r['stop']:.2f}", f"{r['target']:.2f}", int(r["qty"]),
                r["advice"], r["reasoning"]
            ))

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, pady=6)

        def export_csv():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                                initialfile=f"scan_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            if not path:
                return
            try:
                df.to_csv(path, index=False, encoding="utf-8-sig")
                messagebox.showinfo("导出成功", f"已导出到：\n{path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：{e}")

        ttk.Button(btn_frame, text="导出CSV", command=export_csv).pack(side=tk.RIGHT, padx=6)

    def on_evaluate(self):
        """评估历史建议绩效（按基础类别聚合）"""
        # 若表格选中某只股票，则就它评估；否则评估全部
        code = None
        sel = self.tree.selection()
        if sel:
            vals = self.tree.item(sel[0], "values")
            if len(vals) >= 2:
                code = vals[1]
        df = evaluate_history_performance(code=code, limit=500)
        if df is None or df.empty:
            messagebox.showinfo("提示", "历史建议不足或无法计算收益")
            return
        win = tk.Toplevel(self)
        win.title("历史建议绩效" + (f" - {code}" if code else ""))
        cols = ["type", "n", "avg_ret", "win_rate"]
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120 if c == "type" else 100, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True)
        for _, r in df.iterrows():
            tree.insert("", tk.END,
                        values=(r["type"], int(r["n"]), f"{float(r['avg_ret']):.2%}", f"{float(r['win_rate']):.1%}"))

    def on_view_history(self):
        code = None
        sel = self.tree.selection()
        if sel:
            vals = self.tree.item(sel[0], "values")
            if len(vals) >= 2:
                code = vals[1]

        rows = query_advice(code=code, limit=500)
        if not rows:
            messagebox.showinfo("提示", "暂无历史建议")
            return
        # 弹窗展示
        win = tk.Toplevel(self)
        win.title("建议历史" + (f" - {code}" if code else ""))
        tree = ttk.Treeview(win, columns=("date", "code", "score", "advice", "reasoning", "horizon"), show="headings")
        for c in ("date", "code", "score", "advice", "reasoning", "horizon"):
            tree.heading(c, text=c)
            width = 120 if c in ("date", "code", "horizon") else (
                80 if c == "score" else 720 if c == "reasoning" else 140)
            tree.column(c, width=width, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        for r in rows:
            date, code_, score, adv, reason, hz = r
            tree.insert("", tk.END, values=(date, code_, f"{score:.4f}", adv, reason, hz))

    def on_plot(self):
        # 尝试从表格选择股票，否则用输入框第一个
        code = None
        sel = self.tree.selection()
        if sel:
            vals = self.tree.item(sel[0], "values")
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
        fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1]})
        # 价格与均线
        ax[0].plot(pd.to_datetime(df["date"]), df["close"], label="Close", color="black", linewidth=1)
        ax[0].plot(pd.to_datetime(df["date"]), df["sma20"], label="SMA20", color="blue", linewidth=1)
        ax[0].plot(pd.to_datetime(df["date"]), df["sma60"], label="SMA60", color="orange", linewidth=1)
        ax[0].set_title(f"{code} 收盘价与均线")
        ax[0].legend(loc="upper left")
        ax[0].grid(True, linestyle="--", alpha=0.3)

        # MACD/DIF/DEA
        macd_vals = df["macd"].fillna(0).values
        colors = ["red" if x > 0 else "green" for x in macd_vals]
        ax[1].bar(pd.to_datetime(df["date"]), macd_vals, label="MACD", color=colors)
        ax[1].plot(pd.to_datetime(df["date"]), df["dif"], label="DIF", color="purple")
        ax[1].plot(pd.to_datetime(df["date"]), df["dea"], label="DEA", color="brown")
        ax[1].legend(loc="upper left")
        ax[1].grid(True, linestyle="--", alpha=0.3)

        # ATR(14)
        ax[2].plot(pd.to_datetime(df["date"]), df["atr14"], label="ATR14", color="teal")
        ax[2].legend(loc="upper left")
        ax[2].grid(True, linestyle="--", alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def on_close(self):
        try:
            bs_safe_logout()
        except Exception:
            pass
        self.destroy()

def main():
    init_db()
    app = EvoAdvisorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
