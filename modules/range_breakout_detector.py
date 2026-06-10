"""Stage 2 ブレイクアウト検出。

「長期レンジ → 全MA上抜け + RSI回復」という Stan Weinstein 式 Stage 2 入り
の初動を機械的に拾う。5214 日本電気硝子の 2025/7 がリファレンスケース。

検出条件:
  1. ベース期(過去 100 日のうち最新 20 日を除外)が「ほぼレンジ」
     - |slope| < RB_BASE_MAX_ABS_SLOPE
     - range_pct < RB_BASE_MAX_RANGE_PCT
  2. 3 本の MA(13週/26週/52週)が**束ねている**
     - (max - min) / mean < RB_MA_TIGHT_THRESHOLD
  3. 現値が **全ての MA を上抜け済**
     - close > max(MA13W, MA26W, MA52W)
     - かつ close <= max(MA) * (1 + RB_BREAKOUT_MAX_ABOVE_MA)  ← 走り始めの浅さ
  4. **RSI回復シグナル**
     - 直近 RB_RSI_LOW_LOOKBACK_DAYS 日のどこかで RSI < RB_RSI_LOW_THRESHOLD
     - 現 RSI が RB_RSI_NOW_MIN ≤ RSI ≤ RB_RSI_NOW_MAX
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    MA_PERIODS,
    RANGE_BREAKOUT_CSV,
    RB_BASE_EXCLUDE_RECENT_DAYS,
    RB_BASE_LOOKBACK_DAYS,
    RB_BASE_MAX_ABS_SLOPE,
    RB_BASE_MAX_RANGE_PCT,
    RB_BREAKOUT_MAX_ABOVE_MA,
    RB_MA_TIGHT_THRESHOLD,
    RB_RSI_LOW_LOOKBACK_DAYS,
    RB_RSI_LOW_THRESHOLD,
    RB_RSI_NOW_MAX,
    RB_RSI_NOW_MIN,
    RESULTS_CSV,
)
from modules.data_fetcher import load_cached
from modules.wave_classifier import _calc_rsi

logger = logging.getLogger(__name__)

_MA_SHORT = MA_PERIODS["MA13W"]  # 65 日
_MA_MID = MA_PERIODS["MA26W"]  # 130 日
_MA_LONG = MA_PERIODS["MA52W"]  # 260 日


def _slope_norm(close: np.ndarray) -> float:
    if len(close) < 5 or np.isnan(close).any():
        return float("nan")
    x = np.arange(len(close), dtype=float)
    a, _ = np.polyfit(x, close, 1)
    mean = float(np.nanmean(close))
    return float(a / mean) if mean > 0 else float("nan")


def _range_pct(close: np.ndarray) -> float:
    if len(close) < 5:
        return float("nan")
    hi = float(np.nanpercentile(close, 95))
    lo = float(np.nanpercentile(close, 5))
    mid = (hi + lo) / 2
    return (hi - lo) / mid * 100 if mid > 0 else float("nan")


def _evaluate(df: pd.DataFrame, asof_idx: int | None = None) -> dict[str, Any] | None:
    """1銘柄を評価し、シグナル成立なら明細 dict を返す。

    Args:
        df: OHLCV DataFrame (Close 列必須)
        asof_idx: 評価時点のインデックス。None なら最終行。バックテスト時に過去
                   タイミングを指定して検証可能。
    """
    if df is None or len(df) < _MA_LONG + RB_BASE_LOOKBACK_DAYS + 5:
        return None
    if asof_idx is None:
        asof_idx = len(df) - 1
    if asof_idx < _MA_LONG + RB_BASE_LOOKBACK_DAYS:
        return None

    closes = df["Close"].values.astype(float)

    # 1) ベース期 = asof から [-100, -20] 区間
    base_start = asof_idx - RB_BASE_LOOKBACK_DAYS
    base_end = asof_idx - RB_BASE_EXCLUDE_RECENT_DAYS
    if base_start < 0 or base_end - base_start < 30:
        return None
    base_segment = closes[base_start:base_end]
    base_slope = _slope_norm(base_segment)
    base_range = _range_pct(base_segment)
    if np.isnan(base_slope) or np.isnan(base_range):
        return None
    if abs(base_slope) > RB_BASE_MAX_ABS_SLOPE:
        return None
    if base_range > RB_BASE_MAX_RANGE_PCT:
        return None

    # 2) MA 束ね判定
    close_today = float(closes[asof_idx])
    ma_short = float(np.nanmean(closes[asof_idx - _MA_SHORT + 1 : asof_idx + 1]))
    ma_mid = float(np.nanmean(closes[asof_idx - _MA_MID + 1 : asof_idx + 1]))
    ma_long = float(np.nanmean(closes[asof_idx - _MA_LONG + 1 : asof_idx + 1]))
    mas = np.array([ma_short, ma_mid, ma_long])
    if np.isnan(mas).any():
        return None
    ma_mean = float(np.mean(mas))
    if ma_mean <= 0:
        return None
    ma_tightness = (float(np.max(mas)) - float(np.min(mas))) / ma_mean
    if ma_tightness > RB_MA_TIGHT_THRESHOLD:
        return None

    # 3) 全 MA 上抜け & 走り始めの浅さ
    ma_max = float(np.max(mas))
    if close_today <= ma_max:
        return None
    above_pct = (close_today - ma_max) / ma_max
    if above_pct > RB_BREAKOUT_MAX_ABOVE_MA:
        return None

    # 4) RSI 回復シグナル
    rsi_series = _calc_rsi(df["Close"].iloc[: asof_idx + 1])
    if rsi_series.empty:
        return None
    rsi_now = float(rsi_series.iloc[-1])
    if pd.isna(rsi_now):
        return None
    if not (RB_RSI_NOW_MIN <= rsi_now <= RB_RSI_NOW_MAX):
        return None
    rsi_recent = rsi_series.iloc[-RB_RSI_LOW_LOOKBACK_DAYS:]
    rsi_min = float(rsi_recent.min())
    if rsi_min >= RB_RSI_LOW_THRESHOLD:
        return None  # RSI 過去に低圏(<40) を経験していない

    # シグナル強度: 浅さ(0.0-1.0)、ベース硬さ(slope≒0が高い)、RSI位置(中庸が高い)を合成
    shallowness_score = max(0.0, 1.0 - above_pct / RB_BREAKOUT_MAX_ABOVE_MA)
    base_tight_score = 1.0 - min(abs(base_slope) / RB_BASE_MAX_ABS_SLOPE, 1.0)
    rsi_score = 1.0 - abs(rsi_now - 55) / 30  # 55 が最良
    rsi_score = max(0.0, min(1.0, rsi_score))
    signal_strength = round(
        (shallowness_score * 0.4 + base_tight_score * 0.3 + rsi_score * 0.3) * 100, 1
    )

    # MA がゴールデンオーダ(短>中>長) か = 確認シグナル
    golden_order = bool(ma_short > ma_mid > ma_long)

    return {
        "close": round(close_today, 2),
        "ma_short": round(ma_short, 2),
        "ma_mid": round(ma_mid, 2),
        "ma_long": round(ma_long, 2),
        "ma_tightness_pct": round(ma_tightness * 100, 2),
        "above_max_ma_pct": round(above_pct * 100, 2),
        "base_slope": round(base_slope, 5),
        "base_range_pct": round(base_range, 1),
        "rsi_now": round(rsi_now, 1),
        "rsi_min_recent": round(rsi_min, 1),
        "golden_order": golden_order,
        "signal_strength": signal_strength,
    }


def evaluate(code: str, ohlcv: pd.DataFrame) -> dict[str, Any] | None:
    """_evaluate の公開ラッパ。MoatScoreEngine から呼ぶ用。"""
    return _evaluate(ohlcv)


def detect_range_breakouts() -> pd.DataFrame:
    """全銘柄に対し Stage 2 ブレイクアウトシグナルを判定。"""
    results = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"code": str})
    rows: list[dict[str, Any]] = []
    for _, r in results.iterrows():
        ticker = str(r["ticker"])
        df = load_cached(ticker)
        if df is None:
            continue
        eval_result = _evaluate(df)
        if eval_result is None:
            continue
        rows.append(
            {
                "code": r["code"],
                "name": r["name"],
                "ticker": ticker,
                "sector_33": r.get("sector_33"),
                "size_category": r.get("size_category"),
                "per": r.get("per"),
                "pbr": r.get("pbr"),
                "market_cap": r.get("market_cap"),
                "wave_types": r.get("wave_types"),
                **eval_result,
            }
        )

    df_out = (
        pd.DataFrame(rows)
        .sort_values("signal_strength", ascending=False)
        .reset_index(drop=True)
    )
    return df_out


def evaluate_historical(ticker: str, asof_idx: int | None = None) -> dict | None:
    """単一銘柄をバックテスト用に評価(asof_idx を指定して過去時点で判定)。"""
    df = load_cached(ticker)
    return _evaluate(df, asof_idx=asof_idx) if df is not None else None


def update_range_breakout_csv() -> pd.DataFrame:
    df = detect_range_breakouts()
    RANGE_BREAKOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RANGE_BREAKOUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"Stage2 ブレイクアウト検出: {len(df)} 銘柄 → {RANGE_BREAKOUT_CSV}")
    return df


def load_range_breakout() -> pd.DataFrame:
    if not RANGE_BREAKOUT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RANGE_BREAKOUT_CSV, encoding="utf-8-sig", dtype={"code": str})
