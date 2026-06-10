"""下降→上昇トレンド転換検出。

直近 N 日とその前 N 日の2区間で線形回帰の傾き(slope)を計算し、
過去窓では下降していたが直近窓では上昇に転じている銘柄を抽出する。

既存の wave_classifier は 120 営業日の単一窓で slope を計算するため、
転換の途中(直近25日上昇でも全体ではまだマイナス)の銘柄を見逃す。
本モジュールはその「早期トランジション」を捕捉する。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import (
    RESULTS_CSV,
    TREND_TRANSITION_CSV,
    TT_MIN_REBOUND_PCT,
    TT_PAST_SLOPE_MAX,
    TT_RECENT_SLOPE_MIN,
    TT_WINDOW_DAYS,
)
from modules.data_fetcher import load_cached

logger = logging.getLogger(__name__)


def _slope_norm(close: np.ndarray) -> float:
    """価格平均で正規化した線形回帰の傾き(=日次変化率の近似)。

    > 0  = 上昇  / < 0 = 下降。正規化により銘柄間で比較可能。
    """
    if len(close) < 5 or np.isnan(close).any():
        return float("nan")
    x = np.arange(len(close), dtype=float)
    a, _ = np.polyfit(x, close, 1)
    mean = float(np.nanmean(close))
    return float(a / mean) if mean > 0 else float("nan")


def detect_transitions(
    window_days: int = TT_WINDOW_DAYS,
    past_slope_max: float = TT_PAST_SLOPE_MAX,
    recent_slope_min: float = TT_RECENT_SLOPE_MIN,
    min_rebound_pct: float = TT_MIN_REBOUND_PCT,
) -> pd.DataFrame:
    """全銘柄に対しトレンド転換シグナルを判定し DataFrame を返す。

    Args:
        window_days: 過去窓と直近窓の長さ(営業日)。デフォルト 25。
        past_slope_max: 過去窓 slope が**これ未満**で下降と判定。
        recent_slope_min: 直近窓 slope が**これ超**で上昇と判定。
        min_rebound_pct: 直近窓安値からの最低反発率(%)。
    """
    results = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"code": str})

    rows = []
    need_len = window_days * 2 + 5
    for _, r in results.iterrows():
        ticker = str(r["ticker"])
        df = load_cached(ticker)
        if df is None or len(df) < need_len:
            continue
        close = df["Close"].values.astype(float)
        if len(close) < need_len:
            continue

        past_segment = close[-window_days * 2 : -window_days]
        recent_segment = close[-window_days:]
        recent_low = float(np.min(close[-window_days - 5 :]))
        today = float(close[-1])

        s_past = _slope_norm(past_segment)
        s_recent = _slope_norm(recent_segment)
        if np.isnan(s_past) or np.isnan(s_recent):
            continue

        rebound_pct = (today / recent_low - 1) * 100 if recent_low > 0 else 0.0

        # 転換条件
        if not (
            s_past < past_slope_max
            and s_recent > recent_slope_min
            and rebound_pct >= min_rebound_pct
        ):
            continue

        # シグナル強度: 直近 slope - 過去 slope の差で評価(大きいほど明確な転換)
        signal_strength = round((s_recent - s_past) * 1000, 2)

        rows.append(
            {
                "code": r["code"],
                "name": r["name"],
                "ticker": ticker,
                "sector_33": r.get("sector_33"),
                "size_category": r.get("size_category"),
                "slope_past": round(s_past, 5),
                "slope_recent": round(s_recent, 5),
                "signal_strength": signal_strength,
                "rebound_pct": round(rebound_pct, 1),
                "today_close": round(today, 2),
                "recent_low": round(recent_low, 2),
                "per": r.get("per"),
                "pbr": r.get("pbr"),
                "wave_types": r.get("wave_types"),
                "market_cap": r.get("market_cap"),
            }
        )

    df_out = (
        pd.DataFrame(rows)
        .sort_values("signal_strength", ascending=False)
        .reset_index(drop=True)
    )
    return df_out


def update_trend_transition_csv() -> pd.DataFrame:
    """検出結果を CSV に保存(バッチから呼ぶ)。"""
    df = detect_transitions()
    TREND_TRANSITION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TREND_TRANSITION_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"トレンド転換検出: {len(df)} 銘柄 → {TREND_TRANSITION_CSV}")
    return df


def load_trend_transition() -> pd.DataFrame:
    """CSV キャッシュをロード。"""
    if not TREND_TRANSITION_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(TREND_TRANSITION_CSV, encoding="utf-8-sig", dtype={"code": str})
