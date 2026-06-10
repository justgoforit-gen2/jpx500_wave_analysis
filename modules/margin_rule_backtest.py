"""信用倍率シグナルA/B/C/Dの統計的バックテスト。

ルール定義:
    A. 押し目買い   : 上昇トレンド × 信用倍率 < 2
    B. 急低下追従   : 信用倍率 前週比 < -30% (トレンド問わず)
    C. 過熱リスクオフ: 信用倍率 > 5 × 前週比 > +10%
    D. 落ちるナイフ : 下降トレンド × 信用倍率 < 2 (AVOIDシグナルの検証)

トレンド分類 (週次終値ベース):
    uptrend  : close > MA13W > MA26W
    downtrend: close < MA13W < MA26W
    sideways : 上記いずれにも該当しない
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.data_fetcher import load_cached
from modules.margin_fetcher import load_margin_history_combined


MA13_WEEKS = 13
MA26_WEEKS = 26
FORWARD_RETURN_WINDOWS = [1, 4, 12]  # 何週後リターンを評価するか


def _classify_trend(close_series: pd.Series) -> pd.Series:
    """週次終値の時系列からトレンドラベルを返す。"""
    ma13 = close_series.rolling(MA13_WEEKS, min_periods=MA13_WEEKS).mean()
    ma26 = close_series.rolling(MA26_WEEKS, min_periods=MA26_WEEKS).mean()
    up = (close_series > ma13) & (ma13 > ma26)
    down = (close_series < ma13) & (ma13 < ma26)
    trend = pd.Series("sideways", index=close_series.index)
    trend[up] = "uptrend"
    trend[down] = "downtrend"
    # MA未計算期間は NaN
    trend[ma26.isna()] = np.nan
    return trend


def prepare_signal_frame(ticker: str) -> pd.DataFrame | None:
    """1銘柄について、週次の信用倍率+トレンド+将来リターンを並べた表を返す。

    Returns:
        columns = observation_date, margin_ratio, mr_pct_change, close,
                  trend, ret_1w, ret_4w, ret_12w
        データ不足の場合 None。
    """
    margin = load_margin_history_combined(ticker)
    if margin is None or len(margin) < MA26_WEEKS + 1:
        return None

    margin = margin[["observation_date", "margin_ratio"]].copy()
    margin["observation_date"] = pd.to_datetime(margin["observation_date"])
    margin = margin.set_index("observation_date").sort_index()
    # 9999 (売残=0の上限) は除外
    margin = margin[margin["margin_ratio"] < 9999].copy()

    price = load_cached(ticker)
    if price is None or len(price) < MA26_WEEKS * 5:
        return None
    price.index = pd.to_datetime(price.index)
    weekly_close = price["Close"].resample("W-FRI").last()

    df = margin.join(weekly_close.rename("close"), how="left").dropna()
    if len(df) < MA26_WEEKS + max(FORWARD_RETURN_WINDOWS) + 2:
        return None

    df["mr_pct_change"] = df["margin_ratio"].pct_change() * 100
    df["trend"] = _classify_trend(df["close"])

    # 将来リターン (W週後の終値)
    for w in FORWARD_RETURN_WINDOWS:
        df[f"ret_{w}w"] = df["close"].pct_change(w).shift(-w) * 100

    df["ticker"] = ticker
    return df.reset_index().rename(columns={"index": "observation_date"})


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """各観測時点でA/B/C/Dルールの該当フラグを付与。"""
    out = df.copy()
    out["rule_A"] = (out["trend"] == "uptrend") & (out["margin_ratio"] < 2.0)
    out["rule_B"] = out["mr_pct_change"] < -30
    out["rule_C"] = (out["margin_ratio"] > 5.0) & (out["mr_pct_change"] > 10)
    out["rule_D"] = (out["trend"] == "downtrend") & (out["margin_ratio"] < 2.0)
    return out


def aggregate_rule_performance(
    signal_df: pd.DataFrame,
    rule: str,
    return_col: str = "ret_4w",
) -> dict:
    """指定ルールに該当する観測の将来リターン統計を返す。"""
    hits = signal_df[signal_df[rule] & signal_df[return_col].notna()]
    if len(hits) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "std": np.nan,
        }
    rets = hits[return_col]
    return {
        "n": int(len(hits)),
        "mean": float(rets.mean()),
        "median": float(rets.median()),
        "win_rate": float((rets > 0).mean() * 100),
        "std": float(rets.std()),
    }


def run_universe_backtest(
    tickers: list[str],
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全銘柄に対してルール統計を計算。

    Returns:
        per_stock: 各銘柄×ルール×返却窓 の集計
        all_signals: 全銘柄を結合したシグナル発生イベント (per-stock 検証用)
    """
    rows: list[dict] = []
    all_signals: list[pd.DataFrame] = []

    for tk in tickers:
        df = prepare_signal_frame(tk)
        if df is None:
            if verbose:
                print(f"  skip {tk}: insufficient data")
            continue
        sig = apply_rules(df)
        all_signals.append(sig)

        for rule in ["rule_A", "rule_B", "rule_C", "rule_D"]:
            for w in FORWARD_RETURN_WINDOWS:
                stats = aggregate_rule_performance(sig, rule, f"ret_{w}w")
                rows.append(
                    {
                        "ticker": tk,
                        "rule": rule,
                        "window_weeks": w,
                        **stats,
                    }
                )

    per_stock = pd.DataFrame(rows)
    all_sig_df = (
        pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    )
    return per_stock, all_sig_df


def aggregate_by_sector(
    all_signals: pd.DataFrame,
    sector_lookup: dict[str, str],
) -> pd.DataFrame:
    """シグナル発生イベントを業種別に集計。"""
    df = all_signals.copy()
    df["sector"] = df["ticker"].map(sector_lookup)
    df = df[df["sector"].notna()]

    rows: list[dict] = []
    for sector, sub in df.groupby("sector"):
        for rule in ["rule_A", "rule_B", "rule_C", "rule_D"]:
            hits = sub[sub[rule]]
            for w in FORWARD_RETURN_WINDOWS:
                col = f"ret_{w}w"
                rets = hits[col].dropna()
                rows.append(
                    {
                        "sector": sector,
                        "rule": rule,
                        "window_weeks": w,
                        "n_signals": int(len(rets)),
                        "n_tickers": int(hits["ticker"].nunique()),
                        "mean_return": float(rets.mean()) if len(rets) else np.nan,
                        "median_return": float(rets.median()) if len(rets) else np.nan,
                        "win_rate": float((rets > 0).mean() * 100)
                        if len(rets)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)
