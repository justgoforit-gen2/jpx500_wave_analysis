"""海外投資家フローと株価指数の相関分析モジュール。

modules/jpx_investor_flow_fetcher.py が生成する foreign_flow.parquet を入力に、
累積フロー時系列、業種指数との相関、ラグ別相関を算出する。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import JPX_INVESTOR_FLOW_PARQUET
from modules.data_fetcher import compute_sector_index, compute_size_index, load_cached

logger = logging.getLogger(__name__)

# 千円→億円換算係数（千円 × 1e-5 = 億円）
_OKU_FACTOR = 1e-5


def load_foreign_flow(market: str = "TSE Prime", unit: str = "oku") -> pd.DataFrame:
    """parquetから指定市場のフローを読み込む。

    Args:
        market: "TSE Prime" 等
        unit: "oku"=億円換算 / "sen"=千円のまま

    Returns:
        DataFrame: index=date, columns=[sales, purchase, net, total, foreigner_ratio_pct]
    """
    if not JPX_INVESTOR_FLOW_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(JPX_INVESTOR_FLOW_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["market"] == market].sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    out = pd.DataFrame(
        {
            "sales": df["sales_value"].to_numpy(dtype=float),
            "purchase": df["purchase_value"].to_numpy(dtype=float),
            "net": df["net_value"].to_numpy(dtype=float),
            "total": df["total_value"].to_numpy(dtype=float),
            "foreigner_ratio_pct": df["foreigner_ratio_pct"].to_numpy(dtype=float),
        },
        index=pd.to_datetime(df["date"]).to_numpy(),
    )
    out.index.name = "date"

    if unit == "oku":
        for col in ("sales", "purchase", "net", "total"):
            out[col] = out[col] * _OKU_FACTOR

    return out


def compute_cumulative_flow(flow_df: pd.DataFrame) -> pd.Series:
    """net列の累積和（週次）。"""
    if flow_df.empty:
        return pd.Series(dtype=float)
    return flow_df["net"].cumsum().rename("cumulative_net")


def _to_weekly_close(price_df: pd.DataFrame) -> pd.Series:
    """OHLCV DataFrameを週次(金曜)終値Seriesに変換。"""
    if price_df is None or price_df.empty:
        return pd.Series(dtype=float)
    if "Close" not in price_df.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index)
    return close.resample("W-FRI").last().dropna()


def compute_flow_index_correlation(
    flow_net: pd.Series,
    index_close: pd.Series,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """週次フローと指数の週次リターンの相関を、ラグを変えて算出。

    Args:
        flow_net: index=date, value=週次net買い越し額
        index_close: index=date, value=週次終値
        lags: 週ラグのリスト（0=同週、+N=N週遅れて指数が反応）

    Returns:
        DataFrame[lag, corr, n_weeks]
    """
    if lags is None:
        lags = [0, 1, 2, 4]
    if flow_net.empty or index_close.empty:
        return pd.DataFrame(columns=["lag", "corr", "n_weeks"])

    index_ret = index_close.pct_change().dropna()
    # 日付を週末(金)に揃える
    flow_aligned = flow_net.copy()
    flow_aligned.index = pd.to_datetime(flow_aligned.index)
    flow_aligned = flow_aligned.resample("W-FRI").last().dropna()

    rows = []
    for lag in lags:
        if lag >= 0:
            # 指数を lag 週遅らせる（フローが先行）
            shifted = index_ret.shift(-lag)
        else:
            shifted = index_ret.shift(-lag)
        joined = pd.concat(
            [flow_aligned.rename("flow"), shifted.rename("ret")], axis=1
        ).dropna()
        n = len(joined)
        if n < 4:
            rows.append({"lag": lag, "corr": float("nan"), "n_weeks": n})
            continue
        c = float(joined["flow"].corr(joined["ret"]))
        rows.append({"lag": lag, "corr": round(c, 4), "n_weeks": n})
    return pd.DataFrame(rows)


def compute_sector_flow_correlation(
    results_df: pd.DataFrame,
    flow_net: pd.Series,
    lag: int = 0,
) -> pd.DataFrame:
    """33業種ごとに、業種加重指数の週次リターンとフローnetの相関を算出。

    Returns:
        DataFrame[sector_33, corr, n_weeks]、corr降順
    """
    if flow_net.empty or "sector_33" not in results_df.columns:
        return pd.DataFrame(columns=["sector_33", "corr", "n_weeks"])

    flow_w = flow_net.copy()
    flow_w.index = pd.to_datetime(flow_w.index)
    flow_w = flow_w.resample("W-FRI").last().dropna()

    sectors = [s for s in results_df["sector_33"].dropna().unique() if s != ""]
    rows = []
    for sec in sorted(sectors):
        sec_index = compute_sector_index(sec, results_df)
        if sec_index is None or sec_index.empty:
            continue
        weekly = _to_weekly_close(sec_index)
        if weekly.empty:
            continue
        ret = weekly.pct_change().dropna()
        if lag != 0:
            ret = ret.shift(-lag)
        joined = pd.concat([flow_w.rename("flow"), ret.rename("ret")], axis=1).dropna()
        n = len(joined)
        if n < 4:
            continue
        c = float(joined["flow"].corr(joined["ret"]))
        if not np.isnan(c):
            rows.append({"sector_33": sec, "corr": round(c, 4), "n_weeks": n})

    df = pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True)
    return df


def compute_size_flow_correlation(
    results_df: pd.DataFrame,
    flow_net: pd.Series,
    lag: int = 0,
    size_labels: tuple[str, ...] = (
        "TOPIX Core30",
        "TOPIX Large70",
        "TOPIX Mid400",
    ),
) -> pd.DataFrame:
    """size_category 別(Core30/Large70/Mid400)で加重指数とフローの相関を算出。

    Returns:
        DataFrame[size_category, corr, n_weeks]、corr降順
    """
    if flow_net.empty or "size_category" not in results_df.columns:
        return pd.DataFrame(columns=["size_category", "corr", "n_weeks"])

    flow_w = flow_net.copy()
    flow_w.index = pd.to_datetime(flow_w.index)
    flow_w = flow_w.resample("W-FRI").last().dropna()

    rows = []
    for size in size_labels:
        size_index = compute_size_index(size, results_df)
        if size_index is None or size_index.empty:
            continue
        weekly = _to_weekly_close(size_index)
        if weekly.empty:
            continue
        ret = weekly.pct_change().dropna()
        if lag != 0:
            ret = ret.shift(-lag)
        joined = pd.concat([flow_w.rename("flow"), ret.rename("ret")], axis=1).dropna()
        n = len(joined)
        if n < 4:
            continue
        c = float(joined["flow"].corr(joined["ret"]))
        if not np.isnan(c):
            rows.append({"size_category": size, "corr": round(c, 4), "n_weeks": n})

    return (
        pd.DataFrame(rows).sort_values("corr", ascending=False).reset_index(drop=True)
    )


def compute_index_weekly_close(ticker: str) -> pd.Series:
    """ティッカーから週次終値Seriesを取得（既存parquetキャッシュ経由）。

    例: "^N225" (日経225), "1308.T" (TOPIX連動ETF)
    """
    df = load_cached(ticker)
    return _to_weekly_close(df)
