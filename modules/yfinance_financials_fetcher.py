"""yfinance から B/S, CF, 配当指標を取得する補完フェッチャ。

naibu の財務テーブルは JPX500 で 20-44% 程度のフィールドしか埋まっていないため、
total_assets / total_equity / total_debt / operating_cf / net_income / 配当性向 を
yfinance の `balance_sheet`, `cashflow`, `income_stmt`, `info` から拾い直す。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# yfinance balance_sheet / cashflow の行名候補 (バージョンや銘柄で揺れる)
_BS_TOTAL_ASSETS = ("Total Assets",)
_BS_TOTAL_EQUITY = (
    "Stockholders Equity",
    "Common Stock Equity",
    "Total Equity Gross Minority Interest",
)
_BS_TOTAL_DEBT = ("Total Debt",)
_BS_CASH = (
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash And Short Term Investments",
)
_CF_OPERATING = (
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities",
    "Total Cash From Operating Activities",
)
_IS_NET_INCOME = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income Continuous Operations",
)


def _first_value(
    stmt: pd.DataFrame | None, candidates: tuple[str, ...]
) -> float | None:
    """statement DataFrame (rows=item, cols=date 降順) から最新列の最初に見つかった
    候補行の値を返す。見つからない/NaN なら None。"""
    if stmt is None or stmt.empty:
        return None
    latest = stmt.columns[0]
    for name in candidates:
        if name in stmt.index:
            v = stmt.at[name, latest]
            if pd.notna(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
    return None


def fetch_one_yf_financials(ticker: str) -> dict:
    """単一ティッカーの財務指標を yfinance から取得。

    Returns:
        dict (欠損は None):
          total_assets, total_equity, total_debt, cash_yf,
          operating_cf, net_income_yf,
          dividend_yield, payout_ratio, fiscal_year_yf
    """
    out: dict[str, float | int | None] = {
        "total_assets_yf": None,
        "total_equity_yf": None,
        "total_debt_yf": None,
        "cash_yf": None,
        "operating_cf_yf": None,
        "net_income_yf": None,
        "dividend_yield": None,
        "payout_ratio": None,
        "fiscal_year_yf": None,
    }
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet
        cf = t.cashflow
        inc = t.income_stmt

        out["total_assets_yf"] = _first_value(bs, _BS_TOTAL_ASSETS)
        out["total_equity_yf"] = _first_value(bs, _BS_TOTAL_EQUITY)
        out["total_debt_yf"] = _first_value(bs, _BS_TOTAL_DEBT)
        out["cash_yf"] = _first_value(bs, _BS_CASH)
        out["operating_cf_yf"] = _first_value(cf, _CF_OPERATING)
        out["net_income_yf"] = _first_value(inc, _IS_NET_INCOME)
        if bs is not None and not bs.empty:
            out["fiscal_year_yf"] = int(pd.Timestamp(bs.columns[0]).year)

        info = t.info or {}
        # dividendYield は 0-1 (例: 0.025=2.5%) または既に%表示の場合あり
        dy = info.get("dividendYield")
        if dy is not None:
            # heuristic: > 1.0 なら % 表示済みとみなす
            out["dividend_yield"] = float(dy) if dy > 1.0 else float(dy) * 100
        pr = info.get("payoutRatio")
        if pr is not None:
            out["payout_ratio"] = float(pr)
    except Exception as e:
        logger.warning(f"yfinance財務取得失敗 {ticker}: {e}")
    return out


def fetch_all_yf_financials(
    tickers: list[str],
    sleep_sec: float = 0.0,
    progress_callback=None,
) -> pd.DataFrame:
    """複数ティッカーの財務を順次取得。

    Args:
        tickers: ["7203.T", "6758.T", ...] yfinance形式
        sleep_sec: 各リクエスト後の待機 (rate limit対策)
        progress_callback: (i, total, ticker) を受ける任意のコールバック

    Returns:
        DataFrame[ticker, total_assets_yf, ..., dividend_yield, payout_ratio]
    """
    rows = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        d = fetch_one_yf_financials(ticker)
        d["ticker"] = ticker
        rows.append(d)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return pd.DataFrame(rows)


def cache_yf_financials(df: pd.DataFrame, cache_path: Path | str) -> None:
    """parquet にキャッシュ保存 (差分更新用)。"""
    df.to_parquet(cache_path, index=False)


def load_yf_financials_cache(cache_path: Path | str) -> pd.DataFrame | None:
    """parquet キャッシュを読込。存在しなければ None。"""
    p = Path(cache_path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as e:
        logger.warning(f"yf財務キャッシュ読込失敗: {e}")
        return None
