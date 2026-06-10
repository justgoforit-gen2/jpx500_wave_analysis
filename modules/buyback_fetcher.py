"""yfinance のキャッシュフロー計算書から過去2年の自己株買戻額を集計する。

`Repurchase Of Capital Stock` (または `Common Stock Repurchased`) 行は
キャッシュアウト = 負値。最新2会計年度を合計し、絶対値で「過去2年自己株買戻額」とする。

Output:
    data/buybacks_2y.csv : code, ticker, name, market, size_category, sector_33,
                            buyback_2y_jpy, buyback_fy1, buyback_fy2,
                            fy1_year, fy2_year, market_cap, buyback_to_mcap
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR
from modules.data_fetcher import load_stock_list

logger = logging.getLogger(__name__)

BUYBACKS_CSV = DATA_DIR / "buybacks_2y.csv"

# yfinance cashflow 行候補 (バージョン/銘柄で揺れる)
_CF_BUYBACK_KEYS = (
    "Repurchase Of Capital Stock",
    "Common Stock Repurchased",
    "Repurchase Of Common Stock",
)


def _extract_buybacks(
    cf: pd.DataFrame | None,
) -> tuple[float | None, float | None, int | None, int | None]:
    """cashflow から最新2会計年度の自己株買戻額を抽出。

    Returns: (fy1_amount, fy2_amount, fy1_year, fy2_year)
    fy1 = 最新年度, fy2 = 1年前。買戻額は絶対値 (円)。なければ None。
    """
    if cf is None or cf.empty:
        return None, None, None, None
    row = None
    for key in _CF_BUYBACK_KEYS:
        if key in cf.index:
            row = cf.loc[key]
            break
    if row is None:
        return None, None, None, None
    # 列は降順 (新→古)
    cols = list(cf.columns)
    fy1_val = fy2_val = None
    fy1_yr = fy2_yr = None
    if len(cols) >= 1 and pd.notna(row.iloc[0]):
        v = float(row.iloc[0])
        fy1_val = abs(v) if v < 0 else 0.0  # 負値が買戻、正値は (まれだが) 売出
        fy1_yr = int(pd.Timestamp(cols[0]).year)
    if len(cols) >= 2 and pd.notna(row.iloc[1]):
        v = float(row.iloc[1])
        fy2_val = abs(v) if v < 0 else 0.0
        fy2_yr = int(pd.Timestamp(cols[1]).year)
    return fy1_val, fy2_val, fy1_yr, fy2_yr


def fetch_one(ticker: str) -> dict:
    """1銘柄分の自己株買戻データを取得。"""
    out: dict[str, float | int | None] = {
        "buyback_fy1": None,
        "buyback_fy2": None,
        "fy1_year": None,
        "fy2_year": None,
        "market_cap": None,
    }
    try:
        t = yf.Ticker(ticker)
        cf = t.cashflow
        fy1, fy2, y1, y2 = _extract_buybacks(cf)
        out["buyback_fy1"] = fy1
        out["buyback_fy2"] = fy2
        out["fy1_year"] = y1
        out["fy2_year"] = y2
        info = t.info or {}
        mc = info.get("marketCap")
        if mc is not None and mc > 0:
            out["market_cap"] = float(mc)
    except Exception as e:
        logger.debug(f"buyback 取得失敗 {ticker}: {e}")
    return out


def fetch_all_buybacks(
    progress_callback=None,
    sleep_sec: float = 0.0,
) -> pd.DataFrame:
    """JPX500 + Standard Top400 universe 全銘柄の過去2年自己株買戻を集計。"""
    stocks = load_stock_list()
    # ETF 除外
    if "size_category" in stocks.columns:
        stocks = stocks[stocks["size_category"].astype(str).str.upper() != "ETF"].copy()
    stocks = stocks.reset_index(drop=True)
    total = len(stocks)
    logger.info(f"buyback 取得開始: {total} 銘柄")

    rows = []
    for i, row in stocks.iterrows():
        ticker = str(row["ticker"])
        if progress_callback:
            progress_callback(i, total, ticker)
        d = fetch_one(ticker)
        d.update(
            {
                "code": str(row["code"]).zfill(4),
                "ticker": ticker,
                "name": row.get("name", ""),
                "market": row.get("market", "TSE Prime"),
                "size_category": row.get("size_category", ""),
                "sector_33": row.get("sector_33", ""),
            }
        )
        rows.append(d)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    df = pd.DataFrame(rows)
    # 2年合計 (片方欠損ならその分を NaN→0 で吸収しないと「実は買い戻していないだけ」と区別不能)
    fy1 = pd.to_numeric(df["buyback_fy1"], errors="coerce")
    fy2 = pd.to_numeric(df["buyback_fy2"], errors="coerce")
    df["buyback_2y_jpy"] = fy1.fillna(0) + fy2.fillna(0)
    # 両方 None なら NaN として残す
    both_missing = fy1.isna() & fy2.isna()
    df.loc[both_missing, "buyback_2y_jpy"] = pd.NA
    df["buyback_to_mcap"] = pd.to_numeric(
        df["buyback_2y_jpy"], errors="coerce"
    ) / pd.to_numeric(df["market_cap"], errors="coerce")
    cols = [
        "code",
        "ticker",
        "name",
        "market",
        "size_category",
        "sector_33",
        "buyback_2y_jpy",
        "buyback_fy1",
        "buyback_fy2",
        "fy1_year",
        "fy2_year",
        "market_cap",
        "buyback_to_mcap",
    ]
    return (
        df[cols]
        .sort_values("buyback_2y_jpy", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def save_buybacks(df: pd.DataFrame, path: Path | str = BUYBACKS_CSV) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    def _log_progress(i: int, total: int, ticker: str) -> None:
        if i % 50 == 0 or i == total - 1:
            logger.info(f"  buyback 取得 {i + 1}/{total} ({ticker})")

    df = fetch_all_buybacks(progress_callback=_log_progress)
    out = save_buybacks(df)
    logger.info(f"buybacks_2y.csv 保存: {out} ({len(df)} 行)")
    non_zero = df["buyback_2y_jpy"].fillna(0) > 0
    logger.info(f"過去2年で自己株買戻 > 0 の銘柄: {non_zero.sum()} 社")
