"""JPX `data_j.xls` から東証スタンダード市場 Top N の銘柄リストを生成する。

- `fetch_data_j()`        : data_j.xls をキャッシュ付きで DL + パース
- `build_standard_top_n_list()` : スタンダード銘柄を時価総額順に Top N 抽出
- `update_universe()`     : バッチ Step 0 用オーケストレータ（7日以内ならスキップ）
- `load_standard_list()`  : `data/standard_list.csv` を読み込む

JPX500 (data/jpx500_list.csv) と同じスキーマで `data/standard_list.csv` を出力し、
load_stock_list() で両者を結合できるようにする。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from config.settings import (
    CACHE_DIR,
    JPX_DATA_J_CACHE,
    JPX_DATA_J_URL,
    JPX_UNIVERSE_REFRESH_DAYS,
    STANDARD_LIST_CSV,
    STANDARD_MARKET_LABEL,
    STANDARD_TOP100_THRESHOLD,
    STANDARD_TOP_N,
)

logger = logging.getLogger(__name__)

_COL_CODE = "コード"
_COL_NAME = "銘柄名"
_COL_MARKET = "市場・商品区分"
_COL_S33_CODE = "33業種コード"
_COL_S33_NAME = "33業種区分"
_COL_S17_CODE = "17業種コード"
_COL_S17_NAME = "17業種区分"
_COL_SIZE_CODE = "規模コード"
_COL_SIZE_NAME = "規模区分"

_STANDARD_SEGMENT = "スタンダード（内国株式）"

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def fetch_data_j(force: bool = False) -> pd.DataFrame:
    """`data_j.xls` をキャッシュ付きで DL してパース。

    Returns:
        DataFrame[code, name, market_segment, sector_33_code, sector_33,
                  sector_17_code, sector_17, size_code, size_name]
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = Path(JPX_DATA_J_CACHE)

    needs_download = force or not cache_path.exists()
    if not needs_download:
        age_days = (
            datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        ).days
        if age_days >= JPX_UNIVERSE_REFRESH_DAYS:
            needs_download = True

    if needs_download:
        logger.info(f"data_j.xls を取得中: {JPX_DATA_J_URL}")
        resp = requests.get(JPX_DATA_J_URL, headers=_FETCH_HEADERS, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        logger.info(f"data_j.xls 保存完了: {cache_path} ({len(resp.content)} bytes)")

    df = pd.read_excel(cache_path, dtype={_COL_CODE: str})
    df = df.rename(
        columns={
            _COL_CODE: "code",
            _COL_NAME: "name",
            _COL_MARKET: "market_segment",
            _COL_S33_CODE: "sector_33_code",
            _COL_S33_NAME: "sector_33",
            _COL_S17_CODE: "sector_17_code",
            _COL_S17_NAME: "sector_17",
            _COL_SIZE_CODE: "size_code_jpx",
            _COL_SIZE_NAME: "size_name_jpx",
        }
    )
    df["code"] = df["code"].astype(str).str.strip().str.zfill(4)
    return df


def _fetch_market_cap(ticker: str) -> float | None:
    """yfinance `info` から market cap を取得。失敗時 None。"""
    try:
        info = yf.Ticker(ticker).info or {}
        mc = info.get("marketCap")
        if mc is not None and mc > 0:
            return float(mc)
    except Exception as e:
        logger.debug(f"market_cap 取得失敗 {ticker}: {e}")
    return None


def build_standard_top_n_list(
    n: int = STANDARD_TOP_N,
    sleep_sec: float = 0.0,
    progress_callback=None,
) -> pd.DataFrame:
    """スタンダード市場 (内国株式) から時価総額 Top N を抽出。

    Args:
        n: 上位 N 件
        sleep_sec: yfinance リクエスト間スリープ
        progress_callback: (i, total, ticker) を受ける進捗コールバック

    Returns:
        DataFrame[code, name, ticker, market, size_category, sector_33,
                  sector_17, market_cap]
    """
    raw = fetch_data_j()
    std = raw[raw["market_segment"] == _STANDARD_SEGMENT].copy()
    logger.info(f"スタンダード市場対象: {len(std)} 件")

    std["ticker"] = std["code"].astype(str).str.zfill(4) + ".T"

    market_caps: list[float | None] = []
    total = len(std)
    for i, ticker in enumerate(std["ticker"].tolist()):
        if progress_callback:
            progress_callback(i, total, ticker)
        market_caps.append(_fetch_market_cap(ticker))
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    std["market_cap"] = market_caps

    std = std.dropna(subset=["market_cap"])
    std = std.sort_values("market_cap", ascending=False).head(n).reset_index(drop=True)

    def _size_label(rank_idx: int) -> str:
        return (
            f"TSE Standard Top{STANDARD_TOP100_THRESHOLD}"
            if rank_idx < STANDARD_TOP100_THRESHOLD
            else f"TSE Standard Top{n}"
        )

    std["size_category"] = [_size_label(i) for i in range(len(std))]
    std["market"] = STANDARD_MARKET_LABEL

    cols = [
        "code",
        "name",
        "ticker",
        "market",
        "size_category",
        "sector_33",
        "sector_17",
        "market_cap",
    ]
    return std[cols]


def update_universe(force: bool = False) -> Path | None:
    """data_j.xls 更新 + standard_list.csv 再生成。

    既存 standard_list.csv が `JPX_UNIVERSE_REFRESH_DAYS` 以内ならスキップ。
    """
    csv_path = Path(STANDARD_LIST_CSV)
    if not force and csv_path.exists():
        age_days = (
            datetime.now() - datetime.fromtimestamp(csv_path.stat().st_mtime)
        ).days
        if age_days < JPX_UNIVERSE_REFRESH_DAYS:
            logger.info(
                f"standard_list.csv は {age_days} 日前更新 (閾値 {JPX_UNIVERSE_REFRESH_DAYS}) - スキップ"
            )
            return None

    logger.info("Standard Top N リストを再生成中...")

    def _log_progress(i: int, total: int, ticker: str) -> None:
        if i % 100 == 0:
            logger.info(f"  market_cap 取得 {i}/{total} ({ticker})")

    df = build_standard_top_n_list(progress_callback=_log_progress)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"standard_list.csv 保存完了: {csv_path} ({len(df)} 件)")
    return csv_path


def load_standard_list() -> pd.DataFrame | None:
    """`data/standard_list.csv` を読み込む。存在しなければ None。"""
    path = Path(STANDARD_LIST_CSV)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(4)
        if "market" not in df.columns:
            df["market"] = STANDARD_MARKET_LABEL
        return df
    except Exception as e:
        logger.warning(f"standard_list.csv 読込失敗: {e}")
        return None


def _expire_age_days(age_days: timedelta) -> int:
    return age_days.days


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    update_universe(force=True)
