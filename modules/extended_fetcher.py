"""拡張ユニバース(海外ETF/個別株)を取得する。

JPX500 以外のティッカー(VOO, QQQ, NVDA, etc.)を yfinance 経由で
data/cache/ に保存し、最新終値を取得できる状態にする。
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from config.settings import (
    EXTENDED_RESULTS_CSV,
    EXTENDED_UNIVERSE_CSV,
    FETCH_BATCH_SIZE,
)
from modules.data_fetcher import fetch_and_cache, load_cached
from modules.wave_classifier import classify, compute_indicators

logger = logging.getLogger(__name__)


def load_extended_universe() -> pd.DataFrame:
    """extended_universe.csv を読み込む。"""
    if not EXTENDED_UNIVERSE_CSV.exists():
        logger.warning(f"{EXTENDED_UNIVERSE_CSV} が存在しません")
        return pd.DataFrame(columns=["ticker", "name", "category", "currency", "role"])
    return pd.read_csv(EXTENDED_UNIVERSE_CSV, encoding="utf-8-sig")


def fetch_extended_all(progress_callback=None) -> dict[str, str]:
    """拡張ユニバースを全件取得しキャッシュ。失敗銘柄の dict を返す。"""
    universe = load_extended_universe()
    if universe.empty:
        return {}

    failures: dict[str, str] = {}
    tickers = universe["ticker"].tolist()
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        try:
            result = fetch_and_cache(ticker)
            if result is None or result.empty:
                failures[ticker] = "データ取得失敗"
                logger.warning(f"FAILED extended: {ticker}")
        except Exception as e:
            failures[ticker] = str(e)
            logger.warning(f"FAILED extended {ticker}: {e}")
        # yfinance rate limit 対策
        if (i + 1) % FETCH_BATCH_SIZE == 0:
            time.sleep(1)

    logger.info(f"拡張ユニバース取得完了: {total}件中 失敗 {len(failures)}件")
    return failures


def compute_extended_indicators() -> pd.DataFrame:
    """拡張ユニバースの最新指標 + 波形分類を CSV に出力。

    JPX500 と同じ wave_classifier.classify() を流用して
    "上昇トレンド" 等のラベルと slope/range_pct/touch を保存。
    """
    universe = load_extended_universe()
    if universe.empty:
        return pd.DataFrame()

    rows = []
    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        df = load_cached(ticker)
        if df is None or df.empty:
            logger.warning(f"{ticker}: キャッシュなし(extended_results をスキップ)")
            continue

        indicators = compute_indicators(df)
        if indicators is None:
            continue

        wave_types = classify(indicators)
        latest_close = float(df["Close"].iloc[-1])
        latest_date = df.index[-1].strftime("%Y-%m-%d")

        rows.append(
            {
                "ticker": ticker,
                "name": row.get("name", ticker),
                "category": row.get("category", ""),
                "currency": row.get("currency", "USD"),
                "role": row.get("role", ""),
                "latest_close": latest_close,
                "latest_date": latest_date,
                "wave_types": "|".join(wave_types),
                "range_high": indicators.get("range_high"),
                "range_low": indicators.get("range_low"),
                "range_pct": indicators.get("range_pct"),
                "slope": indicators.get("slope"),
                "touch_high": indicators.get("touch_high"),
                "touch_low": indicators.get("touch_low"),
                "atr": indicators.get("atr"),
                "breakout_days": indicators.get("breakout_days"),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.to_csv(EXTENDED_RESULTS_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"拡張ユニバース指標出力: {len(out)}件 → {EXTENDED_RESULTS_CSV}")
    return out


def load_extended_results() -> pd.DataFrame:
    """extended_results.csv を読み込む。"""
    if not EXTENDED_RESULTS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(EXTENDED_RESULTS_CSV, encoding="utf-8-sig")
