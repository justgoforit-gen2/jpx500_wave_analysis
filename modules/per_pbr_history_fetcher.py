"""PER/PBR時系列履歴取得モジュール。

yfinanceから四半期/年次EPS・純資産を取得し、各銘柄の週次（金曜終値ベース）
PER/PBR時系列を合成して data/per_pbr_history.parquet に保存する。

EPSデータソースの優先順位:
1. Ticker.earnings_history.epsActual (直近4Q、最も信頼性が高い)
2. Ticker.quarterly_income_stmt の Diluted EPS（行が揃わない銘柄あり）
3. Ticker.income_stmt の Diluted EPS（年次、過去への延長用）

BSデータソース:
1. Ticker.quarterly_balance_sheet（5四半期、Stockholders Equity / Shares）
2. Ticker.balance_sheet（年次、過去への延長用）

shares が取れないときは info["sharesOutstanding"] にフォールバック。
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Callable

import pandas as pd
import yfinance as yf

from config.settings import (
    FETCH_BATCH_SIZE,
    PER_PBR_FAILURES_CSV,
    PER_PBR_FETCH_RETRY,
    PER_PBR_FETCH_RETRY_DELAY_SEC,
    PER_PBR_HISTORY_PARQUET,
    PER_PBR_LOOKBACK_YEARS,
    PER_PBR_REPORT_LAG_DAYS,
    PER_PBR_SAMPLING_RULE,
)
from modules.data_fetcher import load_cached, load_stock_list

logger = logging.getLogger(__name__)

_EPS_KEYS = ("Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS")
_EQUITY_KEYS = (
    "Stockholders Equity",
    "StockholdersEquity",
    "Common Stock Equity",
    "CommonStockEquity",
    "Total Stockholders Equity",
    "TotalStockholdersEquity",
)
_SHARES_KEYS = (
    "Ordinary Shares Number",
    "OrdinaryShareNumber",
    "Diluted Average Shares",
    "DilutedAverageShares",
    "Basic Average Shares",
    "BasicAverageShares",
)


def _row_as_numeric(df: pd.DataFrame | None, keys: tuple[str, ...]) -> pd.Series:
    """財務諸表DataFrameからキー候補を順に試して数値Seriesを返す（NaN含む生データ）"""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for k in keys:
        if k in df.index:
            row = pd.to_numeric(df.loc[k], errors="coerce")
            return row
    return pd.Series(dtype=float)


def _build_quarterly_eps(t: yf.Ticker) -> pd.Series:
    """四半期EPSのSeries（index=fiscal_end, 昇順）を返す。

    earnings_history のepsActualを最優先、不足分を quarterly_income_stmt で補完。
    """
    eps: dict[pd.Timestamp, float] = {}

    # 1) earnings_history が最も揃いやすい
    try:
        eh = getattr(t, "earnings_history", None)
        if eh is not None and not eh.empty and "epsActual" in eh.columns:
            for idx, val in (
                pd.to_numeric(eh["epsActual"], errors="coerce").dropna().items()
            ):
                eps[pd.Timestamp(idx)] = float(val)
    except Exception:
        pass

    # 2) quarterly_income_stmt の Diluted EPS で補完
    try:
        qi = getattr(t, "quarterly_income_stmt", None)
        if qi is not None and not qi.empty:
            row = _row_as_numeric(qi, _EPS_KEYS).dropna()
            for idx, val in row.items():
                ts = pd.Timestamp(idx)
                if ts not in eps:
                    eps[ts] = float(val)
    except Exception:
        pass

    if not eps:
        return pd.Series(dtype=float)

    s = pd.Series(eps).sort_index()
    return s


def _build_annual_eps(t: yf.Ticker) -> pd.Series:
    """年次EPSのSeries（index=fiscal_end, 昇順）。"""
    try:
        ai = getattr(t, "income_stmt", None)
        if ai is None or ai.empty:
            return pd.Series(dtype=float)
        row = _row_as_numeric(ai, _EPS_KEYS).dropna()
        return row.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _build_equity_and_shares(t: yf.Ticker) -> pd.DataFrame:
    """期末ごとの純資産・株式数を返す。

    Returns:
        DataFrame[fiscal_end, equity, shares] 昇順。
    """
    rows: dict[pd.Timestamp, dict] = {}

    # quarterly BS優先
    try:
        qb = getattr(t, "quarterly_balance_sheet", None)
        if qb is not None and not qb.empty:
            eq = _row_as_numeric(qb, _EQUITY_KEYS)
            sh = _row_as_numeric(qb, _SHARES_KEYS)
            for col in qb.columns:
                ts = pd.Timestamp(col)
                eqv = (
                    float(eq.loc[col])
                    if col in eq.index and pd.notna(eq.loc[col])
                    else None
                )
                shv = (
                    float(sh.loc[col])
                    if col in sh.index and pd.notna(sh.loc[col])
                    else None
                )
                rows[ts] = {"equity": eqv, "shares": shv}
    except Exception:
        pass

    # annual BSで補完
    try:
        ab = getattr(t, "balance_sheet", None)
        if ab is not None and not ab.empty:
            eq = _row_as_numeric(ab, _EQUITY_KEYS)
            sh = _row_as_numeric(ab, _SHARES_KEYS)
            for col in ab.columns:
                ts = pd.Timestamp(col)
                eqv = (
                    float(eq.loc[col])
                    if col in eq.index and pd.notna(eq.loc[col])
                    else None
                )
                shv = (
                    float(sh.loc[col])
                    if col in sh.index and pd.notna(sh.loc[col])
                    else None
                )
                existing = rows.get(ts, {})
                if existing.get("equity") is None and eqv is not None:
                    existing["equity"] = eqv
                if existing.get("shares") is None and shv is not None:
                    existing["shares"] = shv
                if existing:
                    rows[ts] = existing
                else:
                    rows[ts] = {"equity": eqv, "shares": shv}
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=["fiscal_end", "equity", "shares"])

    df = (
        pd.DataFrame(
            [
                {"fiscal_end": ts, "equity": v.get("equity"), "shares": v.get("shares")}
                for ts, v in rows.items()
            ]
        )
        .sort_values("fiscal_end")
        .reset_index(drop=True)
    )

    # shares の前埋め（過去から）/ 後埋め（未来へ）
    df["shares"] = df["shares"].ffill().bfill()
    return df


def _report_date_for(
    fiscal_end: pd.Timestamp, lag_days: int = PER_PBR_REPORT_LAG_DAYS
) -> pd.Timestamp:
    """期末日から発表日の推定値（保守的に lag_days を加算）。"""
    return pd.Timestamp(fiscal_end) + timedelta(days=lag_days)


def build_ttm_eps_and_bps_series(
    ticker: str,
) -> tuple[pd.Series, pd.Series, float | None]:
    """銘柄のTTM EPS時系列とBPS時系列を構築。

    Returns:
        (ttm_eps_series, bps_series, latest_shares)
        - ttm_eps_series: index=report_date, value=TTM EPS（rolling 4Q合計、または annual EPS で代替）
        - bps_series: index=report_date, value=BPS（純資産 / 株式数）
        - latest_shares: 最新の株式数（market_cap算出用、Noneあり）
    """
    last_err: Exception | None = None
    for attempt in range(PER_PBR_FETCH_RETRY):
        try:
            t = yf.Ticker(ticker)
            q_eps = _build_quarterly_eps(t)
            a_eps = _build_annual_eps(t)
            eq_sh = _build_equity_and_shares(t)

            # --- TTM EPS series ---
            ttm_records: dict[pd.Timestamp, float] = {}

            if len(q_eps) >= 4:
                # 各期末について「その期末を含む直近4Q」のEPS合計
                vals = q_eps.values
                idx = q_eps.index
                for i in range(3, len(q_eps)):
                    window = vals[i - 3 : i + 1]
                    if pd.notna(window).all():
                        ttm_records[_report_date_for(idx[i])] = float(window.sum())

            # annual EPS で過去への延長 / 不足補完
            for fe, val in a_eps.items():
                report_dt = _report_date_for(fe)
                if pd.notna(val) and report_dt not in ttm_records:
                    ttm_records[report_dt] = float(val)

            ttm_eps_series = (
                pd.Series(ttm_records).sort_index()
                if ttm_records
                else pd.Series(dtype=float)
            )

            # --- BPS series ---
            bps_records: dict[pd.Timestamp, float] = {}
            latest_shares: float | None = None
            if not eq_sh.empty:
                for _, row in eq_sh.iterrows():
                    eq_v = row["equity"]
                    sh_v = row["shares"]
                    if pd.notna(eq_v) and pd.notna(sh_v) and sh_v > 0:
                        report_dt = _report_date_for(row["fiscal_end"])
                        bps_records[report_dt] = float(eq_v) / float(sh_v)
                # 最新のshares
                sh_clean = pd.to_numeric(eq_sh["shares"], errors="coerce").dropna()
                if not sh_clean.empty:
                    latest_shares = float(sh_clean.iloc[-1])

            # shares が完全に取れない場合は info.sharesOutstanding にフォールバック
            if latest_shares is None or latest_shares <= 0:
                try:
                    info_shares = t.info.get("sharesOutstanding")
                    if info_shares:
                        latest_shares = float(info_shares)
                except Exception:
                    pass

            bps_series = (
                pd.Series(bps_records).sort_index()
                if bps_records
                else pd.Series(dtype=float)
            )
            return ttm_eps_series, bps_series, latest_shares
        except Exception as e:
            last_err = e
            if attempt < PER_PBR_FETCH_RETRY - 1:
                time.sleep(PER_PBR_FETCH_RETRY_DELAY_SEC)

    if last_err is not None:
        logger.debug(f"{ticker}: TTM/BPS構築失敗: {last_err}")
    return pd.Series(dtype=float), pd.Series(dtype=float), None


def _weekly_close_from_cache(ticker: str) -> pd.Series:
    """parquetキャッシュから週次（金曜）終値を抽出。"""
    df = load_cached(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return pd.Series(dtype=float)
    close.index = pd.to_datetime(close.index)
    weekly = close.resample(PER_PBR_SAMPLING_RULE).last().dropna()
    return weekly


def build_weekly_per_pbr(
    ticker: str,
    code: str,
    name: str,
    sector_33: str,
    size_category: str,
    weekly_close: pd.Series,
    ttm_eps: pd.Series,
    bps: pd.Series,
    shares_latest: float | None = None,
    market: str = "TSE Prime",
) -> pd.DataFrame:
    """週次PER/PBR時系列を構築。

    各週末について、その時点で「発表済み」（report_date <= week_end）の
    直近TTM EPS/BPSをffillで当てる。
    """
    if weekly_close.empty:
        return pd.DataFrame()

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(
        days=PER_PBR_LOOKBACK_YEARS * 365 + 30
    )
    weekly_close = weekly_close[weekly_close.index >= cutoff]
    if weekly_close.empty:
        return pd.DataFrame()

    aligned = pd.DataFrame({"close": weekly_close})
    aligned.index.name = "date"

    if not ttm_eps.empty:
        # 発表日を週末indexにffillで割り当て
        eps_reindexed = (
            ttm_eps.reindex(aligned.index.union(ttm_eps.index)).sort_index().ffill()
        )
        aligned["eps_ttm"] = eps_reindexed.reindex(aligned.index)
    else:
        aligned["eps_ttm"] = pd.NA

    if not bps.empty:
        bps_reindexed = bps.reindex(aligned.index.union(bps.index)).sort_index().ffill()
        aligned["bps"] = bps_reindexed.reindex(aligned.index)
    else:
        aligned["bps"] = pd.NA

    eps_num = pd.to_numeric(aligned["eps_ttm"], errors="coerce")
    bps_num = pd.to_numeric(aligned["bps"], errors="coerce")
    aligned["per"] = aligned["close"] / eps_num
    aligned["pbr"] = aligned["close"] / bps_num

    aligned = aligned.dropna(subset=["per", "pbr"], how="all")
    if aligned.empty:
        return pd.DataFrame()

    out = aligned.reset_index()
    out["code"] = code
    out["name"] = name
    out["ticker"] = ticker
    out["sector_33"] = sector_33
    out["size_category"] = size_category
    out["market"] = market
    if shares_latest is not None and shares_latest > 0:
        out["market_cap"] = out["close"] * float(shares_latest)
    else:
        out["market_cap"] = pd.NA
    return out[
        [
            "code",
            "name",
            "ticker",
            "date",
            "close",
            "eps_ttm",
            "bps",
            "per",
            "pbr",
            "market_cap",
            "sector_33",
            "size_category",
            "market",
        ]
    ]


def _process_single(
    code: str,
    name: str,
    ticker: str,
    sector_33: str,
    size_category: str,
    market: str = "TSE Prime",
) -> tuple[pd.DataFrame | None, str | None]:
    """1銘柄分の週次PER/PBR履歴を構築。失敗時は (None, reason)。"""
    weekly_close = _weekly_close_from_cache(ticker)
    if weekly_close.empty:
        return None, "週次終値なし（cache不在）"

    ttm_eps, bps, latest_shares = build_ttm_eps_and_bps_series(ticker)

    if ttm_eps.empty and bps.empty:
        return None, "TTM EPS/BPS両方未算出"

    df = build_weekly_per_pbr(
        ticker=ticker,
        code=code,
        name=name,
        sector_33=sector_33,
        size_category=size_category,
        weekly_close=weekly_close,
        ttm_eps=ttm_eps,
        bps=bps,
        shares_latest=latest_shares,
        market=market,
    )
    if df.empty:
        return None, "週次合成結果が空"
    return df, None


def update_per_pbr_history(
    full_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_tickers: int | None = None,
) -> dict[str, str]:
    """全銘柄ループで per_pbr_history.parquet を出力。

    Args:
        full_refresh: True なら既存parquetを無視してフル再構築。
        progress_callback: (i, total, ticker) を受け取るコールバック。
        max_tickers: 試験的に銘柄数を制限するときに指定。

    Returns:
        失敗銘柄 {ticker: reason}
    """
    stocks = load_stock_list()
    if "size_category" in stocks.columns:
        stocks = stocks[stocks["size_category"].astype(str).str.upper() != "ETF"].copy()
    if max_tickers is not None:
        stocks = stocks.head(max_tickers).copy()

    total = len(stocks)
    failures: dict[str, str] = {}
    new_rows: list[pd.DataFrame] = []

    cutoff_date: pd.Timestamp | None = None
    existing: pd.DataFrame | None = None
    if not full_refresh and PER_PBR_HISTORY_PARQUET.exists():
        try:
            existing = pd.read_parquet(PER_PBR_HISTORY_PARQUET)
            existing["date"] = pd.to_datetime(existing["date"])
            cutoff_date = existing["date"].max()
        except Exception as e:
            logger.warning(f"既存履歴の読み込みに失敗: {e}")
            existing = None

    for i, (_, row) in enumerate(stocks.iterrows()):
        ticker = str(row["ticker"])
        code = str(row["code"])
        name = str(row.get("name", ""))
        sector_33 = str(row.get("sector_33", ""))
        size_category = str(row.get("size_category", ""))
        market = str(row.get("market", "TSE Prime"))

        if progress_callback:
            progress_callback(i, total, ticker)

        try:
            df, reason = _process_single(
                code, name, ticker, sector_33, size_category, market=market
            )
            if df is None or df.empty:
                if reason:
                    failures[ticker] = reason
            else:
                new_rows.append(df)
        except Exception as e:
            failures[ticker] = f"例外: {e}"
            logger.debug(f"{ticker}: per/pbr履歴処理で例外: {e}")

        if (i + 1) % FETCH_BATCH_SIZE == 0:
            time.sleep(1)

    if not new_rows:
        _write_failures(failures)
        return failures

    new_df = pd.concat(new_rows, ignore_index=True)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if existing is not None and not full_refresh and cutoff_date is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["code", "date"], keep="last")
        combined = combined.sort_values(["code", "date"]).reset_index(drop=True)
    else:
        combined = new_df.sort_values(["code", "date"]).reset_index(drop=True)

    keep_from = pd.Timestamp.now().normalize() - pd.Timedelta(
        days=PER_PBR_LOOKBACK_YEARS * 365 + 30
    )
    combined = combined[combined["date"] >= keep_from].reset_index(drop=True)

    PER_PBR_HISTORY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PER_PBR_HISTORY_PARQUET, index=False)

    _write_failures(failures)
    return failures


def _write_failures(failures: dict[str, str]) -> None:
    if not failures:
        if PER_PBR_FAILURES_CSV.exists():
            try:
                PER_PBR_FAILURES_CSV.unlink()
            except Exception:
                pass
        return
    PER_PBR_FAILURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": t, "reason": r} for t, r in failures.items()]).to_csv(
        PER_PBR_FAILURES_CSV, index=False, encoding="utf-8-sig"
    )
