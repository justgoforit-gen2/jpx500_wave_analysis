"""Yahoo Finance からOHLCVを取得し、parquetキャッシュに保存する"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import (
    CACHE_DIR,
    DATA_PERIOD_YEARS,
    FETCH_BATCH_SIZE,
    FETCH_RETRY_COUNT,
    FETCH_RETRY_DELAY_SEC,
    STOCK_LIST_CSV,
)

logger = logging.getLogger(__name__)


def _close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    if "close" in df.columns:
        return pd.to_numeric(df["close"], errors="coerce")
    # 最後の手段: 数値列の先頭
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s
    return pd.Series(dtype=float)


def _fetch_fx_pair(pair: str) -> pd.Series | None:
    """FXペア（例: 'EURJPY=X'）の日次終値Seriesを返す"""
    df = fetch_and_cache(pair)
    if df is None or df.empty:
        return None
    s = _close_series(df)
    if s.empty:
        return None
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    s.sort_index(inplace=True)
    return s


def get_fx_to_jpy_daily(currency: str) -> pd.Series | None:
    """通貨(currency)の 1単位あたりJPYレート（日次）を返す

    例: currency='USD' -> USDJPY（日次、JPY/USD）
    """
    ccy = (currency or "").upper()
    if ccy in ("JPY", ""):
        return None

    # USDJPY は yfinance では "JPY=X" が一般的
    if ccy == "USD":
        return _fetch_fx_pair("JPY=X")

    # 1) 直接ペア: CCYJPY
    direct = _fetch_fx_pair(f"{ccy}JPY=X")
    if direct is not None and not direct.empty:
        return direct

    # 2) CCYUSD * USDJPY
    usdjpy = _fetch_fx_pair("JPY=X")
    if usdjpy is None or usdjpy.empty:
        return None

    ccyusd = _fetch_fx_pair(f"{ccy}USD=X")
    if ccyusd is not None and not ccyusd.empty:
        joined = pd.concat(
            [ccyusd.rename("ccyusd"), usdjpy.rename("usdjpy")], axis=1
        ).dropna()
        if joined.empty:
            return None
        return (joined["ccyusd"] * joined["usdjpy"]).rename(f"{ccy}JPY")

    # 3) 逆ペア: USDCCY を反転して USD/CCY を作る -> * USDJPY
    usdccy = _fetch_fx_pair(f"USD{ccy}=X")
    if usdccy is not None and not usdccy.empty:
        inv = (1.0 / usdccy.replace(0, pd.NA)).dropna()
        joined = pd.concat(
            [inv.rename("ccyusd"), usdjpy.rename("usdjpy")], axis=1
        ).dropna()
        if joined.empty:
            return None
        return (joined["ccyusd"] * joined["usdjpy"]).rename(f"{ccy}JPY")

    return None


def get_fx_to_jpy_monthly_avg(currency: str) -> pd.Series | None:
    """通貨(currency)の 1単位あたりJPYレート（月平均）を返す（indexは月末）"""
    daily = get_fx_to_jpy_daily(currency)
    if daily is None or daily.empty:
        return None
    monthly = daily.resample("ME").mean().dropna()
    monthly.name = f"{currency.upper()}JPY_MA"
    return monthly


def convert_ohlcv_close_to_jpy_by_month_avg(
    df: pd.DataFrame, currency: str
) -> pd.DataFrame:
    """OHLCVのCloseを、月平均為替でJPY換算して返す（元のdfは変更しない）"""
    ccy = (currency or "").upper()
    if df is None or df.empty or ccy in ("JPY", ""):
        return df

    fx_m = get_fx_to_jpy_monthly_avg(ccy)
    if fx_m is None or fx_m.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)

    close = _close_series(out)
    if close.empty:
        return df

    # 月キー（YYYY-MM）で月平均FXを当てる（timestampの時刻ズレを避ける）
    fx_m2 = fx_m.copy()
    fx_m2.index = fx_m2.index.to_period("M")
    month_key = out.index.to_period("M")
    fx_for_days = fx_m2.reindex(month_key).ffill().bfill().to_numpy()

    if "Close" in out.columns:
        out["Close"] = (
            pd.to_numeric(out["Close"], errors="coerce").to_numpy() * fx_for_days
        )
    elif "close" in out.columns:
        out["close"] = (
            pd.to_numeric(out["close"], errors="coerce").to_numpy() * fx_for_days
        )
    else:
        # Close列が無い場合は作成（チャート用途）
        out["Close"] = close.to_numpy() * fx_for_days

    return out


def load_stock_list() -> pd.DataFrame:
    """銘柄リストCSVを読み込む"""
    return pd.read_csv(STOCK_LIST_CSV, encoding="utf-8-sig", dtype={"code": str})


def _cache_path(ticker: str) -> Path:
    """銘柄のparquetキャッシュパス"""
    safe_name = ticker.replace(".", "_")
    return CACHE_DIR / f"{safe_name}.parquet"


def fetch_single(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """1銘柄のOHLCVを取得（リトライ付き）"""
    for attempt in range(FETCH_RETRY_COUNT):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is not None and len(df) > 0:
                # MultiIndex columns対策
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index.name = "Date"
                return df
            logger.warning(f"{ticker}: データなし (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"{ticker}: 取得エラー (attempt {attempt + 1}): {e}")
        if attempt < FETCH_RETRY_COUNT - 1:
            time.sleep(FETCH_RETRY_DELAY_SEC)
    return None


def fetch_and_cache(ticker: str, full_refresh: bool = False) -> pd.DataFrame | None:
    """1銘柄のデータを取得しキャッシュに保存（差分更新対応）"""
    cache = _cache_path(ticker)
    # yfinanceのdownloadは end が実質「排他的」になりやすく、
    # end=今日 だと終値が今日分まで入らず「常に1日遅れ」に見えることがある。
    # そのため end は翌日を指定し、当日分が取れるようにする。
    now = datetime.now()
    end_day = (now + timedelta(days=1)).date()
    end_date = end_day.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=DATA_PERIOD_YEARS * 365 + 30)).strftime(
        "%Y-%m-%d"
    )

    existing = None
    if cache.exists() and not full_refresh:
        existing = pd.read_parquet(cache)
        existing.index = pd.to_datetime(existing.index)
        last_ts = existing.index.max()
        last_day = (
            last_ts.date() if hasattr(last_ts, "date") else pd.Timestamp(last_ts).date()
        )
        # 差分取得: 最終日の翌日から
        diff_start_day = last_day + timedelta(days=1)
        diff_start = diff_start_day.strftime("%Y-%m-%d")
        if diff_start_day >= end_day:
            return existing  # 最新まである
        new_data = fetch_single(ticker, diff_start, end_date)
        if new_data is not None and len(new_data) > 0:
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            # 3年分にトリム
            cutoff = pd.Timestamp(start_date)
            combined = combined[combined.index >= cutoff]
            combined.to_parquet(cache)
            return combined
        return existing

    # フル取得
    df = fetch_single(ticker, start_date, end_date)
    if df is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df


def fetch_all(
    full_refresh: bool = False,
    progress_callback=None,
) -> dict[str, str]:
    """全銘柄を取得。戻り値は失敗銘柄の {ticker: error_reason}"""
    stocks = load_stock_list()
    tickers = stocks["ticker"].tolist()
    total = len(tickers)
    failures: dict[str, str] = {}

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        result = fetch_and_cache(ticker, full_refresh=full_refresh)
        if result is None:
            failures[ticker] = "データ取得失敗"
            logger.error(f"FAILED: {ticker}")
        # yfinance rate limit対策
        if (i + 1) % FETCH_BATCH_SIZE == 0:
            time.sleep(1)

    return failures


def fetch_valuation(ticker: str) -> dict:
    """Yahoo FinanceからPER/PBRを取得する"""
    try:
        info = yf.Ticker(ticker).info
        per = info.get("trailingPE") or info.get("forwardPE")
        pbr = info.get("priceToBook")
        market_cap = info.get("marketCap")
        return {
            "per": round(per, 2) if per is not None else None,
            "pbr": round(pbr, 2) if pbr is not None else None,
            "market_cap": market_cap,
        }
    except Exception as e:
        logger.warning(f"{ticker}: PER/PBR取得失敗: {e}")
        return {"per": None, "pbr": None, "market_cap": None}


def load_cached(ticker: str) -> pd.DataFrame | None:
    """キャッシュからデータを読み込む"""
    cache = _cache_path(ticker)
    if cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        return df
    return None


def get_nikkei225() -> pd.DataFrame | None:
    """日経225（^N225）のデータを取得する（キャッシュ優先）"""
    # キャッシュが存在しても、差分更新(fetch_and_cache)を必ず試みる。
    # 以前は cache があると更新しないため、古い日付で止まり続けることがあった。
    return fetch_and_cache("^N225")


# 海外主要指数マスタ: {ticker: 表示名}
GLOBAL_INDICES: dict[str, str] = {
    "^DJI": "ダウ平均",
    "^GSPC": "S&P500",
    "^GDAXI": "DAX",
    "^STOXX50E": "Euro Stoxx50",
    "^KS11": "KOSPI",
    "^NDX": "Nasdaq100",
    "^HSI": "ハンセン指数",
    "000300.SS": "CSI300",
    "^NSEI": "Nifty50",
    "^AXJO": "ASX200",
}


def get_global_index(ticker: str) -> pd.DataFrame | None:
    """海外指数のデータを取得する（キャッシュ優先）"""
    cached = load_cached(ticker)
    if cached is not None:
        return cached
    return fetch_and_cache(ticker)


def fetch_financials(ticker: str) -> pd.DataFrame | None:
    """過去3年+予測の売上高・営業利益率を取得する

    Returns:
        DataFrame with columns: period, revenue, op_margin, is_forecast
        - period: 表示用文字列 (例: "2023年3月期")
        - revenue: 売上高（億円）
        - op_margin: 営業利益率（%）
        - is_forecast: 予測かどうか
    """
    try:
        t = yf.Ticker(ticker)
        stmt = t.income_stmt  # columns=日付(降順), rows=項目
        if stmt is None or stmt.empty:
            return None

        rows = []
        for col in reversed(stmt.columns):  # 古い順に処理
            revenue = None
            op_income = None
            # Total Revenue
            for key in ["Total Revenue", "TotalRevenue"]:
                if key in stmt.index and pd.notna(stmt.loc[key, col]):
                    revenue = float(stmt.loc[key, col])
                    break
            # Operating Income
            for key in ["Operating Income", "OperatingIncome"]:
                if key in stmt.index and pd.notna(stmt.loc[key, col]):
                    op_income = float(stmt.loc[key, col])
                    break

            if revenue is None or revenue == 0:
                continue

            dt = pd.Timestamp(col)
            period_label = f"{dt.year}年{dt.month}月期"
            op_margin = (op_income / revenue * 100) if op_income is not None else None

            rows.append(
                {
                    "period": period_label,
                    "revenue": round(revenue / 1e8, 1),  # 億円
                    "op_margin": round(op_margin, 2) if op_margin is not None else None,
                    "is_forecast": False,
                }
            )

        # 予測データの取得を試みる
        try:
            rev_est = getattr(t, "revenue_estimate", None)
            earn_est = getattr(t, "earnings_estimate", None)
            if rev_est is not None and not rev_est.empty and "avg" in rev_est.columns:
                # 次年度予測（0Y = 今年度）
                for idx_label in rev_est.index:
                    avg_rev = rev_est.loc[idx_label, "avg"]
                    if pd.notna(avg_rev) and avg_rev > 0:
                        avg_earn = None
                        if (
                            earn_est is not None
                            and not earn_est.empty
                            and "avg" in earn_est.columns
                        ):
                            if idx_label in earn_est.index:
                                avg_earn = earn_est.loc[idx_label, "avg"]

                        # 予測の営業利益率は概算（earnings≒純利益なので参考値）
                        est_margin = None
                        if avg_earn is not None and pd.notna(avg_earn) and avg_rev > 0:
                            est_margin = round(
                                float(avg_earn) / float(avg_rev) * 100, 2
                            )

                        rows.append(
                            {
                                "period": f"{idx_label}(予)",
                                "revenue": round(float(avg_rev) / 1e8, 1),
                                "op_margin": est_margin,
                                "is_forecast": True,
                            }
                        )
        except Exception:
            pass  # 予測データが取得できなくても実績のみで続行

        if not rows:
            return None

        return pd.DataFrame(rows)

    except Exception as e:
        logger.warning(f"{ticker}: 財務データ取得失敗: {e}")
        return None


def compute_sector_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """33業種ごとのPER/PBR/時価総額を集計する

    Returns:
        DataFrame with columns: sector_33, count, per_median, pbr_median,
        market_cap_total (億円), market_cap_count, market_cap_coverage_pct
    """
    if "sector_33" not in results_df.columns:
        return pd.DataFrame()

    has_market_cap = "market_cap" in results_df.columns

    rows = []
    for sector, group in results_df.groupby("sector_33"):
        if pd.isna(sector) or sector == "":
            continue
        per_vals = (
            group["per"].dropna() if "per" in group.columns else pd.Series(dtype=float)
        )
        pbr_vals = (
            group["pbr"].dropna() if "pbr" in group.columns else pd.Series(dtype=float)
        )

        row = {
            "sector_33": sector,
            "count": len(group),
            "per_median": round(float(per_vals.median()), 2)
            if len(per_vals) > 0
            else None,
            "pbr_median": round(float(pbr_vals.median()), 2)
            if len(pbr_vals) > 0
            else None,
        }

        if has_market_cap:
            mc_vals = group["market_cap"].dropna()
            row["market_cap_total"] = (
                round(float(mc_vals.sum()) / 1e8, 0) if len(mc_vals) > 0 else None
            )
            row["market_cap_count"] = int(len(mc_vals))
            row["market_cap_coverage_pct"] = (
                round(float(len(mc_vals)) / float(len(group)) * 100, 1)
                if len(group) > 0
                else None
            )
        else:
            row["market_cap_total"] = None
            row["market_cap_count"] = None
            row["market_cap_coverage_pct"] = None

        rows.append(row)

    return pd.DataFrame(rows).sort_values("sector_33").reset_index(drop=True)


def compute_sector_index(
    sector_33: str, results_df: pd.DataFrame
) -> pd.DataFrame | None:
    """33業種の加重平均指数を算出する（base=100）

    market_cap列がある場合は時価総額加重、ない場合は等加重で算出する。

    Args:
        sector_33: 33業種区分名
        results_df: results.csvを読み込んだDataFrame

    Returns:
        pd.DataFrame with "Close" column indexed by Date, or None
    """
    has_market_cap = "market_cap" in results_df.columns

    # 同一セクターの銘柄を抽出
    sector_stocks = results_df[results_df["sector_33"] == sector_33].copy()

    if has_market_cap:
        # market_capがある銘柄のみ使用
        valid = sector_stocks[
            (sector_stocks["market_cap"].notna()) & (sector_stocks["market_cap"] > 0)
        ]
        if len(valid) > 0:
            sector_stocks = valid
            use_weight = True
        else:
            use_weight = False
    else:
        use_weight = False

    if len(sector_stocks) == 0:
        return None

    # 各銘柄の日次終値を取得
    price_data = {}
    weights = {}
    for _, row in sector_stocks.iterrows():
        ticker = row["ticker"]
        df = load_cached(ticker)
        if df is not None and len(df) > 0:
            price_data[ticker] = df["Close"]
            if use_weight:
                weights[ticker] = float(row["market_cap"])
            else:
                weights[ticker] = 1.0  # 等加重

    if not price_data:
        return None

    # 全銘柄の終値をDataFrameに統合
    prices_df = pd.DataFrame(price_data)
    prices_df = prices_df.dropna(how="all")

    if len(prices_df) < 2:
        return None

    # 日次リターンを計算
    returns_df = prices_df.pct_change()

    # 加重平均リターン
    weight_series = pd.Series(weights)
    weighted_returns = []
    for date_idx in returns_df.index:
        daily = returns_df.loc[date_idx].dropna()
        if len(daily) == 0:
            weighted_returns.append(0.0)
            continue
        w = weight_series[daily.index]
        w_norm = w / w.sum()
        weighted_returns.append(float((daily * w_norm).sum()))

    weighted_returns_series = pd.Series(weighted_returns, index=returns_df.index)

    # 累積して指数化（base=100）
    index_values = (1 + weighted_returns_series).cumprod() * 100

    return pd.DataFrame({"Close": index_values}, index=returns_df.index)
