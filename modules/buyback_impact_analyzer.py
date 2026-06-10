"""自己株買戻の事後影響を会計年度単位で計測する。

各 (ticker, FY) で買戻があった場合:
  buyback_pct_mcap : 買戻額 / FY末時価総額
  pbr_pre, pbr_post: FY末 と +1年の PBR
  roe_pre, roe_post: 当該FY と翌FY の ROE (yfinance income_stmt + balance_sheet)
  price_1y_return  : FY末 close から +1年の上昇率

注意:
- 「アナウンス日」プロキシ = FY末 + 90日 (決算発表後3ヶ月想定)。
  正確なTDnet適時開示日が必要なら別途スクレイピングが必要。
- per_pbr_history.parquet が無い場合は yfinance trailingPE/PBR をFY末snapshotで代用。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR, PER_PBR_HISTORY_PARQUET
from modules.data_fetcher import load_cached, load_stock_list

logger = logging.getLogger(__name__)

BUYBACK_IMPACT_CSV = DATA_DIR / "buyback_impact.csv"

_CF_BUYBACK_KEYS = (
    "Repurchase Of Capital Stock",
    "Common Stock Repurchased",
    "Repurchase Of Common Stock",
)
_BS_TOTAL_EQUITY = (
    "Stockholders Equity",
    "Common Stock Equity",
    "Total Equity Gross Minority Interest",
)
_IS_NET_INCOME = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income Continuous Operations",
)


def _value_for_col(
    stmt: pd.DataFrame | None, keys: tuple[str, ...], col
) -> float | None:
    if stmt is None or stmt.empty or col not in stmt.columns:
        return None
    for k in keys:
        if k in stmt.index:
            v = stmt.at[k, col]
            if pd.notna(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
    return None


def _price_on_or_after(
    cache: pd.DataFrame | None, target: pd.Timestamp
) -> float | None:
    if cache is None or cache.empty:
        return None
    sub = cache[cache.index >= target]
    if sub.empty:
        return None
    v = sub.iloc[0]["Close"]
    return float(v) if pd.notna(v) else None


def _shares_at(equity: float | None, bps_proxy: float | None) -> float | None:
    """株式数を BS equity / BPS で逆算。両者が無ければ None。"""
    if equity is None or bps_proxy is None or bps_proxy <= 0:
        return None
    return float(equity) / float(bps_proxy)


def _pbr_at(history: pd.DataFrame, ticker: str, target: pd.Timestamp) -> float | None:
    if history is None or history.empty:
        return None
    sub = history[(history["ticker"] == ticker) & (history["date"] <= target)]
    if sub.empty:
        sub = history[history["ticker"] == ticker]
        if sub.empty:
            return None
    val = sub.sort_values("date").iloc[-1]["pbr"]
    return float(val) if pd.notna(val) else None


def analyze_ticker(
    ticker: str,
    pbr_history: pd.DataFrame | None,
    lookback_years: int = 2,
) -> list[dict]:
    """1銘柄の過去 N FY 分の buyback impact を返す。"""
    events: list[dict] = []
    try:
        t = yf.Ticker(ticker)
        cf = t.cashflow
        bs = t.balance_sheet
        inc = t.income_stmt
        if cf is None or cf.empty:
            return events

        buyback_row = None
        for k in _CF_BUYBACK_KEYS:
            if k in cf.index:
                buyback_row = cf.loc[k]
                break
        if buyback_row is None:
            return events

        cols = list(cf.columns)
        for i, col in enumerate(cols[:lookback_years]):
            v = buyback_row.iloc[i]
            if pd.isna(v):
                continue
            buyback = abs(float(v)) if float(v) < 0 else 0.0
            if buyback <= 0:
                continue

            fy_end = pd.Timestamp(col)
            announce = fy_end + pd.Timedelta(days=90)  # 決算発表想定
            one_year_later = announce + pd.Timedelta(days=365)

            cache = load_cached(ticker)
            price_at = _price_on_or_after(cache, announce)
            price_1y = _price_on_or_after(cache, one_year_later)
            if price_at is None:
                continue
            price_1y_return = (
                (price_1y / price_at - 1.0) if (price_1y is not None) else None
            )

            equity_pre = _value_for_col(bs, _BS_TOTAL_EQUITY, col)
            net_income_pre = _value_for_col(inc, _IS_NET_INCOME, col)
            roe_pre = (
                (net_income_pre / equity_pre * 100)
                if (equity_pre and net_income_pre is not None and equity_pre > 0)
                else None
            )

            # 翌FY (cf.columns は降順なので i-1 が新しい)
            roe_post = None
            equity_post = None
            net_income_post = None
            if i > 0:
                next_col = cols[i - 1]
                equity_post = _value_for_col(bs, _BS_TOTAL_EQUITY, next_col)
                net_income_post = _value_for_col(inc, _IS_NET_INCOME, next_col)
                if equity_post and net_income_post is not None and equity_post > 0:
                    roe_post = net_income_post / equity_post * 100

            # 時価総額 = price * shares (株式数は equity から逆算しないので info を使う)
            try:
                shares_out = (t.info or {}).get("sharesOutstanding")
            except Exception:
                shares_out = None
            mcap_at = (
                price_at * float(shares_out)
                if (shares_out and shares_out > 0)
                else None
            )
            buyback_pct = (
                (buyback / mcap_at * 100) if (mcap_at and mcap_at > 0) else None
            )

            pbr_pre = (
                _pbr_at(pbr_history, ticker, announce)
                if pbr_history is not None
                else None
            )
            pbr_post = (
                _pbr_at(pbr_history, ticker, one_year_later)
                if pbr_history is not None
                else None
            )

            events.append(
                {
                    "ticker": ticker,
                    "fy_end": fy_end.date(),
                    "announce_proxy_date": announce.date(),
                    "buyback_jpy": buyback,
                    "market_cap_at_announce": mcap_at,
                    "buyback_pct_mcap": buyback_pct,
                    "price_at_announce": price_at,
                    "price_1y_later": price_1y,
                    "price_1y_return_pct": (
                        price_1y_return * 100 if price_1y_return is not None else None
                    ),
                    "pbr_pre": pbr_pre,
                    "pbr_post": pbr_post,
                    "pbr_delta": (
                        (pbr_post - pbr_pre)
                        if (pbr_pre is not None and pbr_post is not None)
                        else None
                    ),
                    "roe_pre_pct": roe_pre,
                    "roe_post_pct": roe_post,
                    "roe_delta_pct": (
                        (roe_post - roe_pre)
                        if (roe_pre is not None and roe_post is not None)
                        else None
                    ),
                    "equity_pre": equity_pre,
                    "equity_post": equity_post,
                    "net_income_pre": net_income_pre,
                    "net_income_post": net_income_post,
                }
            )
    except Exception as e:
        logger.debug(f"analyze_ticker 失敗 {ticker}: {e}")
    return events


def analyze_universe(
    progress_callback=None,
    lookback_years: int = 2,
) -> pd.DataFrame:
    """JPX500 + Standard Top400 universe 全銘柄の buyback impact を集計。"""
    stocks = load_stock_list()
    if "size_category" in stocks.columns:
        stocks = stocks[stocks["size_category"].astype(str).str.upper() != "ETF"].copy()
    stocks = stocks.reset_index(drop=True)
    total = len(stocks)

    pbr_history: pd.DataFrame | None = None
    if Path(PER_PBR_HISTORY_PARQUET).exists():
        try:
            pbr_history = pd.read_parquet(PER_PBR_HISTORY_PARQUET)
            pbr_history["date"] = pd.to_datetime(pbr_history["date"])
        except Exception as e:
            logger.warning(f"per_pbr_history 読込失敗: {e}")

    all_rows: list[dict] = []
    for i, row in stocks.iterrows():
        ticker = str(row["ticker"])
        if progress_callback:
            progress_callback(i, total, ticker)
        events = analyze_ticker(ticker, pbr_history, lookback_years=lookback_years)
        for e in events:
            e.update(
                {
                    "code": str(row["code"]).zfill(4),
                    "name": row.get("name", ""),
                    "market": row.get("market", "TSE Prime"),
                    "size_category": row.get("size_category", ""),
                    "sector_33": row.get("sector_33", ""),
                }
            )
            all_rows.append(e)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    cols = [
        "code",
        "name",
        "ticker",
        "market",
        "size_category",
        "sector_33",
        "fy_end",
        "announce_proxy_date",
        "buyback_jpy",
        "market_cap_at_announce",
        "buyback_pct_mcap",
        "price_at_announce",
        "price_1y_later",
        "price_1y_return_pct",
        "pbr_pre",
        "pbr_post",
        "pbr_delta",
        "roe_pre_pct",
        "roe_post_pct",
        "roe_delta_pct",
        "equity_pre",
        "equity_post",
        "net_income_pre",
        "net_income_post",
    ]
    return (
        df[cols]
        .sort_values("buyback_pct_mcap", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def save_impact(df: pd.DataFrame, path: Path | str = BUYBACK_IMPACT_CSV) -> Path:
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
            logger.info(f"  buyback impact 解析 {i + 1}/{total} ({ticker})")

    df = analyze_universe(progress_callback=_log_progress, lookback_years=2)
    out = save_impact(df)
    logger.info(f"buyback_impact.csv 保存: {out} ({len(df)} イベント)")
    if len(df):
        top10 = df.head(10)[
            [
                "code",
                "name",
                "fy_end",
                "buyback_pct_mcap",
                "price_1y_return_pct",
                "pbr_delta",
                "roe_delta_pct",
            ]
        ]
        logger.info(f"Top10 by buyback_pct_mcap:\n{top10.to_string()}")
