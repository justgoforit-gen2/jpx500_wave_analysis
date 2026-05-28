"""資本効率改善期待スクリーナー (Capital Efficiency Screener, CES)。

「お金を貯め込みすぎていて、PBR/ROE 改善圧力がある健全企業」を機械的に抽出する。

スコアリング (10点満点):
  PBR低 (0-3)        / 市場評価が低い
  ネットキャッシュ豊富 (0-3) / 還元圧力かかりやすい
  ROE低 (0-2)        / 自己資本効率改善余地
  還元余地 (0-2)     / 配当性向低 = 増配/自社株買い余力

ハードフィルタ (どれか1つでも該当 -> 除外):
  - 自己資本比率 < 50%       (財務余力不足)
  - 営業CF <= 0              (還元原資なし)
  - 純利益欠損 / 主要metrics欠損
"""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import (
    CAPITAL_EFFICIENCY_PARQUET,
    CAPITAL_EFFICIENCY_RAW_PARQUET,
    CES_DIVIDEND_YIELD_FALLBACK,
    CES_MIN_EQUITY_RATIO,
    CES_NETCASH_TIERS,
    CES_PAYOUT_TIERS,
    CES_PBR_TIERS,
    CES_ROE_SECONDARY_HI,
    CES_ROE_SECONDARY_LO,
    CES_ROE_SWEET_HI,
    CES_ROE_SWEET_LO,
)
from modules.naibu_client import fetch_jpx500_naibu_data
from modules.yfinance_financials_fetcher import (
    cache_yf_financials,
    fetch_all_yf_financials,
    load_yf_financials_cache,
)

logger = logging.getLogger(__name__)


def _coalesce(*vals: float | None) -> float | None:
    """最初の非NaN/非Noneを返す。"""
    for v in vals:
        if v is not None and pd.notna(v):
            return float(v)
    return None


def _tier_score(value: float | None, tiers: tuple[tuple[float, int], ...]) -> int:
    """値を上限閾値順 tiers と比較し最大配点を返す。

    tiers は (上限, 配点) の降順想定 (例: ((0.7,3),(1.0,2),(1.2,1)))。
    value < 上限 を順にチェック、最初に該当した点数を返す。
    """
    if value is None or pd.isna(value):
        return 0
    for upper, pts in tiers:
        if value < upper:
            return pts
    return 0


def _netcash_tier_score(value: float | None) -> int:
    """ネットキャッシュ/時価 は降順閾値 ((0.5,3),(0.3,2),(0.0,1))。

    value >= upper の最初に該当する点を返す。
    """
    if value is None or pd.isna(value):
        return 0
    for lower, pts in CES_NETCASH_TIERS:
        if value >= lower:
            return pts
    return 0


def _roe_tier_score(roe_pct: float | None) -> int:
    """ROE bin スコア。

    スイート(3-8%) -> 2点 / 副次(0-3% or 8-10%) -> 1点 / それ以外 -> 0
    """
    if roe_pct is None or pd.isna(roe_pct):
        return 0
    if CES_ROE_SWEET_LO <= roe_pct <= CES_ROE_SWEET_HI:
        return 2
    if CES_ROE_SECONDARY_LO < roe_pct < CES_ROE_SWEET_LO:
        return 1
    if CES_ROE_SWEET_HI < roe_pct < CES_ROE_SECONDARY_HI:
        return 1
    return 0


def _payout_tier_score(
    payout_ratio: float | None, dividend_yield_pct: float | None
) -> int:
    """還元余地スコア。配当性向優先、欠損時は dividend_yield フォールバック。"""
    if payout_ratio is not None and pd.notna(payout_ratio):
        return _tier_score(payout_ratio, CES_PAYOUT_TIERS)
    if dividend_yield_pct is not None and pd.notna(dividend_yield_pct):
        # 配当性向不明な場合: 低利回り = 還元余地あり
        if dividend_yield_pct < CES_DIVIDEND_YIELD_FALLBACK:
            return 1
    return 0


def compute_derived_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """naibu + yfinance + jpx500 を結合した DataFrame に派生指標列を追加。

    入力に期待する列 (欠損許容):
        naibu系:  total_assets, total_equity, cash, short_term_debt,
                  long_term_debt, operating_cf, net_income, retained_earnings
        yfinance系: total_assets_yf, total_equity_yf, cash_yf,
                  total_debt_yf, operating_cf_yf, net_income_yf,
                  dividend_yield, payout_ratio
        jpx500系: pbr, market_cap

    出力: equity_ratio, net_cash, net_cash_to_mcap, roe, op_cf
        + 各metricに使ったソース列 (source: naibu / yfinance / nan)
    """
    df = merged.copy()

    def col_or_none(c: str) -> pd.Series:
        return df[c] if c in df.columns else pd.Series([None] * len(df))

    # 各metric: naibu優先、欠損時 yfinance フォールバック
    df["total_assets_final"] = [
        _coalesce(a, b)
        for a, b in zip(col_or_none("total_assets"), col_or_none("total_assets_yf"))
    ]
    df["total_equity_final"] = [
        _coalesce(a, b)
        for a, b in zip(col_or_none("total_equity"), col_or_none("total_equity_yf"))
    ]
    df["cash_final"] = [
        _coalesce(a, b) for a, b in zip(col_or_none("cash"), col_or_none("cash_yf"))
    ]
    # 有利子負債: naibu は short+long, yfinance は total_debt
    naibu_debt: list[float | None] = []
    for s, lo in zip(col_or_none("short_term_debt"), col_or_none("long_term_debt")):
        if pd.notna(s) and pd.notna(lo):
            naibu_debt.append(float(s) + float(lo))
        elif pd.notna(s):
            naibu_debt.append(float(s))
        elif pd.notna(lo):
            naibu_debt.append(float(lo))
        else:
            naibu_debt.append(None)
    df["total_debt_final"] = [
        _coalesce(a, b) for a, b in zip(naibu_debt, col_or_none("total_debt_yf"))
    ]
    df["operating_cf_final"] = [
        _coalesce(a, b)
        for a, b in zip(col_or_none("operating_cf"), col_or_none("operating_cf_yf"))
    ]
    df["net_income_final"] = [
        _coalesce(a, b)
        for a, b in zip(col_or_none("net_income"), col_or_none("net_income_yf"))
    ]

    # 派生指標
    df["equity_ratio"] = df["total_equity_final"] / df["total_assets_final"]
    df["net_cash"] = df["cash_final"] - df["total_debt_final"].fillna(0)
    df["net_cash_to_mcap"] = df["net_cash"] / df["market_cap"]
    # ROE は %
    df["roe"] = df["net_income_final"] / df["total_equity_final"] * 100
    df["retained_earnings_to_mcap"] = df["retained_earnings"] / df["market_cap"]

    return df


def compute_capital_efficiency_score(row: pd.Series) -> dict[str, object]:
    """1銘柄について10点満点スコアと内訳を返す。"""
    # ハードフィルタ
    hard_fail_reason = None
    equity_ratio = row.get("equity_ratio")
    op_cf = row.get("operating_cf_final")
    net_income = row.get("net_income_final")

    if equity_ratio is None or pd.isna(equity_ratio):
        hard_fail_reason = "equity_ratio欠損"
    elif equity_ratio < CES_MIN_EQUITY_RATIO:
        hard_fail_reason = f"自己資本比率{equity_ratio:.2f}<{CES_MIN_EQUITY_RATIO}"
    elif op_cf is None or pd.isna(op_cf):
        hard_fail_reason = "営業CF欠損"
    elif op_cf <= 0:
        hard_fail_reason = "営業CFマイナス"
    elif net_income is None or pd.isna(net_income) or net_income <= 0:
        hard_fail_reason = "純利益欠損/赤字"

    if hard_fail_reason is not None:
        return {
            "pbr_score": 0,
            "netcash_score": 0,
            "roe_score": 0,
            "payout_score": 0,
            "score": 0,
            "hard_filter_failed": True,
            "hard_fail_reason": hard_fail_reason,
        }

    pbr_score = _tier_score(row.get("pbr"), CES_PBR_TIERS)
    netcash_score = _netcash_tier_score(row.get("net_cash_to_mcap"))
    roe_score = _roe_tier_score(row.get("roe"))
    payout_score = _payout_tier_score(
        row.get("payout_ratio"), row.get("dividend_yield")
    )

    return {
        "pbr_score": pbr_score,
        "netcash_score": netcash_score,
        "roe_score": roe_score,
        "payout_score": payout_score,
        "score": pbr_score + netcash_score + roe_score + payout_score,
        "hard_filter_failed": False,
        "hard_fail_reason": None,
    }


def run_screening(
    results_df: pd.DataFrame,
    use_yf_cache: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """JPX500銘柄の完全スクリーニング実行。

    1. naibu から JPX500 財務 + retained_earnings 取得
    2. yfinance から欠損補完用に balance_sheet/cashflow/info 取得 (キャッシュ可)
    3. results.csv (pbr, market_cap, sector_33, size_category) と結合
    4. compute_derived_metrics で派生指標計算
    5. compute_capital_efficiency_score を全行に適用
    6. score 降順でソート、parquet 保存

    Args:
        results_df: data/results.csv をロードしたもの (code, name, pbr, market_cap,
                    sector_33, size_category 必須)
        use_yf_cache: True なら data/capital_efficiency_raw.parquet を再利用
        progress_callback: (i, total, ticker) を受ける任意のコールバック

    Returns:
        スクリーニング結果 DataFrame (score降順)
    """
    # 1) naibu データ取得
    logger.info("naibu DB から JPX500 財務を取得中...")
    naibu_df = fetch_jpx500_naibu_data()
    if naibu_df.empty:
        logger.warning("naibu データ取得失敗、yfinance のみで継続")
        naibu_df = pd.DataFrame(columns=["code"])

    # 2) yfinance 財務取得 (キャッシュ優先)
    yf_df: pd.DataFrame | None = None
    if use_yf_cache:
        yf_df = load_yf_financials_cache(CAPITAL_EFFICIENCY_RAW_PARQUET)
    if yf_df is None or yf_df.empty:
        logger.info(f"yfinance から {len(results_df)}銘柄 の財務を取得中...")
        tickers = [
            f"{str(c).strip()}.T" for c in results_df["code"].astype(str).unique()
        ]
        yf_df = fetch_all_yf_financials(
            tickers, sleep_sec=0.0, progress_callback=progress_callback
        )
        cache_yf_financials(yf_df, CAPITAL_EFFICIENCY_RAW_PARQUET)
    yf_df = yf_df.copy()
    yf_df["code"] = yf_df["ticker"].str.replace(".T", "", regex=False)

    # 3) results.csv (pbr, market_cap, sector_33, size_category) と結合
    base = results_df[
        ["code", "name", "sector_33", "size_category", "pbr", "market_cap"]
    ].copy()
    base["code"] = base["code"].astype(str)
    base["pbr"] = pd.to_numeric(base["pbr"], errors="coerce")
    base["market_cap"] = pd.to_numeric(base["market_cap"], errors="coerce")
    # ETFを除外
    base = base[base["size_category"] != "ETF"].reset_index(drop=True)

    merged = base.merge(
        naibu_df.drop(columns=[c for c in naibu_df.columns if c.endswith("_fy")]),
        on="code",
        how="left",
        suffixes=("", "_naibu"),
    )
    merged = merged.merge(yf_df.drop(columns=["ticker"]), on="code", how="left")

    # 4) 派生指標
    merged = compute_derived_metrics(merged)

    # 5) スコア計算
    score_rows = [compute_capital_efficiency_score(r) for _, r in merged.iterrows()]
    score_df = pd.DataFrame(score_rows)
    out = pd.concat([merged.reset_index(drop=True), score_df], axis=1)

    # 6) 出力列整形
    final_cols = [
        "code",
        "name",
        "sector_33",
        "size_category",
        "score",
        "pbr",
        "roe",
        "equity_ratio",
        "net_cash",
        "net_cash_to_mcap",
        "payout_ratio",
        "dividend_yield",
        "operating_cf_final",
        "net_income_final",
        "total_equity_final",
        "total_assets_final",
        "market_cap",
        "retained_earnings",
        "retained_earnings_to_mcap",
        "fiscal_year",
        "fiscal_year_yf",
        "pbr_score",
        "netcash_score",
        "roe_score",
        "payout_score",
        "hard_filter_failed",
        "hard_fail_reason",
    ]
    available = [c for c in final_cols if c in out.columns]
    out = out[available]

    out = out.sort_values(
        ["hard_filter_failed", "score", "net_cash_to_mcap"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    # parquet 保存
    out.to_parquet(CAPITAL_EFFICIENCY_PARQUET, index=False)
    logger.info(
        f"スクリーニング完了: {len(out)}件 "
        f"(score>=6: {(out['score'] >= 6).sum()}件 / "
        f"hard_filter_failed: {out['hard_filter_failed'].sum()}件)"
    )
    return out


def load_screening_result() -> pd.DataFrame | None:
    """parquetキャッシュをロード。存在しなければNone。"""
    if not CAPITAL_EFFICIENCY_PARQUET.exists():
        return None
    try:
        return pd.read_parquet(CAPITAL_EFFICIENCY_PARQUET)
    except Exception as e:
        logger.warning(f"CES結果parquet読込失敗: {e}")
        return None
