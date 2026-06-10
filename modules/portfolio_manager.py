"""ペーパートレード方式のポートフォリオ管理。

portfolio.csv に仮想保有を記録し、毎日の終値で評価額を計算する。
売買は trades.csv に履歴を保持。日次評価額は portfolio_history.parquet に追記。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    INITIAL_CAPITAL_JPY,
    PORTFOLIO_CSV,
    PORTFOLIO_HISTORY_PARQUET,
    PORTFOLIO_INITIAL_CSV,
    PORTFOLIO_TRADES_CSV,
)
from modules.data_fetcher import get_fx_to_jpy_daily, load_cached

logger = logging.getLogger(__name__)


# ---------- スキーマ ----------

PORTFOLIO_COLS = [
    "code",
    "name",
    "ticker",
    "shares",
    "avg_cost",
    "currency",
    "entry_date",
    "category",  # core / satellite / cash
]

TRADES_COLS = [
    "trade_id",
    "date",
    "code",
    "ticker",
    "side",  # BUY / SELL
    "shares",
    "price",
    "currency",
    "fee",
    "notes",
]


# ---------- ロード / セーブ ----------


def load_portfolio() -> pd.DataFrame:
    """portfolio.csv を読込。存在しなければ空 DataFrame を返す。"""
    if not Path(PORTFOLIO_CSV).exists():
        return pd.DataFrame(columns=PORTFOLIO_COLS)
    df = pd.read_csv(PORTFOLIO_CSV, encoding="utf-8-sig", dtype={"code": str})
    for c in PORTFOLIO_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[PORTFOLIO_COLS]


def save_portfolio(df: pd.DataFrame) -> None:
    """portfolio.csv に書き出し。"""
    Path(PORTFOLIO_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PORTFOLIO_CSV, index=False, encoding="utf-8-sig")


def load_trades() -> pd.DataFrame:
    if not Path(PORTFOLIO_TRADES_CSV).exists():
        return pd.DataFrame(columns=TRADES_COLS)
    return pd.read_csv(PORTFOLIO_TRADES_CSV, encoding="utf-8-sig", dtype={"code": str})


def _append_trade(row: dict) -> None:
    df = load_trades()
    new_row_df = pd.DataFrame([row])
    if df.empty:
        df = new_row_df
    else:
        df = pd.concat([df, new_row_df], ignore_index=True)
    Path(PORTFOLIO_TRADES_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PORTFOLIO_TRADES_CSV, index=False, encoding="utf-8-sig")


# ---------- 操作 ----------


def add_position(
    code: str,
    name: str,
    ticker: str,
    shares: float,
    cost: float,
    currency: str = "JPY",
    category: str = "satellite",
    date: str | None = None,
    notes: str = "",
) -> pd.DataFrame:
    """新規買い。既存ポジションがあれば加重平均で avg_cost を更新。"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    portfolio = load_portfolio()

    existing = portfolio[portfolio["code"].astype(str) == str(code)]
    if len(existing) > 0:
        idx = existing.index[0]
        old_shares = float(portfolio.at[idx, "shares"])
        old_cost = float(portfolio.at[idx, "avg_cost"])
        new_shares = old_shares + shares
        new_avg = (
            (old_shares * old_cost + shares * cost) / new_shares
            if new_shares > 0
            else cost
        )
        portfolio.at[idx, "shares"] = new_shares
        portfolio.at[idx, "avg_cost"] = new_avg
    else:
        new_row = {
            "code": str(code),
            "name": name,
            "ticker": ticker,
            "shares": shares,
            "avg_cost": cost,
            "currency": currency,
            "entry_date": date,
            "category": category,
        }
        new_df = pd.DataFrame([new_row])
        if portfolio.empty:
            portfolio = new_df
        else:
            portfolio = pd.concat([portfolio, new_df], ignore_index=True)

    save_portfolio(portfolio)
    _append_trade(
        {
            "trade_id": f"T{int(datetime.now().timestamp())}",
            "date": date,
            "code": str(code),
            "ticker": ticker,
            "side": "BUY",
            "shares": shares,
            "price": cost,
            "currency": currency,
            "fee": 0,
            "notes": notes,
        }
    )
    return portfolio


def record_sell(
    code: str,
    shares: float,
    price: float,
    date: str | None = None,
    notes: str = "",
) -> pd.DataFrame:
    """売却(部分売却対応)。残株 0 ならポジション削除。avg_cost は維持。"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    portfolio = load_portfolio()
    existing = portfolio[portfolio["code"].astype(str) == str(code)]
    if len(existing) == 0:
        raise ValueError(f"{code}: ポジションが存在しません")

    idx = existing.index[0]
    cur_shares = float(portfolio.at[idx, "shares"])
    if shares > cur_shares + 1e-6:
        raise ValueError(
            f"{code}: 売却株数{shares} が保有株数{cur_shares} を超えています"
        )
    new_shares = cur_shares - shares
    ticker = portfolio.at[idx, "ticker"]
    currency = portfolio.at[idx, "currency"]
    if new_shares < 1e-6:
        portfolio = portfolio.drop(idx).reset_index(drop=True)
    else:
        portfolio.at[idx, "shares"] = new_shares

    save_portfolio(portfolio)
    _append_trade(
        {
            "trade_id": f"T{int(datetime.now().timestamp())}",
            "date": date,
            "code": str(code),
            "ticker": ticker,
            "side": "SELL",
            "shares": shares,
            "price": price,
            "currency": currency,
            "fee": 0,
            "notes": notes,
        }
    )
    return portfolio


def initialize_from_template(
    template_csv: Path | None = None, default_prices: dict | None = None
) -> pd.DataFrame:
    """portfolio_initial.csv から仮想ポジションを一括投入する。

    各銘柄について最新終値(キャッシュ)を取得し、target_jpy を満たす株数を
    avg_cost = 最新終値で計算してポジション化する。CASH 行は現金として保持。
    """
    template_csv = template_csv or PORTFOLIO_INITIAL_CSV
    if not Path(template_csv).exists():
        raise FileNotFoundError(f"{template_csv} がありません")

    tpl = pd.read_csv(template_csv, encoding="utf-8-sig", dtype={"code": str})
    default_prices = default_prices or {}

    # 既存をクリアして上書き(初期化用途)
    Path(PORTFOLIO_CSV).unlink(missing_ok=True)
    Path(PORTFOLIO_TRADES_CSV).unlink(missing_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")

    for _, row in tpl.iterrows():
        ticker = str(row["ticker"])
        category = str(row.get("category", "satellite"))
        target_jpy = float(row.get("target_jpy", 0))
        name = str(row.get("name", ticker))
        code = str(row.get("code", ticker))

        if category == "cash" or ticker == "CASH":
            # キャッシュは特別ポジション(shares=現金額, avg_cost=1, currency=JPY)
            add_position(
                code="CASH",
                name=name,
                ticker="CASH",
                shares=target_jpy,
                cost=1.0,
                currency="JPY",
                category="cash",
                date=today,
                notes="初期キャッシュ",
            )
            continue

        # ティッカーから通貨を推定
        currency = _infer_currency(ticker)
        # 最新終値を取得
        price_local = default_prices.get(ticker)
        if price_local is None:
            df = load_cached(ticker)
            if df is None or df.empty:
                logger.warning(
                    f"{ticker}: キャッシュなし、初期化スキップ(extended_fetcher を先に実行してください)"
                )
                continue
            price_local = float(df["Close"].iloc[-1])

        # 円換算で target_jpy 満たす株数を計算
        if currency == "JPY":
            jpy_per_share = price_local
        else:
            fx = get_fx_to_jpy_daily(currency)
            if fx is None or fx.empty:
                logger.warning(f"{ticker}: FX 取得失敗、スキップ")
                continue
            jpy_per_share = price_local * float(fx.iloc[-1])

        if jpy_per_share <= 0:
            continue
        shares = round(target_jpy / jpy_per_share, 4)

        add_position(
            code=code,
            name=name,
            ticker=ticker,
            shares=shares,
            cost=price_local,
            currency=currency,
            category=category,
            date=today,
            notes="初期投入",
        )

    return load_portfolio()


# ---------- 評価額計算 ----------


def _infer_currency(ticker: str) -> str:
    """ticker から通貨を推定: .T -> JPY, それ以外で英大文字のみ -> USD"""
    if ticker == "CASH":
        return "JPY"
    if str(ticker).upper().endswith(".T"):
        return "JPY"
    return "USD"


def _get_latest_price(ticker: str) -> tuple[float | None, str | None]:
    """ticker の最新終値 (現地通貨) と最新日付を返す。"""
    if ticker == "CASH":
        return 1.0, datetime.now().strftime("%Y-%m-%d")
    df = load_cached(ticker)
    if df is None or df.empty:
        return None, None
    last_price = float(df["Close"].iloc[-1])
    last_date = df.index[-1].strftime("%Y-%m-%d")
    return last_price, last_date


def _get_fx_rate(currency: str) -> float:
    """currency -> JPY の最新レート。JPY は 1.0。"""
    if currency == "JPY":
        return 1.0
    fx = get_fx_to_jpy_daily(currency)
    if fx is None or fx.empty:
        logger.warning(f"FX レート取得失敗: {currency}, 暫定 150.0 を使用")
        return 150.0
    return float(fx.iloc[-1])


def compute_current_valuation() -> dict:
    """全銘柄の最新終値で時価評価し、サマリ + 銘柄明細 dict を返す。

    Returns: {
        total_value, core_value, satellite_value, cash, pnl_jpy, pnl_pct,
        latest_date, by_holding: [{code, name, shares, avg_cost, last_price,
            value_jpy, pnl_pct, weight_pct, category}, ...]
    }
    """
    portfolio = load_portfolio()
    if portfolio.empty:
        return {
            "total_value": 0.0,
            "core_value": 0.0,
            "satellite_value": 0.0,
            "cash": 0.0,
            "pnl_jpy": 0.0,
            "pnl_pct": 0.0,
            "latest_date": None,
            "by_holding": [],
        }

    # FX キャッシュ(同一通貨を何度も取らない)
    fx_cache: dict[str, float] = {"JPY": 1.0}

    from typing import Any

    rows: list[dict[str, Any]] = []
    latest_dates: list[str] = []
    for _, p in portfolio.iterrows():
        ticker = str(p["ticker"])
        shares = float(p["shares"])
        avg_cost = float(p["avg_cost"])
        currency = str(p["currency"])
        category = str(p["category"])

        last_price, last_date = _get_latest_price(ticker)
        if last_price is None:
            logger.warning(f"{ticker}: 終値取得失敗")
            last_price = avg_cost  # フォールバック
            last_date = None
        if last_date:
            latest_dates.append(last_date)

        if currency not in fx_cache:
            fx_cache[currency] = _get_fx_rate(currency)
        fx = fx_cache[currency]

        cost_jpy = shares * avg_cost * fx if category != "cash" else shares
        value_jpy = shares * last_price * fx if category != "cash" else shares
        pnl_jpy = value_jpy - cost_jpy
        pnl_pct = (pnl_jpy / cost_jpy * 100) if cost_jpy > 0 else 0.0

        rows.append(
            {
                "code": str(p["code"]),
                "name": str(p["name"]),
                "ticker": ticker,
                "category": category,
                "shares": shares,
                "avg_cost": avg_cost,
                "last_price": last_price,
                "currency": currency,
                "fx": fx,
                "cost_jpy": cost_jpy,
                "value_jpy": value_jpy,
                "pnl_jpy": pnl_jpy,
                "pnl_pct": pnl_pct,
                "last_date": last_date,
            }
        )

    total_value = float(sum(float(r["value_jpy"]) for r in rows))
    core_value = float(
        sum(float(r["value_jpy"]) for r in rows if r["category"] == "core")
    )
    satellite_value = float(
        sum(float(r["value_jpy"]) for r in rows if r["category"] == "satellite")
    )
    cash = float(sum(float(r["value_jpy"]) for r in rows if r["category"] == "cash"))
    total_cost = float(sum(float(r["cost_jpy"]) for r in rows))
    pnl_jpy = total_value - total_cost
    pnl_pct = (pnl_jpy / total_cost * 100) if total_cost > 0 else 0.0

    # weight_pct を計算
    for r in rows:
        r["weight_pct"] = (
            (float(r["value_jpy"]) / total_value * 100) if total_value > 0 else 0.0
        )

    return {
        "total_value": total_value,
        "core_value": core_value,
        "satellite_value": satellite_value,
        "cash": cash,
        "total_cost": total_cost,
        "pnl_jpy": pnl_jpy,
        "pnl_pct": pnl_pct,
        "latest_date": max(latest_dates) if latest_dates else None,
        "by_holding": rows,
    }


# ---------- 履歴更新 ----------


def update_portfolio_history() -> pd.DataFrame:
    """当日評価額を portfolio_history.parquet に追記。同日重複なら上書き。"""
    val = compute_current_valuation()
    if val["total_value"] == 0 and val["cash"] == 0:
        logger.info("ポートフォリオ未設定のため履歴更新スキップ")
        return pd.DataFrame()

    today = datetime.now().strftime("%Y-%m-%d")
    snapshot = {
        "date": today,
        "total_value_jpy": val["total_value"],
        "core_value_jpy": val["core_value"],
        "satellite_value_jpy": val["satellite_value"],
        "cash_jpy": val["cash"],
        "total_cost_jpy": val["total_cost"],
        "pnl_jpy": val["pnl_jpy"],
        "pnl_pct": val["pnl_pct"],
    }

    history_path = Path(PORTFOLIO_HISTORY_PARQUET)
    if history_path.exists():
        hist = pd.read_parquet(history_path)
        hist = hist[hist["date"] != today]
        hist = pd.concat([hist, pd.DataFrame([snapshot])], ignore_index=True)
    else:
        hist = pd.DataFrame([snapshot])

    hist = hist.sort_values("date").reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(history_path, index=False)
    logger.info(f"ポートフォリオ履歴更新: {today} → {history_path}")
    return hist


def load_portfolio_history() -> pd.DataFrame:
    if not Path(PORTFOLIO_HISTORY_PARQUET).exists():
        return pd.DataFrame()
    return pd.read_parquet(PORTFOLIO_HISTORY_PARQUET)


# ---------- パフォーマンス指標 ----------


def compute_performance_metrics() -> dict:
    """累積リターン、年率換算、最大DD、シャープ等を返す。

    INITIAL_CAPITAL_JPY を基準にした絶対パフォーマンスを使う。
    """
    hist = load_portfolio_history()
    if hist.empty or len(hist) < 2:
        val = compute_current_valuation()
        if val["total_value"] > 0:
            cum = (val["total_value"] - INITIAL_CAPITAL_JPY) / INITIAL_CAPITAL_JPY * 100
            return {
                "cumulative_return_pct": cum,
                "annualized_return_pct": None,
                "max_drawdown_pct": None,
                "sharpe": None,
                "days": 0,
            }
        return {
            "cumulative_return_pct": 0.0,
            "annualized_return_pct": None,
            "max_drawdown_pct": None,
            "sharpe": None,
            "days": 0,
        }

    hist = hist.copy().sort_values("date")
    hist["date"] = pd.to_datetime(hist["date"])

    initial = INITIAL_CAPITAL_JPY
    final = float(hist["total_value_jpy"].iloc[-1])
    cum_ret = (final - initial) / initial * 100

    days = (hist["date"].iloc[-1] - hist["date"].iloc[0]).days
    years = max(days / 365.25, 1 / 365.25)
    annualized = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 else None

    # 最大DD
    cummax = hist["total_value_jpy"].cummax()
    dd = (hist["total_value_jpy"] - cummax) / cummax * 100
    max_dd = float(dd.min()) if len(dd) else None

    # シャープ(簡易、無リスク金利 0)
    rets = hist["total_value_jpy"].pct_change().dropna()
    sharpe = (
        float(rets.mean() / rets.std() * np.sqrt(252))
        if len(rets) > 1 and rets.std() > 0
        else None
    )

    return {
        "cumulative_return_pct": cum_ret,
        "annualized_return_pct": annualized,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "days": days,
        "initial_capital_jpy": initial,
        "final_value_jpy": final,
    }
