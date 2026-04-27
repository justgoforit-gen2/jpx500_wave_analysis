from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.backtester import build_contexts, run_backtest
from modules.data_fetcher import load_cached
from modules.strategy_engine import fetch_eps_data
from modules.strategy_loader import load_strategy


def _load_stock_list() -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    csv_path = data_dir / "jpx500_list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing stock list: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ["code", "name", "ticker"]:
        if col not in df.columns:
            raise ValueError(f"jpx500_list.csv must include '{col}'")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly backtest -> CSV outputs")
    ap.add_argument(
        "--years",
        type=int,
        default=None,
        help="Lookback years (default: strategy.yaml evaluation.lookback_years or 3)",
    )
    ap.add_argument(
        "--initial", type=float, default=10_000_000.0, help="Initial capital (JPY)"
    )
    ap.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Max positions (default: strategy.yaml execution.max_positions or 20)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tickers (debug/smoke test)",
    )
    ap.add_argument(
        "--with-eps",
        action="store_true",
        help="Fetch EPS data from yfinance and apply C/A conditions to E_can_slim",
    )
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    strategy = load_strategy(project_dir / "config" / "strategy.yaml")
    lookback_years = args.years
    if lookback_years is None:
        lookback_years = int(strategy.get("evaluation", {}).get("lookback_years", 3))

    max_positions = args.max_positions
    if max_positions is None:
        max_positions = int(strategy.get("execution", {}).get("max_positions", 20))

    # benchmark to define trading days
    bench = load_cached("^N225")
    if bench is None or len(bench) < 250:
        raise RuntimeError(
            "Missing benchmark cache for ^N225. Run daily update first to populate data/cache."
        )
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)
    bench.sort_index(inplace=True)
    end_date = pd.Timestamp(bench.index.max())
    start_date = end_date - pd.DateOffset(years=lookback_years)
    trading_days = pd.DatetimeIndex(bench.index)

    stock_list_df = _load_stock_list()

    # EPS フィルタ (--with-eps 指定時のみ)
    eps_eligible: set[str] | None = None
    if args.with_eps:
        tickers = stock_list_df["ticker"].tolist()
        if args.limit:
            tickers = tickers[: args.limit]
        print(f"[EPS] Fetching EPS data for {len(tickers)} tickers ...")
        eps_eligible = set()
        e_cfg = strategy.get("patterns", {}).get("E_can_slim", {})
        qtr_thresh = 0.25
        annual_thresh = 0.25
        for cond in e_cfg.get("entry", []):
            if isinstance(cond, dict) and "eps_qtr_yoy_growth_ge" in cond:
                qtr_thresh = cond["eps_qtr_yoy_growth_ge"]
            if isinstance(cond, dict) and "eps_annual_growth_3y_ge" in cond:
                annual_thresh = cond["eps_annual_growth_3y_ge"]
        for i, tkr in enumerate(tickers):
            eps = fetch_eps_data(tkr)
            if not eps.get("ok"):
                continue
            qg = eps.get("eps_qtr_yoy_growth")
            ag = eps.get("eps_annual_growth_3y", [])
            # 四半期EPS が None の場合は日本株でデータ未取得のためスキップ
            if qg is not None and qg < qtr_thresh:
                continue
            if len(ag) < 3 or not all(
                g is not None and g >= annual_thresh for g in ag[:3]
            ):
                continue
            eps_eligible.add(tkr)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tickers)} done, eligible={len(eps_eligible)}")
        print(f"[EPS] Eligible tickers: {len(eps_eligible)}")

    contexts = build_contexts(
        stock_list_df=stock_list_df,
        load_cached_fn=load_cached,
        strategy=strategy,
        end_date=end_date,
        limit=args.limit,
        eps_eligible=eps_eligible,
    )
    if len(contexts) == 0:
        raise RuntimeError("No usable ticker contexts built (check cache & stock list)")

    summaries = []
    trades_all = []
    equity_all = []

    for policy in ["fixed_amount", "fixed_rate"]:
        summary_df, trades_df, equity_df = run_backtest(
            contexts=contexts,
            trading_days=trading_days,
            start_date=pd.Timestamp(start_date),
            end_date=end_date,
            strategy=strategy,
            policy=policy,
            initial_capital=float(args.initial),
            max_positions=max_positions,
        )
        summaries.append(summary_df)
        trades_all.append(trades_df)
        equity_all.append(equity_df)

    summary = pd.concat(summaries, ignore_index=True)
    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    equity = pd.concat(equity_all, ignore_index=True) if equity_all else pd.DataFrame()

    # monthly performance (based on equity curve + closed trades)
    if len(equity) > 0:
        equity2 = equity.copy()
        equity2["date"] = pd.to_datetime(equity2["date"])
        equity2.sort_values(["policy", "date"], inplace=True)
        equity2["ym"] = equity2["date"].dt.to_period("M").astype(str)

        monthly_equity = (
            equity2.groupby(["policy", "ym"], dropna=False)
            .agg(
                month_start_equity=("equity", "first"),
                month_end_equity=("equity", "last"),
            )
            .reset_index()
        )
        monthly_equity["month_return_pct"] = (
            monthly_equity["month_end_equity"] / monthly_equity["month_start_equity"]
            - 1.0
        ) * 100.0
    else:
        monthly_equity = pd.DataFrame(
            columns=[
                "policy",
                "ym",
                "month_start_equity",
                "month_end_equity",
                "month_return_pct",
            ]
        )

    if len(trades) > 0:
        trades2 = trades.copy()
        trades2["exit_date"] = pd.to_datetime(trades2["exit_date"], errors="coerce")
        trades2["ym"] = trades2["exit_date"].dt.to_period("M").astype(str)
        trades2["is_win"] = trades2["pnl"] > 0
        trades2["is_loss"] = trades2["pnl"] < 0

        monthly_trades = (
            trades2.groupby(["policy", "ym"], dropna=False)
            .agg(
                trades_closed=("ticker", "size"),
                wins=("is_win", "sum"),
                losses=("is_loss", "sum"),
                win_rate=("is_win", "mean"),
                total_pnl=("pnl", "sum"),
            )
            .reset_index()
        )

        # exit reason breakdown -> wide columns
        if "exit_reason" in trades2.columns:
            reason_counts = trades2.pivot_table(
                index=["policy", "ym"],
                columns="exit_reason",
                values="ticker",
                aggfunc="size",
                fill_value=0,
            ).reset_index()
        else:
            reason_counts = pd.DataFrame(columns=["policy", "ym"])  # no-op
    else:
        monthly_trades = pd.DataFrame(
            columns=[
                "policy",
                "ym",
                "trades_closed",
                "wins",
                "losses",
                "win_rate",
                "total_pnl",
            ]
        )
        reason_counts = pd.DataFrame(columns=["policy", "ym"])  # no-op

    monthly = monthly_equity.merge(monthly_trades, on=["policy", "ym"], how="left")
    if len(reason_counts) > 0:
        monthly = monthly.merge(reason_counts, on=["policy", "ym"], how="left")

    # Always present these exit columns
    for col in ["time_exit", "trailing_atr", "trend_exit", "rebalance_drop"]:
        if col not in monthly.columns:
            monthly[col] = 0

    for col in ["trades_closed", "wins", "losses"]:
        if col in monthly.columns:
            monthly[col] = monthly[col].fillna(0).astype(int)

    for col in ["time_exit", "trailing_atr", "trend_exit", "rebalance_drop"]:
        monthly[col] = monthly[col].fillna(0).astype(int)
    if "win_rate" in monthly.columns:
        monthly["win_rate"] = monthly["win_rate"].fillna(0.0)
    if "total_pnl" in monthly.columns:
        monthly["total_pnl"] = monthly["total_pnl"].fillna(0.0)
    if "month_return_pct" in monthly.columns:
        monthly["month_return_pct"] = monthly["month_return_pct"].fillna(0.0)

    # per-symbol aggregates
    if len(trades) > 0:
        per_symbol = (
            trades.groupby(
                ["policy", "code", "name", "ticker", "pattern"], dropna=False
            )
            .agg(
                trade_count=("ticker", "size"),
                total_pnl=("pnl", "sum"),
                avg_return_pct=("return_pct", "mean"),
                win_rate=("pnl", lambda s: float((s > 0).mean())),
            )
            .reset_index()
            .sort_values(["policy", "total_pnl"], ascending=[True, False])
        )
    else:
        per_symbol = pd.DataFrame(
            columns=[
                "policy",
                "code",
                "name",
                "ticker",
                "pattern",
                "trade_count",
                "total_pnl",
                "avg_return_pct",
                "win_rate",
            ]
        )

    summary_path = data_dir / "backtest_summary.csv"
    trades_path = data_dir / "backtest_trades.csv"
    equity_path = data_dir / "backtest_equity_curve.csv"
    per_symbol_path = data_dir / "backtest_by_symbol.csv"
    monthly_path = data_dir / "backtest_monthly.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    per_symbol.to_csv(per_symbol_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    print("Wrote:")
    print(f"- {summary_path}")
    print(f"- {equity_path}")
    print(f"- {trades_path}")
    print(f"- {per_symbol_path}")
    print(f"- {monthly_path}")


if __name__ == "__main__":
    main()
