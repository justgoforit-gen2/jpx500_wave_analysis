from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.backtester import build_contexts, run_backtest
from modules.data_fetcher import load_cached
from modules.strategy_loader import load_strategy


@dataclass(frozen=True)
class Scenario:
    name: str
    entry_rsi_max: float | None
    trailing_override_mult: float | None
    c_breakout_rsi_max: float | None


def _load_stock_list(project_dir: Path) -> pd.DataFrame:
    csv_path = project_dir / "data" / "jpx500_list.csv"
    df = pd.read_csv(csv_path)
    return df


def _monthly_from_equity_and_trades(
    equity: pd.DataFrame, trades: pd.DataFrame
) -> pd.DataFrame:
    if len(equity) == 0:
        return pd.DataFrame()
    equity2 = equity.copy()
    equity2["date"] = pd.to_datetime(equity2["date"])
    equity2.sort_values(["policy", "date"], inplace=True)
    equity2["ym"] = equity2["date"].dt.to_period("M").astype(str)

    monthly_equity = (
        equity2.groupby(["policy", "ym"], dropna=False)
        .agg(
            month_start_equity=("equity", "first"), month_end_equity=("equity", "last")
        )
        .reset_index()
    )
    monthly_equity["month_return_pct"] = (
        monthly_equity["month_end_equity"] / monthly_equity["month_start_equity"] - 1.0
    ) * 100.0

    if len(trades) == 0:
        monthly = monthly_equity.copy()
        for col in [
            "trades_closed",
            "wins",
            "losses",
            "win_rate",
            "total_pnl",
            "rebalance_drop",
            "time_exit",
            "trailing_atr",
            "trend_exit",
        ]:
            monthly[col] = 0
        return monthly

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

    if "exit_reason" in trades2.columns:
        reason_counts = trades2.pivot_table(
            index=["policy", "ym"],
            columns="exit_reason",
            values="ticker",
            aggfunc="size",
            fill_value=0,
        ).reset_index()
    else:
        reason_counts = pd.DataFrame(columns=["policy", "ym"])

    monthly = monthly_equity.merge(monthly_trades, on=["policy", "ym"], how="left")
    if len(reason_counts) > 0:
        monthly = monthly.merge(reason_counts, on=["policy", "ym"], how="left")

    for col in ["rebalance_drop", "time_exit", "trailing_atr", "trend_exit"]:
        if col not in monthly.columns:
            monthly[col] = 0

    for col in [
        "trades_closed",
        "wins",
        "losses",
        "rebalance_drop",
        "time_exit",
        "trailing_atr",
        "trend_exit",
    ]:
        monthly[col] = monthly[col].fillna(0).astype(int)
    monthly["win_rate"] = monthly["win_rate"].fillna(0.0)
    monthly["total_pnl"] = monthly["total_pnl"].fillna(0.0)

    return monthly


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run scenario grid for (1)(2)(3) improvements"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Limit tickers for smoke test"
    )
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--initial", type=float, default=10_000_000.0)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument(
        "--positions-grid",
        type=str,
        default=None,
        help="Comma-separated max_positions grid (e.g. '5,10,20'). If omitted, uses --max-positions or strategy default.",
    )
    ap.add_argument(
        "--regime-grid",
        type=str,
        default="none",
        help="Comma-separated regime grid. Supported: none, n225_sma200",
    )
    ap.add_argument("--high-rsi-threshold", type=float, default=70.0)
    ap.add_argument("--trailing-mult-high", type=float, default=2.5)
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    out_root = data_dir / "scenarios"
    out_root.mkdir(parents=True, exist_ok=True)

    strategy = load_strategy(project_dir / "config" / "strategy.yaml")
    lookback_years = (
        int(args.years)
        if args.years is not None
        else int(strategy.get("evaluation", {}).get("lookback_years", 3))
    )
    max_positions_default = (
        int(args.max_positions)
        if args.max_positions is not None
        else int(strategy.get("execution", {}).get("max_positions", 20))
    )
    if args.positions_grid:
        try:
            max_positions_grid = [
                int(x.strip()) for x in str(args.positions_grid).split(",") if x.strip()
            ]
        except Exception:
            raise ValueError(
                "Invalid --positions-grid. Use comma-separated integers like '5,10,20'."
            )
        if not max_positions_grid:
            raise ValueError("Invalid --positions-grid. Provide at least one integer.")
    else:
        max_positions_grid = [max_positions_default]

    bench = load_cached("^N225")
    if bench is None or len(bench) < 250:
        raise RuntimeError("Missing benchmark cache for ^N225")
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)
    bench.sort_index(inplace=True)

    close_col = (
        "Close"
        if "Close" in bench.columns
        else ("close" if "close" in bench.columns else None)
    )
    if close_col is None:
        raise RuntimeError("Benchmark cache for ^N225 has no Close column")

    bench_close = pd.to_numeric(bench[close_col], errors="coerce")
    bench_sma200 = bench_close.rolling(200, min_periods=200).mean()
    regime_n225_sma200 = (bench_close > bench_sma200).fillna(True)

    regimes = [x.strip() for x in str(args.regime_grid).split(",") if x.strip()]
    supported = {"none", "n225_sma200"}
    unknown = [x for x in regimes if x not in supported]
    if unknown:
        raise ValueError(
            f"Unknown regime(s): {unknown}. Supported: {sorted(supported)}"
        )
    trading_days = pd.DatetimeIndex(bench.index)
    end_date = pd.Timestamp(bench.index.max())
    start_date = end_date - pd.DateOffset(years=lookback_years)

    stock_list_df = _load_stock_list(project_dir)
    contexts = build_contexts(
        stock_list_df=stock_list_df,
        load_cached_fn=load_cached,
        strategy=strategy,
        end_date=end_date,
        limit=args.limit,
    )
    if len(contexts) == 0:
        raise RuntimeError("No usable ticker contexts built")

    # Define 12 scenarios:
    # - baseline, (2), (3), (2+3)
    # - (1@70) with combos of (2)(3)
    # - (1@65) with combos of (2)(3)
    scenarios: list[Scenario] = []
    scenarios.append(Scenario("S0_baseline", None, None, None))
    scenarios.append(
        Scenario("S2_trailingHighRSI", None, float(args.trailing_mult_high), None)
    )
    scenarios.append(
        Scenario(
            "S3_suppressCbreakoutHighRSI", None, None, float(args.high_rsi_threshold)
        )
    )
    scenarios.append(
        Scenario(
            "S2S3_trailing+suppressC",
            None,
            float(args.trailing_mult_high),
            float(args.high_rsi_threshold),
        )
    )

    for thr in [70.0, 65.0]:
        tlabel = f"{int(thr)}"
        scenarios.append(Scenario(f"S1_rsiMax{tlabel}", thr, None, None))
        scenarios.append(
            Scenario(
                f"S1S2_rsiMax{tlabel}+trailing",
                thr,
                float(args.trailing_mult_high),
                None,
            )
        )
        scenarios.append(
            Scenario(
                f"S1S3_rsiMax{tlabel}+suppressC",
                thr,
                None,
                float(args.high_rsi_threshold),
            )
        )
        scenarios.append(
            Scenario(
                f"S1S2S3_rsiMax{tlabel}+trailing+suppressC",
                thr,
                float(args.trailing_mult_high),
                float(args.high_rsi_threshold),
            )
        )

    grid_rows: list[dict[str, object]] = []

    for max_positions in max_positions_grid:
        for regime in regimes:
            if regime == "none":
                market_regime = None
                exit_all_on_regime_off = False
            elif regime == "n225_sma200":
                market_regime = regime_n225_sma200
                exit_all_on_regime_off = True
            else:
                continue

            for sc in scenarios:
                scen_id = f"{sc.name}__pos{int(max_positions)}__reg_{regime}"

                out_dir = out_root / scen_id
                out_dir.mkdir(parents=True, exist_ok=True)

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
                        max_positions=int(max_positions),
                        entry_rsi_max=sc.entry_rsi_max,
                        c_breakout_rsi_max=sc.c_breakout_rsi_max,
                        high_rsi_threshold=float(args.high_rsi_threshold),
                        trailing_atr_mult_high_rsi=sc.trailing_override_mult,
                        market_regime=market_regime,
                        exit_all_on_regime_off=bool(exit_all_on_regime_off),
                    )
                    summaries.append(summary_df)
                    trades_all.append(trades_df)
                    equity_all.append(equity_df)

                summary = pd.concat(summaries, ignore_index=True)
                trades = (
                    pd.concat(trades_all, ignore_index=True)
                    if trades_all
                    else pd.DataFrame()
                )
                equity = (
                    pd.concat(equity_all, ignore_index=True)
                    if equity_all
                    else pd.DataFrame()
                )
                monthly = _monthly_from_equity_and_trades(equity, trades)

                summary.to_csv(
                    out_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig"
                )
                equity.to_csv(
                    out_dir / "backtest_equity_curve.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                trades.to_csv(
                    out_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig"
                )
                monthly.to_csv(
                    out_dir / "backtest_monthly.csv", index=False, encoding="utf-8-sig"
                )

                for _, row in summary.iterrows():
                    grid_rows.append(
                        {
                            "scenario": scen_id,
                            "base_scenario": sc.name,
                            "max_positions": int(max_positions),
                            "regime": regime,
                            "policy": row.get("policy"),
                            "entry_rsi_max": sc.entry_rsi_max,
                            "trailing_mult_high_rsi": sc.trailing_override_mult,
                            "c_breakout_rsi_max": sc.c_breakout_rsi_max,
                            "final_equity": row.get("final_equity"),
                            "total_return_pct": row.get("total_return_pct"),
                            "cagr": row.get("cagr"),
                            "annual_vol": row.get("annual_vol"),
                            "sharpe": row.get("sharpe"),
                            "sortino": row.get("sortino"),
                            "max_drawdown": row.get("max_drawdown"),
                            "trade_count": row.get("trade_count"),
                            "win_rate": row.get("win_rate"),
                            "profit_factor": row.get("profit_factor"),
                        }
                    )

                print(f"Done: {scen_id}")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_root / "summary_grid.csv", index=False, encoding="utf-8-sig")
    print("Wrote:")
    print(f"- {out_root / 'summary_grid.csv'}")


if __name__ == "__main__":
    main()
