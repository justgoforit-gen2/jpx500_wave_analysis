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
class PatternSet:
    name: str
    patterns: set[str] | None


def _load_stock_list(project_dir: Path) -> pd.DataFrame:
    csv_path = project_dir / "data" / "jpx500_list.csv"
    df = pd.read_csv(csv_path)
    return df


def _parse_pattern_sets(s: str) -> list[PatternSet]:
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    if not items:
        return [PatternSet("ALL", None)]

    out: list[PatternSet] = []
    for item in items:
        if item.upper() == "ALL":
            out.append(PatternSet("ALL", None))
            continue
        pats = {p.strip() for p in item.split("+") if p.strip()}
        if not pats:
            continue
        name = item
        out.append(PatternSet(name, pats))
    if not out:
        out = [PatternSet("ALL", None)]
    return out


def _compute_regime(bench: pd.DataFrame, regime: str) -> tuple[pd.Series | None, bool]:
    if regime == "none":
        return None, False

    close_col = "Close" if "Close" in bench.columns else ("close" if "close" in bench.columns else None)
    if close_col is None:
        raise RuntimeError("Benchmark cache for ^N225 has no Close column")

    close = pd.to_numeric(bench[close_col], errors="coerce")

    if regime == "n225_sma200":
        sma200 = close.rolling(200, min_periods=200).mean()
        ok = (close > sma200).fillna(True)
        return ok, True

    raise ValueError(f"Unknown regime: {regime}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate multiple pattern-set styles (A-only/B-only/C-only etc.)")
    ap.add_argument("--limit", type=int, default=None, help="Limit tickers for quick run")
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--initial", type=float, default=10_000_000.0)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument(
        "--pattern-sets",
        type=str,
        default="ALL,A_trend,B_pullback,C_breakout",
        help="Comma-separated sets. Use '+' to combine (e.g. 'A_trend+B_pullback'). Default compares ALL,A,B,C.",
    )
    ap.add_argument(
        "--regime",
        type=str,
        default="none",
        choices=["none", "n225_sma200"],
        help="Benchmark regime filter. none or n225_sma200",
    )
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    out_root = data_dir / "pattern_sets"
    out_root.mkdir(parents=True, exist_ok=True)

    strategy = load_strategy(project_dir / "config" / "strategy.yaml")
    lookback_years = int(args.years) if args.years is not None else int(strategy.get("evaluation", {}).get("lookback_years", 3))
    max_positions = int(args.max_positions) if args.max_positions is not None else int(strategy.get("execution", {}).get("max_positions", 20))

    bench = load_cached("^N225")
    if bench is None or len(bench) < 250:
        raise RuntimeError("Missing benchmark cache for ^N225")
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)
    bench.sort_index(inplace=True)

    trading_days = pd.DatetimeIndex(bench.index)
    end_date = pd.Timestamp(bench.index.max())
    start_date = end_date - pd.DateOffset(years=lookback_years)

    market_regime, exit_all_on_regime_off = _compute_regime(bench, str(args.regime))

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

    pattern_sets = _parse_pattern_sets(args.pattern_sets)

    rows: list[dict[str, object]] = []

    for ps in pattern_sets:
        safe_name = ps.name.replace("+", "-")
        scen_id = f"PAT_{safe_name}__pos{int(max_positions)}__reg_{args.regime}"
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
                max_positions=max_positions,
                allowed_patterns=ps.patterns,
                market_regime=market_regime,
                exit_all_on_regime_off=bool(exit_all_on_regime_off),
            )
            summaries.append(summary_df)
            trades_all.append(trades_df)
            equity_all.append(equity_df)

        summary = pd.concat(summaries, ignore_index=True)
        trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
        equity = pd.concat(equity_all, ignore_index=True) if equity_all else pd.DataFrame()

        summary.to_csv(out_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        equity.to_csv(out_dir / "backtest_equity_curve.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(out_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig")

        for _, r in summary.iterrows():
            rows.append(
                {
                    "scenario": scen_id,
                    "pattern_set": ps.name,
                    "max_positions": int(max_positions),
                    "regime": str(args.regime),
                    "policy": r.get("policy"),
                    "final_equity": r.get("final_equity"),
                    "total_return_pct": r.get("total_return_pct"),
                    "cagr": r.get("cagr"),
                    "annual_vol": r.get("annual_vol"),
                    "sharpe": r.get("sharpe"),
                    "sortino": r.get("sortino"),
                    "max_drawdown": r.get("max_drawdown"),
                    "trade_count": r.get("trade_count"),
                    "win_rate": r.get("win_rate"),
                    "profit_factor": r.get("profit_factor"),
                }
            )

        print(f"Done: {scen_id}")

    grid = pd.DataFrame(rows)
    out_fp = out_root / "summary_grid.csv"
    grid.to_csv(out_fp, index=False, encoding="utf-8-sig")
    print("Wrote:")
    print(f"- {out_fp}")

    # Quick peek: best by CAGR (fixed_rate)
    try:
        g = grid.copy()
        g["cagr"] = pd.to_numeric(g["cagr"], errors="coerce")
        best = g[g["policy"] == "fixed_rate"].sort_values("cagr", ascending=False).head(10)
        if len(best) > 0:
            print("Top CAGR (fixed_rate):")
            print(best[["pattern_set", "regime", "max_positions", "cagr", "max_drawdown", "total_return_pct", "trade_count"]].to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
