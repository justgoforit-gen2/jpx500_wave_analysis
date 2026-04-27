"""
ABCD Investment Style Optimizer
================================
Tests all combinations of:
  - Style (short/mid/long) x holding days
  - Pattern combinations (A/B/C/D subsets)
  - Exit parameters (trailing ATR multiplier, trend exit SMA period, time exit on/off)

Policy: fixed_amount, max 20 positions.
Benchmark: Nikkei225 Buy & Hold.

Usage:
    python batch/style_optimizer.py
    python batch/style_optimizer.py --limit 100   # debug: limit tickers
    python batch/style_optimizer.py --years 3 --initial 10000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.backtester import build_contexts, run_backtest
from modules.data_fetcher import load_cached, load_stock_list
from modules.strategy_loader import load_strategy

# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------
STYLES: dict[str, list[int]] = {
    "short": [5, 10, 15, 20],
    "mid": [30, 60],
    "long": [120, 252],
}

# ---------------------------------------------------------------------------
# Pattern combinations
# ---------------------------------------------------------------------------
PATTERN_COMBOS: dict[str, set[str]] = {
    "ALL(A+B+C+D)": {"A_trend", "B_pullback", "C_breakout", "D_reversal"},
    "ALL(A+B+C+D+E)": {
        "A_trend",
        "B_pullback",
        "C_breakout",
        "D_reversal",
        "E_can_slim",
    },
    "A+B+C": {"A_trend", "B_pullback", "C_breakout"},
    "A+B+C+E": {"A_trend", "B_pullback", "C_breakout", "E_can_slim"},
    "A+B": {"A_trend", "B_pullback"},
    "A+C": {"A_trend", "C_breakout"},
    "B+C": {"B_pullback", "C_breakout"},
    "A_only": {"A_trend"},
    "C_only": {"C_breakout"},
    "E_only": {"E_can_slim"},
}

# Exit-grid: focus on the best-performing patterns with varied exit conditions
EXIT_GRID_COMBOS: dict[str, set[str]] = {
    "A+B+C": {"A_trend", "B_pullback", "C_breakout"},
    "A+B+C+E": {"A_trend", "B_pullback", "C_breakout", "E_can_slim"},
    "A_only": {"A_trend"},
}
EXIT_GRID_TRAILING = [2.0, 3.0, 4.0]
EXIT_GRID_SMA_PERIOD = [50, 100, 200]
EXIT_GRID_TIME_EXIT = [True, False]  # True = use 60-day time exit
EXIT_GRID_HOLDING = 60  # days used when time_exit=True


# ---------------------------------------------------------------------------
# Nikkei225 benchmark metrics
# ---------------------------------------------------------------------------
def _nikkei_bench_metrics(
    bench_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    close_col = "Close" if "Close" in bench_df.columns else "close"
    nk = bench_df.copy()
    nk.index = pd.to_datetime(nk.index)
    nk = nk[(nk.index >= start_date) & (nk.index <= end_date)]
    if len(nk) < 2:
        return {}
    closes = pd.to_numeric(nk[close_col], errors="coerce").dropna()
    s, e = float(closes.iloc[0]), float(closes.iloc[-1])
    total_return = (e / s - 1.0) * 100.0
    years = max((closes.index[-1] - closes.index[0]).days / 365.25, 1e-9)
    cagr = (e / s) ** (1.0 / years) - 1.0
    peak, mdd = -np.inf, 0.0
    for v in closes:
        peak = max(peak, v)
        mdd = min(mdd, (v / peak) - 1.0 if peak > 0 else 0.0)
    rets = closes.pct_change().dropna()
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {
        "total_return_pct": round(total_return, 3),
        "cagr": round(cagr, 6),
        "max_drawdown": round(mdd, 6),
        "sharpe": round(sharpe, 6),
    }


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------
def _run_one(
    *,
    contexts,
    trading_days: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategy: dict,
    initial_capital: float,
    max_positions: int,
    scenario_id: str,
    style: str,
    holding_days: int,
    pattern_combo: str,
    allowed_patterns: set[str],
    trailing_atr_mult: float,
    trend_exit_period: int,
    use_time_exit: bool,
    nikkei_total: float,
    nikkei_cagr: float,
    verbose: bool,
) -> tuple[dict | None, pd.DataFrame | None]:
    try:
        summary_df, _trades, equity_df = run_backtest(
            contexts=contexts,
            trading_days=trading_days,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            policy="fixed_amount",
            initial_capital=initial_capital,
            max_positions=max_positions,
            allowed_patterns=allowed_patterns,
            time_exit_days_override=holding_days if use_time_exit else None,
            trailing_atr_mult_override=trailing_atr_mult,
            trend_exit_period_override=trend_exit_period,
            use_time_exit_override=use_time_exit,
        )
        if summary_df.empty:
            if verbose:
                print("empty")
            return None, None

        row = summary_df.iloc[0].to_dict()
        total_ret = float(row.get("total_return_pct", 0.0))
        row_cagr = float(row.get("cagr", 0.0))
        beats = total_ret > nikkei_total

        row.update(
            {
                "scenario_id": scenario_id,
                "style": style,
                "holding_days": holding_days,
                "pattern_combo": pattern_combo,
                "trailing_atr_mult": trailing_atr_mult,
                "trend_exit_period": trend_exit_period,
                "use_time_exit": use_time_exit,
                "beats_nikkei": beats,
                "nikkei_total_return_pct": round(nikkei_total, 3),
                "nikkei_cagr": round(nikkei_cagr, 6),
                "excess_return_pct": round(total_ret - nikkei_total, 3),
                "cagr_excess": round(row_cagr - nikkei_cagr, 6),
            }
        )

        eq = equity_df.copy()
        eq["scenario_id"] = scenario_id

        if verbose:
            flag = "OK" if beats else "--"
            print(
                f"{flag} Return={total_ret:+6.1f}%  CAGR={row_cagr:.4f}"
                f"  DD={row.get('max_drawdown', 0):.4f}  Trades={row.get('trade_count', 0)}"
            )
        return row, eq

    except Exception as exc:
        if verbose:
            print(f"ERROR: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Phase 1: style x pattern grid (default exit params)
# ---------------------------------------------------------------------------
def run_style_grid(
    contexts,
    trading_days: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategy: dict,
    nikkei_bench: dict,
    initial_capital: float = 10_000_000.0,
    max_positions: int = 20,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nikkei_total = nikkei_bench.get("total_return_pct", 0.0)
    nikkei_cagr = nikkei_bench.get("cagr", 0.0)
    summaries, curves = [], []
    total = sum(len(v) for v in STYLES.values()) * len(PATTERN_COMBOS)
    done = 0

    for style_name, days_list in STYLES.items():
        for holding_days in days_list:
            for combo_name, allowed in PATTERN_COMBOS.items():
                done += 1
                sid = f"{style_name}_{holding_days}d_{combo_name}"
                if verbose:
                    print(
                        f"[{done:2d}/{total}] {style_name:5s} {holding_days:3d}d x {combo_name}",
                        end=" ... ",
                        flush=True,
                    )
                row, eq = _run_one(
                    contexts=contexts,
                    trading_days=trading_days,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    initial_capital=initial_capital,
                    max_positions=max_positions,
                    scenario_id=sid,
                    style=style_name,
                    holding_days=holding_days,
                    pattern_combo=combo_name,
                    allowed_patterns=allowed,
                    trailing_atr_mult=2.0,
                    trend_exit_period=50,
                    use_time_exit=True,
                    nikkei_total=nikkei_total,
                    nikkei_cagr=nikkei_cagr,
                    verbose=verbose,
                )
                if row:
                    summaries.append(row)
                if eq is not None:
                    curves.append(eq)

    df = pd.DataFrame(summaries) if summaries else pd.DataFrame()
    eq_all = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return df, eq_all


# ---------------------------------------------------------------------------
# Phase 2: exit parameter grid (best patterns, varied exits)
# ---------------------------------------------------------------------------
def run_exit_grid(
    contexts,
    trading_days: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategy: dict,
    nikkei_bench: dict,
    initial_capital: float = 10_000_000.0,
    max_positions: int = 20,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nikkei_total = nikkei_bench.get("total_return_pct", 0.0)
    nikkei_cagr = nikkei_bench.get("cagr", 0.0)
    summaries, curves = [], []

    total = (
        len(EXIT_GRID_COMBOS)
        * len(EXIT_GRID_TRAILING)
        * len(EXIT_GRID_SMA_PERIOD)
        * len(EXIT_GRID_TIME_EXIT)
    )
    done = 0
    print(f"\n[exit-grid] {total} scenarios  (pattern x trailing x sma x time_exit)")

    for combo_name, allowed in EXIT_GRID_COMBOS.items():
        for tr_mult in EXIT_GRID_TRAILING:
            for sma_p in EXIT_GRID_SMA_PERIOD:
                for use_te in EXIT_GRID_TIME_EXIT:
                    done += 1
                    te_label = f"te{EXIT_GRID_HOLDING}d" if use_te else "noTE"
                    sid = f"exit_{combo_name}_tr{tr_mult}_sma{sma_p}_{te_label}"
                    if verbose:
                        print(
                            f"[{done:2d}/{total}] {combo_name:6s}  "
                            f"tr={tr_mult:.1f}x  sma{sma_p:3d}  {te_label}",
                            end=" ... ",
                            flush=True,
                        )
                    row, eq = _run_one(
                        contexts=contexts,
                        trading_days=trading_days,
                        start_date=start_date,
                        end_date=end_date,
                        strategy=strategy,
                        initial_capital=initial_capital,
                        max_positions=max_positions,
                        scenario_id=sid,
                        style="exit_grid",
                        holding_days=EXIT_GRID_HOLDING,
                        pattern_combo=combo_name,
                        allowed_patterns=allowed,
                        trailing_atr_mult=tr_mult,
                        trend_exit_period=sma_p,
                        use_time_exit=use_te,
                        nikkei_total=nikkei_total,
                        nikkei_cagr=nikkei_cagr,
                        verbose=verbose,
                    )
                    if row:
                        summaries.append(row)
                    if eq is not None:
                        curves.append(eq)

    df = pd.DataFrame(summaries) if summaries else pd.DataFrame()
    eq_all = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return df, eq_all


# ---------------------------------------------------------------------------
# Add Nikkei225 equity curve to equity DataFrame
# ---------------------------------------------------------------------------
def _add_nikkei_equity(
    equity_all: pd.DataFrame,
    bench_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
) -> pd.DataFrame:
    close_col = "Close" if "Close" in bench_df.columns else "close"
    nk = bench_df.copy()
    nk.index = pd.to_datetime(nk.index)
    nk = nk[(nk.index >= start_date) & (nk.index <= end_date)]
    closes = pd.to_numeric(nk[close_col], errors="coerce").dropna()
    if len(closes) < 2:
        return equity_all
    c0 = float(closes.iloc[0])
    nk_eq = pd.DataFrame(
        {
            "date": closes.index.strftime("%Y-%m-%d"),
            "equity": (initial_capital * closes / c0).values,
            "cash": 0.0,
            "positions": 0,
            "policy": "fixed_amount",
            "scenario_id": "Nikkei225_BuyHold",
        }
    )
    return pd.concat([equity_all, nk_eq], ignore_index=True)


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def _print_summary(results_df: pd.DataFrame, nikkei_total: float) -> None:
    beats_df = results_df[results_df["beats_nikkei"]].sort_values(
        "cagr", ascending=False
    )
    print(f"\n{'=' * 70}")
    print(f"[result] beats Nikkei225: {len(beats_df)} / {len(results_df)} scenarios")
    print(f"{'=' * 70}")
    cols = [
        "scenario_id",
        "total_return_pct",
        "excess_return_pct",
        "cagr",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "trade_count",
    ]
    cols = [c for c in cols if c in beats_df.columns]
    if not beats_df.empty:
        print(beats_df[cols].head(20).to_string(index=False))
    else:
        print("  No scenario beat Nikkei225.")

    # Best per style
    print(f"\n{'=' * 70}")
    print("[best] by style:")
    for style_name in ["short", "mid", "long"]:
        sub = results_df[results_df["style"] == style_name].sort_values(
            "cagr", ascending=False
        )
        if sub.empty:
            continue
        best = sub.iloc[0]
        flag = "[BEAT]" if best["beats_nikkei"] else "[MISS]"
        print(
            f"  {style_name:5s}: {int(best['holding_days']):3d}d x {best['pattern_combo']}"
            f"  tr={best['trailing_atr_mult']:.1f}x  sma{int(best['trend_exit_period'])}"
            f"  Return={best.get('total_return_pct', 0):+.1f}%"
            f"  CAGR={best.get('cagr', 0):.4f}"
            f"  DD={best.get('max_drawdown', 0):.4f}  {flag}"
        )

    # Best overall from exit_grid
    eg = results_df[results_df["style"] == "exit_grid"]
    if not eg.empty:
        print("\n[exit-grid] best (CAGR):")
        eg_top = eg.sort_values("cagr", ascending=False).head(5)
        eg_cols = [
            "scenario_id",
            "total_return_pct",
            "excess_return_pct",
            "cagr",
            "max_drawdown",
            "sharpe",
            "beats_nikkei",
        ]
        eg_cols = [c for c in eg_cols if c in eg_top.columns]
        print(eg_top[eg_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="ABCD Investment Style Optimizer")
    ap.add_argument("--initial", type=float, default=10_000_000.0)
    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="Limit tickers (debug)")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: latest cache date)",
    )
    ap.add_argument(
        "--skip-style-grid",
        action="store_true",
        help="Skip phase-1 style grid, run only exit-grid",
    )
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    strategy = load_strategy(project_dir / "config" / "strategy.yaml")

    bench = load_cached("^N225")
    if bench is None or len(bench) < 250:
        raise RuntimeError("^N225 cache not found. Run batch/update.py first.")
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)
    bench.sort_index(inplace=True)

    latest_date = pd.Timestamp(bench.index.max())
    if args.end_date:
        end_date = min(pd.Timestamp(args.end_date), latest_date)
    else:
        end_date = latest_date
    start_date = pd.Timestamp(end_date - pd.DateOffset(years=args.years))
    # clamp to earliest available data
    earliest = pd.Timestamp(bench.index.min())
    if start_date < earliest:
        start_date = earliest
    trading_days = pd.DatetimeIndex(bench.index)

    nikkei_bench = _nikkei_bench_metrics(bench, start_date, end_date)
    nikkei_total = nikkei_bench.get("total_return_pct", 0.0)

    print(f"[benchmark] Nikkei225 ({start_date.date()} -> {end_date.date()}):")
    print(f"   total_return : {nikkei_total:.2f}%")
    print(f"   CAGR         : {nikkei_bench.get('cagr', 0.0):.4f}")
    print(f"   max_drawdown : {nikkei_bench.get('max_drawdown', 0.0):.4f}")
    print(f"   Sharpe       : {nikkei_bench.get('sharpe', 0.0):.4f}")
    print()

    print("[loading] stock data...")
    stock_list_df = load_stock_list()
    contexts = build_contexts(
        stock_list_df=stock_list_df,
        load_cached_fn=load_cached,
        strategy=strategy,
        end_date=end_date,
        limit=args.limit,
    )
    if not contexts:
        raise RuntimeError("No valid contexts. Check cache and stock list.")
    print(f"   {len(contexts)} tickers loaded\n")

    all_summaries: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []

    # --- Phase 1: style x pattern grid ---
    if not args.skip_style_grid:
        n_phase1 = sum(len(v) for v in STYLES.values()) * len(PATTERN_COMBOS)
        print(f"[phase-1] style x pattern grid: {n_phase1} scenarios")
        sg_df, sg_eq = run_style_grid(
            contexts=contexts,
            trading_days=trading_days,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            nikkei_bench=nikkei_bench,
            initial_capital=args.initial,
            max_positions=args.max_positions,
        )
        if not sg_df.empty:
            all_summaries.append(sg_df)
        if not sg_eq.empty:
            all_equity.append(sg_eq)

    # --- Phase 2: exit parameter grid ---
    n_phase2 = (
        len(EXIT_GRID_COMBOS)
        * len(EXIT_GRID_TRAILING)
        * len(EXIT_GRID_SMA_PERIOD)
        * len(EXIT_GRID_TIME_EXIT)
    )
    print(f"\n[phase-2] exit parameter grid: {n_phase2} scenarios")
    eg_df, eg_eq = run_exit_grid(
        contexts=contexts,
        trading_days=trading_days,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy,
        nikkei_bench=nikkei_bench,
        initial_capital=args.initial,
        max_positions=args.max_positions,
    )
    if not eg_df.empty:
        all_summaries.append(eg_df)
    if not eg_eq.empty:
        all_equity.append(eg_eq)

    # --- Merge & save ---
    if not all_summaries:
        print("[warn] no results.")
        return

    results_df = pd.concat(all_summaries, ignore_index=True)
    equity_all = (
        pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    )

    equity_all = _add_nikkei_equity(
        equity_all, bench, start_date, end_date, args.initial
    )

    out_summary = data_dir / "style_optimization.csv"
    out_equity = data_dir / "style_optimization_equity.csv"
    results_df.to_csv(out_summary, index=False, encoding="utf-8-sig")
    equity_all.to_csv(out_equity, index=False, encoding="utf-8-sig")

    print(f"\n[saved] summary : {out_summary}")
    print(f"[saved] equity  : {out_equity}")

    _print_summary(results_df, nikkei_total)

    print(f"\n{'=' * 70}")
    print("Done. Check the [Investment Style Optimizer] tab in the Streamlit app.")


if __name__ == "__main__":
    main()
