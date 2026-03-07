from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.data_fetcher import load_cached
from modules.strategy_engine import compute_all_features
from modules.strategy_loader import load_strategy


def _rsi_bins() -> list[tuple[float, float, str]]:
    return [
        (-np.inf, 30.0, "<30"),
        (30.0, 40.0, "30-40"),
        (40.0, 50.0, "40-50"),
        (50.0, 60.0, "50-60"),
        (60.0, 70.0, "60-70"),
        (70.0, np.inf, ">=70"),
    ]


def _bin_rsi(v: float | None) -> str | None:
    if v is None or pd.isna(v):
        return None
    for lo, hi, label in _rsi_bins():
        if lo <= float(v) < hi:
            return label
    return None


def _prev_trading_day_map(trading_days: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp | None]:
    trading_days = pd.DatetimeIndex(pd.to_datetime(trading_days)).sort_values()
    m: dict[pd.Timestamp, pd.Timestamp | None] = {}
    prev: pd.Timestamp | None = None
    for d in trading_days:
        d2 = pd.Timestamp(d)
        m[d2] = prev
        prev = d2
    return m


def _get_series_value_at(series: pd.Series, dt: pd.Timestamp) -> float | None:
    if series is None or len(series) == 0:
        return None
    if dt in series.index:
        v = series.get(dt)
        return None if pd.isna(v) else float(v)
    # if missing, use last available prior value
    idx = series.index
    pos = idx.searchsorted(dt, side="right") - 1
    if pos < 0:
        return None
    v = series.iloc[int(pos)]
    return None if pd.isna(v) else float(v)


def enrich_trades_with_rsi(
    trades_df: pd.DataFrame,
    *,
    strategy: dict,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    prev_map = _prev_trading_day_map(trading_days)

    tickers = sorted(set(trades_df["ticker"].dropna().astype(str).tolist()))
    rsi_cache: dict[str, pd.Series] = {}
    for tkr in tickers:
        df = load_cached(tkr)
        if df is None or len(df) < 50:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        feats = compute_all_features(df, strategy)
        rsi = feats.get("rsi")
        if rsi is None:
            continue
        rsi_cache[tkr] = rsi

    out = trades_df.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce")
    if "exit_trigger_date" in out.columns:
        out["exit_trigger_date"] = pd.to_datetime(out["exit_trigger_date"], errors="coerce")
    else:
        out["exit_trigger_date"] = pd.NaT

    rsi_signal_list: list[float | None] = []
    rsi_entry_close_list: list[float | None] = []
    rsi_exit_trigger_list: list[float | None] = []
    rsi_exit_close_list: list[float | None] = []

    for _, row in out.iterrows():
        tkr = str(row.get("ticker", ""))
        rsi = rsi_cache.get(tkr)
        if rsi is None:
            rsi_signal_list.append(None)
            rsi_entry_close_list.append(None)
            rsi_exit_trigger_list.append(None)
            rsi_exit_close_list.append(None)
            continue

        entry_dt = row.get("entry_date")
        exit_dt = row.get("exit_date")
        trigger_dt = row.get("exit_trigger_date")

        signal_dt = None
        if pd.notna(entry_dt):
            signal_dt = prev_map.get(pd.Timestamp(entry_dt), None)

        rsi_signal_list.append(_get_series_value_at(rsi, pd.Timestamp(signal_dt)) if signal_dt is not None else None)
        rsi_entry_close_list.append(_get_series_value_at(rsi, pd.Timestamp(entry_dt)) if pd.notna(entry_dt) else None)
        rsi_exit_trigger_list.append(_get_series_value_at(rsi, pd.Timestamp(trigger_dt)) if pd.notna(trigger_dt) else None)
        rsi_exit_close_list.append(_get_series_value_at(rsi, pd.Timestamp(exit_dt)) if pd.notna(exit_dt) else None)

    out["rsi_signal"] = rsi_signal_list
    out["rsi_entry_close"] = rsi_entry_close_list
    out["rsi_exit_trigger"] = rsi_exit_trigger_list
    out["rsi_exit_close"] = rsi_exit_close_list
    out["rsi_bin"] = out["rsi_signal"].map(_bin_rsi)
    out["is_win"] = out["pnl"] > 0
    out["is_loss"] = out["pnl"] < 0
    return out


def summarize_factors(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Overall by policy
    overall = (
        enriched.groupby(["policy"], dropna=False)
        .agg(
            trades=("ticker", "size"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            win_rate=("is_win", "mean"),
            avg_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            avg_rsi_signal=("rsi_signal", "mean"),
        )
        .reset_index()
    )

    # RSI bins
    by_rsi = (
        enriched.groupby(["policy", "rsi_bin"], dropna=False)
        .agg(
            trades=("ticker", "size"),
            win_rate=("is_win", "mean"),
            avg_return_pct=("return_pct", "mean"),
            avg_rsi_signal=("rsi_signal", "mean"),
        )
        .reset_index()
        .sort_values(["policy", "rsi_bin"])
    )

    # Exit reason x RSI bin (most directly actionable)
    if "exit_reason" in enriched.columns:
        by_exit_rsi = (
            enriched.groupby(["policy", "exit_reason", "rsi_bin"], dropna=False)
            .agg(
                trades=("ticker", "size"),
                win_rate=("is_win", "mean"),
                avg_return_pct=("return_pct", "mean"),
                avg_rsi_signal=("rsi_signal", "mean"),
            )
            .reset_index()
            .sort_values(["policy", "exit_reason", "rsi_bin"])
        )
    else:
        by_exit_rsi = pd.DataFrame(columns=["policy", "exit_reason", "rsi_bin", "trades", "win_rate", "avg_return_pct", "avg_rsi_signal"])

    return overall, by_rsi, by_exit_rsi


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze win/loss factors incl. RSI")
    ap.add_argument("--trades", type=str, default=None, help="Path to backtest_trades.csv")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: jpx500_wave_analysis/data)")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    outdir = Path(args.outdir) if args.outdir else data_dir
    outdir.mkdir(parents=True, exist_ok=True)

    trades_path = Path(args.trades) if args.trades else (data_dir / "backtest_trades.csv")
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades CSV: {trades_path}")

    strategy = load_strategy(project_dir / "config" / "strategy.yaml")
    bench = load_cached("^N225")
    if bench is None or len(bench) < 250:
        raise RuntimeError("Missing benchmark cache for ^N225")
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)
    bench.sort_index(inplace=True)
    trading_days = pd.DatetimeIndex(bench.index)

    trades_df = pd.read_csv(trades_path, encoding="utf-8-sig")
    enriched = enrich_trades_with_rsi(trades_df, strategy=strategy, trading_days=trading_days)
    overall, by_rsi, by_exit_rsi = summarize_factors(enriched)

    enriched_path = outdir / "backtest_trades_enriched.csv"
    overall_path = outdir / "backtest_factor_overall.csv"
    by_rsi_path = outdir / "backtest_factor_by_rsi.csv"
    by_exit_rsi_path = outdir / "backtest_factor_by_exit_reason_rsi.csv"

    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    by_rsi.to_csv(by_rsi_path, index=False, encoding="utf-8-sig")
    by_exit_rsi.to_csv(by_exit_rsi_path, index=False, encoding="utf-8-sig")

    print("Wrote:")
    print(f"- {enriched_path}")
    print(f"- {overall_path}")
    print(f"- {by_rsi_path}")
    print(f"- {by_exit_rsi_path}")


if __name__ == "__main__":
    main()
