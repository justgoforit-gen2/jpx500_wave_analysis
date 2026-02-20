from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modules.strategy_engine import (
    _calc_multiplier_linear,
    _calc_multiplier_piecewise,
    compute_all_features,
)
from modules.strategy_loader import get_patterns, get_scoring


@dataclass(frozen=True)
class TickerContext:
    code: str
    name: str
    ticker: str
    is_etf: bool
    df: pd.DataFrame
    features: dict[str, pd.Series]
    signals: dict[str, pd.Series]


def _rolling_atr_pct_rank_le(atr_pct: pd.Series, window: int) -> pd.Series:
    """Return rolling percentile-rank (%), computed like the live checker.

    For each t, pct_rank = (# of past values < current) / window * 100.
    Window includes current value.
    """
    values = atr_pct.to_numpy(dtype=float, copy=False)
    out = np.full(values.shape, np.nan, dtype=float)
    if window <= 1:
        return pd.Series(out, index=atr_pct.index)

    for i in range(window - 1, len(values)):
        w = values[i - window + 1 : i + 1]
        if np.isnan(w).any() or np.isnan(w[-1]):
            continue
        out[i] = float((w < w[-1]).sum()) / float(len(w)) * 100.0

    return pd.Series(out, index=atr_pct.index)


def compute_signals(features: dict[str, pd.Series], strategy: dict) -> dict[str, pd.Series]:
    patterns_cfg = get_patterns(strategy)
    signals: dict[str, pd.Series] = {}

    close = features["close"]
    high = features["high"]
    low = features["low"]
    open_ = features["open"]

    # --- A_trend ---
    a_cfg = patterns_cfg.get("A_trend", {})
    lookback_a = 10
    for cond in a_cfg.get("entry", []):
        if isinstance(cond, dict) and "sma_slope_positive" in cond:
            lookback_a = int(cond["sma_slope_positive"].get("lookback", 10))

    sma50 = features.get("sma_50")
    sma200 = features.get("sma_200")
    if sma50 is not None and sma200 is not None:
        a = (close > sma200) & (sma50 > sma200) & ((sma50 - sma50.shift(lookback_a)) > 0)
        signals["A_trend"] = a.fillna(False)

    # --- B_pullback ---
    b_cfg = patterns_cfg.get("B_pullback", {})
    lookback_b = int(b_cfg.get("pullback_high_lookback", 10))
    rsi_range = (40.0, 55.0)
    atr_range = (1.0, 2.5)
    for cond in b_cfg.get("entry", []):
        if isinstance(cond, dict) and "rsi_between" in cond:
            rr = cond["rsi_between"]
            rsi_range = (float(rr[0]), float(rr[1]))
        if isinstance(cond, dict) and "pullback_atr_between" in cond:
            ar = cond["pullback_atr_between"]
            atr_range = (float(ar[0]), float(ar[1]))

    sma100 = features.get("sma_100")
    rsi = features.get("rsi")
    atr = features.get("atr")
    if sma100 is not None and rsi is not None and atr is not None:
        recent_high = high.rolling(window=lookback_b, min_periods=lookback_b).max()
        pullback_atr = (recent_high - close) / atr
        b = (
            (close > sma100)
            & (rsi >= rsi_range[0])
            & (rsi <= rsi_range[1])
            & (pullback_atr >= atr_range[0])
            & (pullback_atr <= atr_range[1])
            & (close > high.shift(1))
        )
        signals["B_pullback"] = b.fillna(False)

    # --- C_breakout ---
    c_cfg = patterns_cfg.get("C_breakout", {})
    breakout_n = 20
    vr_threshold = 1.5
    rank_threshold = 40.0
    for cond in c_cfg.get("entry", []):
        if isinstance(cond, dict) and "breakout_high" in cond:
            breakout_n = int(cond["breakout_high"])
        if isinstance(cond, dict) and "volume_ratio_gt" in cond:
            vr_threshold = float(cond["volume_ratio_gt"])
        if isinstance(cond, dict) and "atr_percent_rank_le" in cond:
            rank_threshold = float(cond["atr_percent_rank_le"])
    rank_window = int(c_cfg.get("atr_percent_rank_window", 60))

    volume_ratio = features.get("volume_ratio")
    atr_pct = features.get("atr_pct")
    if volume_ratio is not None and atr_pct is not None:
        past_high = high.shift(1).rolling(window=breakout_n, min_periods=breakout_n).max()
        pct_rank = _rolling_atr_pct_rank_le(atr_pct, rank_window)
        c = (high > past_high) & (volume_ratio > vr_threshold) & (pct_rank <= rank_threshold)
        signals["C_breakout"] = c.fillna(False)

    # --- D_reversal ---
    d_cfg = patterns_cfg.get("D_reversal", {})
    n = 5
    drop_range = (-0.15, -0.08)
    for cond in d_cfg.get("entry", []):
        if isinstance(cond, dict) and "drop_pct_last_n_days_between" in cond:
            params = cond["drop_pct_last_n_days_between"]
            n = int(params.get("n", 5))
            rr = params.get("range", [-0.15, -0.08])
            drop_range = (float(rr[0]), float(rr[1]))

    sma200_d = features.get("sma_200")
    if sma200_d is not None:
        drop_pct = (close - close.shift(n)) / close.shift(n)
        body = (close - open_).abs()
        day_range = (high - low)
        lower_shadow = np.minimum(open_.to_numpy(dtype=float, copy=False), close.to_numpy(dtype=float, copy=False)) - low.to_numpy(
            dtype=float, copy=False
        )

        # hammer rules
        ratio = float(strategy.get("candle_patterns", {})
                      .get("bullish_reversal_candle", {})
                      .get("rules", {})
                      .get("lower_shadow_min_body_ratio", 1.5))
        max_body_ratio = float(strategy.get("candle_patterns", {})
                               .get("bullish_reversal_candle", {})
                               .get("rules", {})
                               .get("body_max_range_ratio", 0.40))

        body_np = body.to_numpy(dtype=float, copy=False)
        range_np = day_range.to_numpy(dtype=float, copy=False)

        cond_range = range_np > 0
        cond_lower = np.where(body_np > 0, lower_shadow >= body_np * ratio, lower_shadow > 0)
        cond_body = body_np <= range_np * max_body_ratio
        cond_close = close >= open_
        hammer = pd.Series(cond_range & cond_lower & cond_body, index=close.index) & cond_close

        d = (
            (close > sma200_d)
            & (drop_pct >= drop_range[0])
            & (drop_pct <= drop_range[1])
            & hammer
        )
        signals["D_reversal"] = d.fillna(False)

    return signals


def score_at_date(
    *,
    pattern: str,
    features: dict[str, pd.Series],
    dt: pd.Timestamp,
    turnover_rank_pct: float,
    is_etf: bool,
    strategy: dict,
) -> float:
    scoring_cfg = get_scoring(strategy)
    base_points = scoring_cfg.get("base_points", {})
    base = float(base_points.get(pattern, 0.0))

    if is_etf:
        overrides = scoring_cfg.get("instrument_overrides", {}).get("ETF", {})
        base = float(overrides.get("base_points", {}).get(pattern, base))

    apply_list = scoring_cfg.get("apply_to_pattern", {}).get(pattern, [])
    multipliers_common = scoring_cfg.get("multipliers", {}).get("common", {})

    if is_etf:
        etf_common = (
            scoring_cfg.get("instrument_overrides", {})
            .get("ETF", {})
            .get("multipliers", {})
            .get("common", {})
        )
        merged_mult = {**multipliers_common, **etf_common}
    else:
        merged_mult = multipliers_common

    product = 1.0
    for mult_name in apply_list:
        mult_cfg = merged_mult.get(mult_name)
        if not mult_cfg:
            continue
        mult_type = mult_cfg.get("type")

        if mult_name == "turnover_rank":
            value = float(turnover_rank_pct)
        elif mult_name == "atr_percent":
            s = features.get("atr_pct")
            value = float(s.get(dt, np.nan)) if s is not None else np.nan
        elif mult_name == "volume_ratio":
            s = features.get("volume_ratio")
            value = float(s.get(dt, np.nan)) if s is not None else np.nan
        else:
            continue

        if np.isnan(value):
            # fallbacks (same spirit as live scoring)
            if mult_name == "atr_percent":
                value = 0.03
            elif mult_name == "volume_ratio":
                value = 1.0
            else:
                value = 50.0

        if mult_type == "piecewise":
            m = float(_calc_multiplier_piecewise(value, mult_cfg.get("points", [])))
        elif mult_type == "linear":
            m = float(_calc_multiplier_linear(value, mult_cfg.get("params", {}), mult_cfg.get("clip")))
        else:
            m = 1.0
        product *= m

    return round(base * product, 2)


def _turnover_rank_pct_at_date(contexts: list[TickerContext], dt: pd.Timestamp) -> dict[str, float]:
    vals: dict[str, float] = {}
    for ctx in contexts:
        s = ctx.features.get("turnover")
        if s is None:
            continue
        v = s.get(dt, np.nan)
        if pd.notna(v) and float(v) > 0:
            vals[ctx.ticker] = float(v)

    if not vals:
        return {}

    ser = pd.Series(vals)
    n = len(ser)
    if n == 1:
        return {ser.index[0]: 0.0}

    # 0 = most liquid
    ranks = ser.rank(ascending=False, method="min")
    pct = (ranks - 1.0) / float(n - 1) * 100.0
    return {t: round(float(p), 1) for t, p in pct.items()}


def build_contexts(
    *,
    stock_list_df: pd.DataFrame,
    load_cached_fn,
    strategy: dict,
    end_date: pd.Timestamp | None = None,
    limit: int | None = None,
) -> list[TickerContext]:
    contexts: list[TickerContext] = []
    for _, row in stock_list_df.iterrows():
        if limit is not None and len(contexts) >= limit:
            break
        ticker = str(row["ticker"])
        df = load_cached_fn(ticker)
        if df is None or len(df) < 250:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        if end_date is not None:
            df = df[df.index <= end_date]
        if len(df) < 250:
            continue

        features = compute_all_features(df, strategy)
        signals = compute_signals(features, strategy)
        is_etf = str(row.get("size_category", "")).upper() == "ETF"
        contexts.append(
            TickerContext(
                code=str(row.get("code", "")),
                name=str(row.get("name", "")),
                ticker=ticker,
                is_etf=is_etf,
                df=df,
                features=features,
                signals=signals,
            )
        )
    return contexts


def weekly_rebalance_dates(trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if len(trading_days) == 0:
        return []
    s = pd.Series(trading_days, index=trading_days)
    firsts = s.groupby(trading_days.to_period("W")).first()
    return [pd.Timestamp(x) for x in firsts.to_list()]


def _next_trading_day(trading_days: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp | None:
    if dt not in trading_days:
        return None
    i = trading_days.get_loc(dt)
    if isinstance(i, slice):
        i = i.start
    if i is None:
        return None
    if i + 1 >= len(trading_days):
        return None
    return pd.Timestamp(trading_days[i + 1])


def _get_price(ctx: TickerContext, dt: pd.Timestamp, field: str) -> float:
    col = None
    if field == "open":
        col = "Open"
    elif field == "close":
        col = "Close"
    elif field == "high":
        col = "High"
    elif field == "low":
        col = "Low"
    else:
        raise ValueError(field)

    s = ctx.df.get(col)
    if s is None:
        return float("nan")
    v = float(s.get(dt, np.nan))
    return v


def _time_exit_days(strategy: dict) -> int:
    holding = strategy.get("holding", {})
    candidates = holding.get("holding_days_candidates") or []
    # Prefer a common horizon if present (20 trading days ~ 1 month).
    if isinstance(candidates, list):
        if 20 in candidates:
            return 20
        if len(candidates) > 0:
            try:
                return int(candidates[0])
            except Exception:
                pass
    return 20


def run_backtest(
    *,
    contexts: list[TickerContext],
    trading_days: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategy: dict,
    policy: str,
    initial_capital: float = 10_000_000.0,
    max_positions: int = 20,
    # --- scenario knobs ---
    entry_rsi_max: float | None = None,
    c_breakout_rsi_max: float | None = None,
    high_rsi_threshold: float = 70.0,
    trailing_atr_mult_high_rsi: float | None = None,
    allowed_patterns: set[str] | None = None,
    time_exit_days_override: int | None = None,
    trailing_atr_mult_override: float | None = None,
    trend_exit_period_override: int | None = None,
    use_time_exit_override: bool | None = None,
    use_trend_exit_override: bool | None = None,
    # --- market regime knobs ---
    market_regime: pd.Series | None = None,
    exit_all_on_regime_off: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily exit rules + weekly rebalance entries.

        Exit reasons recorded in trades:
      - time_exit: holding days reached
      - trailing_atr: close fell below trailing stop
      - trend_exit: close < SMA(period)
      - rebalance_drop: removed from weekly target basket
            - regime_off: benchmark regime filter turned off
    """
    exec_cfg = strategy.get("execution", {})
    roundtrip_cost = float(exec_cfg.get("transaction_cost_roundtrip", 0.003))
    cost_side = roundtrip_cost / 2.0

    exits_cfg = strategy.get("holding", {}).get("exits", {})
    use_time_exit = bool(exits_cfg.get("use_time_exit", True)) if use_time_exit_override is None else bool(use_time_exit_override)
    use_trailing = bool(exits_cfg.get("use_trailing_atr", True))
    trailing_mult_base = float(exits_cfg.get("trailing_atr_mult", 2.0)) if trailing_atr_mult_override is None else float(trailing_atr_mult_override)
    use_trend_exit = bool(exits_cfg.get("use_trend_exit", True)) if use_trend_exit_override is None else bool(use_trend_exit_override)
    trend_period = int(exits_cfg.get("trend_exit_rule", {}).get("period", 50)) if trend_exit_period_override is None else int(trend_exit_period_override)
    time_exit_days = int(time_exit_days_override) if time_exit_days_override is not None else _time_exit_days(strategy)

    td = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]
    if len(td) < 60:
        raise RuntimeError("Not enough trading days for backtest period")

    rebalance_days = set(weekly_rebalance_dates(td))

    positions: dict[str, dict[str, Any]] = {}
    cash = float(initial_capital)

    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    ctx_by_ticker = {c.ticker: c for c in contexts}
    pending_exits: dict[str, dict[str, Any]] = {}
    last_target_set: set[str] = set()

    if market_regime is not None:
        try:
            market_regime = market_regime.copy()
            market_regime.index = pd.to_datetime(market_regime.index)
        except Exception:
            market_regime = None

    for dt in td:
        dt = pd.Timestamp(dt)

        # 1) execute any pending exits scheduled for today at open
        for tkr, info in list(pending_exits.items()):
            if info.get("exit_dt") != dt:
                continue
            pos = positions.get(tkr)
            if pos is None:
                del pending_exits[tkr]
                continue
            ctx = ctx_by_ticker.get(tkr)
            if ctx is None:
                del pending_exits[tkr]
                continue

            exit_px = _get_price(ctx, dt, "open")
            if np.isnan(exit_px) or exit_px <= 0:
                exit_px = _get_price(ctx, dt, "close")
            if np.isnan(exit_px) or exit_px <= 0:
                # cannot execute -> keep pending
                continue

            shares = int(pos["shares"])
            gross = exit_px * shares
            fee = gross * cost_side
            net = gross - fee
            cash += net

            entry_px = float(pos["entry_price"])
            entry_gross = entry_px * shares
            entry_fee = entry_gross * cost_side
            pnl = net - (entry_gross + entry_fee)
            ret = pnl / (entry_gross + entry_fee) if (entry_gross + entry_fee) > 0 else 0.0

            trades.append(
                {
                    "policy": policy,
                    "code": pos["code"],
                    "name": pos["name"],
                    "ticker": tkr,
                    "pattern": pos["pattern"],
                    "entry_date": pos["entry_date"],
                    "exit_date": dt.strftime("%Y-%m-%d"),
                    "exit_reason": info.get("exit_reason"),
                    "exit_trigger_date": info.get("trigger_dt"),
                    "entry_price": round(entry_px, 4),
                    "exit_price": round(exit_px, 4),
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret * 100, 3),
                }
            )
            del positions[tkr]
            del pending_exits[tkr]

        # 2) evaluate exits at close (schedule for next trading day open)
        next_dt = _next_trading_day(td, dt)
        for tkr, pos in list(positions.items()):
            if tkr in pending_exits:
                continue
            ctx = ctx_by_ticker.get(tkr)
            if ctx is None:
                continue
            if dt not in ctx.df.index:
                continue

            close = _get_price(ctx, dt, "close")
            if np.isnan(close) or close <= 0:
                continue

            # update trailing state
            pos["days_held"] = int(pos.get("days_held", 0)) + 1
            pos["highest_close"] = float(max(float(pos.get("highest_close", close)), close))

            reason = None
            if use_time_exit and pos["days_held"] >= time_exit_days:
                reason = "time_exit"
            if reason is None and use_trailing:
                atr = ctx.features.get("atr")
                atr_v = float(atr.get(dt, np.nan)) if atr is not None else np.nan
                if not np.isnan(atr_v) and atr_v > 0:
                    tmult = float(pos.get("trailing_mult_used", trailing_mult_base))
                    stop = float(pos["highest_close"]) - tmult * atr_v
                    if close <= stop:
                        reason = "trailing_atr"
            if reason is None and use_trend_exit:
                sma_key = f"sma_{trend_period}"
                sma = ctx.features.get(sma_key)
                sma_v = float(sma.get(dt, np.nan)) if sma is not None else np.nan
                if not np.isnan(sma_v) and close < sma_v:
                    reason = "trend_exit"

            if reason is not None and next_dt is not None:
                pending_exits[tkr] = {
                    "exit_dt": next_dt,
                    "trigger_dt": dt.strftime("%Y-%m-%d"),
                    "exit_reason": reason,
                }

        # 3) weekly rebalance: compute target basket using signal from previous trading day
        if dt in rebalance_days:
            idx = td.get_loc(dt)
            if isinstance(idx, slice):
                idx = idx.start
            if idx is not None and idx > 0:
                signal_dt = pd.Timestamp(td[idx - 1])

                regime_ok = True
                if market_regime is not None:
                    try:
                        v = market_regime.get(signal_dt, True)
                        regime_ok = bool(v) if pd.notna(v) else True
                    except Exception:
                        regime_ok = True

                if not regime_ok:
                    target: list[dict[str, Any]] = []
                    target_set: set[str] = set()
                else:
                    turnover_rank = _turnover_rank_pct_at_date(contexts, signal_dt)
                    candidates: list[dict[str, Any]] = []
                    for ctx in contexts:
                        if signal_dt not in ctx.df.index:
                            continue

                        rsi_series = ctx.features.get("rsi")
                        rsi_signal = float(rsi_series.get(signal_dt, np.nan)) if rsi_series is not None else np.nan
                        if entry_rsi_max is not None and not np.isnan(rsi_signal):
                            if rsi_signal >= float(entry_rsi_max):
                                continue

                        matched: list[str] = []
                        for pat, sig in ctx.signals.items():
                            try:
                                if bool(sig.get(signal_dt, False)):
                                    if allowed_patterns is None or pat in allowed_patterns:
                                        matched.append(pat)
                            except Exception:
                                continue
                        if not matched:
                            continue

                        rank_pct = float(turnover_rank.get(ctx.ticker, 50.0))
                        best_pat = None
                        best_score = -1e18
                        for pat in matched:
                            if (
                                c_breakout_rsi_max is not None
                                and pat == "C_breakout"
                                and not np.isnan(rsi_signal)
                                and rsi_signal >= float(c_breakout_rsi_max)
                            ):
                                continue
                            sc = score_at_date(
                                pattern=pat,
                                features=ctx.features,
                                dt=signal_dt,
                                turnover_rank_pct=rank_pct,
                                is_etf=ctx.is_etf,
                                strategy=strategy,
                            )
                            if sc > best_score:
                                best_score = sc
                                best_pat = pat
                        if best_pat is None:
                            continue
                        candidates.append(
                            {
                                "ctx": ctx,
                                "best_pattern": best_pat,
                                "best_score": float(best_score),
                                "rsi_signal": None if np.isnan(rsi_signal) else float(rsi_signal),
                            }
                        )
                    candidates.sort(key=lambda x: x["best_score"], reverse=True)
                    target = candidates[:max_positions]
                    target_set = {x["ctx"].ticker for x in target}

                # schedule exits for holdings dropped from basket (execute next open)
                dropped = set(positions.keys()) - target_set
                if next_dt is not None:
                    for tkr in dropped:
                        if tkr in pending_exits:
                            continue
                        pending_exits[tkr] = {
                            "exit_dt": next_dt,
                            "trigger_dt": dt.strftime("%Y-%m-%d"),
                            "exit_reason": "rebalance_drop",
                        }

                # optional: force exit all positions when regime turns off
                if (not regime_ok) and exit_all_on_regime_off and next_dt is not None:
                    for tkr in list(positions.keys()):
                        if tkr in pending_exits:
                            continue
                        pending_exits[tkr] = {
                            "exit_dt": next_dt,
                            "trigger_dt": dt.strftime("%Y-%m-%d"),
                            "exit_reason": "regime_off",
                        }

                # compute current equity at open (approx) to size fixed_rate
                equity_for_alloc = cash
                for tkr, pos in positions.items():
                    ctx = ctx_by_ticker.get(tkr)
                    if ctx is None:
                        continue
                    px = _get_price(ctx, dt, "open")
                    if np.isnan(px) or px <= 0:
                        px = _get_price(ctx, dt, "close")
                    if np.isnan(px) or px <= 0:
                        continue
                    equity_for_alloc += int(pos["shares"]) * px

                if policy == "fixed_amount":
                    slot_yen = float(initial_capital) / float(max_positions)
                elif policy == "fixed_rate":
                    slot_yen = float(equity_for_alloc) / float(max_positions)
                else:
                    raise ValueError(f"Unknown policy: {policy}")

                # enter missing tickers
                for item in target:
                    ctx = item["ctx"]
                    tkr = ctx.ticker
                    if tkr in positions or tkr in pending_exits:
                        continue
                    entry_px = _get_price(ctx, dt, "open")
                    if np.isnan(entry_px) or entry_px <= 0:
                        continue
                    alloc = min(cash, slot_yen)
                    if alloc <= 0:
                        continue
                    denom = entry_px * (1.0 + cost_side)
                    shares = int(np.floor(alloc / denom))
                    if shares <= 0:
                        continue
                    gross = entry_px * shares
                    fee = gross * cost_side
                    cash -= (gross + fee)

                    rsi_sig = item.get("rsi_signal")
                    if rsi_sig is not None and trailing_atr_mult_high_rsi is not None:
                        tmult_used = float(trailing_atr_mult_high_rsi) if float(rsi_sig) >= float(high_rsi_threshold) else float(trailing_mult_base)
                    else:
                        tmult_used = float(trailing_mult_base)

                    positions[tkr] = {
                        "code": ctx.code,
                        "name": ctx.name,
                        "ticker": tkr,
                        "pattern": item["best_pattern"],
                        "entry_date": dt.strftime("%Y-%m-%d"),
                        "entry_price": entry_px,
                        "shares": shares,
                        "days_held": 0,
                        "highest_close": float(_get_price(ctx, dt, "close")) if not np.isnan(_get_price(ctx, dt, "close")) else float(entry_px),
                        "entry_rsi_signal": rsi_sig,
                        "trailing_mult_used": tmult_used,
                    }

                last_target_set = target_set

        # 4) mark daily equity at close
        equity_close = cash
        pos_count = 0
        for tkr, pos in positions.items():
            ctx = ctx_by_ticker.get(tkr)
            if ctx is None:
                continue
            px = _get_price(ctx, dt, "close")
            if np.isnan(px) or px <= 0:
                continue
            equity_close += int(pos["shares"]) * px
            pos_count += 1
        equity_rows.append(
            {
                "policy": policy,
                "date": dt.strftime("%Y-%m-%d"),
                "equity": round(float(equity_close), 2),
                "cash": round(float(cash), 2),
                "positions": int(pos_count),
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    if len(equity_df) == 0:
        raise RuntimeError("Backtest produced no equity points")

    # ---- summary metrics ----
    eq = equity_df["equity"].astype(float).to_numpy()
    rets = pd.Series(eq).pct_change().dropna().to_numpy(dtype=float)
    if len(rets) < 2:
        ann_vol = 0.0
        sharpe = 0.0
        sortino = 0.0
    else:
        ann_vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
        mean = float(np.mean(rets))
        std = float(np.std(rets, ddof=1))
        sharpe = float((mean / std) * np.sqrt(52.0)) if std > 0 else 0.0
        downside = rets[rets < 0]
        dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = float((mean / dstd) * np.sqrt(52.0)) if dstd > 0 else 0.0

    # max drawdown
    peak = -np.inf
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)
    mdd = float(mdd)

    start_eq = float(eq[0])
    end_eq = float(eq[-1])
    start_dt = pd.to_datetime(equity_df["date"].iloc[0])
    end_dt = pd.to_datetime(equity_df["date"].iloc[-1])
    years = max(1e-9, (end_dt - start_dt).days / 365.25)
    cagr = float((end_eq / start_eq) ** (1.0 / years) - 1.0) if start_eq > 0 else 0.0

    trade_count = int(len(trades_df))
    if trade_count > 0:
        wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        losses = trades_df[trades_df["pnl"] < 0]["pnl"].sum()
        win_rate = float((trades_df["pnl"] > 0).mean())
        profit_factor = float(wins / abs(losses)) if losses < 0 else float("inf")
    else:
        win_rate = 0.0
        profit_factor = 0.0

    summary_df = pd.DataFrame(
        [
            {
                "policy": policy,
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_dt.strftime("%Y-%m-%d"),
                "initial_capital": round(float(initial_capital), 2),
                "final_equity": round(float(end_eq), 2),
                "total_return_pct": round((end_eq / start_eq - 1.0) * 100.0, 3) if start_eq > 0 else 0.0,
                "cagr": round(cagr, 6),
                "annual_vol": round(ann_vol, 6),
                "sharpe": round(sharpe, 6),
                "sortino": round(sortino, 6),
                "max_drawdown": round(mdd, 6),
                "trade_count": trade_count,
                "win_rate": round(win_rate, 6),
                "profit_factor": round(profit_factor, 6) if np.isfinite(profit_factor) else profit_factor,
                "avg_positions": round(float(equity_df["positions"].mean()), 3),
            }
        ]
    )

    return summary_df, trades_df, equity_df


def run_weekly_backtest(
    *,
    contexts: list[TickerContext],
    trading_days: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategy: dict,
    policy: str,
    initial_capital: float = 10_000_000.0,
    max_positions: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Backward-compat alias
    return run_backtest(
        contexts=contexts,
        trading_days=trading_days,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy,
        policy=policy,
        initial_capital=initial_capital,
        max_positions=max_positions,
    )
