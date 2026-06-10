"""売買シグナル判定エンジン。

保有銘柄: 損切/利確/警告/トレンド転換
監視銘柄: 押し目買い/ブレイクアウト/目標価格到達

シグナルは「発生時のみ」を表示する設計のため、
発生したものだけ DataFrame として返す。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    SIGNAL_BREAKOUT_LOOKBACK_DAYS,
    SIGNAL_BREAKOUT_VOLUME_RATIO,
    SIGNAL_LOG_PARQUET,
    SIGNAL_LOSS_CUT_PCT,
    SIGNAL_LOSS_WARNING_PCT,
    SIGNAL_RSI_BUY_THRESHOLD,
    SIGNAL_TAKE_PROFIT_PCT,
)
from modules.data_fetcher import load_cached
from modules.wave_classifier import _calc_rsi

logger = logging.getLogger(__name__)


SIGNAL_COLS = [
    "date",
    "code",
    "name",
    "ticker",
    "signal_type",  # LOSS_CUT / TAKE_PROFIT / LOSS_WARNING / TREND_REVERSAL /
    #                  BUY_DIP / BREAKOUT / TARGET_PRICE
    "severity",  # info / warning / critical
    "side",  # BUY / SELL / HOLD
    "current_price",
    "trigger_price",
    "message",
]


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLS)


# ---------- 保有銘柄向け ----------


def compute_signals_for_holdings(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """保有銘柄に対するシグナルを判定。"""
    if portfolio_df is None or portfolio_df.empty:
        return _empty_signals()

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for _, p in portfolio_df.iterrows():
        ticker = str(p["ticker"])
        category = str(p.get("category", ""))
        if ticker == "CASH" or category == "cash":
            continue

        avg_cost = float(p["avg_cost"])
        code = str(p["code"])
        name = str(p["name"])

        df = load_cached(ticker)
        if df is None or df.empty:
            continue

        last_price = float(df["Close"].iloc[-1])
        change_pct = (last_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0

        # 1. 損切シグナル
        if change_pct <= -SIGNAL_LOSS_CUT_PCT:
            rows.append(
                {
                    "date": today,
                    "code": code,
                    "name": name,
                    "ticker": ticker,
                    "signal_type": "LOSS_CUT",
                    "severity": "critical",
                    "side": "SELL",
                    "current_price": last_price,
                    "trigger_price": avg_cost * (1 - SIGNAL_LOSS_CUT_PCT),
                    "message": (
                        f"損切ライン到達 ({change_pct * 100:+.1f}%)。"
                        f"取得 {avg_cost:.2f} → 現在 {last_price:.2f}"
                    ),
                }
            )
            continue  # 損切が出たら他のシグナル不要

        # 2. 損切警告 (-7% 圏)
        if -SIGNAL_LOSS_CUT_PCT < change_pct <= -SIGNAL_LOSS_WARNING_PCT:
            rows.append(
                {
                    "date": today,
                    "code": code,
                    "name": name,
                    "ticker": ticker,
                    "signal_type": "LOSS_WARNING",
                    "severity": "warning",
                    "side": "HOLD",
                    "current_price": last_price,
                    "trigger_price": avg_cost * (1 - SIGNAL_LOSS_CUT_PCT),
                    "message": (
                        f"損切ライン接近 ({change_pct * 100:+.1f}%)。"
                        f"損切目安 {avg_cost * (1 - SIGNAL_LOSS_CUT_PCT):.2f}"
                    ),
                }
            )

        # 3. 利確シグナル
        if change_pct >= SIGNAL_TAKE_PROFIT_PCT:
            rows.append(
                {
                    "date": today,
                    "code": code,
                    "name": name,
                    "ticker": ticker,
                    "signal_type": "TAKE_PROFIT",
                    "severity": "info",
                    "side": "SELL",
                    "current_price": last_price,
                    "trigger_price": avg_cost * (1 + SIGNAL_TAKE_PROFIT_PCT),
                    "message": (
                        f"利確目安到達 ({change_pct * 100:+.1f}%)。部分利確を検討"
                    ),
                }
            )

        # 4. トレンド転換 (slope マイナス & 直近20日安値割れ)
        if len(df) >= 30:
            recent = df.tail(20)
            recent_low = float(recent["Low"].min())
            if last_price < recent_low * 1.005 and last_price < avg_cost:
                # 弱いシグナル(slope は wave_classifier 側で判定済みのため簡易判定)
                rows.append(
                    {
                        "date": today,
                        "code": code,
                        "name": name,
                        "ticker": ticker,
                        "signal_type": "TREND_REVERSAL",
                        "severity": "warning",
                        "side": "HOLD",
                        "current_price": last_price,
                        "trigger_price": recent_low,
                        "message": ("直近20日安値割れ。トレンド転換の可能性"),
                    }
                )

    return pd.DataFrame(rows, columns=SIGNAL_COLS) if rows else _empty_signals()


# ---------- 監視銘柄向け ----------


def compute_signals_for_watchlist(
    watchlist_df: pd.DataFrame,
) -> pd.DataFrame:
    """監視銘柄に対するシグナルを判定。"""
    if watchlist_df is None or watchlist_df.empty:
        return _empty_signals()

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for _, w in watchlist_df.iterrows():
        ticker = str(w["ticker"])
        code = str(w["code"])
        name = str(w["name"])
        target_price = w.get("target_price")
        try:
            target_price = float(target_price) if pd.notna(target_price) else None
        except (TypeError, ValueError):
            target_price = None

        df = load_cached(ticker)
        if df is None or df.empty or len(df) < 30:
            continue

        last_price = float(df["Close"].iloc[-1])

        # 1. 押し目買いシグナル: RSI < 40 + 直近20日中央値より下
        rsi_series = _calc_rsi(df["Close"])
        if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]):
            rsi_val = float(rsi_series.iloc[-1])
            recent_median = float(df["Close"].tail(20).median())
            if rsi_val < SIGNAL_RSI_BUY_THRESHOLD and last_price < recent_median:
                rows.append(
                    {
                        "date": today,
                        "code": code,
                        "name": name,
                        "ticker": ticker,
                        "signal_type": "BUY_DIP",
                        "severity": "info",
                        "side": "BUY",
                        "current_price": last_price,
                        "trigger_price": recent_median,
                        "message": (
                            f"押し目買い候補 (RSI {rsi_val:.0f} < 40, "
                            f"20日中央値 {recent_median:.2f} 以下)"
                        ),
                    }
                )

        # 2. ブレイクアウト: 直近20日高値 + 出来高 1.5x
        recent = df.tail(SIGNAL_BREAKOUT_LOOKBACK_DAYS + 1)
        if len(recent) > SIGNAL_BREAKOUT_LOOKBACK_DAYS:
            prior_high = float(recent["High"].iloc[:-1].max())
            avg_vol = float(recent["Volume"].iloc[:-1].mean())
            last_vol = float(recent["Volume"].iloc[-1])
            if (
                last_price > prior_high
                and avg_vol > 0
                and last_vol >= avg_vol * SIGNAL_BREAKOUT_VOLUME_RATIO
            ):
                rows.append(
                    {
                        "date": today,
                        "code": code,
                        "name": name,
                        "ticker": ticker,
                        "signal_type": "BREAKOUT",
                        "severity": "info",
                        "side": "BUY",
                        "current_price": last_price,
                        "trigger_price": prior_high,
                        "message": (
                            f"20日高値ブレイク({prior_high:.2f})、"
                            f"出来高 {last_vol / avg_vol:.1f}x"
                        ),
                    }
                )

        # 3. 目標価格到達(当日クロスのみ発火、常時発火回避)
        if target_price is not None and target_price > 0 and len(df) >= 2:
            prev_close = float(df["Close"].iloc[-2])
            crossed_down = prev_close > target_price and last_price <= target_price
            if crossed_down:
                rows.append(
                    {
                        "date": today,
                        "code": code,
                        "name": name,
                        "ticker": ticker,
                        "signal_type": "TARGET_PRICE",
                        "severity": "info",
                        "side": "BUY",
                        "current_price": last_price,
                        "trigger_price": target_price,
                        "message": (
                            f"目標買い価格 {target_price:.2f} に下抜け到達 "
                            f"(前日 {prev_close:.2f} → 当日 {last_price:.2f})"
                        ),
                    }
                )

    return pd.DataFrame(rows, columns=SIGNAL_COLS) if rows else _empty_signals()


# ---------- ロギング ----------


def log_signals(signal_df: pd.DataFrame) -> pd.DataFrame:
    """signal_log.parquet に追記。同日 code × signal_type の重複はスキップ。"""
    if signal_df is None or signal_df.empty:
        return load_signal_log()

    today = datetime.now().strftime("%Y-%m-%d")
    log = load_signal_log()
    if not log.empty:
        existing_keys = set(
            zip(
                log[log["date"] == today]["code"].astype(str),
                log[log["date"] == today]["signal_type"].astype(str),
            )
        )
    else:
        existing_keys = set()

    new_rows = signal_df[
        ~signal_df.apply(
            lambda r: (str(r["code"]), str(r["signal_type"])) in existing_keys, axis=1
        )
    ]
    if new_rows.empty:
        return log

    combined = (
        pd.concat([log, new_rows], ignore_index=True) if not log.empty else new_rows
    )
    Path(SIGNAL_LOG_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SIGNAL_LOG_PARQUET, index=False)
    logger.info(f"シグナルログ追記: {len(new_rows)}件")
    return combined


def load_signal_log() -> pd.DataFrame:
    if not Path(SIGNAL_LOG_PARQUET).exists():
        return _empty_signals()
    return pd.read_parquet(SIGNAL_LOG_PARQUET)


def get_today_signals() -> pd.DataFrame:
    """当日のシグナルのみ返す(画面表示用)。"""
    log = load_signal_log()
    if log.empty:
        return _empty_signals()
    today = datetime.now().strftime("%Y-%m-%d")
    return log[log["date"] == today].reset_index(drop=True)
