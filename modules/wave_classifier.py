"""波形タイプ分類ロジック: 7指標を計算し6タイプを判定する"""

import numpy as np
import pandas as pd

from config.settings import (
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    BREAKOUT_LOOKBACK_DAYS,
    BREAKOUT_MIN_DAYS,
    DAILY_PICK_LOOKBACK,
    DAILY_PICKS_CSV,
    DEFAULT_WINDOW,
    HIGH_VOLATILITY_THRESHOLD,
    RANGE_MIN_TOUCHES,
    RANGE_PERCENTILE_HIGH,
    RANGE_PERCENTILE_LOW,
    RSI_PERIOD,
    RSI_OVERSOLD,
    SLOPE_THRESHOLD,
    SQUEEZE_BANDWIDTH_SHRINK,
    STOCK_LIST_CSV,
    RESULTS_CSV,
    TOUCH_THRESHOLD_PCT,
)
from modules.data_fetcher import fetch_valuation, load_cached


def _calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSIを計算する"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> dict | None:
    """評価窓内で各指標を計算する。データ不足の場合はNoneを返す。"""
    if df is None or len(df) < window:
        return None

    w = df.tail(window).copy()
    close = w["Close"].values.astype(float)
    high = w["High"].values.astype(float)
    low = w["Low"].values.astype(float)

    if len(close) < 20 or np.isnan(close).all():
        return None

    # 1. range_high / range_low (パーセンタイル)
    range_high = float(np.nanpercentile(close, RANGE_PERCENTILE_HIGH))
    range_low = float(np.nanpercentile(close, RANGE_PERCENTILE_LOW))

    # 2. range_pct
    midpoint = (range_high + range_low) / 2
    range_pct = (range_high - range_low) / midpoint * 100 if midpoint > 0 else 0.0

    # 3. slope (正規化: 傾き / 平均価格)
    x = np.arange(len(close), dtype=float)
    valid = ~np.isnan(close)
    if valid.sum() < 10:
        return None
    coeffs = np.polyfit(x[valid], close[valid], 1)
    slope = coeffs[0] / np.nanmean(close)

    # 4. touch_high / touch_low
    touch_band = midpoint * TOUCH_THRESHOLD_PCT / 100
    touch_high = int(np.sum(close >= (range_high - touch_band)))
    touch_low = int(np.sum(close <= (range_low + touch_band)))

    # 5. ATR (Average True Range)
    full_df = df.tail(window + ATR_PERIOD)
    tr_high_low = full_df["High"].values - full_df["Low"].values
    tr_high_close = np.abs(full_df["High"].values[1:] - full_df["Close"].values[:-1])
    tr_low_close = np.abs(full_df["Low"].values[1:] - full_df["Close"].values[:-1])
    tr = np.maximum(tr_high_low[1:], np.maximum(tr_high_close, tr_low_close))
    atr = (
        float(np.nanmean(tr[-ATR_PERIOD:]))
        if len(tr) >= ATR_PERIOD
        else float(np.nanmean(tr))
    )

    # 6. bandwidth (ボリンジャーバンド幅)
    close_series = pd.Series(close)
    sma = close_series.rolling(BB_PERIOD).mean()
    std = close_series.rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    bandwidth = ((upper - lower) / sma).dropna()

    # bandwidth縮小率: 後半 vs 前半
    if len(bandwidth) >= 20:
        half = len(bandwidth) // 2
        bw_first = bandwidth.iloc[:half].mean()
        bw_second = bandwidth.iloc[half:].mean()
        bw_shrink_ratio = bw_second / bw_first if bw_first > 0 else 1.0
    else:
        bw_shrink_ratio = 1.0

    bw_latest = float(bandwidth.iloc[-1]) if len(bandwidth) > 0 else 0.0

    # 7. breakout_days
    recent_close = close[-BREAKOUT_LOOKBACK_DAYS:]
    breakout_days = int(
        np.sum((recent_close > range_high) | (recent_close < range_low))
    )

    # 高値切り上げ / 安値切り下げ判定
    half = len(close) // 2
    first_half_high = np.nanmax(high[:half])
    second_half_high = np.nanmax(high[half:])
    first_half_low = np.nanmin(low[:half])
    second_half_low = np.nanmin(low[half:])
    higher_highs = second_half_high > first_half_high
    lower_lows = second_half_low < first_half_low

    return {
        "range_high": round(range_high, 2),
        "range_low": round(range_low, 2),
        "range_pct": round(range_pct, 2),
        "slope": round(slope, 6),
        "touch_high": touch_high,
        "touch_low": touch_low,
        "touch_total": touch_high + touch_low,
        "atr": round(atr, 2),
        "bandwidth": round(bw_latest, 4),
        "bw_shrink_ratio": round(bw_shrink_ratio, 4),
        "breakout_days": breakout_days,
        "higher_highs": higher_highs,
        "lower_lows": lower_lows,
    }


def classify(indicators: dict) -> list[str]:
    """指標から波形タイプを判定する。複数タイプ付与可。"""
    types = []
    slope = indicators["slope"]
    abs_slope = abs(slope)

    # レンジ（波型）
    if abs_slope < SLOPE_THRESHOLD and indicators["touch_total"] >= RANGE_MIN_TOUCHES:
        types.append("レンジ（波型）")

    # 上昇トレンド
    if slope > SLOPE_THRESHOLD and indicators["higher_highs"]:
        types.append("上昇トレンド")

    # 下降トレンド
    if slope < -SLOPE_THRESHOLD and indicators["lower_lows"]:
        types.append("下降トレンド")

    # 収束（スクイーズ）
    if indicators["bw_shrink_ratio"] < SQUEEZE_BANDWIDTH_SHRINK:
        types.append("収束（スクイーズ）")

    # ブレイク気味
    if indicators["breakout_days"] >= BREAKOUT_MIN_DAYS:
        types.append("ブレイク気味")

    # 高ボラ（荒い）
    avg_price = (indicators["range_high"] + indicators["range_low"]) / 2
    if avg_price > 0 and indicators["atr"] / avg_price > HIGH_VOLATILITY_THRESHOLD:
        types.append("高ボラ（荒い）")

    if not types:
        types.append("未分類")

    return types


def classify_all(window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """全銘柄を分類してDataFrameを返す"""
    stocks = pd.read_csv(STOCK_LIST_CSV, encoding="utf-8-sig", dtype={"code": str})
    results = []

    for _, row in stocks.iterrows():
        ticker = row["ticker"]
        df = load_cached(ticker)
        indicators = compute_indicators(df, window=window)
        valuation = fetch_valuation(ticker)

        if indicators is None:
            results.append(
                {
                    "code": row["code"],
                    "name": row["name"],
                    "size_category": row["size_category"],
                    "sector_33": row.get("sector_33", ""),
                    "ticker": ticker,
                    "wave_types": "データ不足",
                    "range_high": None,
                    "range_low": None,
                    "range_pct": None,
                    "slope": None,
                    "touch_high": None,
                    "touch_low": None,
                    "touch_total": None,
                    "atr": None,
                    "bandwidth": None,
                    "breakout_days": None,
                    "per": valuation["per"],
                    "pbr": valuation["pbr"],
                    "market_cap": valuation["market_cap"],
                }
            )
            continue

        wave_types = classify(indicators)
        results.append(
            {
                "code": row["code"],
                "name": row["name"],
                "size_category": row["size_category"],
                "sector_33": row.get("sector_33", ""),
                "ticker": ticker,
                "wave_types": "|".join(wave_types),
                "range_high": indicators["range_high"],
                "range_low": indicators["range_low"],
                "range_pct": indicators["range_pct"],
                "slope": indicators["slope"],
                "touch_high": indicators["touch_high"],
                "touch_low": indicators["touch_low"],
                "touch_total": indicators["touch_total"],
                "atr": indicators["atr"],
                "bandwidth": indicators["bandwidth"],
                "breakout_days": indicators["breakout_days"],
                "per": valuation["per"],
                "pbr": valuation["pbr"],
                "market_cap": valuation["market_cap"],
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    return result_df


def generate_daily_picks(window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """レンジ銘柄の中から直近でタッチした銘柄を抽出する。

    - 下タッチ銘柄 = レンジ下限付近（買い候補）
    - 上タッチ銘柄 = レンジ上限付近（利確/ブレイク監視候補）
    """
    results = pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"code": str})
    range_stocks = results[
        results["wave_types"].apply(
            lambda x: "レンジ（波型）" in str(x) if pd.notna(x) else False
        )
    ]

    picks = []
    for _, row in range_stocks.iterrows():
        ticker = row["ticker"]
        df = load_cached(ticker)
        if df is None or len(df) < window:
            continue

        indicators = compute_indicators(df, window=window)
        if indicators is None:
            continue

        range_high = indicators["range_high"]
        range_low = indicators["range_low"]
        midpoint = (range_high + range_low) / 2
        touch_band = midpoint * TOUCH_THRESHOLD_PCT / 100

        # 直近N日の終値
        recent = df.tail(DAILY_PICK_LOOKBACK)
        recent_close = recent["Close"].values.astype(float)
        latest_close = float(recent_close[-1])
        latest_date = recent.index[-1].strftime("%Y-%m-%d")

        # RSI計算
        rsi_series = _calc_rsi(df["Close"])
        rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None
        if rsi_val is not None:
            if rsi_val <= RSI_OVERSOLD:
                rsi_signal = "売られすぎ（反発期待）"
            elif rsi_val <= 50:
                rsi_signal = "下落中（様子見）"
            else:
                rsi_signal = "割高（様子見）"
        else:
            rsi_signal = "算出不可"

        # 直近N日以内に下限タッチしたか
        near_low = bool((recent_close <= (range_low + touch_band)).any())
        # 直近N日以内に上限タッチしたか
        near_high = bool((recent_close >= (range_high - touch_band)).any())

        if not near_low and not near_high:
            continue

        # レンジ内での位置 (0%=下限, 100%=上限)
        range_width = range_high - range_low
        position_pct = (
            ((latest_close - range_low) / range_width * 100)
            if range_width > 0
            else 50.0
        )

        pick_type = []
        if near_low:
            pick_type.append("下タッチ（買い候補）")
        if near_high:
            pick_type.append("上タッチ（利確/ブレイク監視）")

        picks.append(
            {
                "date": latest_date,
                "code": row["code"],
                "name": row["name"],
                "size_category": row["size_category"],
                "ticker": ticker,
                "pick_type": "|".join(pick_type),
                "latest_close": round(latest_close, 2),
                "range_high": range_high,
                "range_low": range_low,
                "range_pct": indicators["range_pct"],
                "position_pct": round(position_pct, 1),
                "slope": indicators["slope"],
                "touch_high": indicators["touch_high"],
                "touch_low": indicators["touch_low"],
                "atr": indicators["atr"],
                "rsi": round(rsi_val, 1) if rsi_val is not None else None,
                "rsi_signal": rsi_signal,
            }
        )

    picks_df = pd.DataFrame(picks)
    if len(picks_df) > 0:
        picks_df.sort_values(
            ["pick_type", "position_pct"], ascending=[True, True], inplace=True
        )
    picks_df.to_csv(DAILY_PICKS_CSV, index=False, encoding="utf-8-sig")
    return picks_df
