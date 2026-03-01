"""ABCD戦略エンジン: パターン判定・スコアリング・ランキング生成

strategy.yaml の定義に基づいて、各銘柄に対して A/B/C/D パターンの判定と
補正付きスコアリングを行い、推奨ランキングを出力する。
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from modules.strategy_loader import (
    get_candle_patterns,
    get_features_config,
    get_patterns,
    get_scoring,
    load_strategy,
)

logger = logging.getLogger(__name__)


# ============================================================================
# テクニカル指標の計算
# ============================================================================

def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_volume_ratio(volume: pd.Series, ma_period: int = 20) -> pd.Series:
    vol_ma = volume.rolling(window=ma_period, min_periods=ma_period).mean()
    return volume / vol_ma


def compute_turnover(close: pd.Series, volume: pd.Series,
                     window: int = 20) -> pd.Series:
    daily_turnover = close * volume
    return daily_turnover.rolling(window=window, min_periods=window).mean()


def compute_all_features(df: pd.DataFrame,
                         strategy: dict | None = None) -> dict[str, pd.Series]:
    """strategy.yaml の features 設定に基づき全指標を計算して辞書で返す。"""
    cfg = get_features_config(strategy)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    features: dict[str, pd.Series] = {"close": close, "high": high,
                                       "low": low, "volume": volume}

    # SMA
    for p in cfg.get("moving_averages", [20, 50, 100, 200]):
        features[f"sma_{p}"] = compute_sma(close, p)

    # RSI
    rsi_period = cfg.get("rsi", {}).get("period", 14)
    features["rsi"] = compute_rsi(close, rsi_period)

    # ATR
    atr_period = cfg.get("atr", {}).get("period", 14)
    features["atr"] = compute_atr(high, low, close, atr_period)
    features["atr_pct"] = features["atr"] / close  # ATR%

    # Volume ratio
    vol_ma = cfg.get("volume_ratio", {}).get("ma_period", 20)
    features["volume_ratio"] = compute_volume_ratio(volume, vol_ma)

    # Turnover (avg daily turnover JPY)
    to_window = cfg.get("turnover", {}).get("window_days", 20)
    features["turnover"] = compute_turnover(close, volume, to_window)

    # Open (ローソク足パターン用)
    features["open"] = df["Open"].astype(float)

    return features


# ============================================================================
# ローソク足パターン判定
# ============================================================================

def is_bullish_reversal_candle(open_: float, high: float, low: float,
                               close: float,
                               strategy: dict | None = None) -> bool:
    """strategy.yaml の candle_patterns.bullish_reversal_candle に基づき判定。"""
    cp = get_candle_patterns(strategy).get("bullish_reversal_candle", {})
    rules = cp.get("rules", {})

    body = abs(close - open_)
    day_range = high - low
    if day_range == 0:
        return False

    lower_shadow = min(open_, close) - low

    # 下ヒゲ ≥ 実体 × ratio
    min_ratio = rules.get("lower_shadow_min_body_ratio", 1.5)
    if body > 0 and lower_shadow < body * min_ratio:
        return False
    if body == 0 and lower_shadow == 0:
        return False

    # 実体 ≤ レンジ × ratio
    max_body_ratio = rules.get("body_max_range_ratio", 0.40)
    if body > day_range * max_body_ratio:
        return False

    # 終値 ≥ 始値
    if rules.get("close_gte_open", True) and close < open_:
        return False

    return True


# ============================================================================
# カップウィズハンドル 検出ヘルパー
# ============================================================================

def _detect_cup_with_handle(
    close: pd.Series,
    volume_ratio: pd.Series,
    cfg: dict,
) -> bool:
    """カップウィズハンドル・ブレイクアウトを検出する。

    アルゴリズム:
      1. 本日終値 > ハンドル高値 (ブレイクアウト)
      2. ハンドル深さ ≤ handle_depth_pct_le
      3. 出来高比率 ≥ breakout_volume_ratio_ge
      4. カップ深さ ∈ cup_depth_pct_between
      5. カップ底が中央60%に存在（丸底判定）
      6. ハンドル高値がカップ右縁から12%以内
    """
    cup_depth_min, cup_depth_max = cfg.get("cup_depth_pct_between", [0.12, 0.35])
    cup_len_min, cup_len_max = cfg.get("cup_length_days_between", [35, 325])
    handle_depth_max = cfg.get("handle_depth_pct_le", 0.12)
    handle_len_min, handle_len_max = cfg.get("handle_length_days_between", [5, 25])
    breakout_vol_min = cfg.get("breakout_volume_ratio_ge", 1.4)

    n = len(close)
    if n < cup_len_min + handle_len_min + 5:
        return False

    latest_close = float(close.iloc[-1])

    # 出来高チェック（先に弾く）
    latest_vr = float(volume_ratio.iloc[-1]) if pd.notna(volume_ratio.iloc[-1]) else 0.0
    if latest_vr < breakout_vol_min:
        return False

    # ハンドル長を総当たり（短い方から）
    for h_len in range(handle_len_min, handle_len_max + 1):
        h_start = n - h_len - 1   # ハンドル開始インデックス
        h_end = n - 1              # ハンドル終了（今日の前日まで）
        if h_start < cup_len_min:
            break

        handle_close = close.iloc[h_start:h_end]
        if len(handle_close) < handle_len_min:
            continue

        handle_high = float(handle_close.max())
        handle_low = float(handle_close.min())

        # ブレイクアウト: 本日終値 > ハンドル最高値
        if latest_close <= handle_high:
            continue

        # ハンドル深さ
        if handle_high <= 0 or (handle_high - handle_low) / handle_high > handle_depth_max:
            continue

        # カップ長を総当たり
        for c_len in range(cup_len_min, min(cup_len_max, h_start) + 1):
            c_start = h_start - c_len
            if c_start < 0:
                break

            cup_close = close.iloc[c_start:h_start]

            # カップ左縁・右縁（それぞれ端20%の最高値）
            edge = max(1, c_len // 5)
            left_rim = float(cup_close.iloc[:edge].max())
            right_rim = float(cup_close.iloc[-edge:].max())
            cup_rim = max(left_rim, right_rim)
            if cup_rim <= 0:
                continue

            cup_low = float(cup_close.min())
            cup_depth = (cup_rim - cup_low) / cup_rim

            if not (cup_depth_min <= cup_depth <= cup_depth_max):
                continue

            # ハンドル開始値がカップ右縁に近い（12%以内）
            if right_rim > 0 and (right_rim - handle_high) / right_rim > 0.12:
                continue

            # 丸底チェック: カップ底がカップ期間の中央60%に存在
            cup_low_pos = float(np.argmin(cup_close.values)) / max(1, c_len)
            if not (0.20 <= cup_low_pos <= 0.80):
                continue

            return True

    return False


# ============================================================================
# EPS データ取得
# ============================================================================

def fetch_eps_data(ticker_str: str) -> dict:
    """yfinance から四半期・年次 EPS を取得し、CAN-SLIM 判定用の指標を返す。

    Returns:
        {
            "eps_qtr_yoy_growth": float | None,  # 直近四半期 EPS 前年同期比成長率
            "eps_annual_growth_3y": list,         # 過去3年の年次 EPS 成長率（新→旧）
            "ok": bool,
        }
    """
    import yfinance as yf

    result: dict = {"eps_qtr_yoy_growth": None, "eps_annual_growth_3y": [], "ok": False}
    try:
        t = yf.Ticker(ticker_str)

        # ---- 四半期EPS 前年同期比 ----
        q_stmt = t.quarterly_income_stmt
        if q_stmt is not None and not q_stmt.empty:
            eps_row = None
            for idx in q_stmt.index:
                idx_str = str(idx)
                if "Basic EPS" in idx_str or "Diluted EPS" in idx_str:
                    eps_row = q_stmt.loc[idx].dropna().sort_index(ascending=False)
                    break
            if eps_row is not None and len(eps_row) >= 5:
                latest = float(eps_row.iloc[0])
                year_ago = float(eps_row.iloc[4])
                if year_ago != 0:
                    result["eps_qtr_yoy_growth"] = (latest - year_ago) / abs(year_ago)

        # ---- 年次EPS 成長率（直近3年分）----
        a_stmt = t.income_stmt
        if a_stmt is not None and not a_stmt.empty:
            eps_row = None
            for idx in a_stmt.index:
                idx_str = str(idx)
                if "Basic EPS" in idx_str or "Diluted EPS" in idx_str:
                    eps_row = a_stmt.loc[idx].dropna().sort_index(ascending=False)
                    break
            if eps_row is not None and len(eps_row) >= 4:
                annual_growth = []
                for i in range(min(3, len(eps_row) - 1)):
                    cur = float(eps_row.iloc[i])
                    prev = float(eps_row.iloc[i + 1])
                    if prev != 0:
                        annual_growth.append((cur - prev) / abs(prev))
                    else:
                        annual_growth.append(None)
                result["eps_annual_growth_3y"] = annual_growth

        result["ok"] = True
    except Exception as e:
        logger.debug(f"EPS取得エラー ({ticker_str}): {e}")

    return result


# ============================================================================
# パターン判定 (A/B/C/D)
# ============================================================================

def check_pattern_A(features: dict[str, pd.Series],
                    pattern_cfg: dict) -> bool:
    """A_trend: トレンド継続"""
    close = features["close"]
    if len(close) < 200:
        return False
    latest_close = float(close.iloc[-1])
    sma_200 = features.get("sma_200")
    sma_50 = features.get("sma_50")

    if sma_200 is None or sma_50 is None:
        return False
    if pd.isna(sma_200.iloc[-1]) or pd.isna(sma_50.iloc[-1]):
        return False

    # close > SMA200
    if latest_close <= float(sma_200.iloc[-1]):
        return False

    # SMA50 > SMA200 (ゴールデンクロス状態)
    if float(sma_50.iloc[-1]) <= float(sma_200.iloc[-1]):
        return False

    # SMA50の傾きが正 (lookback期間)
    lookback = 10
    for cond in pattern_cfg.get("entry", []):
        if isinstance(cond, dict) and "sma_slope_positive" in cond:
            lookback = cond["sma_slope_positive"].get("lookback", 10)
    if len(sma_50.dropna()) < lookback + 1:
        return False
    slope = float(sma_50.iloc[-1]) - float(sma_50.iloc[-1 - lookback])
    if slope <= 0:
        return False

    return True


def check_pattern_B(features: dict[str, pd.Series],
                    pattern_cfg: dict) -> bool:
    """B_pullback: 押し目"""
    close = features["close"]
    high = features["high"]
    if len(close) < 100:
        return False
    latest_close = float(close.iloc[-1])
    sma_100 = features.get("sma_100")
    rsi = features.get("rsi")
    atr = features.get("atr")

    if sma_100 is None or rsi is None or atr is None:
        return False
    if pd.isna(sma_100.iloc[-1]) or pd.isna(rsi.iloc[-1]) or pd.isna(atr.iloc[-1]):
        return False

    # close > SMA100
    if latest_close <= float(sma_100.iloc[-1]):
        return False

    # RSI between [40, 55]
    rsi_val = float(rsi.iloc[-1])
    rsi_range = [40, 55]
    for cond in pattern_cfg.get("entry", []):
        if isinstance(cond, dict) and "rsi_between" in cond:
            rsi_range = cond["rsi_between"]
    if not (rsi_range[0] <= rsi_val <= rsi_range[1]):
        return False

    # pullback_atr_between: (直近N日高値 - 当日終値) / ATR
    lookback = pattern_cfg.get("pullback_high_lookback", 10)
    atr_range = [1.0, 2.5]
    for cond in pattern_cfg.get("entry", []):
        if isinstance(cond, dict) and "pullback_atr_between" in cond:
            atr_range = cond["pullback_atr_between"]
    recent_high = float(high.iloc[-lookback:].max())
    atr_val = float(atr.iloc[-1])
    if atr_val <= 0:
        return False
    pullback_atr = (recent_high - latest_close) / atr_val
    if not (atr_range[0] <= pullback_atr <= atr_range[1]):
        return False

    # reversal_break_prev_high: 当日終値 > 前日高値
    if len(high) < 2:
        return False
    prev_high = float(high.iloc[-2])
    if latest_close <= prev_high:
        return False

    return True


def check_pattern_C(features: dict[str, pd.Series],
                    pattern_cfg: dict) -> bool:
    """C_breakout: ブレイクアウト"""
    close = features["close"]
    high = features["high"]
    volume_ratio = features.get("volume_ratio")
    atr_pct = features.get("atr_pct")

    if volume_ratio is None or atr_pct is None:
        return False

    # breakout_high: N日高値更新
    breakout_n = 20
    for cond in pattern_cfg.get("entry", []):
        if isinstance(cond, dict) and "breakout_high" in cond:
            breakout_n = cond["breakout_high"]
    if len(high) < breakout_n + 1:
        return False
    past_high = float(high.iloc[-(breakout_n + 1):-1].max())
    latest_high = float(high.iloc[-1])
    if latest_high <= past_high:
        return False

    # volume_ratio > threshold
    vr_threshold = 1.5
    for cond in pattern_cfg.get("entry", []):
        if isinstance(cond, dict) and "volume_ratio_gt" in cond:
            vr_threshold = cond["volume_ratio_gt"]
    if pd.isna(volume_ratio.iloc[-1]) or float(volume_ratio.iloc[-1]) <= vr_threshold:
        return False

    # atr_percent_rank <= threshold (ボラ収縮)
    rank_threshold = 40
    for cond in pattern_cfg.get("entry", []):
        if isinstance(cond, dict) and "atr_percent_rank_le" in cond:
            rank_threshold = cond["atr_percent_rank_le"]
    rank_window = pattern_cfg.get("atr_percent_rank_window", 60)
    if len(atr_pct.dropna()) < rank_window:
        return False
    recent_atr_pct = atr_pct.dropna().iloc[-rank_window:]
    current_atr_pct = float(atr_pct.iloc[-1])
    if pd.isna(current_atr_pct):
        return False
    pct_rank = float((recent_atr_pct < current_atr_pct).sum()) / len(recent_atr_pct) * 100
    if pct_rank > rank_threshold:
        return False

    return True


def check_pattern_D(features: dict[str, pd.Series],
                    pattern_cfg: dict,
                    strategy: dict | None = None) -> bool:
    """D_reversal: リバーサル"""
    close = features["close"]
    sma_200 = features.get("sma_200")
    open_ = features.get("open")
    high = features["high"]
    low = features["low"]

    if sma_200 is None or open_ is None:
        return False
    if len(close) < 200:
        return False
    if pd.isna(sma_200.iloc[-1]):
        return False

    latest_close = float(close.iloc[-1])

    # close > SMA200 (下落トレンド銘柄は避ける)
    if latest_close <= float(sma_200.iloc[-1]):
        return False

    # drop_pct_last_n_days_between
    n = 5
    drop_range = [-0.15, -0.08]
    for cond in pattern_cfg.get("entry", []):
        if isinstance(cond, dict) and "drop_pct_last_n_days_between" in cond:
            params = cond["drop_pct_last_n_days_between"]
            n = params.get("n", 5)
            drop_range = params.get("range", [-0.15, -0.08])
    if len(close) < n + 1:
        return False
    price_n_days_ago = float(close.iloc[-1 - n])
    if price_n_days_ago == 0:
        return False
    drop_pct = (latest_close - price_n_days_ago) / price_n_days_ago
    if not (drop_range[0] <= drop_pct <= drop_range[1]):
        return False

    # bullish_reversal_candle (ハンマー判定)
    if not is_bullish_reversal_candle(
        float(open_.iloc[-1]),
        float(high.iloc[-1]),
        float(low.iloc[-1]),
        latest_close,
        strategy,
    ):
        return False

    return True


def check_pattern_E(
    features: dict[str, pd.Series],
    pattern_cfg: dict,
    eps_data: dict | None = None,
) -> bool:
    """E_can_slim: CAN-SLIM (業績成長 × カップウィズハンドル・ブレイクアウト)

    C: 直近四半期 EPS 前年同期比 ≥ 25%
    A: 過去3年の年次 EPS が毎年 ≥ 25% 成長
    N: カップウィズハンドル・ブレイクアウト
    """
    close = features["close"]
    volume_ratio = features.get("volume_ratio")

    if volume_ratio is None or len(close) < 50:
        return False

    entry = pattern_cfg.get("entry", [])

    # --- C: 直近四半期 EPS 前年同期比 ---
    qtr_threshold = 0.25
    for cond in entry:
        if isinstance(cond, dict) and "eps_qtr_yoy_growth_ge" in cond:
            qtr_threshold = cond["eps_qtr_yoy_growth_ge"]

    # --- A: 過去3年 年次EPS 成長率 ---
    annual_threshold = 0.25
    for cond in entry:
        if isinstance(cond, dict) and "eps_annual_growth_3y_ge" in cond:
            annual_threshold = cond["eps_annual_growth_3y_ge"]

    if eps_data is not None and eps_data.get("ok"):
        # C: 四半期EPS — None の場合は日本株でデータ未取得のためスキップ
        qtr_growth = eps_data.get("eps_qtr_yoy_growth")
        if qtr_growth is not None and qtr_growth < qtr_threshold:
            return False

        # A: 年次EPS 3年連続成長
        annual_growths = eps_data.get("eps_annual_growth_3y", [])
        if len(annual_growths) < 3:
            return False
        if not all(g is not None and g >= annual_threshold for g in annual_growths[:3]):
            return False
    else:
        # EPS データなし → 業績条件スキップ（チャートのみ評価）
        logger.debug("E_can_slim: EPS データなし。チャートパターンのみ判定。")

    # --- N: カップウィズハンドル ---
    cup_handle_cfg: dict = {}
    for cond in entry:
        if isinstance(cond, dict) and "cup_handle_breakout" in cond:
            cup_handle_cfg = cond["cup_handle_breakout"]
            break

    return _detect_cup_with_handle(close, volume_ratio, cup_handle_cfg)


_PATTERN_CHECKERS = {
    "A_trend": check_pattern_A,
    "B_pullback": check_pattern_B,
    "C_breakout": check_pattern_C,
    "D_reversal": check_pattern_D,
    "E_can_slim": check_pattern_E,
}


def detect_patterns(
    features: dict[str, pd.Series],
    strategy: dict | None = None,
    eps_data: dict | None = None,
) -> list[str]:
    """全パターンをチェックし、成立したパターン名リストを返す。"""
    s = strategy or load_strategy()
    patterns_cfg = get_patterns(s)
    matched = []
    for pat_name, cfg in patterns_cfg.items():
        checker = _PATTERN_CHECKERS.get(pat_name)
        if checker is None:
            continue
        try:
            if pat_name == "D_reversal":
                result = checker(features, cfg, s)
            elif pat_name == "E_can_slim":
                result = checker(features, cfg, eps_data)
            else:
                result = checker(features, cfg)
            if result:
                matched.append(pat_name)
        except Exception as e:
            logger.debug(f"Pattern check {pat_name} error: {e}")
    return matched


# ============================================================================
# スコアリング
# ============================================================================

def _calc_multiplier_piecewise(value: float, points: list[dict]) -> float:
    """piecewise型の補正乗数を計算する。"""
    for p in points:
        if "top_pct" in p:
            if value <= p["top_pct"]:
                return p["m"]
        elif "lte" in p:
            if value <= p["lte"]:
                return p["m"]
        elif "gt" in p:
            if value > p["gt"]:
                return p["m"]
        elif "else" in p:
            return p["m"]
    return 1.0


def _calc_multiplier_linear(value: float, params: dict,
                            clip: list[float] | None = None) -> float:
    """linear型の補正乗数を計算する。"""
    x0 = params["x0"]
    x1 = params["x1"]
    m0 = params["m0"]
    m1 = params["m1"]
    if x1 == x0:
        return m0
    t = (value - x0) / (x1 - x0)
    m = m0 + t * (m1 - m0)
    if clip:
        m = max(clip[0], min(clip[1], m))
    return m


def compute_score(pattern: str,
                  features: dict[str, pd.Series],
                  turnover_rank_pct: float,
                  is_etf: bool = False,
                  strategy: dict | None = None) -> float:
    """パターンに対するスコアを計算する。

    Args:
        pattern: パターン名 (例: "A_trend")
        features: compute_all_features の戻り値
        turnover_rank_pct: 売買代金の順位パーセンタイル (0=最も多い, 100=最も少ない)
        is_etf: ETFかどうか
        strategy: strategy辞書（省略時はload_strategy()）

    Returns:
        補正済みスコア
    """
    s = strategy or load_strategy()
    scoring_cfg = get_scoring(s)

    # 基本点
    base_points = scoring_cfg.get("base_points", {})
    base = base_points.get(pattern, 0)

    # ETFオーバーライド
    if is_etf:
        overrides = scoring_cfg.get("instrument_overrides", {}).get("ETF", {})
        override_bp = overrides.get("base_points", {})
        if pattern in override_bp:
            base = override_bp[pattern]

    # 適用する補正リスト
    apply_list = scoring_cfg.get("apply_to_pattern", {}).get(pattern, [])

    # 補正乗数の定義を取得
    multipliers_common = scoring_cfg.get("multipliers", {}).get("common", {})
    if is_etf:
        etf_mult = scoring_cfg.get("instrument_overrides", {}).get(
            "ETF", {}).get("multipliers", {}).get("common", {})
        # ETF用があればそちらを優先
        merged_mult = {**multipliers_common, **etf_mult}
    else:
        merged_mult = multipliers_common

    # 各乗数を計算
    product = 1.0
    for mult_name in apply_list:
        mult_cfg = merged_mult.get(mult_name)
        if mult_cfg is None:
            continue
        mult_type = mult_cfg.get("type")

        if mult_name == "turnover_rank":
            value = turnover_rank_pct
        elif mult_name == "atr_percent":
            atr_pct = features.get("atr_pct")
            value = float(atr_pct.iloc[-1]) if atr_pct is not None and not pd.isna(atr_pct.iloc[-1]) else 0.03
        elif mult_name == "volume_ratio":
            vr = features.get("volume_ratio")
            value = float(vr.iloc[-1]) if vr is not None and not pd.isna(vr.iloc[-1]) else 1.0
        else:
            continue

        if mult_type == "piecewise":
            m = _calc_multiplier_piecewise(value, mult_cfg.get("points", []))
        elif mult_type == "linear":
            m = _calc_multiplier_linear(
                value, mult_cfg.get("params", {}), mult_cfg.get("clip"))
        else:
            m = 1.0

        product *= m

    return round(base * product, 2)


# ============================================================================
# ランキング生成
# ============================================================================

def evaluate_single(ticker: str, df: pd.DataFrame,
                    turnover_rank_pct: float,
                    is_etf: bool = False,
                    strategy: dict | None = None,
                    eps_data: dict | None = None) -> dict[str, Any] | None:
    """1銘柄に対してパターン判定 + スコアリングを実行する。

    Returns:
        {
            "ticker": ...,
            "matched_patterns": ["A_trend", ...],
            "scores": {"A_trend": 33.0, ...},
            "best_score": 33.0,
            "best_pattern": "A_trend",
            "features_summary": { ... },
        }
        パターン成立なしの場合は None
    """
    s = strategy or load_strategy()

    if df is None or len(df) < 50:
        return None

    features = compute_all_features(df, s)
    matched = detect_patterns(features, s, eps_data)

    if not matched:
        return None

    scores = {}
    for pat in matched:
        scores[pat] = compute_score(pat, features, turnover_rank_pct,
                                    is_etf, s)

    best_pattern = max(scores, key=scores.get)
    best_score = scores[best_pattern]

    # サマリー指標
    close_val = float(features["close"].iloc[-1])
    summary = {
        "close": close_val,
        "rsi": round(float(features["rsi"].iloc[-1]), 1) if not pd.isna(features["rsi"].iloc[-1]) else None,
        "atr_pct": round(float(features["atr_pct"].iloc[-1]) * 100, 2) if not pd.isna(features["atr_pct"].iloc[-1]) else None,
        "volume_ratio": round(float(features["volume_ratio"].iloc[-1]), 2) if not pd.isna(features["volume_ratio"].iloc[-1]) else None,
    }
    # SMA状態
    for p in [50, 100, 200]:
        sma_key = f"sma_{p}"
        if sma_key in features and not pd.isna(features[sma_key].iloc[-1]):
            summary[f"vs_sma{p}"] = round(
                (close_val / float(features[sma_key].iloc[-1]) - 1) * 100, 2)

    return {
        "ticker": ticker,
        "matched_patterns": matched,
        "scores": scores,
        "best_score": best_score,
        "best_pattern": best_pattern,
        "features_summary": summary,
    }


def generate_ranking(stock_list: pd.DataFrame,
                     load_cached_fn,
                     strategy: dict | None = None,
                     max_positions: int | None = None,
                     eps_data_map: dict[str, dict] | None = None) -> pd.DataFrame:
    """全銘柄を評価し、スコア順にランキングを生成する。

    Args:
        stock_list: 銘柄リスト DataFrame (code, name, ticker, size_category 等)
        load_cached_fn: ticker → DataFrame を返す関数
        strategy: strategy辞書
        max_positions: 上位N件に絞る（None=全件）
        eps_data_map: {ticker: fetch_eps_data(ticker) の戻り値} の辞書。
                      E_can_slim の業績条件チェックに使用。None の場合は業績条件スキップ。

    Returns:
        ランキング DataFrame
    """
    s = strategy or load_strategy()
    exec_cfg = s.get("execution", {})
    if max_positions is None:
        max_positions = exec_cfg.get("max_positions", 20)

    # 売買代金ランキングを先に計算（全銘柄の turnover を集計）
    turnover_values = {}
    for _, row in stock_list.iterrows():
        ticker = row["ticker"]
        df = load_cached_fn(ticker)
        if df is not None and len(df) >= 20:
            close = df["Close"].astype(float)
            volume = df["Volume"].astype(float)
            avg_turnover = float((close * volume).tail(20).mean())
            turnover_values[ticker] = avg_turnover

    # 順位パーセンタイル (0=最も多い)
    if turnover_values:
        sorted_tickers = sorted(turnover_values, key=turnover_values.get,
                                reverse=True)
        total = len(sorted_tickers)
        turnover_rank = {t: (i / total) * 100
                         for i, t in enumerate(sorted_tickers)}
    else:
        turnover_rank = {}

    # 各銘柄を評価
    results = []
    for _, row in stock_list.iterrows():
        ticker = row["ticker"]
        df = load_cached_fn(ticker)
        is_etf = str(row.get("size_category", "")).upper() == "ETF"
        rank_pct = turnover_rank.get(ticker, 50.0)

        ticker_eps = eps_data_map.get(ticker) if eps_data_map else None
        eval_result = evaluate_single(ticker, df, rank_pct, is_etf, s, ticker_eps)
        if eval_result is None:
            continue

        summary = eval_result["features_summary"]
        results.append({
            "code": row["code"],
            "name": row["name"],
            "size_category": row.get("size_category", ""),
            "sector_33": row.get("sector_33", ""),
            "ticker": ticker,
            "matched_patterns": "|".join(eval_result["matched_patterns"]),
            "best_pattern": eval_result["best_pattern"],
            "best_score": eval_result["best_score"],
            "all_scores": str(eval_result["scores"]),
            "close": summary.get("close"),
            "rsi": summary.get("rsi"),
            "atr_pct": summary.get("atr_pct"),
            "volume_ratio": summary.get("volume_ratio"),
            "vs_sma50": summary.get("vs_sma50"),
            "vs_sma100": summary.get("vs_sma100"),
            "vs_sma200": summary.get("vs_sma200"),
            "turnover_rank_pct": round(rank_pct, 1),
        })

    if not results:
        return pd.DataFrame()

    ranking = pd.DataFrame(results)
    ranking.sort_values("best_score", ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    ranking.index = ranking.index + 1  # 1始まりのランク
    ranking.index.name = "rank"

    if max_positions:
        ranking = ranking.head(max_positions)

    return ranking
