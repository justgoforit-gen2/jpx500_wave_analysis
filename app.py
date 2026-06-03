"""JPX500 波形タイプ分類アプリ - Streamlit MVP"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import plotly.express as px

from config.settings import (
    CACHE_DIR,
    DAILY_PICKS_CSV,
    STOCK_LIST_CSV,
    RANGE_PCT_LARGE,
    RANGE_PCT_SMALL,
    RESULTS_CSV,
    WAVE_TYPES,
    WINDOW_OPTIONS,
    TOUCH_THRESHOLD_PCT,
    SLOPE_THRESHOLD,
    RANGE_MIN_TOUCHES,
    RANGE_PERCENTILE_HIGH,
    RANGE_PERCENTILE_LOW,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    BREAKOUT_LOOKBACK_DAYS,
    BREAKOUT_MIN_DAYS,
    HIGH_VOLATILITY_THRESHOLD,
    SQUEEZE_BANDWIDTH_SHRINK,
    DAILY_PICK_LOOKBACK,
    PER_PBR_HISTORY_PARQUET,
    PER_PBR_DEFAULT_PER_CAP,
    PER_PBR_DEFAULT_PBR_CAP,
)
from modules.chart_builder import (
    build_chart,
    build_comparison_chart,
    build_financials_chart,
    build_flow_index_dual_chart,
    build_index_chart,
)
from modules.data_fetcher import (
    load_cached,
    load_stock_list,
    get_nikkei225,
    get_topix,
    compute_sector_index,
    compute_size_index,
    fetch_financials,
    compute_sector_stats,
    fetch_and_cache,
    GLOBAL_INDICES,
    convert_ohlcv_close_to_jpy_by_month_avg,
)
from modules.foreign_flow_analyzer import (
    compute_cumulative_flow,
    compute_flow_index_correlation,
    compute_sector_flow_correlation,
    compute_size_flow_correlation,
    load_foreign_flow,
)
from modules.capital_efficiency_screener import (
    load_screening_result as _load_ces_result,
)
from modules.earnings_fetcher import load_earnings_dates, get_earnings_dates_for_code
from modules.wave_classifier import compute_indicators, classify
from modules.strategy_engine import generate_ranking
from modules.strategy_loader import load_strategy
from modules.backtester import build_contexts, run_backtest
from modules.portfolio_manager import (
    add_position,
    compute_current_valuation,
    compute_performance_metrics,
    initialize_from_template,
    load_portfolio,
    load_portfolio_history,
    record_sell,
)
from modules.signal_engine import get_today_signals
from modules.trend_transition_detector import (
    detect_transitions,
    load_trend_transition,
)
from modules.range_breakout_detector import (
    load_range_breakout,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_price_data_cached(ticker: str) -> pd.DataFrame | None:
    return fetch_and_cache(ticker)


# PER/PBR 散布図・アニメーションで使う固定カラーマップ。
# Prime: Core30=赤 / Large70=橙 / Mid400=緑 (時価総額が大きい順に暖色)。
# Standard: Top100=紫 / Top400=青(青系で Prime と差別化、Standard 内も濃淡を持たせる)。
SIZE_COLOR_MAP = {
    "TOPIX Core30": "#D32F2F",
    "TOPIX Large70": "#F57C00",
    "TOPIX Mid400": "#388E3C",
    "TSE Standard Top100": "#7B1FA2",
    "TSE Standard Top400": "#5C6BC0",
}

# 市場ラベル (UI フィルタ用)
MARKET_OPTIONS = ["TSE Prime", "TSE Standard"]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _get_financials_cached(ticker: str) -> pd.DataFrame | None:
    return fetch_financials(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_per_pbr_history() -> pd.DataFrame | None:
    """週次PER/PBR履歴 parquet をロード（存在しなければ None）"""
    if not PER_PBR_HISTORY_PARQUET.exists():
        return None
    try:
        df = pd.read_parquet(PER_PBR_HISTORY_PARQUET)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


st.set_page_config(
    page_title="JPX500 波形分類",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 用語・指標の定義（ヘルプ用）
# ------------------------------------------------------------------
HELP_TEXT = f"""
### 波形タイプの定義

| タイプ | 判定条件 | 意味 |
|--------|----------|------|
| **レンジ（波型）** | 傾き(slope)の絶対値 < {SLOPE_THRESHOLD} かつ タッチ合計 ≥ {RANGE_MIN_TOUCHES}回 | トレンドが弱く、一定の価格帯で上下を反復している状態 |
| **上昇トレンド** | 傾き > {SLOPE_THRESHOLD} かつ 高値が切り上がっている | 中長期的に価格が右肩上がり |
| **下降トレンド** | 傾き < -{SLOPE_THRESHOLD} かつ 安値が切り下がっている | 中長期的に価格が右肩下がり |
| **収束（スクイーズ）** | ボリンジャーバンド幅が後半で前半の{int(SQUEEZE_BANDWIDTH_SHRINK * 100)}%以下に縮小 | ボラティリティが低下し、次の大きな動きの前兆となりうる状態 |
| **ブレイク気味** | 直近{BREAKOUT_LOOKBACK_DAYS}日中{BREAKOUT_MIN_DAYS}日以上がレンジ外 | レンジの上限/下限を明確に超えた状態（終値基準） |
| **高ボラ（荒い）** | ATR ÷ 平均価格 > {HIGH_VOLATILITY_THRESHOLD * 100:.0f}% | 価格変動が大きく、安定した反復が弱い |

### 指標の定義

| 指標 | 定義 |
|------|------|
| **レンジ幅%** (range_pct) | (上限 - 下限) ÷ 中央値 × 100。評価窓内の価格の上位{RANGE_PERCENTILE_HIGH}%ile と下位{RANGE_PERCENTILE_LOW}%ile の差 |
| **タッチ回数** (touch_total) | 終値がレンジ上限付近（±{TOUCH_THRESHOLD_PCT}%以内）または下限付近に到達した日数の合計 |
| **上タッチ / 下タッチ** | それぞれレンジ上限/下限付近への到達日数 |
| **傾き** (slope) | 評価窓内の終値に線形回帰を適用した傾き。平均価格で正規化済み。正=上昇、負=下降 |
| **ATR** | Average True Range（{ATR_PERIOD}日平均）。1日の価格変動幅の平均。大きいほどボラティリティが高い |
| **バンド幅** (bandwidth) | ボリンジャーバンド（{BB_PERIOD}日, {BB_STD}σ）の幅 ÷ SMA。収束判定に使用 |
| **レンジ外日数** (breakout_days) | 直近{BREAKOUT_LOOKBACK_DAYS}日中、終値がレンジ上限/下限の外にあった日数 |

### 推奨銘柄の定義

**「レンジ→上昇転換候補」**: 以下の条件をすべて満たす銘柄
1. 波形タイプに「レンジ（波型）」を含む
2. 傾き(slope)がわずかにプラス（0 〜 {SLOPE_THRESHOLD}）
3. 直近で上限付近にいる（ブレイク気味を含む、または上タッチが多い）

### 本日の推奨銘柄

レンジ（波型）と判定された銘柄の中から、**直近{DAILY_PICK_LOOKBACK}営業日以内**にレンジ端にタッチした銘柄を抽出。

| 種類 | 意味 | 使い方 |
|------|------|--------|
| **下タッチ（買い候補）** | 終値がレンジ下限付近（±{TOUCH_THRESHOLD_PCT}%以内）に到達 | レンジ内での反発を狙う買いエントリーポイント |
| **上タッチ（利確/ブレイク監視）** | 終値がレンジ上限付近（±{TOUCH_THRESHOLD_PCT}%以内）に到達 | 保有中なら利確検討、上抜けならブレイクアウト狙い |

**位置%**: レンジ内での現在位置（0%=下限、100%=上限）。0%に近いほど下限寄り。

### RSIシグナルの見方

| RSI値 | シグナル | 意味 |
|--------|----------|------|
| **≤ 30** | 売られすぎ（反発期待） | 過度に売られた状態。反発の可能性が高い |
| **30〜50** | 下落中（様子見） | まだ下落トレンドの途中。追加下落リスクあり |
| **> 50** | 割高（様子見） | 下タッチでもRSIは高め。反発は限定的かも |

### チャートの見方

- **赤▼マーカー**: 終値がレンジ上限付近（タッチ）した日
- **青▲マーカー**: 終値がレンジ下限付近（タッチ）した日
- **オレンジ破線（横）**: レンジ上限/下限ライン
- **オレンジ破線（縦）**: 決算発表予定日
- **青い背景**: 評価窓（分析対象期間）
- **MA13週/26週/52週**: 13/26/52週移動平均線
"""


def get_data_dates() -> dict[str, str]:
    """データの最新日付と更新実行日時を取得する"""
    info = {}
    # results.csvのファイル更新日時 = バッチ実行日時
    if RESULTS_CSV.exists():
        mtime = RESULTS_CSV.stat().st_mtime
        info["batch_run"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    latest_dates = []

    # daily_picks.csv の date 列はバッチが生成するため、Cloudでも追随しやすい
    if DAILY_PICKS_CSV.exists():
        try:
            picks = pd.read_csv(
                DAILY_PICKS_CSV, encoding="utf-8-sig", dtype={"code": str}
            )
            if "date" in picks.columns and len(picks) > 0:
                d = pd.to_datetime(picks["date"], errors="coerce").max()
                if pd.notna(d):
                    latest_dates.append(pd.Timestamp(d))
        except Exception:
            pass

    # キャッシュparquetから実際の最新株価日付を取得
    if CACHE_DIR.exists():
        parquets = list(CACHE_DIR.glob("*.parquet"))
        if parquets:
            try:
                # 任意の先頭ファイルだと古いキャッシュを引くことがあるため、
                # 更新日時が一番新しいファイルを参照する。
                newest = max(parquets, key=lambda p: p.stat().st_mtime)
                sample = pd.read_parquet(newest)
                sample.index = pd.to_datetime(sample.index)
                d = sample.index.max()
                if pd.notna(d):
                    latest_dates.append(pd.Timestamp(d))
            except Exception:
                pass

    if latest_dates:
        info["latest_data"] = max(latest_dates).strftime("%Y-%m-%d")
    return info


def load_results() -> pd.DataFrame | None:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"code": str})
    return None


def load_daily_picks() -> pd.DataFrame | None:
    if DAILY_PICKS_CSV.exists():
        return pd.read_csv(DAILY_PICKS_CSV, encoding="utf-8-sig", dtype={"code": str})
    return None


def load_abcd_ranking() -> pd.DataFrame | None:
    fp = Path(__file__).resolve().parent / "data" / "abcd_ranking.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp, encoding="utf-8-sig", dtype={"code": str})
        if "rank" in df.columns:
            df.set_index("rank", inplace=True)
        return df
    except Exception:
        return None


def is_recommended(row: pd.Series) -> bool:
    """推奨銘柄判定: レンジ→上昇転換候補"""
    if pd.isna(row.get("wave_types")) or pd.isna(row.get("slope")):
        return False
    types = str(row["wave_types"])
    slope = float(row["slope"])
    touch_high = int(row["touch_high"]) if pd.notna(row.get("touch_high")) else 0
    has_breakout = "ブレイク気味" in types

    return (
        "レンジ（波型）" in types and slope >= 0 and (touch_high >= 3 or has_breakout)
    )


# ------------------------------------------------------------------
# 一覧画面
# ------------------------------------------------------------------
def _show_trend_transition_section() -> None:
    """下降→上昇トレンド転換候補セクション(波形分類タブの冒頭に置く)。

    過去 25 日と直近 25 日の slope を別計算し、符号が反転した銘柄を抽出。
    既存の wave_classifier は 120 日窓のため転換中の銘柄を取りこぼすが、
    この検出器はその「早期トランジション」をピンポイントで拾う。
    """
    with st.expander(
        "🔄 トレンド転換検出 — 下降→上昇に切り替わった銘柄(直近25日)",
        expanded=False,
    ):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            window = st.number_input(
                "窓(日)",
                min_value=10,
                max_value=60,
                value=25,
                step=5,
                key="tt_window",
            )
        with col2:
            past_max = st.number_input(
                "過去 slope 上限",
                min_value=-0.01,
                max_value=0.0,
                value=-0.0005,
                step=0.0001,
                format="%.4f",
                key="tt_past_max",
            )
        with col3:
            recent_min = st.number_input(
                "直近 slope 下限",
                min_value=0.0,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
                key="tt_recent_min",
            )
        with col4:
            min_rebound = st.slider(
                "最低反発率(%)",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=1.0,
                key="tt_min_rebound",
            )

        use_cache = st.checkbox(
            "キャッシュ(日次バッチで生成)を使う(高速)",
            value=True,
            key="tt_use_cache",
            help="OFF にすると全銘柄を再計算します(数十秒)",
        )

        if use_cache:
            tt_df = load_trend_transition()
            if not tt_df.empty:
                tt_df = tt_df[
                    (tt_df["slope_past"] < past_max)
                    & (tt_df["slope_recent"] > recent_min)
                    & (tt_df["rebound_pct"] >= min_rebound)
                ]
        else:
            with st.spinner("全銘柄を再計算中..."):
                tt_df = detect_transitions(
                    window_days=int(window),
                    past_slope_max=float(past_max),
                    recent_slope_min=float(recent_min),
                    min_rebound_pct=float(min_rebound),
                )

        if tt_df is None or tt_df.empty:
            st.info(
                "条件を満たす銘柄がありません。閾値を緩めるか、"
                "バッチ更新を実行してください: `python batch/update.py`"
            )
            return

        st.caption(
            f"該当: **{len(tt_df)} 銘柄** "
            f"(過去 slope < {past_max:.4f} / 直近 slope > {recent_min:.4f} / "
            f"反発 ≥ {min_rebound:.0f}%)"
        )

        display = tt_df.copy()
        if "signal_strength" not in display.columns:
            display["signal_strength"] = (
                (display["slope_recent"] - display["slope_past"]) * 1000
            ).round(2)
        display = display.sort_values("signal_strength", ascending=False)
        show_cols = [
            "code",
            "name",
            "sector_33",
            "size_category",
            "slope_past",
            "slope_recent",
            "signal_strength",
            "rebound_pct",
            "today_close",
            "per",
            "pbr",
            "wave_types",
        ]
        show_cols = [c for c in show_cols if c in display.columns]
        renamed = display[show_cols].rename(
            columns={
                "code": "コード",
                "name": "銘柄",
                "sector_33": "業種",
                "size_category": "規模",
                "slope_past": "過去slope",
                "slope_recent": "直近slope",
                "signal_strength": "強度",
                "rebound_pct": "反発(%)",
                "today_close": "現値",
                "per": "PER",
                "pbr": "PBR",
                "wave_types": "現状の波形",
            }
        )

        event = st.dataframe(
            renamed,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="trend_transition_table",
            column_config={
                "過去slope": st.column_config.NumberColumn(format="%.5f"),
                "直近slope": st.column_config.NumberColumn(format="%.5f"),
                "強度": st.column_config.NumberColumn(format="%.2f"),
                "反発(%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        if event and event.selection.rows:
            r = renamed.iloc[event.selection.rows[0]]
            row_full = display.iloc[event.selection.rows[0]]
            st.session_state["selected_code"] = str(r["コード"])
            st.session_state["selected_name"] = str(r["銘柄"])
            st.session_state["selected_ticker"] = str(row_full["ticker"])
            st.session_state["view"] = "detail"
            st.rerun()


def _show_range_breakout_section() -> None:
    """Stage 2 ブレイクアウト検出セクション。

    「長期レンジ → 全 MA 上抜け + RSI 回復」という Stan Weinstein 式の
    Stage 2 入り初動を機械的に拾う。5214 日本電気硝子の 2025/7 がリファレンス。
    """
    with st.expander(
        "🚀 Stage 2 ブレイクアウト — レンジ脱出+全MA上抜け+RSI回復(初動候補)",
        expanded=False,
    ):
        st.caption(
            "**条件**: 過去ベース期がレンジ ／ MA13/26/52週が束ねる ／ "
            "現値が全 MA を浅く上抜け(+15% 以内) ／ "
            "RSI が直近30日で 40 を割って回復中(40-70)。"
            "5214 日本電気硝子の 2025/7 型を捕捉する設定。"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            max_above = st.slider(
                "走り始めの浅さ上限(% above max MA)",
                min_value=0.0,
                max_value=30.0,
                value=15.0,
                step=1.0,
                key="rb_max_above",
            )
        with col2:
            rsi_now_min = st.slider(
                "現 RSI 下限",
                min_value=30,
                max_value=60,
                value=40,
                step=1,
                key="rb_rsi_now_min",
            )
        with col3:
            rsi_now_max = st.slider(
                "現 RSI 上限(過熱除外)",
                min_value=50,
                max_value=85,
                value=70,
                step=1,
                key="rb_rsi_now_max",
            )

        golden_only = st.checkbox(
            "MA がゴールデンオーダ(短>中>長)のみ表示",
            value=False,
            key="rb_golden_only",
            help="ON にすると Stage 2 確定の銘柄に絞られます(銘柄数減)",
        )

        df_rb = load_range_breakout()
        if df_rb.empty:
            st.info(
                "キャッシュが空です。バッチを実行してください: `python batch/update.py`"
            )
            return

        # UI フィルタ適用
        filtered = df_rb[
            (df_rb["above_max_ma_pct"] <= max_above)
            & (df_rb["rsi_now"] >= rsi_now_min)
            & (df_rb["rsi_now"] <= rsi_now_max)
        ]
        if golden_only:
            filtered = filtered[filtered["golden_order"] == True]  # noqa: E712

        if filtered.empty:
            st.info("条件を満たす銘柄がありません。閾値を緩めてください")
            return

        st.caption(f"該当: **{len(filtered)} 銘柄** (強度降順)")

        show_cols = [
            "code",
            "name",
            "sector_33",
            "size_category",
            "signal_strength",
            "above_max_ma_pct",
            "rsi_now",
            "rsi_min_recent",
            "ma_tightness_pct",
            "golden_order",
            "close",
            "ma_short",
            "ma_mid",
            "ma_long",
            "per",
            "pbr",
            "wave_types",
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]
        renamed = filtered[show_cols].rename(
            columns={
                "code": "コード",
                "name": "銘柄",
                "sector_33": "業種",
                "size_category": "規模",
                "signal_strength": "強度",
                "above_max_ma_pct": "浅さ(%)",
                "rsi_now": "現RSI",
                "rsi_min_recent": "30日最小RSI",
                "ma_tightness_pct": "MA束ね(%)",
                "golden_order": "GoldenOrder",
                "close": "現値",
                "ma_short": "MA13週",
                "ma_mid": "MA26週",
                "ma_long": "MA52週",
                "per": "PER",
                "pbr": "PBR",
                "wave_types": "波形分類",
            }
        )

        event = st.dataframe(
            renamed,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="range_breakout_table",
            column_config={
                "強度": st.column_config.NumberColumn(format="%.1f"),
                "浅さ(%)": st.column_config.NumberColumn(format="%.2f%%"),
                "現RSI": st.column_config.NumberColumn(format="%.1f"),
                "30日最小RSI": st.column_config.NumberColumn(format="%.1f"),
                "MA束ね(%)": st.column_config.NumberColumn(format="%.2f%%"),
                "GoldenOrder": st.column_config.CheckboxColumn(),
            },
        )
        if event and event.selection.rows:
            r = renamed.iloc[event.selection.rows[0]]
            row_full = filtered.iloc[event.selection.rows[0]]
            st.session_state["selected_code"] = str(r["コード"])
            st.session_state["selected_name"] = str(r["銘柄"])
            st.session_state["selected_ticker"] = str(row_full["ticker"])
            st.session_state["view"] = "detail"
            st.rerun()


def show_list_view():
    st.title("JPX500 波形タイプ分類")

    # データ更新日時の表示
    dates = get_data_dates()
    if dates:
        data_date = dates.get("latest_data", "不明")
        batch_date = dates.get("batch_run", "不明")
        st.caption(f"株価データ: **{data_date}** 時点 ｜ バッチ実行: {batch_date}")

    # 🚀 Stage 2 ブレイクアウト検出
    _show_range_breakout_section()

    # 🔄 下降→上昇トレンド転換検出(エクスパンダー、既存の filter には影響しない)
    _show_trend_transition_section()

    results = load_results()
    if results is None:
        st.error("分析結果がありません。先にバッチ更新を実行してください。")
        st.code("cd jpx500_wave_analysis && python batch/update.py")
        return

    # --- サイドバー ---
    st.sidebar.header("フィルタ")

    # 検索窓
    search_query = st.sidebar.text_input(
        "銘柄検索（コード or 銘柄名）",
        placeholder="例: 6758, ソニー",
        key="search_query",
    )

    # 推奨銘柄フィルタ
    st.sidebar.subheader("推奨銘柄")
    show_recommended = st.sidebar.checkbox(
        "レンジ→上昇転換候補のみ表示",
        value=False,
        key="show_recommended",
    )

    # 波形タイプ
    st.sidebar.subheader("波形タイプ")
    selected_types = []
    for wt in WAVE_TYPES:
        if st.sidebar.checkbox(wt, value=True, key=f"wt_{wt}"):
            selected_types.append(wt)

    # 市場 (Prime + TSE Standard Top400)
    st.sidebar.subheader("市場")
    selected_markets = st.sidebar.multiselect(
        "市場を選択",
        options=MARKET_OPTIONS,
        default=MARKET_OPTIONS,
        key="market_filter",
    )

    # 規模区分
    st.sidebar.subheader("規模区分")
    size_options = [
        "TOPIX Core30",
        "TOPIX Large70",
        "TOPIX Mid400",
        "TSE Standard Top100",
        "TSE Standard Top400",
        "ETF",
    ]
    selected_sizes = []
    for s in size_options:
        label = s.replace("TOPIX ", "").replace("TSE ", "")
        if st.sidebar.checkbox(label, value=True, key=f"sz_{s}"):
            selected_sizes.append(s)

    # 33業種区分
    st.sidebar.subheader("33業種区分")
    all_sectors = (
        sorted(results["sector_33"].dropna().unique().tolist())
        if "sector_33" in results.columns
        else []
    )
    selected_sectors = st.sidebar.multiselect(
        "33業種を選択",
        options=all_sectors,
        default=all_sectors,
        key="sector_filter",
    )

    # レンジ幅フィルタ
    st.sidebar.subheader("レンジ幅")
    range_filter = st.sidebar.radio(
        "レンジ幅フィルタ",
        ["全て", f"小波 (≤{RANGE_PCT_SMALL}%)", "中波", f"大波 (≥{RANGE_PCT_LARGE}%)"],
        index=0,
        label_visibility="collapsed",
    )

    # 信用取引フィルタ (results に margin_ratio 列がある場合のみ)
    margin_filter = "全て"
    if "margin_ratio" in results.columns:
        st.sidebar.subheader("信用取引")
        margin_filter = st.sidebar.radio(
            "信用倍率フィルタ",
            [
                "全て",
                "買い偏重 (倍率≥5) しこり警戒",
                "売り偏重 (倍率<1) 踏み上げ余地",
                "上値しこり重い (買残≥上場5%)",
                "データあり",
            ],
            index=0,
            label_visibility="collapsed",
        )

    # ヘルプ表示
    st.sidebar.divider()
    with st.sidebar.expander("使い方・用語の定義"):
        st.markdown(HELP_TEXT, unsafe_allow_html=False)

    # --- フィルタ適用 ---
    filtered = results.copy()

    # 検索
    if search_query.strip():
        q = search_query.strip()
        filtered = filtered[
            filtered["code"].str.contains(q, na=False)
            | filtered["name"].str.contains(q, case=False, na=False)
        ]

    # 推奨銘柄フィルタ
    if show_recommended:
        filtered = filtered[filtered.apply(is_recommended, axis=1)]

    # 市場 (旧データは market 列無し -> TSE Prime 扱い)
    if "market" not in filtered.columns:
        filtered["market"] = "TSE Prime"
    filtered["market"] = filtered["market"].fillna("TSE Prime")
    filtered = filtered[filtered["market"].isin(selected_markets)]

    # 規模
    filtered = filtered[filtered["size_category"].isin(selected_sizes)]

    # 33業種
    if "sector_33" in filtered.columns and selected_sectors:
        filtered = filtered[filtered["sector_33"].isin(selected_sectors)]

    # 波形タイプ
    if selected_types:
        mask = filtered["wave_types"].apply(
            lambda x: any(t in str(x) for t in selected_types) if pd.notna(x) else False
        )
        filtered = filtered[mask]

    # レンジ幅
    if range_filter != "全て" and "range_pct" in filtered.columns:
        rp = filtered["range_pct"]
        if "小波" in range_filter:
            filtered = filtered[rp <= RANGE_PCT_SMALL]
        elif "大波" in range_filter:
            filtered = filtered[rp >= RANGE_PCT_LARGE]
        else:
            filtered = filtered[(rp > RANGE_PCT_SMALL) & (rp < RANGE_PCT_LARGE)]

    # 信用取引フィルタ
    if margin_filter != "全て" and "margin_ratio" in filtered.columns:
        if "買い偏重" in margin_filter:
            filtered = filtered[filtered["margin_ratio"] >= 5.0]
        elif "売り偏重" in margin_filter:
            filtered = filtered[
                (filtered["margin_ratio"] > 0) & (filtered["margin_ratio"] < 1.0)
            ]
        elif "上値しこり" in margin_filter:
            filtered = filtered[filtered["margin_buy_pct_listed"] >= 5.0]
        elif "データあり" in margin_filter:
            filtered = filtered[filtered["margin_ratio"].notna()]

    # --- サマリーカード ---
    st.markdown("### サマリー")
    cols = st.columns(len(WAVE_TYPES))
    for i, wt in enumerate(WAVE_TYPES):
        count = (
            results["wave_types"]
            .apply(lambda x, _wt=wt: _wt in str(x) if pd.notna(x) else False)
            .sum()
        )
        cols[i].metric(wt, f"{count}銘柄")

    # 推奨銘柄数
    rec_count = results.apply(is_recommended, axis=1).sum()
    st.info(
        f"レンジ→上昇転換候補: **{rec_count}銘柄** （サイドバーの「推奨銘柄」で絞り込み可能）"
    )

    # --- マーケット概況 ---
    st.markdown("---")
    st.markdown("### マーケット概況")

    mkt_col1, mkt_col2, mkt_col3 = st.columns(3)
    with mkt_col1:
        st.markdown("**国内指数**")
        show_nikkei225 = st.checkbox("日経225", value=True, key="mkt_nikkei")
    with mkt_col2:
        st.markdown("**33業種指数（複数可）**")
        sector_list = (
            sorted(results["sector_33"].dropna().unique().tolist())
            if "sector_33" in results.columns
            else []
        )
        selected_mkt_sectors = st.multiselect(
            "33業種を選択",
            options=sector_list,
            default=[],
            key="mkt_sectors",
            label_visibility="collapsed",
        )
    with mkt_col3:
        st.markdown("**海外指数（複数可）**")
        selected_global_names = st.multiselect(
            "海外指数を選択",
            options=list(GLOBAL_INDICES.values()),
            default=[],
            key="mkt_global",
            label_visibility="collapsed",
        )

    mkt_period_labels = {
        60: "60日",
        120: "120日",
        180: "180日",
        260: "1年",
        520: "2年",
        0: "全期間",
    }
    mkt_period_sel = st.radio(
        "表示期間",
        list(mkt_period_labels.keys()),
        index=1,
        horizontal=True,
        format_func=lambda x: mkt_period_labels[x],
        key="mkt_window",
    )

    fx_jpy_mode = st.checkbox(
        "円換算（為替: 各月の平均レート）",
        value=False,
        key="mkt_fx_jpy_mode",
    )

    _SECTOR_COLORS = [
        "#E91E63",
        "#00BCD4",
        "#FF9800",
        "#8BC34A",
        "#9C27B0",
        "#795548",
        "#FF5722",
        "#03A9F4",
        "#607D8B",
        "#CDDC39",
    ]
    _GLOBAL_COLORS = {
        "^DJI": "#FF5722",
        "^GSPC": "#FF9800",
        "^GDAXI": "#4CAF50",
        "^STOXX50E": "#009688",
        "^KS11": "#F48FB1",
        "^NDX": "#3F51B5",
        "^HSI": "#F44336",
        "000300.SS": "#C2185B",
        "^NSEI": "#9C27B0",
        "^AXJO": "#795548",
    }
    _GLOBAL_TICKER_TO_CCY = {
        "^DJI": "USD",
        "^GSPC": "USD",
        "^NDX": "USD",
        "^GDAXI": "EUR",
        "^STOXX50E": "EUR",
        "^KS11": "KRW",
        "^HSI": "HKD",
        "000300.SS": "CNY",
        "^NSEI": "INR",
        "^AXJO": "AUD",
    }
    _GLOBAL_NAME_TO_TICKER = {v: k for k, v in GLOBAL_INDICES.items()}

    mkt_indices = []
    if show_nikkei225:
        nk_data = get_nikkei225()
        if nk_data is not None:
            mkt_indices.append({"name": "日経225", "data": nk_data, "color": "#1976d2"})

    for i, sector in enumerate(selected_mkt_sectors):
        sec_data = compute_sector_index(sector, results)
        if sec_data is not None:
            color = _SECTOR_COLORS[i % len(_SECTOR_COLORS)]
            mkt_indices.append(
                {"name": f"33業種: {sector}", "data": sec_data, "color": color}
            )

    for gname in selected_global_names:
        gticker = _GLOBAL_NAME_TO_TICKER.get(gname)
        if gticker:
            gdata = _get_price_data_cached(gticker)
            if gdata is not None:
                if fx_jpy_mode:
                    ccy = _GLOBAL_TICKER_TO_CCY.get(gticker)
                    if ccy:
                        gdata = convert_ohlcv_close_to_jpy_by_month_avg(gdata, ccy)
                color = _GLOBAL_COLORS.get(gticker, "#607D8B")
                mkt_indices.append({"name": gname, "data": gdata, "color": color})

    if mkt_indices:
        fig_mkt = build_index_chart(mkt_indices, window=mkt_period_sel)
        st.plotly_chart(fig_mkt, use_container_width=True)
    else:
        st.caption("表示する指数を選択してください。")

    # --- 海外投資家フロー × 指数 連動分析 ---
    st.markdown("---")
    st.markdown("### 海外投資家フロー × 指数 連動分析")
    st.caption(
        "週次の投資部門別売買と指数の連動を可視化。相関は因果ではない点に注意。"
        "データソース: JPX公式「投資部門別取引状況」"
    )

    flow_df_all = load_foreign_flow(market="TSE Prime")
    if flow_df_all.empty:
        st.info(
            "海外投資家フローデータがまだありません。"
            "`python batch/update.py` または `python -c 'from modules.jpx_investor_flow_fetcher import fetch_all_investor_flow; fetch_all_investor_flow(force=True)'` を実行してください。"
        )
    else:
        ff_col1, ff_col2, ff_col3 = st.columns([1, 2, 1.5])
        with ff_col1:
            ff_market = st.radio(
                "市場",
                ["TSE Prime", "TSE Standard", "TSE Growth", "Tokyo & Nagoya"],
                index=0,
                key="ff_market",
            )
        with ff_col2:
            # 比較対象: 日経225 + TOPIX + サイズ別 + 33業種
            _size_labels_present = [
                s
                for s in (
                    "TOPIX Core30",
                    "TOPIX Large70",
                    "TOPIX Mid400",
                    "TSE Standard Top100",
                    "TSE Standard Top400",
                )
                if s in set(results["size_category"].dropna().unique())
            ]
            ff_index_options = (
                ["日経225", "TOPIX (1308.T)"]
                + _size_labels_present
                + sorted([s for s in results["sector_33"].dropna().unique() if s != ""])
            )
            ff_targets = st.multiselect(
                "比較対象（複数可）",
                options=ff_index_options,
                default=["日経225", "TOPIX (1308.T)"],
                key="ff_targets",
            )
        with ff_col3:
            ff_view = st.radio(
                "表示形式",
                [
                    "累積フロー vs 指数 (2軸)",
                    "週次フロー × リターン散布図",
                    "業種別相関バー",
                    "サイズ別相関バー",
                ],
                index=0,
                key="ff_view",
            )

        flow_df = load_foreign_flow(market=ff_market)
        if flow_df.empty:
            st.caption(f"{ff_market}のフローデータがありません。")
        else:
            cumulative = compute_cumulative_flow(flow_df)
            flow_net = flow_df["net"]

            # === 表示1: 累積フロー × 指数（2軸）===
            if ff_view.startswith("累積フロー"):
                ff_period_labels = {
                    0: "全期間",
                    130: "6ヶ月",
                    260: "1年",
                    520: "2年",
                    780: "3年",
                }
                ff_period_sel = st.radio(
                    "表示期間",
                    list(ff_period_labels.keys()),
                    index=0,
                    horizontal=True,
                    format_func=lambda x: ff_period_labels[x],
                    key="ff_period",
                )
                indices_for_chart = []
                _sector_palette = [
                    "#E91E63",
                    "#FF9800",
                    "#9C27B0",
                    "#4CAF50",
                    "#795548",
                    "#00BCD4",
                ]
                _size_color_map = {
                    "TOPIX Core30": "#2E7D32",
                    "TOPIX Large70": "#43A047",
                    "TOPIX Mid400": "#7CB342",
                    "TSE Standard Top100": "#7B1FA2",
                    "TSE Standard Top400": "#5C6BC0",
                }
                _sec_idx = 0
                for name in ff_targets:
                    if name == "日経225":
                        d = get_nikkei225()
                        if d is not None:
                            indices_for_chart.append(
                                {"name": name, "data": d, "color": "#1976d2"}
                            )
                    elif name == "TOPIX (1308.T)":
                        d = get_topix()
                        if d is not None:
                            indices_for_chart.append(
                                {"name": name, "data": d, "color": "#E91E63"}
                            )
                    elif name in _size_color_map:
                        d = compute_size_index(name, results)
                        if d is not None:
                            indices_for_chart.append(
                                {
                                    "name": f"サイズ: {name}",
                                    "data": d,
                                    "color": _size_color_map[name],
                                }
                            )
                    else:
                        d = compute_sector_index(name, results)
                        if d is not None:
                            indices_for_chart.append(
                                {
                                    "name": f"33業種: {name}",
                                    "data": d,
                                    "color": _sector_palette[
                                        _sec_idx % len(_sector_palette)
                                    ],
                                }
                            )
                            _sec_idx += 1

                fig_flow = build_flow_index_dual_chart(
                    flow_cumulative=cumulative,
                    indices=indices_for_chart,
                    window=ff_period_sel,
                )
                st.plotly_chart(fig_flow, use_container_width=True)

                latest = flow_df.iloc[-1]
                _net = latest["net"]
                _ratio = latest["foreigner_ratio_pct"]
                st.caption(
                    f"{ff_market} 直近週 ({flow_df.index[-1].date()}): "
                    f"net = {_net:+,.0f} 億円 / "
                    f"委託売買シェア = {_ratio:.1f}%"
                )

            # === 表示2: 週次散布図 ===
            elif ff_view.startswith("週次フロー"):
                from modules.foreign_flow_analyzer import (
                    compute_index_weekly_close,
                )

                if not ff_targets:
                    st.caption("比較対象を1つ以上選択してください。")
                else:
                    target_name = ff_targets[0]
                    if target_name == "日経225":
                        idx_close = compute_index_weekly_close("^N225")
                    elif target_name == "TOPIX (1308.T)":
                        idx_close = compute_index_weekly_close("1308.T")
                    elif target_name in (
                        "TOPIX Core30",
                        "TOPIX Large70",
                        "TOPIX Mid400",
                        "TSE Standard Top100",
                        "TSE Standard Top400",
                    ):
                        size_idx = compute_size_index(target_name, results)
                        if size_idx is None:
                            idx_close = pd.Series(dtype=float)
                        else:
                            idx_close = (
                                pd.to_numeric(size_idx["Close"], errors="coerce")
                                .dropna()
                                .resample("W-FRI")
                                .last()
                                .dropna()
                            )
                    else:
                        sec = compute_sector_index(target_name, results)
                        if sec is None:
                            idx_close = pd.Series(dtype=float)
                        else:
                            idx_close = (
                                pd.to_numeric(sec["Close"], errors="coerce")
                                .dropna()
                                .resample("W-FRI")
                                .last()
                                .dropna()
                            )

                    if idx_close.empty:
                        st.caption(f"{target_name}の指数データが取得できません。")
                    else:
                        # 散布図 + 回帰線
                        index_ret = idx_close.pct_change().dropna() * 100
                        joined = pd.concat(
                            [
                                flow_net.rename("net_flow_oku"),
                                index_ret.rename("index_return_pct"),
                            ],
                            axis=1,
                        ).dropna()
                        joined.index.name = "date"
                        if len(joined) < 3:
                            st.caption(
                                f"データが少なすぎます（{len(joined)}週）。3年バックフィル後に再確認してください。"
                            )
                        else:
                            corr_value = joined["net_flow_oku"].corr(
                                joined["index_return_pct"]
                            )
                            scatter_kwargs = dict(
                                x="net_flow_oku",
                                y="index_return_pct",
                                hover_data={"date": "|%Y-%m-%d"},
                                title=(
                                    f"{ff_market} 海外投資家 net (億円) vs "
                                    f"{target_name} 週次リターン (%) | "
                                    f"相関={corr_value:.3f}, n={len(joined)}週"
                                ),
                                height=500,
                            )
                            try:
                                fig_scatter = px.scatter(
                                    joined.reset_index(),
                                    trendline="ols",
                                    **scatter_kwargs,
                                )
                            except Exception:
                                # statsmodels未導入やStreamlitプロセス内
                                # キャッシュ不一致のフォールバック
                                fig_scatter = px.scatter(
                                    joined.reset_index(),
                                    **scatter_kwargs,
                                )
                            fig_scatter.update_layout(
                                xaxis_title="海外投資家 net 買い越し（億円）",
                                yaxis_title=f"{target_name} 週次リターン (%)",
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            if len(ff_targets) > 1:
                                st.caption(
                                    "散布図は複数選択不可。最初の比較対象のみ表示しています。"
                                )

                            # ラグ別相関表
                            lag_df = compute_flow_index_correlation(
                                flow_net=flow_net,
                                index_close=idx_close,
                                lags=[-2, -1, 0, 1, 2, 4],
                            )
                            with st.expander("ラグ別相関係数"):
                                st.dataframe(lag_df, use_container_width=True)
                                st.caption(
                                    "lag=+N: 指数が N週遅れて反応。lag=-N: 指数が N週先行。"
                                )

            # === 表示3: 業種別相関バー ===
            elif ff_view.startswith("業種別"):
                ff_lag = st.radio(
                    "ラグ（週）",
                    [0, 1, 2, 4],
                    index=0,
                    horizontal=True,
                    key="ff_corr_lag",
                    help="0=同週、+N=N週遅れて指数が反応",
                )
                sector_corr = compute_sector_flow_correlation(
                    results_df=results,
                    flow_net=flow_net,
                    lag=ff_lag,
                )
                if sector_corr.empty:
                    st.caption(
                        "業種別相関が計算できません。データ蓄積が少ない可能性があります。"
                    )
                else:
                    fig_corr = px.bar(
                        sector_corr,
                        x="corr",
                        y="sector_33",
                        orientation="h",
                        color="corr",
                        color_continuous_scale="RdBu",
                        range_color=[-1, 1],
                        title=(
                            f"{ff_market} 海外投資家フロー vs 業種指数リターン 相関係数 "
                            f"(lag={ff_lag}週, n={sector_corr['n_weeks'].iloc[0]}週)"
                        ),
                        height=750,
                    )
                    fig_corr.update_layout(
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="相関係数",
                        yaxis_title="33業種",
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption(
                        "正の相関: 海外投資家の買い越しと業種上昇が連動。"
                        "負の相関: 売り越しで業種が上昇。"
                        "相関は因果ではない点に注意。"
                    )

            # === 表示4: サイズ別相関バー (Prime: Core30/Large70/Mid400, Standard: Top100/Top400) ===
            else:
                ff_lag_size = st.radio(
                    "ラグ（週）",
                    [0, 1, 2, 4],
                    index=0,
                    horizontal=True,
                    key="ff_corr_lag_size",
                    help="0=同週、+N=N週遅れて指数が反応",
                )
                if ff_market == "TSE Standard":
                    _size_labels = ("TSE Standard Top100", "TSE Standard Top400")
                elif ff_market == "TSE Prime":
                    _size_labels = ("TOPIX Core30", "TOPIX Large70", "TOPIX Mid400")
                else:
                    _size_labels = (
                        "TOPIX Core30",
                        "TOPIX Large70",
                        "TOPIX Mid400",
                        "TSE Standard Top100",
                        "TSE Standard Top400",
                    )
                size_corr = compute_size_flow_correlation(
                    results_df=results,
                    flow_net=flow_net,
                    lag=ff_lag_size,
                    size_labels=_size_labels,
                )
                if size_corr.empty:
                    st.caption(
                        "サイズ別相関が計算できません。size_category 列または market_cap が不足している可能性があります。"
                    )
                else:
                    fig_size = px.bar(
                        size_corr,
                        x="corr",
                        y="size_category",
                        orientation="h",
                        color="corr",
                        color_continuous_scale="RdBu",
                        range_color=[-1, 1],
                        text="corr",
                        title=(
                            f"{ff_market} 海外投資家フロー vs サイズ別指数リターン 相関係数 "
                            f"(lag={ff_lag_size}週, n={size_corr['n_weeks'].iloc[0]}週)"
                        ),
                        height=320,
                    )
                    fig_size.update_traces(
                        texttemplate="%{text:.2f}", textposition="outside"
                    )
                    fig_size.update_layout(
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="相関係数",
                        yaxis_title="サイズ区分",
                    )
                    st.plotly_chart(fig_size, use_container_width=True)
                    st.caption(
                        "Prime: Core30(上位30) / Large70(31-100位) / Mid400(101-500位)。"
                        "Standard: Top100(時価総額上位100) / Top400(101-400位)。"
                        "サイズ別の相関差から海外資金の波及順序が読み取れる。"
                    )

    # --- 資本効率改善期待スクリーナー ---
    st.markdown("---")
    st.markdown("### 資本効率改善期待スクリーナー")
    st.caption(
        "PBR低 × ROE低 × ネットキャッシュ豊富 × 還元余地大 の組合せで "
        '増配/自社株買い/政策保有株売却の "カタリスト待ち" 銘柄を10点満点スコアで抽出。'
        "ハードフィルタ: 自己資本比率≥50% & 営業CF>0 & 純利益>0。"
        "データソース: naibu (内部留保 + 一部財務) + yfinance (B/S・CF・配当)"
    )

    ces_df = _load_ces_result()
    if ces_df is None or len(ces_df) == 0:
        st.info(
            "スクリーニング結果がまだありません。"
            '`python -c "import pandas as pd; from modules.capital_efficiency_screener import run_screening; '
            "run_screening(pd.read_csv('data/results.csv', encoding='utf-8-sig', dtype={'code':str}))\"` "
            "または `python batch/update.py` でスクリーニングを実行してください。"
        )
    else:
        ces_c1, ces_c2, ces_c3, ces_c4 = st.columns([1, 1, 1.5, 1.5])
        with ces_c1:
            ces_min_score = st.slider(
                "最低スコア",
                min_value=0,
                max_value=12,
                value=7,
                step=1,
                key="ces_min_score",
                help="12点満点: PBR低3+ネットキャッシュ3+ROE低2+還元余地2+株主構造2",
            )
        with ces_c2:
            ces_pbr_max = st.slider(
                "PBR上限",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="ces_pbr_max",
            )
        with ces_c3:
            _sectors_avail = sorted(
                [s for s in ces_df["sector_33"].dropna().unique() if s != ""]
            )
            ces_sectors = st.multiselect(
                "業種フィルタ",
                options=_sectors_avail,
                default=_sectors_avail,
                key="ces_sectors",
            )
        with ces_c4:
            ces_sizes = st.multiselect(
                "サイズフィルタ",
                options=[
                    "TOPIX Core30",
                    "TOPIX Large70",
                    "TOPIX Mid400",
                    "TSE Standard Top100",
                    "TSE Standard Top400",
                ],
                default=[
                    "TOPIX Core30",
                    "TOPIX Large70",
                    "TOPIX Mid400",
                    "TSE Standard Top100",
                    "TSE Standard Top400",
                ],
                key="ces_sizes",
            )

        # フィルタ適用 (ハードフィルタ落ちは常に除外)
        view = ces_df[~ces_df["hard_filter_failed"]].copy()
        view = view[view["score"] >= ces_min_score]
        view = view[view["pbr"].fillna(99) <= ces_pbr_max]
        if ces_sectors:
            view = view[view["sector_33"].isin(ces_sectors)]
        if ces_sizes:
            view = view[view["size_category"].isin(ces_sizes)]
        view = view.sort_values("score", ascending=False).reset_index(drop=True)

        st.caption(
            f"該当: {len(view)}件 / 全候補(ハードフィルタ通過): "
            f"{(~ces_df['hard_filter_failed']).sum()}件 / "
            f"ハードフィルタ落ち: {ces_df['hard_filter_failed'].sum()}件"
        )

        if len(view) == 0:
            st.caption("該当銘柄なし。スライダー/フィルタを緩めてください。")
        else:
            # --- ランキングテーブル ---
            display = view.copy()
            # 兆/億単位への変換 (兆 = 1e12, 億 = 1e8)
            display["net_cash_億円"] = (display["net_cash"].fillna(0) / 1e8).round(0)
            display["営業CF_億円"] = (
                display["operating_cf_final"].fillna(0) / 1e8
            ).round(0)
            display["時価総額_億円"] = (display["market_cap"].fillna(0) / 1e8).round(0)
            display["内部留保_億円"] = (
                display["retained_earnings"].fillna(0) / 1e8
            ).round(0)
            display["自己資本比率_%"] = (display["equity_ratio"].fillna(0) * 100).round(
                1
            )
            display["配当性向_%"] = (display["payout_ratio"].fillna(0) * 100).round(1)
            display["ROE_%"] = display["roe"].round(2)
            display["余剰/時価_%"] = (
                display["net_cash_to_mcap"].fillna(0) * 100
            ).round(1)
            # 株主構造 (v1.1)
            display["機関投資家_%"] = display.get(
                "institution_pct", pd.Series([])
            ).round(1)
            display["インサイダー_%"] = display.get("insider_pct", pd.Series([])).round(
                1
            )
            display["自社株_%"] = display.get("treasury_pct", pd.Series([])).round(1)
            display["浮動株_%"] = display.get("float_pct", pd.Series([])).round(1)

            table_cols = [
                "code",
                "name",
                "sector_33",
                "size_category",
                "score",
                "pbr",
                "ROE_%",
                "自己資本比率_%",
                "余剰/時価_%",
                "配当性向_%",
                "機関投資家_%",
                "インサイダー_%",
                "自社株_%",
                "浮動株_%",
                "net_cash_億円",
                "営業CF_億円",
                "時価総額_億円",
                "内部留保_億円",
            ]
            st.caption("テーブルの行をクリックすると、その銘柄の詳細画面に移動します。")
            ces_event = st.dataframe(
                display[table_cols],
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "score",
                        min_value=0,
                        max_value=12,
                        format="%d",
                    ),
                    "pbr": st.column_config.NumberColumn("PBR", format="%.2f"),
                    "ROE_%": st.column_config.NumberColumn("ROE(%)", format="%.2f"),
                    "余剰/時価_%": st.column_config.NumberColumn(
                        "余剰/時価(%)", format="%.1f"
                    ),
                    "配当性向_%": st.column_config.NumberColumn(
                        "配当性向(%)", format="%.1f"
                    ),
                    "機関投資家_%": st.column_config.NumberColumn(
                        "機関投資家(%)", format="%.1f"
                    ),
                    "インサイダー_%": st.column_config.NumberColumn(
                        "インサイダー(%)", format="%.1f"
                    ),
                    "自社株_%": st.column_config.NumberColumn(
                        "自社株(%)", format="%.1f"
                    ),
                    "浮動株_%": st.column_config.NumberColumn(
                        "浮動株(%)", format="%.1f"
                    ),
                },
                use_container_width=True,
                hide_index=True,
                height=400,
                on_select="rerun",
                selection_mode="single-row",
                key="ces_table",
            )

            # 行クリック → 銘柄詳細へ遷移 (既存パターン踏襲)
            if ces_event and ces_event.selection and ces_event.selection.rows:
                _sel_idx = ces_event.selection.rows[0]
                _sel_code = str(view.iloc[_sel_idx]["code"])
                _sel_match = results[results["code"].astype(str) == _sel_code]
                if not _sel_match.empty:
                    _sel = _sel_match.iloc[0]
                    st.session_state["selected_ticker"] = _sel.get(
                        "ticker", f"{_sel_code}.T"
                    )
                    st.session_state["selected_code"] = _sel_code
                    st.session_state["selected_name"] = _sel.get("name", "")
                    st.session_state["view"] = "detail"
                    st.rerun()

            # --- 散布図 (PBR × ROE) ---
            scatter_view = view.dropna(subset=["pbr", "roe"]).copy()
            if len(scatter_view) > 0:
                scatter_view["net_cash_to_mcap_pct"] = (
                    scatter_view["net_cash_to_mcap"].fillna(0) * 100
                ).clip(lower=0)
                # size は 0 だと px.scatter が落ちることがあるので min=1
                scatter_view["size_for_plot"] = (
                    scatter_view["net_cash_to_mcap_pct"].clip(lower=1).fillna(1)
                )
                fig_ces = px.scatter(
                    scatter_view,
                    x="pbr",
                    y="roe",
                    size="size_for_plot",
                    color="score",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 10],
                    hover_name="name",
                    hover_data={
                        "code": True,
                        "sector_33": True,
                        "size_category": True,
                        "score": True,
                        "net_cash_to_mcap_pct": ":.1f",
                        "payout_ratio": ":.2f",
                        "size_for_plot": False,
                    },
                    title="PBR × ROE 散布図 (size=余剰/時価%, color=score)",
                    height=500,
                )
                fig_ces.add_vline(
                    x=0.7, line_dash="dash", line_color="gray", opacity=0.4
                )
                fig_ces.add_vline(
                    x=1.0, line_dash="dash", line_color="gray", opacity=0.4
                )
                fig_ces.add_hline(
                    y=8.0, line_dash="dash", line_color="gray", opacity=0.4
                )
                fig_ces.add_hline(
                    y=3.0, line_dash="dash", line_color="gray", opacity=0.4
                )
                fig_ces.update_layout(xaxis_title="PBR (倍)", yaxis_title="ROE (%)")
                st.plotly_chart(fig_ces, use_container_width=True)

            # --- スコア内訳バー (上位10銘柄) ---
            top10 = view.head(10).copy()
            if len(top10) > 0:
                # shareholder_score 列が無い古い parquet との後方互換
                if "shareholder_score" not in top10.columns:
                    top10["shareholder_score"] = 0
                bar_long = top10.melt(
                    id_vars=["name", "code"],
                    value_vars=[
                        "pbr_score",
                        "netcash_score",
                        "roe_score",
                        "payout_score",
                        "shareholder_score",
                    ],
                    var_name="軸",
                    value_name="点",
                )
                _axis_label = {
                    "pbr_score": "PBR低",
                    "netcash_score": "ネットキャッシュ",
                    "roe_score": "ROE低",
                    "payout_score": "還元余地",
                    "shareholder_score": "株主構造",
                }
                bar_long["軸"] = bar_long["軸"].map(_axis_label)
                bar_long["銘柄"] = bar_long["name"] + " (" + bar_long["code"] + ")"
                fig_bar = px.bar(
                    bar_long,
                    x="点",
                    y="銘柄",
                    color="軸",
                    orientation="h",
                    title="上位10銘柄 スコア内訳 (12点満点)",
                    color_discrete_map={
                        "PBR低": "#D32F2F",
                        "ネットキャッシュ": "#1976D2",
                        "ROE低": "#F57C00",
                        "還元余地": "#388E3C",
                        "株主構造": "#9C27B0",
                    },
                    category_orders={
                        "軸": [
                            "PBR低",
                            "ネットキャッシュ",
                            "ROE低",
                            "還元余地",
                            "株主構造",
                        ],
                        "銘柄": (top10["name"] + " (" + top10["code"] + ")").tolist(),
                    },
                    height=400,
                )
                fig_bar.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(range=[0, 12]),
                    legend_title_text="スコア軸",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.caption(
                "ネットキャッシュ = 現金 - 有利子負債(short_term_debt + long_term_debt または yfinance total_debt)。"
                "ROE = 純利益 / 自己資本 × 100 (B/Sベース)。"
                "株主構造スコア: インサイダー<50% かつ 機関投資家≥30% で 2点 / ≥20% で 1点 / "
                "インサイダー≥50% でオーナー支配強と判定し 0点。"
                "naibu に欠損する財務は yfinance の balance_sheet/cashflow/info で補完。"
            )

    # --- 33業種サマリー ---
    st.markdown("---")
    st.markdown("### 33業種 PER・PBR・時価総額")
    st.caption("行をクリックすると銘柄一覧をその業種で絞り込みます")
    sector_stats = compute_sector_stats(results)
    if len(sector_stats) > 0:
        sector_display = sector_stats.rename(
            columns={
                "sector_33": "33業種",
                "count": "銘柄数",
                "per_median": "PER（中央値）",
                "pbr_median": "PBR（中央値）",
                "market_cap_total": "時価総額合計（億円）",
                "market_cap_count": "時価総額取得数",
                "market_cap_coverage_pct": "時価総額取得率（%）",
            }
        )
        # 時価総額がすべてNoneの場合は列を非表示
        display_cols_sector = [
            "33業種",
            "銘柄数",
            "PER（中央値）",
            "PBR（中央値）",
            "時価総額合計（億円）",
            "時価総額取得数",
            "時価総額取得率（%）",
        ]
        if (
            "時価総額合計（億円）" in sector_display.columns
            and sector_display["時価総額合計（億円）"].isna().all()
        ):
            for col in [
                "時価総額合計（億円）",
                "時価総額取得数",
                "時価総額取得率（%）",
            ]:
                if col in display_cols_sector:
                    display_cols_sector.remove(col)
        # 解除ボタンが押された直後はテーブル選択をリセット
        _sector_table_key = "sector_table"
        if st.session_state.get("_clear_sector_flag"):
            _sector_table_key = (
                f"sector_table_{st.session_state.get('_sector_table_ver', 0)}"
            )

        sector_event = st.dataframe(
            sector_display[
                [c for c in display_cols_sector if c in sector_display.columns]
            ],
            use_container_width=True,
            hide_index=True,
            height=400,
            on_select="rerun",
            selection_mode="multi-row",
            key=_sector_table_key,
        )
        # 業種クリック → 絞り込み（複数選択対応）
        if st.session_state.get("_clear_sector_flag"):
            # 解除直後: フラグをクリアして選択なし状態にする
            st.session_state["_clear_sector_flag"] = False
            st.session_state["quick_sector_filter"] = None
        elif sector_event and sector_event.selection and sector_event.selection.rows:
            selected_sector_names = [
                sector_stats.iloc[i]["sector_33"] for i in sector_event.selection.rows
            ]
            st.session_state["quick_sector_filter"] = selected_sector_names
        elif "quick_sector_filter" not in st.session_state:
            st.session_state["quick_sector_filter"] = None
    else:
        st.caption("業種データがありません。")

    # 業種絞り込み表示（複数対応）
    quick_sectors = st.session_state.get("quick_sector_filter")
    if quick_sectors:
        if "sector_33" in filtered.columns:
            filtered = filtered[filtered["sector_33"].isin(quick_sectors)]
        label = (
            "、".join(quick_sectors)
            if len(quick_sectors) <= 3
            else f"{quick_sectors[0]} 他{len(quick_sectors) - 1}業種"
        )
        if st.button(f"業種絞り込み解除（現在: {label}）", key="btn_clear_sector"):
            st.session_state["quick_sector_filter"] = None
            # テーブルキーを変えて選択状態をリセット
            st.session_state["_clear_sector_flag"] = True
            st.session_state["_sector_table_ver"] = (
                st.session_state.get("_sector_table_ver", 0) + 1
            )
            st.rerun()

    # --- PER×PBR 散布図 ---
    st.markdown("---")
    st.markdown("### PER × PBR 散布図")
    pp_c1, pp_c2 = st.columns(2)
    with pp_c1:
        perpbr_scope = st.radio(
            "対象",
            ["絞り込み後", "全銘柄"],
            index=0,
            horizontal=True,
            key="perpbr_scope",
        )
    with pp_c2:
        perpbr_color_by = st.radio(
            "色分け",
            ["33業種", "サイズ区分 (Prime + Standard)"],
            index=0,
            horizontal=True,
            key="perpbr_color_by",
        )
    base_df = filtered if perpbr_scope == "絞り込み後" else results
    if (
        base_df is not None
        and len(base_df) > 0
        and ("per" in base_df.columns)
        and ("pbr" in base_df.columns)
    ):
        scatter_df = base_df.copy()
        scatter_df["per"] = pd.to_numeric(scatter_df["per"], errors="coerce")
        scatter_df["pbr"] = pd.to_numeric(scatter_df["pbr"], errors="coerce")
        scatter_df = scatter_df.dropna(subset=["per", "pbr"])
        scatter_df = scatter_df[(scatter_df["per"] > 0) & (scatter_df["pbr"] > 0)]

        if len(scatter_df) == 0:
            st.caption("PER/PBRのデータがありません（欠損または0以下）。")
        else:
            use_size_color = perpbr_color_by.startswith("サイズ")
            if use_size_color and "size_category" in scatter_df.columns:
                color_col = "size_category"
                color_map = SIZE_COLOR_MAP
                legend_title = "サイズ区分"
            else:
                color_col = "sector_33" if "sector_33" in scatter_df.columns else None
                color_map = None
                legend_title = "33業種" if color_col else None
            fig_scatter = px.scatter(
                scatter_df,
                x="per",
                y="pbr",
                color=color_col,
                color_discrete_map=color_map,
                category_orders=(
                    {"size_category": list(SIZE_COLOR_MAP.keys())}
                    if use_size_color
                    else None
                ),
                hover_name="name" if "name" in scatter_df.columns else None,
                hover_data={
                    "code": True if "code" in scatter_df.columns else False,
                    "size_category": True
                    if "size_category" in scatter_df.columns
                    else False,
                    "wave_types": True if "wave_types" in scatter_df.columns else False,
                    "per": ":.2f",
                    "pbr": ":.2f",
                },
            )
            fig_scatter.update_layout(
                height=520,
                xaxis_title="PER",
                yaxis_title="PBR",
                legend_title_text=legend_title,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"表示件数: {len(scatter_df)}")
    else:
        st.caption("PER/PBR列がないため散布図を表示できません。")

    # --- PER × PBR 時系列アニメーション ---
    st.markdown("---")
    st.markdown("### PER × PBR 時系列アニメーション")
    st.caption(
        "週次（金曜終値）のPER/PBRを時系列で再生。"
        "業績(EPS/BPS)と株価期待値の変化を観測するためのビューです。"
    )

    pp_hist = _load_per_pbr_history()
    if pp_hist is None or len(pp_hist) == 0:
        st.info(
            "PER/PBR履歴データがまだありません。"
            "`python batch/update.py` を実行して per_pbr_history.parquet を生成してください。"
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            anim_period = st.radio(
                "期間",
                ["直近1年", "直近2年", "直近3年"],
                index=1,
                horizontal=False,
                key="pp_anim_period",
            )
        with c2:
            anim_scope = st.radio(
                "対象",
                ["絞り込み後", "全銘柄"],
                index=0,
                horizontal=False,
                key="pp_anim_scope",
            )
            anim_unit = st.radio(
                "表示単位",
                ["個別銘柄", "業種加重平均"],
                index=0,
                horizontal=False,
                key="pp_anim_unit",
                help=(
                    "業種加重平均: 各業種を1点として表示。"
                    "業種PER = Σ(時価総額) / Σ(純利益)、点サイズは業種合計時価総額。"
                ),
            )
            sector_choices = ["（業種で絞らない）"] + sorted(
                [s for s in pp_hist["sector_33"].dropna().unique() if s != ""]
            )
            anim_sector = st.selectbox(
                "業種",
                sector_choices,
                index=0,
                key="pp_anim_sector",
                help="単一業種に絞ってアニメ再生します（個別銘柄モード時のみ有効）",
                disabled=(anim_unit == "業種加重平均"),
            )
        with c3:
            per_cap = st.slider(
                "PER上限",
                min_value=10,
                max_value=200,
                value=PER_PBR_DEFAULT_PER_CAP,
                step=5,
                key="pp_anim_per_cap",
            )
            pbr_cap = st.slider(
                "PBR上限",
                min_value=1,
                max_value=30,
                value=PER_PBR_DEFAULT_PBR_CAP,
                step=1,
                key="pp_anim_pbr_cap",
            )
        with c4:
            speed_label = st.select_slider(
                "再生速度",
                options=["遅い", "標準", "速い"],
                value="標準",
                key="pp_anim_speed",
            )
            size_metric = st.radio(
                "点サイズ",
                ["時価総額", "ROE", "固定"],
                index=0,
                horizontal=True,
                key="pp_anim_size_metric",
                help="ROE(%) = PBR / PER × 100。負値（赤字）は最小サイズに丸める。",
            )
            use_log_x = st.checkbox("PER軸を対数表示", value=False, key="pp_anim_log_x")
            include_loss = st.checkbox(
                "赤字銘柄も含める",
                value=False,
                key="pp_anim_include_loss",
                help="TTM EPSが負（PERが負値）の銘柄を表示するかどうか",
            )
            anim_color_by = st.radio(
                "色分け",
                ["33業種", "サイズ区分"],
                index=0,
                horizontal=True,
                key="pp_anim_color_by",
                help="個別銘柄モード時のみ有効。Prime(Core30/Large70/Mid400)+ Standard(Top100/Top400)で色分け",
            )

        years_map = {"直近1年": 1, "直近2年": 2, "直近3年": 3}
        years = years_map.get(anim_period, 2)
        speed_map = {"遅い": 1000, "標準": 500, "速い": 250}
        speed_ms = speed_map.get(speed_label, 500)

        cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=years * 365)
        plot_df = pp_hist[pp_hist["date"] >= cutoff_date].copy()

        if anim_scope == "絞り込み後" and filtered is not None and len(filtered) > 0:
            target_codes = set(filtered["code"].astype(str).tolist())
            plot_df = plot_df[plot_df["code"].astype(str).isin(target_codes)]

        # 業種絞り込みは個別銘柄モードでのみ適用（加重平均モードは全業種を表示）
        if anim_unit == "個別銘柄" and anim_sector != "（業種で絞らない）":
            plot_df = plot_df[plot_df["sector_33"] == anim_sector]

        plot_df["per"] = pd.to_numeric(plot_df["per"], errors="coerce")
        plot_df["pbr"] = pd.to_numeric(plot_df["pbr"], errors="coerce")
        plot_df["market_cap"] = pd.to_numeric(
            plot_df.get("market_cap"), errors="coerce"
        )
        plot_df = plot_df.dropna(subset=["per", "pbr"])

        # 加重平均モードは赤字銘柄を除外（合算PER分母が負になるため）
        if anim_unit == "業種加重平均" or not include_loss:
            plot_df = plot_df[(plot_df["per"] > 0) & (plot_df["pbr"] > 0)]
        else:
            plot_df = plot_df[plot_df["pbr"] > 0]

        # 加重平均モードでは業種×日付で集計してから上限クリップ
        if anim_unit == "業種加重平均":
            plot_df = plot_df.dropna(subset=["market_cap"])
            plot_df = plot_df[plot_df["market_cap"] > 0]
            if len(plot_df) > 0:
                plot_df = plot_df.assign(
                    _earn=plot_df["market_cap"] / plot_df["per"],
                    _equity=plot_df["market_cap"] / plot_df["pbr"],
                )
                agg = (
                    plot_df.groupby(["date", "sector_33"])
                    .agg(
                        mc_sum=("market_cap", "sum"),
                        earn_sum=("_earn", "sum"),
                        equity_sum=("_equity", "sum"),
                        n=("code", "nunique"),
                    )
                    .reset_index()
                )
                agg["per"] = agg["mc_sum"] / agg["earn_sum"]
                agg["pbr"] = agg["mc_sum"] / agg["equity_sum"]
                agg = agg.rename(columns={"sector_33": "name", "mc_sum": "market_cap"})
                agg["code"] = agg["name"]
                agg["sector_33"] = agg["name"]
                agg["size_category"] = "業種加重"
                agg["close"] = pd.NA
                plot_df = agg[
                    [
                        "code",
                        "name",
                        "date",
                        "per",
                        "pbr",
                        "market_cap",
                        "sector_33",
                        "size_category",
                        "close",
                        "n",
                    ]
                ]

        plot_df = plot_df[(plot_df["per"] <= per_cap) & (plot_df["pbr"] <= pbr_cap)]

        if len(plot_df) == 0:
            st.caption("選択条件に該当するデータがありません。")
        else:
            plot_df["date_str"] = plot_df["date"].dt.strftime("%Y-%m-%d")
            plot_df = plot_df.sort_values(["date_str", "code"])

            # 色分け方針
            anim_use_size_color = (
                anim_color_by == "サイズ区分"
                and anim_unit == "個別銘柄"
                and "size_category" in plot_df.columns
            )
            if anim_unit == "業種加重平均":
                color_col = "sector_33"
            elif anim_use_size_color:
                color_col = "size_category"
            elif anim_sector != "（業種で絞らない）":
                color_col = "name" if "name" in plot_df.columns else None
            else:
                color_col = "sector_33" if "sector_33" in plot_df.columns else None

            # 点サイズの計算列
            size_col: str | None = None
            # market_cap は hover でも使うので常に整形しておく
            if "market_cap" in plot_df.columns:
                plot_df["market_cap"] = pd.to_numeric(
                    plot_df["market_cap"], errors="coerce"
                )
                if plot_df["market_cap"].notna().any():
                    min_mc = plot_df["market_cap"].dropna().min()
                    plot_df["market_cap"] = (
                        plot_df["market_cap"].fillna(min_mc).clip(lower=1.0)
                    )

            # ROE = PBR / PER × 100 を計算
            plot_df["roe"] = (
                pd.to_numeric(plot_df["pbr"], errors="coerce")
                / pd.to_numeric(plot_df["per"], errors="coerce")
                * 100
            )
            # size用に常に roe_size 列を計算（hover_data から確実に参照可能にする）
            plot_df["roe_size"] = plot_df["roe"].fillna(0.1).clip(lower=0.1)

            if size_metric == "時価総額" and plot_df["market_cap"].notna().any():
                size_col = "market_cap"
            elif size_metric == "ROE":
                size_col = "roe_size"

            # hover_dataはモードによって変える
            if anim_unit == "業種加重平均":
                hover_data = {
                    "n": True,  # 業種内の銘柄数
                    "per": ":.2f",
                    "pbr": ":.2f",
                    "roe": ":.1f",
                    "market_cap": ":.0f",
                    "roe_size": False,
                    "date_str": False,
                    "code": False,
                    "sector_33": False,
                }
            else:
                hover_data = {
                    "code": True,
                    "size_category": "size_category" in plot_df.columns,
                    "close": ":.0f",
                    "per": ":.2f",
                    "pbr": ":.2f",
                    "roe": ":.1f",
                    "market_cap": ":.0f",
                    "roe_size": False,
                    "date_str": False,
                }

            # 個別銘柄モードかつ表示銘柄数が多すぎないときはドット上にラベル表示
            n_unique = plot_df["code"].nunique()
            anim_text_col = (
                "name"
                if (
                    anim_unit == "個別銘柄"
                    and "name" in plot_df.columns
                    and n_unique <= 80
                )
                else None
            )

            fig_anim = px.scatter(
                plot_df,
                x="per",
                y="pbr",
                animation_frame="date_str",
                animation_group="code",
                color=color_col,
                color_discrete_map=SIZE_COLOR_MAP if anim_use_size_color else None,
                category_orders=(
                    {"size_category": list(SIZE_COLOR_MAP.keys())}
                    if anim_use_size_color
                    else None
                ),
                size=size_col,
                size_max=55 if anim_unit == "業種加重平均" else 45,
                text=anim_text_col,
                hover_name="name" if "name" in plot_df.columns else None,
                hover_data=hover_data,
                range_x=[
                    0 if not use_log_x else max(plot_df["per"].min(), 0.1),
                    per_cap,
                ],
                range_y=[0, pbr_cap],
                log_x=use_log_x,
                height=620,
            )
            if anim_unit == "業種加重平均":
                legend_title = "33業種"
            elif anim_use_size_color:
                legend_title = "サイズ区分"
            elif anim_sector != "（業種で絞らない）":
                legend_title = "銘柄"
            else:
                legend_title = "33業種" if color_col else None
            fig_anim.update_layout(
                xaxis_title="PER",
                yaxis_title="PBR",
                legend_title_text=legend_title,
            )
            if anim_text_col is not None:
                fig_anim.update_traces(
                    textposition="top center",
                    textfont=dict(size=10),
                )
                # アニメーションフレームにも text を伝播
                for frame in fig_anim.frames or []:
                    for tr in frame.data:
                        tr.textposition = "top center"
                        tr.textfont = dict(size=10)
            # スライダー/再生速度調整
            try:
                if fig_anim.layout.updatemenus:
                    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"][
                        "duration"
                    ] = speed_ms
                    fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"][
                        "duration"
                    ] = speed_ms // 2
            except Exception:
                pass

            st.plotly_chart(fig_anim, use_container_width=True)
            if anim_unit == "業種加重平均":
                st.caption(
                    f"単位: 業種加重平均 / 対象: {plot_df['code'].nunique()}業種 × "
                    f"{plot_df['date_str'].nunique()}週 "
                    f"(期間: {plot_df['date'].min().date()} ～ {plot_df['date'].max().date()})"
                )
            else:
                sector_label = (
                    ""
                    if anim_sector == "（業種で絞らない）"
                    else f"業種={anim_sector} / "
                )
                st.caption(
                    f"{sector_label}対象: {plot_df['code'].nunique()}銘柄 × "
                    f"{plot_df['date_str'].nunique()}週 "
                    f"(期間: {plot_df['date'].min().date()} ～ {plot_df['date'].max().date()})"
                )

            # 業種別中央値の時系列（個別銘柄モード時のみ。加重平均モードでは散布図と重複するため非表示）
            if anim_unit == "個別銘柄":
                with st.expander("業種別 PER/PBR 中央値の時系列推移"):
                    if "sector_33" in plot_df.columns:
                        agg_method = st.radio(
                            "集計方法",
                            ["中央値", "時価総額加重平均"],
                            index=0,
                            horizontal=True,
                            key="pp_anim_agg_method",
                            help=(
                                "加重平均: 業種PER = Σ(時価総額) / Σ(純利益)、"
                                "業種PBR = Σ(時価総額) / Σ(純資産)。"
                                "規模の大きい銘柄ほど影響が大きい。"
                            ),
                        )

                        if agg_method == "中央値":
                            agg_df = (
                                plot_df.groupby(["date", "sector_33"])[["per", "pbr"]]
                                .median()
                                .reset_index()
                            )
                            per_title = "業種別 PER中央値"
                            pbr_title = "業種別 PBR中央値"
                        else:
                            # 時価総額加重: market_cap, eps_ttm, bps を業種・日付で合計
                            # PER_w = Σ(market_cap) / Σ(market_cap / per) = Σ(close*shares) / Σ(eps_ttm*shares)
                            # PBR_w = Σ(market_cap) / Σ(market_cap / pbr) = Σ(close*shares) / Σ(bps*shares)
                            w_df = plot_df.copy()
                            w_df["per"] = pd.to_numeric(w_df["per"], errors="coerce")
                            w_df["pbr"] = pd.to_numeric(w_df["pbr"], errors="coerce")
                            w_df["market_cap"] = pd.to_numeric(
                                w_df["market_cap"], errors="coerce"
                            )
                            # 加重PERは赤字銘柄を除外（純利益合計が負だと意味を失う）
                            per_valid = w_df[
                                (w_df["per"] > 0) & (w_df["market_cap"] > 0)
                            ].copy()
                            per_valid["earnings"] = (
                                per_valid["market_cap"] / per_valid["per"]
                            )
                            per_agg = per_valid.groupby(["date", "sector_33"]).agg(
                                mc_sum=("market_cap", "sum"),
                                e_sum=("earnings", "sum"),
                            )
                            per_agg["per"] = per_agg["mc_sum"] / per_agg["e_sum"]

                            pbr_valid = w_df[
                                (w_df["pbr"] > 0) & (w_df["market_cap"] > 0)
                            ].copy()
                            pbr_valid["equity"] = (
                                pbr_valid["market_cap"] / pbr_valid["pbr"]
                            )
                            pbr_agg = pbr_valid.groupby(["date", "sector_33"]).agg(
                                mc_sum=("market_cap", "sum"),
                                b_sum=("equity", "sum"),
                            )
                            pbr_agg["pbr"] = pbr_agg["mc_sum"] / pbr_agg["b_sum"]

                            agg_df = (
                                per_agg[["per"]]
                                .join(pbr_agg[["pbr"]], how="outer")
                                .reset_index()
                            )
                            per_title = "業種別 PER（時価総額加重）"
                            pbr_title = "業種別 PBR（時価総額加重）"

                        col_per, col_pbr = st.columns(2)
                        with col_per:
                            fig_per = px.line(
                                agg_df,
                                x="date",
                                y="per",
                                color="sector_33",
                                title=per_title,
                                height=400,
                            )
                            st.plotly_chart(fig_per, use_container_width=True)
                        with col_pbr:
                            fig_pbr = px.line(
                                agg_df,
                                x="date",
                                y="pbr",
                                color="sector_33",
                                title=pbr_title,
                                height=400,
                            )
                            st.plotly_chart(fig_pbr, use_container_width=True)

    # --- 本日の推奨銘柄 ---
    picks = load_daily_picks()
    if picks is not None and len(picks) > 0:
        st.markdown("---")
        st.markdown("### 本日の推奨銘柄（レンジ銘柄の直近タッチ）")
        st.caption(
            f"直近{DAILY_PICK_LOOKBACK}営業日以内にレンジ上限/下限付近にタッチした銘柄"
        )

        low_picks = picks[picks["pick_type"].str.contains("下タッチ", na=False)]
        high_picks = picks[picks["pick_type"].str.contains("上タッチ", na=False)]

        col_low, col_high = st.columns(2)

        with col_low:
            st.markdown(f"#### 下タッチ - 買い候補 ({len(low_picks)}銘柄)")
            st.caption("レンジ下限付近まで下落 → 反発を狙える位置")
            if len(low_picks) > 0:
                pick_display = low_picks[
                    [
                        "code",
                        "name",
                        "latest_close",
                        "range_low",
                        "range_high",
                        "range_pct",
                        "position_pct",
                        "slope",
                        "rsi",
                        "rsi_signal",
                    ]
                ].rename(
                    columns={
                        "code": "コード",
                        "name": "銘柄名",
                        "latest_close": "直近終値",
                        "range_low": "下限",
                        "range_high": "上限",
                        "range_pct": "レンジ幅%",
                        "position_pct": "位置%",
                        "slope": "傾き",
                        "rsi": "RSI",
                        "rsi_signal": "RSIシグナル",
                    }
                )
                event_low = st.dataframe(
                    pick_display,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="picks_low",
                )
                if event_low and event_low.selection and event_low.selection.rows:
                    idx = event_low.selection.rows[0]
                    r = low_picks.iloc[idx]
                    st.session_state["selected_ticker"] = r["ticker"]
                    st.session_state["selected_code"] = r["code"]
                    st.session_state["selected_name"] = r["name"]
                    st.session_state["view"] = "detail"
                    st.rerun()
            else:
                st.write("該当なし")

        with col_high:
            st.markdown(f"#### 上タッチ - 利確/ブレイク監視 ({len(high_picks)}銘柄)")
            st.caption("レンジ上限付近まで上昇 → 利確 or 上抜け監視")
            if len(high_picks) > 0:
                pick_display = high_picks[
                    [
                        "code",
                        "name",
                        "latest_close",
                        "range_low",
                        "range_high",
                        "range_pct",
                        "position_pct",
                        "slope",
                        "rsi",
                        "rsi_signal",
                    ]
                ].rename(
                    columns={
                        "code": "コード",
                        "name": "銘柄名",
                        "latest_close": "直近終値",
                        "range_low": "下限",
                        "range_high": "上限",
                        "range_pct": "レンジ幅%",
                        "position_pct": "位置%",
                        "slope": "傾き",
                        "rsi": "RSI",
                        "rsi_signal": "RSIシグナル",
                    }
                )
                event_high = st.dataframe(
                    pick_display,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="picks_high",
                )
                if event_high and event_high.selection and event_high.selection.rows:
                    idx = event_high.selection.rows[0]
                    r = high_picks.iloc[idx]
                    st.session_state["selected_ticker"] = r["ticker"]
                    st.session_state["selected_code"] = r["code"]
                    st.session_state["selected_name"] = r["name"]
                    st.session_state["view"] = "detail"
                    st.rerun()
            else:
                st.write("該当なし")

        st.markdown("---")

    # --- テーブル ---
    st.markdown(f"### 銘柄一覧 ({len(filtered)}銘柄)")

    # 時価総額（億円）と業種内比率を計算
    if "market_cap" in filtered.columns:
        filtered = filtered.copy()
        filtered["market_cap_oku"] = filtered["market_cap"].apply(
            lambda v: round(float(v) / 1e8, 0) if pd.notna(v) else None
        )
        if "sector_33" in filtered.columns:
            sector_mc_totals = results.groupby("sector_33")["market_cap"].sum()
            filtered["sector_mc_pct"] = filtered.apply(
                lambda r: (
                    round(
                        float(r["market_cap"])
                        / float(sector_mc_totals[r["sector_33"]])
                        * 100,
                        1,
                    )
                    if pd.notna(r.get("market_cap"))
                    and r.get("sector_33") in sector_mc_totals.index
                    and sector_mc_totals[r["sector_33"]] > 0
                    else None
                ),
                axis=1,
            )

    display_cols = [
        "code",
        "name",
        "size_category",
        "sector_33",
        "wave_types",
        "range_pct",
        "touch_total",
        "slope",
        "atr",
        "breakout_days",
        "per",
        "pbr",
        "market_cap_oku",
        "sector_mc_pct",
    ]
    display_names = {
        "code": "コード",
        "name": "銘柄名",
        "size_category": "規模",
        "sector_33": "33業種",
        "wave_types": "波形タイプ",
        "range_pct": "レンジ幅%",
        "touch_total": "タッチ回数",
        "slope": "傾き",
        "atr": "ATR",
        "breakout_days": "レンジ外日数",
        "per": "PER",
        "pbr": "PBR",
        "market_cap_oku": "時価総額(億円)",
        "sector_mc_pct": "業種内比率%",
    }

    available_cols = [c for c in display_cols if c in filtered.columns]
    display_df = (
        filtered[available_cols].rename(columns=display_names).reset_index(drop=True)
    )

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=600,
    )

    if event and event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_row = filtered.iloc[selected_idx]
        st.session_state["selected_ticker"] = selected_row["ticker"]
        st.session_state["selected_code"] = selected_row["code"]
        st.session_state["selected_name"] = selected_row["name"]
        st.session_state["view"] = "detail"
        st.rerun()


# ------------------------------------------------------------------
# 詳細画面
# ------------------------------------------------------------------
def show_detail_view():
    ticker = st.session_state.get("selected_ticker")
    code = st.session_state.get("selected_code", "")
    name = st.session_state.get("selected_name", "")

    if not ticker:
        st.session_state["view"] = "list"
        st.rerun()
        return

    if st.button("← 一覧に戻る"):
        st.session_state["view"] = "list"
        st.rerun()
        return

    results = load_results()
    row = None
    if results is not None:
        match = results[results["ticker"] == ticker]
        if len(match) > 0:
            row = match.iloc[0]

    # ヘッダー
    size_cat = row["size_category"].replace("TOPIX ", "") if row is not None else ""
    sector_33 = row.get("sector_33", "") if row is not None else ""
    wave_types_str = (
        row["wave_types"] if row is not None and pd.notna(row["wave_types"]) else ""
    )
    st.title(f"{code} {name}")

    # 波形タイプをタグ風に表示（results.csvの値 = デフォルト窓）
    type_tags = wave_types_str.replace("|", " | ") if wave_types_str else ""
    rec_label = " ★推奨" if row is not None and is_recommended(row) else ""
    sector_label = f" | {sector_33}" if sector_33 else ""
    st.markdown(f"**{size_cat}**{sector_label} | {type_tags}{rec_label}")

    # チャート設定（指標カードの前に配置 → 窓選択を先に取得）
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        chart_type = st.radio(
            "チャートタイプ", ["ローソク足", "ライン"], horizontal=True
        )
    with col_b:
        window = st.radio("評価窓", WINDOW_OPTIONS, index=1, horizontal=True)
    with col_c:
        show_touch = st.checkbox("タッチポイント表示", value=True)
    with col_d:
        show_bb = st.checkbox("ボリンジャーバンド", value=False)

    chart_type_key = "candlestick" if chart_type == "ローソク足" else "line"

    # データ読み込み & 選択された窓でリアルタイム再計算
    # Cloudの最小デプロイでは data/cache/*.parquet を含めないため、
    # キャッシュが無い場合は必要な銘柄のみオンデマンド取得する。
    df = load_cached(ticker)
    if df is None:
        with st.spinner("株価データを取得中..."):
            df = _get_price_data_cached(ticker)
    live_indicators = None
    if df is not None:
        live_indicators = compute_indicators(df, window=window)

    # 指標カード（窓に応じたリアルタイム値を表示）
    if live_indicators is not None:
        live_types = classify(live_indicators)
        st.markdown(f"**評価窓 {window}日での判定:** {' | '.join(live_types)}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("レンジ幅", f"{live_indicators['range_pct']}%")
        c2.metric("上タッチ", f"{live_indicators['touch_high']}回")
        c3.metric("下タッチ", f"{live_indicators['touch_low']}回")
        c4.metric("傾き", f"{live_indicators['slope']}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("ATR", f"{live_indicators['atr']}")
        c6.metric("レンジ外日数", f"{live_indicators['breakout_days']}日")
        c7.metric("上限", f"¥{live_indicators['range_high']:,.0f}")
        c8.metric("下限", f"¥{live_indicators['range_low']:,.0f}")

        if row is not None:
            c9, c10, _, _ = st.columns(4)
            per_val = row.get("per")
            pbr_val = row.get("pbr")
            c9.metric("PER", f"{per_val:.2f}" if pd.notna(per_val) else "-")
            c10.metric("PBR", f"{pbr_val:.2f}" if pd.notna(pbr_val) else "-")
    elif row is not None:
        # フォールバック: results.csvの値
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("レンジ幅", f"{row.get('range_pct', '-')}%")
        c2.metric(
            "上タッチ",
            f"{int(row['touch_high'])}回" if pd.notna(row.get("touch_high")) else "-",
        )
        c3.metric(
            "下タッチ",
            f"{int(row['touch_low'])}回" if pd.notna(row.get("touch_low")) else "-",
        )
        c4.metric("傾き", f"{row.get('slope', '-')}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("ATR", f"{row.get('atr', '-')}")
        c6.metric(
            "レンジ外日数",
            f"{int(row['breakout_days'])}日"
            if pd.notna(row.get("breakout_days"))
            else "-",
        )
        c7.metric(
            "上限",
            f"¥{row['range_high']:,.0f}" if pd.notna(row.get("range_high")) else "-",
        )
        c8.metric(
            "下限",
            f"¥{row['range_low']:,.0f}" if pd.notna(row.get("range_low")) else "-",
        )

        c9, c10, _, _ = st.columns(4)
        per_val = row.get("per")
        pbr_val = row.get("pbr")
        c9.metric("PER", f"{per_val:.2f}" if pd.notna(per_val) else "-")
        c10.metric("PBR", f"{pbr_val:.2f}" if pd.notna(pbr_val) else "-")

    # === 業種情報（PER/PBR/時価総額比率） ===
    if row is not None and sector_33 and results is not None:
        sector_stats = compute_sector_stats(results)
        sector_row = sector_stats[sector_stats["sector_33"] == sector_33]
        if len(sector_row) > 0:
            sr = sector_row.iloc[0]
            st.markdown(f"#### 業種比較（{sector_33}）")
            sc1, sc2, sc3, sc4 = st.columns(4)
            # 業種PER
            sector_per = sr.get("per_median")
            sc1.metric(
                "業種PER（中央値）",
                f"{sector_per:.2f}" if pd.notna(sector_per) else "-",
            )
            # 業種PBR
            sector_pbr = sr.get("pbr_median")
            sc2.metric(
                "業種PBR（中央値）",
                f"{sector_pbr:.2f}" if pd.notna(sector_pbr) else "-",
            )
            # 個別銘柄の時価総額比率
            has_mc = "market_cap" in results.columns
            stock_mc = (
                float(row["market_cap"])
                if has_mc and pd.notna(row.get("market_cap"))
                else None
            )
            sector_mc_total = sr.get("market_cap_total")
            if stock_mc is not None and sector_mc_total and sector_mc_total > 0:
                stock_mc_oku = stock_mc / 1e8
                ratio_pct = stock_mc_oku / sector_mc_total * 100
                sc3.metric("時価総額", f"{stock_mc_oku:,.0f}億円")
                sc4.metric("業種内比率", f"{ratio_pct:.1f}%")
            else:
                sc3.metric("時価総額", "-")
                sc4.metric("業種内比率", "-")
                if not has_mc:
                    st.caption("※ 時価総額データはバッチ再実行後に表示されます")

    # === 業績 ===
    st.markdown("---")
    st.markdown("#### 業績")
    with st.spinner("財務データを取得中..."):
        financials = _get_financials_cached(ticker)
    if financials is not None and len(financials) > 0:
        fig_fin = build_financials_chart(financials)
        st.plotly_chart(fig_fin, use_container_width=True)
        # 表形式でも表示
        fin_display = financials.rename(
            columns={
                "period": "期間",
                "revenue": "売上高（億円）",
                "op_margin": "営業利益率（%）",
                "eps": "EPS（円/株）",
                "is_forecast": "予測",
            }
        )
        st.dataframe(fin_display, use_container_width=True, hide_index=True)
    else:
        st.caption("財務データを取得できませんでした。")

    # === 信用取引残高 ===
    st.markdown("---")
    st.markdown("#### 信用取引残高 (JPX公表)")
    st.caption(
        "信用倍率=買残÷売残。買偏重(>5)→戻り売り重い／売偏重(<1)→踏み上げ余地。"
        "JPXは直近1営業日分しか公開しないため、本アプリは日次累積方式。"
    )

    try:
        from modules.margin_fetcher import (
            compute_deadline_calendar,
            load_margin_history,
        )
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        m_hist = load_margin_history(ticker)
        if m_hist is None or len(m_hist) == 0:
            st.caption(
                "本銘柄の信用残データは未収集です。バッチが累積するに従い表示されます。"
            )
        else:
            latest_row = m_hist.iloc[-1]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                "信用倍率",
                f"{latest_row['margin_ratio']:.2f}倍",
                help="買残÷売残。5倍以上で買偏重、1倍未満で売偏重。",
            )
            buy_pct = latest_row.get("buy_pct_listed")
            mc2.metric(
                "買残/上場株式",
                f"{buy_pct:.2f}%" if pd.notna(buy_pct) else "-",
            )
            sell_pct = latest_row.get("sell_pct_listed")
            mc3.metric(
                "売残/上場株式",
                f"{sell_pct:.2f}%" if pd.notna(sell_pct) else "-",
            )
            try:
                df_price = load_cached(ticker)
                avg_vol = (
                    float(df_price["Volume"].tail(20).mean())
                    if df_price is not None and len(df_price) >= 20
                    else None
                )
                voldays = (
                    latest_row["buy_balance"] / avg_vol
                    if avg_vol and avg_vol > 0
                    else None
                )
                mc4.metric(
                    "しこり消化日数",
                    f"{voldays:.1f}日" if voldays is not None else "-",
                    help="買残÷20日平均出来高。20日分以上は上値しこり重い。",
                )
            except Exception:
                mc4.metric("しこり消化日数", "-")

            obs_dt = latest_row["observation_date"]
            st.caption(f"観測日: {obs_dt} / 累積観測数: {len(m_hist)}件")

            # 時系列チャート (2段: 買残/売残 + 信用倍率)
            if len(m_hist) >= 2:
                hist_plot = m_hist.copy()
                hist_plot["observation_date"] = pd.to_datetime(
                    hist_plot["observation_date"]
                )

                fig_m = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    row_heights=[0.6, 0.4],
                    vertical_spacing=0.08,
                    subplot_titles=("信用残高推移 (株数)", "信用倍率"),
                )
                fig_m.add_trace(
                    go.Scatter(
                        x=hist_plot["observation_date"],
                        y=hist_plot["buy_balance"],
                        mode="lines+markers",
                        name="買残",
                        line=dict(color="#E91E63", width=2),
                    ),
                    row=1,
                    col=1,
                )
                fig_m.add_trace(
                    go.Scatter(
                        x=hist_plot["observation_date"],
                        y=hist_plot["sell_balance"],
                        mode="lines+markers",
                        name="売残",
                        line=dict(color="#1976d2", width=2),
                    ),
                    row=1,
                    col=1,
                )
                fig_m.add_trace(
                    go.Scatter(
                        x=hist_plot["observation_date"],
                        y=hist_plot["margin_ratio"],
                        mode="lines+markers",
                        name="信用倍率",
                        line=dict(color="#FF6F00", width=2),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
                fig_m.add_hline(
                    y=5.0,
                    line_dash="dash",
                    line_color="rgba(255,82,82,0.5)",
                    row=2,
                    col=1,
                    annotation_text="買偏重 (5倍)",
                    annotation_position="right",
                )
                fig_m.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="rgba(76,175,80,0.5)",
                    row=2,
                    col=1,
                    annotation_text="売偏重 (1倍)",
                    annotation_position="right",
                )
                fig_m.update_layout(
                    height=420,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.04,
                        xanchor="right",
                        x=1,
                    ),
                )
                st.plotly_chart(fig_m, use_container_width=True)
            else:
                st.caption(
                    f"履歴は{len(m_hist)}件のみ。"
                    "時系列チャートには2件以上の観測が必要です。"
                )

            # 期日カレンダー
            calendar = compute_deadline_calendar(ticker)
            if calendar is not None and len(calendar) > 0:
                st.markdown("##### 期日売り予想カレンダー (6ヶ月後の制度信用期日)")
                st.caption(
                    "新規買い建てから6ヶ月後の期日に強制決済売りが発生。"
                    "週次集計で予想売り圧を可視化。"
                )
                cal_display = calendar.copy()
                cal_display["deadline_date"] = cal_display["deadline_date"].dt.strftime(
                    "%Y-%m-%d"
                )
                cal_display = cal_display.rename(
                    columns={
                        "deadline_date": "期日週",
                        "expected_selling_shares": "予想売圧(株数)",
                    }
                )
                st.dataframe(
                    cal_display.head(20), use_container_width=True, hide_index=True
                )
    except Exception as e:
        st.caption(f"信用残データ表示に失敗: {e}")

    # === 比較分析 ===
    st.markdown("---")
    st.markdown("#### 比較分析")
    col_ov1, col_ov2 = st.columns(2)
    with col_ov1:
        show_sector = st.checkbox(
            f"33業種指数（{sector_33}）" if sector_33 else "33業種指数",
            value=False,
            disabled=not bool(sector_33),
            key="overlay_sector",
        )
    with col_ov2:
        show_nikkei = st.checkbox("日経225", value=False, key="overlay_nikkei")

    # 比較銘柄選択
    comparison_tickers = []
    if results is not None:
        ticker_options = results[results["ticker"] != ticker][
            ["ticker", "code", "name", "sector_33"]
        ].copy()
        ticker_options["label"] = ticker_options["code"] + " " + ticker_options["name"]

        # 同業種抽出ボタン
        comp_col1, comp_col2 = st.columns([1, 3])
        with comp_col1:
            if sector_33 and st.button(
                f"同業種を選択（{sector_33}）", key="btn_same_sector"
            ):
                same_sector = ticker_options[ticker_options["sector_33"] == sector_33][
                    "label"
                ].tolist()
                st.session_state["comparison_stocks"] = same_sector[:5]
                st.rerun()

        selected_labels = st.multiselect(
            "比較銘柄を追加",
            options=ticker_options["label"].tolist(),
            max_selections=5,
            key="comparison_stocks",
        )
        if selected_labels:
            comparison_tickers = ticker_options[
                ticker_options["label"].isin(selected_labels)
            ]["ticker"].tolist()

    # オーバーレイデータ構築
    overlays = []
    sector_index_df = None
    nikkei_df = None

    if show_sector and sector_33 and results is not None:
        sector_index_df = compute_sector_index(sector_33, results)
        if sector_index_df is not None:
            overlays.append(
                {
                    "name": f"33業種: {sector_33}",
                    "data": sector_index_df,
                    "color": "#FF5722",
                }
            )

    if show_nikkei:
        nikkei_df = get_nikkei225()
        if nikkei_df is not None:
            overlays.append({"name": "日経225", "data": nikkei_df, "color": "#9C27B0"})

    # 決算発表予定日の取得
    earnings_df = load_earnings_dates()
    earnings_dates = (
        get_earnings_dates_for_code(code, earnings_df)
        if earnings_df is not None
        else []
    )

    # チャート描画（リアルタイムのrange値を使用）
    if df is not None:
        range_h = live_indicators["range_high"] if live_indicators else None
        range_l = live_indicators["range_low"] if live_indicators else None

        fig = build_chart(
            df,
            ticker=ticker,
            name=name,
            range_high=range_h,
            range_low=range_l,
            window=window,
            chart_type=chart_type_key,
            show_touch_points=show_touch,
            show_bb=show_bb,
            earnings_dates=earnings_dates,
            overlays=overlays if overlays else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("株価データが見つかりません。バッチ更新を実行してください。")

    # === 比較チャート＋相関係数 ===
    if df is not None and (
        comparison_tickers or (overlays and comparison_tickers is not None)
    ):
        # 比較対象のデータを収集
        comp_data_list = []
        return_data = {}

        # メイン銘柄の日次リターン（評価窓内）
        main_w = df.tail(window)
        main_returns = main_w["Close"].pct_change().dropna()
        return_data[f"{code} {name}"] = main_returns

        # 選択された比較銘柄
        for comp_ticker in comparison_tickers:
            comp_df = load_cached(comp_ticker)
            if comp_df is None:
                with st.spinner(f"比較銘柄の株価データを取得中: {comp_ticker} ..."):
                    comp_df = _get_price_data_cached(comp_ticker)
            if comp_df is not None:
                comp_row = (
                    results[results["ticker"] == comp_ticker].iloc[0]
                    if results is not None
                    else None
                )
                comp_label = (
                    f"{comp_row['code']} {comp_row['name']}"
                    if comp_row is not None
                    else comp_ticker
                )
                comp_data_list.append({"name": comp_label, "data": comp_df})
                comp_w = comp_df[comp_df.index >= main_w.index[0]]
                if len(comp_w) > 1:
                    return_data[comp_label] = comp_w["Close"].pct_change().dropna()

        # セクター指数・日経225も比較対象に含める
        if show_sector and sector_index_df is not None:
            sector_label = f"33業種: {sector_33}"
            comp_data_list.append({"name": sector_label, "data": sector_index_df})
            sec_w = sector_index_df[sector_index_df.index >= main_w.index[0]]
            if len(sec_w) > 1:
                return_data[sector_label] = sec_w["Close"].pct_change().dropna()

        if show_nikkei and nikkei_df is not None:
            nikkei_label = "日経225"
            comp_data_list.append({"name": nikkei_label, "data": nikkei_df})
            nk_w = nikkei_df[nikkei_df.index >= main_w.index[0]]
            if len(nk_w) > 1:
                return_data[nikkei_label] = nk_w["Close"].pct_change().dropna()

        if comp_data_list:
            st.markdown("---")
            st.markdown("#### 比較チャート")
            comp_fig = build_comparison_chart(
                main_name=f"{code} {name}",
                main_df=df,
                comparisons=comp_data_list,
                window=window,
            )
            st.plotly_chart(comp_fig, use_container_width=True)

            # 相関係数
            if len(return_data) >= 2:
                st.markdown("#### 相関係数（評価窓内の日次リターン）")
                corr_df = pd.DataFrame(return_data).corr()
                st.dataframe(corr_df.style.format("{:.4f}"), use_container_width=True)

    # 指標の説明（展開式）
    with st.expander("指標の見方"):
        st.markdown(HELP_TEXT, unsafe_allow_html=False)


# ------------------------------------------------------------------
# ABCD戦略タブ
# ------------------------------------------------------------------
PATTERN_LABELS = {
    "A_trend": "A: トレンド継続",
    "B_pullback": "B: 押し目",
    "C_breakout": "C: ブレイクアウト",
    "D_reversal": "D: リバーサル",
    "E_can_slim": "E: CAN-SLIM",
    "F_turnaround": "F: ターンアラウンド",
}


def show_strategy_view():
    """ABCD戦略の推奨ランキングを表示する"""
    st.title("ABCD戦略 推奨ランキング")

    # strategy.yaml の読み込み状態を表示
    try:
        strategy = load_strategy()
    except FileNotFoundError:
        st.error("config/strategy.yaml が見つかりません。")
        return

    # 設定サマリー
    exec_cfg = strategy.get("execution", {})
    with st.expander("戦略設定サマリー", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最大保有数", exec_cfg.get("max_positions", 20))
        col2.metric("リバランス", exec_cfg.get("rebalance", "weekly"))
        col3.metric("エントリー", exec_cfg.get("entry_timing", "next_open"))
        col4.metric(
            "取引コスト",
            f"{exec_cfg.get('transaction_cost_roundtrip', 0.003) * 100:.1f}%",
        )

        st.markdown("**パターン定義:**")
        patterns_cfg = strategy.get("patterns", {})
        for pat_name, pat_cfg in patterns_cfg.items():
            label = PATTERN_LABELS.get(pat_name, pat_name)
            notes = pat_cfg.get("notes", "")
            base_pts = (
                strategy.get("scoring", {}).get("base_points", {}).get(pat_name, "?")
            )
            st.markdown(f"- **{label}** (基本点: {base_pts}) — {notes}")

    # データ更新日時
    dates = get_data_dates()
    if dates:
        data_date = dates.get("latest_data", "不明")
        batch_date = dates.get("batch_run", "不明")
        st.caption(f"株価データ: **{data_date}** 時点 ｜ バッチ実行: {batch_date}")

    # サイドバーフィルタ
    st.sidebar.header("ABCD戦略フィルタ")

    # パターンフィルタ
    st.sidebar.subheader("パターン")
    selected_patterns = []
    for pat_key, pat_label in PATTERN_LABELS.items():
        if st.sidebar.checkbox(pat_label, value=True, key=f"strat_{pat_key}"):
            selected_patterns.append(pat_key)

    # 表示件数
    max_show = st.sidebar.slider("表示件数", 10, 100, 20, key="strat_max_show")

    # ランキング生成
    stock_list_df = None
    if STOCK_LIST_CSV.exists():
        stock_list_df = pd.read_csv(
            STOCK_LIST_CSV, encoding="utf-8-sig", dtype={"code": str}
        )

    if stock_list_df is None or len(stock_list_df) == 0:
        st.error("銘柄リストがありません。先にバッチ更新を実行してください。")
        return

    # Cloud向け: 事前計算済みランキングがあればそれを使用
    precomputed = load_abcd_ranking()
    if precomputed is not None and len(precomputed) > 0:
        ranking = precomputed
    else:
        # キャッシュデータの存在確認
        if not CACHE_DIR.exists() or len(list(CACHE_DIR.glob("*.parquet"))) == 0:
            st.warning(
                "株価キャッシュデータがありません。バッチ更新を実行してください。"
            )
            st.code("cd jpx500_wave_analysis && python batch/update.py")
            return

        # キャッシュ付きランキング生成
        @st.cache_data(ttl=3600, show_spinner="ABCD戦略を計算中...")
        def _cached_ranking(_stock_hash: str) -> pd.DataFrame:
            return generate_ranking(
                stock_list_df,
                load_cached_fn=load_cached,
                strategy=strategy,
                max_positions=None,  # 全件計算してからフィルタ
            )

        # stock_list のハッシュ（キャッシュキー用）
        stock_hash = str(len(stock_list_df)) + "_" + str(dates.get("latest_data", ""))
        ranking = _cached_ranking(stock_hash)

    if ranking is None or len(ranking) == 0:
        st.warning("現在シグナルが成立している銘柄はありません。")
        return

    # パターンフィルタ適用
    if selected_patterns:
        mask = ranking["matched_patterns"].apply(
            lambda x: any(p in str(x) for p in selected_patterns)
        )
        filtered_ranking = ranking[mask]
    else:
        filtered_ranking = ranking

    filtered_ranking = filtered_ranking.head(max_show)

    # サマリーカード
    st.markdown("### シグナル集計")
    pat_cols = st.columns(len(PATTERN_LABELS))
    for i, (pat_key, pat_label) in enumerate(PATTERN_LABELS.items()):
        count = (
            ranking["matched_patterns"].apply(lambda x, _p=pat_key: _p in str(x)).sum()
        )
        pat_cols[i].metric(pat_label, f"{count}銘柄")

    # ランキングテーブル
    st.markdown(f"### 推奨ランキング ({len(filtered_ranking)}銘柄)")

    display_cols = [
        "code",
        "name",
        "size_category",
        "sector_33",
        "best_pattern",
        "best_score",
        "matched_patterns",
        "close",
        "rsi",
        "atr_pct",
        "volume_ratio",
        "vs_sma50",
        "vs_sma200",
        "turnover_rank_pct",
    ]
    display_names = {
        "code": "コード",
        "name": "銘柄名",
        "size_category": "規模",
        "sector_33": "33業種",
        "best_pattern": "最良パターン",
        "best_score": "スコア",
        "matched_patterns": "成立パターン",
        "close": "終値",
        "rsi": "RSI",
        "atr_pct": "ATR%",
        "volume_ratio": "出来高比",
        "vs_sma50": "vs SMA50(%)",
        "vs_sma200": "vs SMA200(%)",
        "turnover_rank_pct": "売買代金順位%",
    }

    available = [c for c in display_cols if c in filtered_ranking.columns]
    display_df = filtered_ranking[available].rename(columns=display_names)

    # パターン名を日本語に変換
    if "最良パターン" in display_df.columns:
        display_df["最良パターン"] = display_df["最良パターン"].map(
            lambda x: PATTERN_LABELS.get(x, x)
        )
    if "成立パターン" in display_df.columns:
        display_df["成立パターン"] = display_df["成立パターン"].apply(
            lambda x: " | ".join(PATTERN_LABELS.get(p, p) for p in str(x).split("|"))
        )

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
        key="strategy_ranking",
    )

    # 行選択 → 詳細画面へ遷移
    if event and event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_row = filtered_ranking.iloc[selected_idx]
        st.session_state["selected_ticker"] = selected_row["ticker"]
        st.session_state["selected_code"] = selected_row["code"]
        st.session_state["selected_name"] = selected_row["name"]
        st.session_state["view"] = "detail"
        st.session_state["active_tab"] = "波形分類"
        st.rerun()


# ------------------------------------------------------------------
# バックテスト可視化タブ（Streamlit + Plotly）
# ------------------------------------------------------------------
def show_backtest_view():
    st.title("バックテスト / シナリオ結果")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    scenarios_dir = data_dir / "scenarios"

    summary_fp = data_dir / "backtest_summary.csv"
    trades_fp = data_dir / "backtest_trades.csv"
    monthly_fp = data_dir / "backtest_monthly.csv"
    enriched_fp = data_dir / "backtest_trades_enriched.csv"
    grid_fp = scenarios_dir / "summary_grid.csv"

    missing = [p for p in [summary_fp, trades_fp, monthly_fp] if not p.exists()]
    if missing:
        st.error(
            "バックテスト結果CSVが不足しています。先に batch/backtest.py を実行してください。"
        )
        st.write("不足:")
        for p in missing:
            st.write(f"- {p}")
        return

    @st.cache_data(ttl=3600)
    def _read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8-sig")

    summary = _read_csv(summary_fp)
    trades = _read_csv(trades_fp)
    monthly = _read_csv(monthly_fp)
    enriched = _read_csv(enriched_fp) if enriched_fp.exists() else None
    grid = _read_csv(grid_fp) if grid_fp.exists() else None

    # normalize
    for c in [
        "initial_capital",
        "final_equity",
        "total_return_pct",
        "cagr",
        "annual_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "profit_factor",
        "avg_positions",
    ]:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce")

    for c in ["pnl", "return_pct"]:
        if c in trades.columns:
            trades[c] = pd.to_numeric(trades[c], errors="coerce")

    policies = [
        p for p in ["fixed_amount", "fixed_rate"] if p in set(summary.get("policy", []))
    ]
    if not policies:
        policies = sorted(summary["policy"].dropna().unique().tolist())

    default_policy = "fixed_rate" if "fixed_rate" in policies else policies[0]
    policy = st.selectbox("ポリシー", policies, index=policies.index(default_policy))

    srow = summary[summary["policy"] == policy].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("元本", f"{int(srow['initial_capital']):,}円")
    c2.metric("最終資産", f"{int(round(srow['final_equity'])):,}円")
    c3.metric("総リターン", f"{float(srow['total_return_pct']):.3f}%")
    c4.metric("CAGR", f"{float(srow['cagr']):.4f}")
    c5.metric("最大DD", f"{float(srow['max_drawdown']):.4f}")

    # --- 月次: 収益・回数・勝率・exit内訳 ---
    st.subheader("月次成績（収益・取引回数・勝率・exit内訳）")

    m = monthly[monthly["policy"] == policy].copy()
    for c in [
        "month_return_pct",
        "trades_closed",
        "win_rate",
        "total_pnl",
        "rebalance_drop",
        "time_exit",
        "trailing_atr",
        "trend_exit",
    ]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=m["ym"], y=m["trades_closed"], name="trades_closed", opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=m["ym"], y=m["win_rate"] * 100.0, name="win_rate(%)", mode="lines+markers"
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_tickangle=45,
        legend_orientation="h",
    )
    fig.update_yaxes(title_text="Trades closed", secondary_y=False)
    fig.update_yaxes(title_text="Win rate (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    exit_cols = [
        c
        for c in ["rebalance_drop", "time_exit", "trailing_atr", "trend_exit"]
        if c in m.columns
    ]
    for col in exit_cols:
        fig2.add_trace(go.Bar(x=m["ym"], y=m[col], name=col))
    fig2.update_layout(
        barmode="stack",
        height=380,
        title="exit理由（件数）",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickangle=45,
        legend_orientation="h",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 月次リターン
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=m["ym"],
            y=m["month_return_pct"],
            name="month_return_pct",
            mode="lines+markers",
        )
    )
    fig3.update_layout(
        height=320,
        title="月次リターン(%)",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickangle=45,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- パターン（入口）× exit（出口） ---
    st.subheader("パターン別（入口）× exit理由（出口）")
    t = trades[trades["policy"] == policy].copy()
    t["is_win"] = t["pnl"] > 0

    pat = (
        t.groupby("pattern", dropna=False)
        .agg(
            trade_count=("pnl", "size"),
            win_rate=("is_win", "mean"),
            avg_return_pct=("return_pct", "mean"),
        )
        .reset_index()
        .sort_values("trade_count", ascending=False)
    )
    top_k = st.slider("表示パターン数（上位）", 5, 30, 12)
    pat_top = pat.head(top_k)
    pat_top = pat_top.copy()
    pat_top["win_rate_pct"] = pat_top["win_rate"] * 100.0

    fig4 = px.bar(
        pat_top,
        x="pattern",
        y="win_rate_pct",
        text="trade_count",
        labels={"win_rate_pct": "Win rate (%)", "pattern": "pattern"},
        title="パターン別 勝率（棒） / 取引回数（テキスト）",
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig4, use_container_width=True)

    ct = pd.crosstab(t["pattern"].fillna("unknown"), t["exit_reason"].fillna("unknown"))
    if len(ct.index) > top_k:
        keep = pat_top["pattern"].astype(str).tolist()
        ct = ct.loc[[p for p in ct.index if str(p) in set(keep)]]

    fig5 = px.imshow(
        ct,
        text_auto=True,
        aspect="auto",
        title="Pattern × Exit reason（件数）",
    )
    fig5.update_layout(
        height=max(380, 28 * len(ct.index)), margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig5, use_container_width=True)

    # --- 負け要因（RSI） ---
    st.subheader("負け要因（RSI帯 × exit理由 など）")
    if enriched is None or enriched.empty:
        st.info(
            "data/backtest_trades_enriched.csv が無いため、RSI要因の可視化はスキップします。"
        )
    else:
        e = enriched[enriched["policy"] == policy].copy()
        for c in ["pnl", "return_pct", "rsi_signal"]:
            if c in e.columns:
                e[c] = pd.to_numeric(e[c], errors="coerce")
        loss = (
            e[e.get("is_loss", False) == True].copy()  # noqa: E712  (pandas mask)
            if "is_loss" in e.columns
            else e[e["pnl"] < 0].copy()
        )
        if loss.empty:
            st.write("負けトレードがありません。")
        else:
            if "rsi_bin" in loss.columns and "exit_reason" in loss.columns:
                ctl = pd.crosstab(
                    loss["rsi_bin"].fillna("unknown"),
                    loss["exit_reason"].fillna("unknown"),
                )
                share = ctl.div(ctl.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)

                fig6 = go.Figure()
                for col in share.columns:
                    fig6.add_trace(
                        go.Bar(x=share.index.astype(str), y=share[col], name=str(col))
                    )
                fig6.update_layout(
                    barmode="stack",
                    height=380,
                    title="負けトレード: RSI帯ごとの exit理由構成比",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig6, use_container_width=True)

            if "rsi_bin" in loss.columns and "return_pct" in loss.columns:
                fig7 = px.box(
                    loss,
                    x="rsi_bin",
                    y="return_pct",
                    points=False,
                    title="負けトレード: RSI帯別の損失分布（return %）",
                )
                fig7.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig7, use_container_width=True)

            if (
                "rsi_signal" in loss.columns
                and "return_pct" in loss.columns
                and "exit_reason" in loss.columns
            ):
                fig8 = px.scatter(
                    loss,
                    x="rsi_signal",
                    y="return_pct",
                    color="exit_reason",
                    hover_data=["ticker", "pattern", "entry_date", "exit_date"],
                    title="負けトレード: rsi_signal × return %（色=exit理由）",
                )
                fig8.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig8, use_container_width=True)

    # --- シナリオ比較（summary_grid.csv） ---
    st.subheader("シナリオ比較（summary_grid.csv）")
    if grid is None or grid.empty:
        st.info(
            "data/scenarios/summary_grid.csv が無いため、シナリオ比較はスキップします。"
        )
        return

    g = grid[grid["policy"] == policy].copy()
    for c in ["cagr", "max_drawdown", "total_return_pct", "trade_count"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    fig9 = px.scatter(
        g,
        x="max_drawdown",
        y="cagr",
        color="scenario",
        hover_data=["total_return_pct", "trade_count"],
        title="CAGR × 最大DD（点=シナリオ）",
    )
    fig9.update_layout(
        height=420, margin=dict(l=10, r=10, t=50, b=10), showlegend=False
    )
    st.plotly_chart(fig9, use_container_width=True)

    top_n = st.slider("上位表示数", 5, 30, 12)
    metric = st.selectbox(
        "ランキング指標",
        ["total_return_pct", "cagr", "max_drawdown", "trade_count"],
        index=0,
    )
    ascending = metric in {"max_drawdown"}
    top = g.sort_values(metric, ascending=ascending).head(top_n)
    fig10 = px.bar(
        top,
        x="scenario",
        y=metric,
        title=f"上位{top_n}: {metric}",
    )
    fig10.update_layout(
        height=380, margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=45
    )
    st.plotly_chart(fig10, use_container_width=True)

    # equity curve compare
    st.subheader("エクイティカーブ比較（シナリオ別）")
    scenario_options = sorted(g["scenario"].dropna().unique().tolist())
    baseline = next(
        (s for s in scenario_options if str(s).startswith("S0_baseline")), None
    )
    best = (
        str(g.sort_values("cagr", ascending=False).iloc[0]["scenario"])
        if len(g)
        else None
    )
    default_sel = [x for x in [baseline, best] if x]
    selected = st.multiselect(
        "比較するシナリオ",
        options=scenario_options,
        default=default_sel,
    )
    include_nikkei = st.checkbox(
        "日経225（Buy&Hold）も重ねる（元本1000万に補正）", value=True
    )
    if selected:
        curves = []
        for scen in selected:
            fp = scenarios_dir / scen / "backtest_equity_curve.csv"
            if not fp.exists():
                continue
            eq = _read_csv(fp)
            eq = eq[eq["policy"] == policy].copy()
            if eq.empty:
                continue
            eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
            eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
            eq["scenario"] = scen
            curves.append(eq[["date", "equity", "scenario"]])
        if curves:
            dfc = pd.concat(curves, ignore_index=True)

            if include_nikkei:

                @st.cache_data(ttl=3600)
                def _load_nikkei() -> pd.DataFrame | None:
                    return get_nikkei225()

                nikkei = _load_nikkei()
                if nikkei is None or nikkei.empty:
                    st.info(
                        "日経225データ（^N225）を取得できませんでした。ネットワークやキャッシュを確認してください。"
                    )
                else:
                    close_col = (
                        "Close"
                        if "Close" in nikkei.columns
                        else ("close" if "close" in nikkei.columns else None)
                    )
                    if close_col is None:
                        st.info("日経225データに Close 列がありませんでした。")
                    else:
                        nk = nikkei.copy()
                        nk.index = pd.to_datetime(nk.index, errors="coerce")
                        nk = nk[nk.index.notna()]

                        dmin = pd.to_datetime(dfc["date"].min(), errors="coerce")
                        dmax = pd.to_datetime(dfc["date"].max(), errors="coerce")
                        if pd.notna(dmin) and pd.notna(dmax):
                            nk = nk[(nk.index >= dmin) & (nk.index <= dmax)]

                        if len(nk) >= 2:
                            base = float(srow.get("initial_capital", 10_000_000.0))
                            c0 = float(
                                pd.to_numeric(nk[close_col].iloc[0], errors="coerce")
                            )
                            if c0 and pd.notna(c0):
                                equity_nk = base * (
                                    pd.to_numeric(nk[close_col], errors="coerce") / c0
                                )
                                add = pd.DataFrame(
                                    {
                                        "date": nk.index,
                                        "equity": equity_nk,
                                        "scenario": "Nikkei225_BuyHold",
                                    }
                                )
                                dfc = pd.concat([dfc, add], ignore_index=True)

            fig11 = px.line(
                dfc, x="date", y="equity", color="scenario", title="Equity curve"
            )
            fig11.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig11, use_container_width=True)
        else:
            st.info("選択したシナリオのエクイティカーブCSVが見つかりませんでした。")


# ------------------------------------------------------------------
# 投資スタイル最適化タブ
# ------------------------------------------------------------------
import numpy as np  # noqa: E402  (already available via plotly deps)


_STYLE_OPTIMIZER_PATTERNS: dict[str, set[str]] = {
    "全(A+B+C+D)": {"A_trend", "B_pullback", "C_breakout", "D_reversal"},
    "全(A+B+C+D+E)": {
        "A_trend",
        "B_pullback",
        "C_breakout",
        "D_reversal",
        "E_can_slim",
    },
    "全(A+B+C+D+E+F)": {
        "A_trend",
        "B_pullback",
        "C_breakout",
        "D_reversal",
        "E_can_slim",
        "F_turnaround",
    },
    "A+B+C": {"A_trend", "B_pullback", "C_breakout"},
    "A+B+C+E": {"A_trend", "B_pullback", "C_breakout", "E_can_slim"},
    "A+B+C+F": {"A_trend", "B_pullback", "C_breakout", "F_turnaround"},
    "A+B": {"A_trend", "B_pullback"},
    "A+C": {"A_trend", "C_breakout"},
    "B+C": {"B_pullback", "C_breakout"},
    "Aのみ": {"A_trend"},
    "Cのみ": {"C_breakout"},
    "Eのみ": {"E_can_slim"},
    "Fのみ": {"F_turnaround"},
}

_STYLE_HOLDING_DAYS = {
    "短期（5〜20日）": [5, 10, 15, 20],
    "中期（30〜60日）": [30, 60],
    "長期（120〜252日）": [120, 252],
}


def _nikkei_cache_signature() -> str:
    """^N225 キャッシュ更新に追従して cache_resource を更新するためのシグネチャ"""
    try:
        fp = CACHE_DIR / "^N225.parquet"
        if fp.exists():
            return str(fp.stat().st_mtime_ns)
    except Exception:
        pass
    return "missing"


@st.cache_resource(show_spinner="銘柄データを読み込み中（初回のみ時間がかかります）...")
def _load_optimizer_contexts(_bench_sig: str):
    """コンテキストをキャッシュ（1アプリ起動につき1回だけ構築）"""
    base_dir = Path(__file__).resolve().parent
    strategy = load_strategy(base_dir / "config" / "strategy.yaml")
    stock_list_df = load_stock_list()
    bench = get_nikkei225()
    if bench is None or len(bench) < 250:
        return None, None, None, None, None
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)
    bench.sort_index(inplace=True)
    end_date = pd.Timestamp(bench.index.max())
    contexts = build_contexts(
        stock_list_df=stock_list_df,
        load_cached_fn=load_cached,
        strategy=strategy,
        end_date=end_date,
    )
    trading_days = pd.DatetimeIndex(bench.index)
    return contexts, strategy, trading_days, end_date, bench


def _nikkei_bench_metrics_app(bench_df, start_date, end_date):
    close_col = "Close" if "Close" in bench_df.columns else "close"
    nk = bench_df.copy()
    nk.index = pd.to_datetime(nk.index)
    nk = nk[(nk.index >= start_date) & (nk.index <= end_date)]
    if len(nk) < 2:
        return {}
    closes = pd.to_numeric(nk[close_col], errors="coerce").dropna()
    start_price, end_price = float(closes.iloc[0]), float(closes.iloc[-1])
    total_return = (end_price / start_price - 1.0) * 100.0
    years = max((closes.index[-1] - closes.index[0]).days / 365.25, 1e-9)
    cagr = (end_price / start_price) ** (1.0 / years) - 1.0
    peak = -np.inf
    mdd = 0.0
    for v in closes:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)
    rets = closes.pct_change().dropna()
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {
        "total_return_pct": round(total_return, 3),
        "cagr": round(cagr, 6),
        "max_drawdown": round(mdd, 6),
        "sharpe": round(sharpe, 6),
    }


def _show_precomputed_results(opt_csv: Path, equity_csv: Path) -> None:
    import plotly.graph_objects as go

    df = pd.read_csv(opt_csv, encoding="utf-8-sig")
    for c in [
        "total_return_pct",
        "cagr",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "trade_count",
        "beats_nikkei",
        "excess_return_pct",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    nikkei_total = (
        float(df["nikkei_total_return_pct"].dropna().iloc[0])
        if "nikkei_total_return_pct" in df.columns
        else 0.0
    )
    nikkei_cagr = (
        float(df["nikkei_cagr"].dropna().iloc[0])
        if "nikkei_cagr" in df.columns
        else 0.0
    )

    # 日経225指標
    st.subheader("📊 日経225 ベンチマーク（Buy & Hold）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総リターン", f"{nikkei_total:+.2f}%")
    c2.metric("CAGR", f"{nikkei_cagr:.4f}")
    if "nikkei_max_drawdown" in df.columns:
        nk_dd = float(df["nikkei_max_drawdown"].dropna().iloc[0])
        c3.metric("最大DD", f"{nk_dd:.4f}")
    if "nikkei_sharpe" in df.columns:
        nk_sh = float(df["nikkei_sharpe"].dropna().iloc[0])
        c4.metric("Sharpe", f"{nk_sh:.4f}")

    # スタイル別ベスト
    st.subheader("🏆 スタイル別 ベスト設定（CAGR順）")
    style_cols = st.columns(3)
    for i, style_name in enumerate(["短期", "中期", "長期"]):
        sub = df[df["style"] == style_name].sort_values("cagr", ascending=False)
        with style_cols[i]:
            if sub.empty:
                st.info(f"{style_name}: データなし")
                continue
            best = sub.iloc[0]
            beats = bool(best.get("beats_nikkei", False))
            badge = "✅ 日経225超" if beats else "❌ 日経225未達"
            st.markdown(f"**【{style_name}】** {badge}")
            st.markdown(f"保有日数: **{int(best['holding_days'])}日**")
            st.markdown(f"パターン: **{best['pattern_combo']}**")
            st.metric(
                "総リターン",
                f"{best.get('total_return_pct', 0):+.2f}%",
                delta=f"vs 日経 {best.get('excess_return_pct', 0):+.2f}%",
            )
            st.metric("CAGR", f"{best.get('cagr', 0):.4f}")
            st.metric("最大DD", f"{best.get('max_drawdown', 0):.4f}")
            st.metric("Sharpe", f"{best.get('sharpe', 0):.4f}")
            st.metric("勝率", f"{best.get('win_rate', 0):.1%}")

    # フィルタ付きランキング表
    st.subheader("📋 全シナリオ 成績一覧")
    filter_beats = st.checkbox(
        "日経225を上回る設定のみ表示", value=True, key="opt_filter_beats"
    )
    filter_styles = st.multiselect(
        "スタイル絞り込み",
        options=["短期", "中期", "長期"],
        default=["短期", "中期", "長期"],
        key="opt_style_filter",
    )

    show = df.copy()
    if filter_beats:
        show = show[show["beats_nikkei"] == True]  # noqa: E712
    if filter_styles:
        show = show[show["style"].isin(filter_styles)]

    show = show.sort_values("cagr", ascending=False)

    display_cols = [
        "style",
        "holding_days",
        "pattern_combo",
        "total_return_pct",
        "excess_return_pct",
        "cagr",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "trade_count",
    ]
    display_cols = [c for c in display_cols if c in show.columns]
    rename_map = {
        "style": "スタイル",
        "holding_days": "保有日数",
        "pattern_combo": "パターン",
        "total_return_pct": "総リターン%",
        "excess_return_pct": "超過リターン%",
        "cagr": "CAGR",
        "max_drawdown": "最大DD",
        "sharpe": "Sharpe",
        "win_rate": "勝率",
        "trade_count": "取引数",
    }
    st.dataframe(
        show[display_cols].rename(columns=rename_map),
        use_container_width=True,
        hide_index=True,
    )

    # エクイティカーブ
    if equity_csv.exists():
        st.subheader("📈 エクイティカーブ比較")
        try:
            eq_df = pd.read_csv(equity_csv, encoding="utf-8-sig")
            eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
            eq_df["equity"] = pd.to_numeric(eq_df["equity"], errors="coerce")

            scenario_options = [
                s
                for s in eq_df["scenario_id"].dropna().unique()
                if s != "Nikkei225_BuyHold"
            ]
            beats_set = set(df[df["beats_nikkei"] == True]["scenario_id"].tolist())  # noqa: E712
            default_scenarios = [s for s in scenario_options if s in beats_set][:5]

            selected_scens = st.multiselect(
                "比較するシナリオを選択（日経225は常に表示）",
                options=sorted(scenario_options),
                default=default_scenarios,
                key="opt_equity_select",
            )

            plot_scenarios = selected_scens + ["Nikkei225_BuyHold"]
            plot_df = eq_df[eq_df["scenario_id"].isin(plot_scenarios)].copy()

            if not plot_df.empty:
                fig = go.Figure()
                for scen in plot_scenarios:
                    sub = plot_df[plot_df["scenario_id"] == scen].sort_values("date")
                    if sub.empty:
                        continue
                    is_nikkei = scen == "Nikkei225_BuyHold"
                    fig.add_trace(
                        go.Scatter(
                            x=sub["date"],
                            y=sub["equity"],
                            name=scen,
                            line=dict(
                                width=3 if is_nikkei else 1.5,
                                dash="dash" if is_nikkei else "solid",
                                color="orange" if is_nikkei else None,
                            ),
                            mode="lines",
                        )
                    )
                fig.update_layout(
                    height=480,
                    title="エクイティカーブ（定額スタイル vs 日経225 Buy&Hold）",
                    xaxis_title="日付",
                    yaxis_title="資産額（円）",
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend_orientation="v",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"エクイティカーブの読み込みに失敗しました: {exc}")


def _show_custom_simulation(base_dir: Path, bench_df_ref) -> None:
    """インタラクティブなカスタムシミュレーション"""
    import plotly.graph_objects as go

    st.markdown("パラメータを指定してリアルタイムでバックテストを実行します。")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        style_choice = st.selectbox(
            "投資スタイル",
            options=[
                "短期（5〜20日）",
                "中期（30〜60日）",
                "長期（120〜252日）",
                "カスタム",
            ],
            key="custom_style",
        )

        style_default_days = {
            "短期（5〜20日）": 10,
            "中期（30〜60日）": 30,
            "長期（120〜252日）": 120,
            "カスタム": 20,
        }
        default_days = style_default_days.get(style_choice, 20)

        holding_days = st.slider(
            "保有日数（time_exit）",
            min_value=5,
            max_value=252,
            value=default_days,
            step=5,
            key="custom_holding_days",
        )

        combo_choice = st.selectbox(
            "パターン組み合わせ",
            options=list(_STYLE_OPTIMIZER_PATTERNS.keys()),
            key="custom_pattern_combo",
        )

        initial_capital = st.number_input(
            "初期資金（円）",
            min_value=1_000_000,
            max_value=100_000_000,
            value=10_000_000,
            step=1_000_000,
            format="%d",
            key="custom_initial",
        )

        max_positions = st.number_input(
            "最大同時保有数",
            min_value=1,
            max_value=20,
            value=20,
            key="custom_max_pos",
        )

        years = st.slider(
            "バックテスト期間（年）",
            min_value=1,
            max_value=5,
            value=3,
            key="custom_years",
        )

        run_btn = st.button(
            "▶ シミュレーション実行", type="primary", key="custom_run_btn"
        )

    with col_right:
        if not run_btn:
            st.info(
                "左のパラメータを設定して「シミュレーション実行」ボタンを押してください。"
            )
            return

        with st.spinner(
            "シミュレーション実行中...（初回はコンテキスト構築に時間がかかります）"
        ):
            contexts, strategy, trading_days, end_date, bench = (
                _load_optimizer_contexts(_nikkei_cache_signature())
            )

        if contexts is None:
            st.error(
                "^N225 キャッシュが見つかりません。先に batch/update.py を実行してください。"
            )
            return

        allowed_patterns = _STYLE_OPTIMIZER_PATTERNS[combo_choice]
        start_date = pd.Timestamp(end_date - pd.DateOffset(years=years))
        td_period = trading_days[
            (trading_days >= start_date) & (trading_days <= end_date)
        ]

        if len(td_period) < 60:
            st.error("期間が短すぎます。年数を増やしてください。")
            return

        nikkei_bench = _nikkei_bench_metrics_app(bench, start_date, end_date)
        nikkei_total = nikkei_bench.get("total_return_pct", 0.0)

        with st.spinner("バックテスト計算中..."):
            try:
                summary_df, trades_df, equity_df = run_backtest(
                    contexts=contexts,
                    trading_days=trading_days,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    policy="fixed_amount",
                    initial_capital=float(initial_capital),
                    max_positions=int(max_positions),
                    allowed_patterns=allowed_patterns,
                    time_exit_days_override=holding_days,
                )
            except Exception as exc:
                st.error(f"バックテストエラー: {exc}")
                return

        if summary_df.empty:
            st.warning(
                "有効な結果がありませんでした。パラメータや期間を見直してください。"
            )
            return

        srow = summary_df.iloc[0]
        total_ret = float(srow.get("total_return_pct", 0))
        beats = total_ret > nikkei_total

        # 結果メトリクス
        badge = (
            "✅ 日経225を上回りました！" if beats else "❌ 日経225に届きませんでした"
        )
        st.markdown(f"### {badge}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "総リターン",
            f"{total_ret:+.2f}%",
            delta=f"vs 日経 {total_ret - nikkei_total:+.2f}%",
        )
        m2.metric("CAGR", f"{float(srow.get('cagr', 0)):.4f}")
        m3.metric("最大DD", f"{float(srow.get('max_drawdown', 0)):.4f}")
        m4.metric("Sharpe", f"{float(srow.get('sharpe', 0)):.4f}")
        m5.metric("勝率", f"{float(srow.get('win_rate', 0)):.1%}")

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("取引数", f"{int(srow.get('trade_count', 0))}回")
        n2.metric("日経225 総リターン", f"{nikkei_total:+.2f}%")
        n3.metric("日経225 CAGR", f"{nikkei_bench.get('cagr', 0):.4f}")
        n4.metric("日経225 最大DD", f"{nikkei_bench.get('max_drawdown', 0):.4f}")

        # エクイティカーブ
        equity_df["date"] = pd.to_datetime(equity_df["date"], errors="coerce")
        equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")

        close_col = "Close" if "Close" in bench.columns else "close"
        nk = bench.copy()
        nk.index = pd.to_datetime(nk.index)
        nk = nk[(nk.index >= start_date) & (nk.index <= end_date)]
        nk_closes = pd.to_numeric(nk[close_col], errors="coerce").dropna()
        nk_equity = float(initial_capital) * nk_closes / float(nk_closes.iloc[0])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_df["date"],
                y=equity_df["equity"],
                name=f"ABCD戦略（{holding_days}日/{combo_choice}）",
                line=dict(color="royalblue", width=2),
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=nk.index,
                y=nk_equity,
                name="日経225 Buy&Hold",
                line=dict(color="orange", width=2, dash="dash"),
                mode="lines",
            )
        )
        fig.update_layout(
            height=420,
            title="エクイティカーブ（定額スタイル vs 日経225）",
            xaxis_title="日付",
            yaxis_title="資産額（円）",
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # パターン別成績
        if not trades_df.empty:
            st.subheader("パターン別 成績")
            pat_summary = (
                trades_df.groupby("pattern")
                .agg(
                    取引数=("pnl", "size"),
                    勝率=("pnl", lambda s: (s > 0).mean()),
                    平均損益=("return_pct", "mean"),
                    合計損益=("pnl", "sum"),
                )
                .reset_index()
                .sort_values("合計損益", ascending=False)
            )
            pat_summary["勝率"] = pat_summary["勝率"].map("{:.1%}".format)
            pat_summary["平均損益"] = pat_summary["平均損益"].map("{:+.2f}%".format)
            pat_summary["合計損益"] = pat_summary["合計損益"].map("{:,.0f}円".format)
            st.dataframe(pat_summary, use_container_width=True, hide_index=True)


def show_style_optimizer_view():
    st.title("📊 投資スタイル最適化")
    st.markdown(
        """
        **ABCD戦略**を**短期・中期・長期**でバックテストし、過去3年間で
        **日経225 Buy & Hold** を上回る投資スタイルを特定します。

        | 条件 | 設定 |
        |------|------|
        | 投資スタイル | **定額スタイル**（初期資金を最大20銘柄に等分） |
        | 最大同時保有 | **20銘柄** |
        | リバランス | **週次**（毎週月曜日） |
        | 売買コスト | **0.3%**（往復込み） |
        | ベンチマーク | **日経225 Buy & Hold** |
        """
    )

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    opt_csv = data_dir / "style_optimization.csv"
    equity_csv = data_dir / "style_optimization_equity.csv"

    # ① 事前計算済み結果の表示
    if opt_csv.exists():
        with st.expander("✅ 最適化済み結果（全シナリオ）", expanded=True):
            _show_precomputed_results(opt_csv, equity_csv)
    else:
        st.warning(
            "最適化結果CSVが見つかりません。以下のコマンドで全シナリオを実行してください:"
        )
        st.code("python batch/style_optimizer.py")
        st.info(
            "初回実行には20〜30分かかる場合があります。"
            "実行後、ページをリロードすると結果が表示されます。"
        )

    # ② インタラクティブシミュレーション
    st.divider()
    st.subheader("🔧 カスタムシミュレーション（リアルタイム実行）")

    bench_data = get_nikkei225()
    if bench_data is None:
        st.error(
            "^N225 キャッシュが見つかりません。先に batch/update.py を実行してください。"
        )
        return

    _show_custom_simulation(base_dir, bench_data)


# ------------------------------------------------------------------
# ポートフォリオ・ダッシュボード
# ------------------------------------------------------------------
def _signal_color_map(severity: str) -> str:
    return {
        "critical": "🔴",
        "warning": "🟡",
        "info": "🟢",
    }.get(str(severity), "ℹ️")


def _navigate_to_detail(code: str, name: str, ticker: str) -> None:
    """波形分類タブの詳細ビューへ遷移する。"""
    st.session_state["selected_code"] = str(code)
    st.session_state["selected_name"] = str(name)
    st.session_state["selected_ticker"] = str(ticker)
    st.session_state["view"] = "detail"
    st.rerun()


def show_portfolio_view():
    """ポートフォリオ・ダッシュボード(新規タブ)。

    - 投資状況サマリ(総資産 / 損益 / コア-衛星比)
    - 本日のシグナル(発生時のみ)
    - 累積パフォーマンスチャート
    - 保有銘柄テーブル(行クリックで詳細遷移)
    - 監視銘柄テーブル(同上)
    """
    st.title("📊 ポートフォリオ・ダッシュボード")

    # ----- サイドバー操作 -----
    with st.sidebar:
        st.header("ポートフォリオ操作")

        with st.expander("🎯 初期ポートフォリオ生成", expanded=False):
            st.caption(
                "data/portfolio_initial.csv の構成で 3,000万円分の "
                "仮想ポジションを一括投入します。"
            )
            if st.button("初期投入を実行", key="port_init_btn"):
                try:
                    df = initialize_from_template()
                    st.success(f"初期ポートフォリオを生成しました ({len(df)} 件)")
                    st.rerun()
                except Exception as e:
                    st.error(f"初期化失敗: {e}")

        with st.expander("➕ ポジション追加(買い)", expanded=False):
            with st.form("add_position_form"):
                add_ticker = st.text_input(
                    "ティッカー (例: 6855.T, NVDA, VOO)", key="add_ticker"
                )
                add_name = st.text_input("銘柄名", key="add_name")
                add_shares = st.number_input(
                    "株数", min_value=0.0, step=1.0, key="add_shares"
                )
                add_cost = st.number_input(
                    "取得単価(現地通貨)", min_value=0.0, step=0.01, key="add_cost"
                )
                add_category = st.selectbox(
                    "カテゴリ", ["satellite", "core"], key="add_category"
                )
                add_submitted = st.form_submit_button("追加")
            if add_submitted and add_ticker:
                code_in = add_ticker.replace(".T", "")
                currency = "JPY" if add_ticker.upper().endswith(".T") else "USD"
                try:
                    add_position(
                        code=code_in,
                        name=add_name or add_ticker,
                        ticker=add_ticker,
                        shares=float(add_shares),
                        cost=float(add_cost),
                        currency=currency,
                        category=add_category,
                    )
                    st.success(f"{add_ticker} を追加しました")
                    st.rerun()
                except Exception as e:
                    st.error(f"追加失敗: {e}")

        with st.expander("➖ 売却記録", expanded=False):
            portfolio_for_sell = load_portfolio()
            sellable = portfolio_for_sell[portfolio_for_sell["category"] != "cash"]
            if len(sellable) == 0:
                st.info("売却可能ポジションがありません")
            else:
                with st.form("sell_position_form"):
                    sell_code = st.selectbox(
                        "銘柄",
                        sellable["code"].astype(str).tolist(),
                        format_func=lambda c: (
                            f"{c} {sellable[sellable['code'].astype(str) == c]['name'].iloc[0]}"
                        ),
                        key="sell_code",
                    )
                    sell_shares = st.number_input(
                        "売却株数", min_value=0.0, step=1.0, key="sell_shares"
                    )
                    sell_price = st.number_input(
                        "売却単価(現地通貨)",
                        min_value=0.0,
                        step=0.01,
                        key="sell_price",
                    )
                    sell_submitted = st.form_submit_button("売却")
                if sell_submitted:
                    try:
                        record_sell(
                            code=sell_code,
                            shares=float(sell_shares),
                            price=float(sell_price),
                        )
                        st.success(f"{sell_code} を {sell_shares} 株売却")
                        st.rerun()
                    except Exception as e:
                        st.error(f"売却失敗: {e}")

    # ----- メインエリア -----
    val = compute_current_valuation()
    perf = compute_performance_metrics()

    if val["total_value"] == 0 and val["cash"] == 0:
        st.info(
            "ポートフォリオが空です。サイドバーの「初期ポートフォリオ生成」または "
            "「ポジション追加」から開始してください。"
        )
        return

    # サマリーカード
    st.caption(f"株価更新: **{val['latest_date'] or '未取得'}** 時点")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "総資産",
        f"¥{val['total_value']:,.0f}",
        f"{val['pnl_jpy']:+,.0f}",
    )
    col2.metric("損益率", f"{val['pnl_pct']:+.2f}%")
    col3.metric(
        "コア",
        f"¥{val['core_value']:,.0f}",
        f"{val['core_value'] / val['total_value'] * 100:.1f}%"
        if val["total_value"] > 0
        else "—",
    )
    col4.metric(
        "衛星",
        f"¥{val['satellite_value']:,.0f}",
        f"{val['satellite_value'] / val['total_value'] * 100:.1f}%"
        if val["total_value"] > 0
        else "—",
    )
    col5.metric(
        "キャッシュ",
        f"¥{val['cash']:,.0f}",
        f"{val['cash'] / val['total_value'] * 100:.1f}%"
        if val["total_value"] > 0
        else "—",
    )

    if perf.get("cumulative_return_pct") is not None:
        st.caption(
            f"初期資金 ¥{perf.get('initial_capital_jpy', 30_000_000):,.0f} → 現在 "
            f"¥{val['total_value']:,.0f} | 累積 {perf['cumulative_return_pct']:+.2f}%"
            + (
                f" | 年率換算 {perf['annualized_return_pct']:+.2f}%"
                if perf.get("annualized_return_pct") is not None
                else ""
            )
            + (
                f" | 最大DD {perf['max_drawdown_pct']:.2f}%"
                if perf.get("max_drawdown_pct") is not None
                else ""
            )
        )

    st.divider()

    # ----- 本日のシグナル -----
    st.subheader("⚡ 本日のシグナル")
    signals = get_today_signals()
    if signals.empty:
        st.info("本日のシグナルなし。アクション不要")
    else:
        # 重要度順
        sev_order = {"critical": 0, "warning": 1, "info": 2}
        signals = signals.copy()
        signals["_ord"] = signals["severity"].map(sev_order).fillna(99)
        signals = signals.sort_values("_ord").drop(columns=["_ord"])

        for _, s in signals.iterrows():
            badge = _signal_color_map(s["severity"])
            side = s["side"]
            label = (
                "売却推奨"
                if side == "SELL"
                else ("買い推奨" if side == "BUY" else "様子見")
            )
            msg = (
                f"{badge} **[{label}] {s['code']} {s['name']}** — "
                f"{s['signal_type']} | 現値 {s['current_price']:.2f} | "
                f"{s['message']}"
            )
            if s["severity"] == "critical":
                st.error(msg)
            elif s["severity"] == "warning":
                st.warning(msg)
            else:
                st.success(msg)

    st.divider()

    # ----- パフォーマンスチャート -----
    st.subheader("📈 累積パフォーマンス")
    hist = load_portfolio_history()
    if hist.empty or len(hist) < 2:
        st.info("履歴データが2日分未満です。バッチを継続実行することで蓄積されます")
    else:
        hist_plot = hist.copy()
        hist_plot["date"] = pd.to_datetime(hist_plot["date"])
        hist_plot = hist_plot.sort_values("date")
        fig = px.line(
            hist_plot,
            x="date",
            y="total_value_jpy",
            title="ポートフォリオ評価額の推移",
            labels={"date": "日付", "total_value_jpy": "総資産(円)"},
        )
        fig.add_hline(
            y=30_000_000,
            line_dash="dot",
            annotation_text="初期資金 3,000万円",
            annotation_position="top left",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----- 保有銘柄テーブル -----
    st.subheader("💼 保有銘柄")
    if val["by_holding"]:
        holdings_df = pd.DataFrame(val["by_holding"])
        holdings_df = holdings_df.sort_values("value_jpy", ascending=False)
        display = holdings_df[
            [
                "code",
                "name",
                "category",
                "shares",
                "avg_cost",
                "last_price",
                "currency",
                "value_jpy",
                "pnl_jpy",
                "pnl_pct",
                "weight_pct",
            ]
        ].rename(
            columns={
                "code": "コード",
                "name": "銘柄",
                "category": "区分",
                "shares": "株数",
                "avg_cost": "取得単価",
                "last_price": "現値",
                "currency": "通貨",
                "value_jpy": "評価額(円)",
                "pnl_jpy": "損益(円)",
                "pnl_pct": "損益率(%)",
                "weight_pct": "比率(%)",
            }
        )
        event = st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="portfolio_table",
            column_config={
                "評価額(円)": st.column_config.NumberColumn(format="¥%.0f"),
                "損益(円)": st.column_config.NumberColumn(format="¥%.0f"),
                "損益率(%)": st.column_config.NumberColumn(format="%.2f%%"),
                "比率(%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        if event and event.selection.rows:
            r = display.iloc[event.selection.rows[0]]
            ticker_lookup = holdings_df.iloc[event.selection.rows[0]]["ticker"]
            _navigate_to_detail(r["コード"], r["銘柄"], ticker_lookup)
    else:
        st.info("保有なし")

    st.divider()

    # ----- 監視銘柄テーブル -----
    st.subheader("👁 監視銘柄")
    from config.settings import WATCHLIST_CSV

    if Path(WATCHLIST_CSV).exists():
        watch = pd.read_csv(WATCHLIST_CSV, encoding="utf-8-sig", dtype={"code": str})
        if len(watch) > 0:
            # 最新価格を併記
            latest_prices = []
            for _, w in watch.iterrows():
                df_cache = load_cached(str(w["ticker"]))
                if df_cache is not None and not df_cache.empty:
                    latest_prices.append(float(df_cache["Close"].iloc[-1]))
                else:
                    latest_prices.append(None)
            watch_disp = watch.copy()
            watch_disp["last_price"] = latest_prices
            watch_disp = watch_disp[
                [
                    "code",
                    "name",
                    "ticker",
                    "category",
                    "target_price",
                    "last_price",
                    "notes",
                ]
            ].rename(
                columns={
                    "code": "コード",
                    "name": "銘柄",
                    "ticker": "Ticker",
                    "category": "区分",
                    "target_price": "目標価格",
                    "last_price": "現値",
                    "notes": "メモ",
                }
            )
            event2 = st.dataframe(
                watch_disp,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="watchlist_table",
            )
            if event2 and event2.selection.rows:
                r = watch_disp.iloc[event2.selection.rows[0]]
                _navigate_to_detail(r["コード"], r["銘柄"], r["Ticker"])
        else:
            st.info("監視銘柄なし")
    else:
        st.info("watchlist.csv がありません")


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------
def main():
    if "view" not in st.session_state:
        st.session_state["view"] = "list"

    # 詳細画面は波形分類タブ側で表示（タブ外で表示）
    if st.session_state["view"] == "detail":
        show_detail_view()
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["ポートフォリオ", "波形分類", "ABCD戦略", "バックテスト", "投資スタイル最適化"]
    )

    with tab1:
        show_portfolio_view()

    with tab2:
        show_list_view()

    with tab3:
        show_strategy_view()

    with tab4:
        show_backtest_view()

    with tab5:
        show_style_optimizer_view()


if __name__ == "__main__":
    main()
