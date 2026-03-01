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
    DEFAULT_WINDOW,
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
)
from modules.chart_builder import build_chart, build_comparison_chart, build_financials_chart, build_index_chart
from modules.data_fetcher import (
    load_cached,
    load_stock_list,
    get_nikkei225,
    compute_sector_index,
    fetch_financials,
    compute_sector_stats,
    fetch_and_cache,
)
from modules.earnings_fetcher import load_earnings_dates, get_earnings_dates_for_code
from modules.wave_classifier import compute_indicators, classify
from modules.strategy_engine import generate_ranking
from modules.strategy_loader import load_strategy
from modules.backtester import build_contexts, run_backtest


@st.cache_data(ttl=3600, show_spinner=False)
def _get_price_data_cached(ticker: str) -> pd.DataFrame | None:
    return fetch_and_cache(ticker)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _get_financials_cached(ticker: str) -> pd.DataFrame | None:
    return fetch_financials(ticker)

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
| **収束（スクイーズ）** | ボリンジャーバンド幅が後半で前半の{int(SQUEEZE_BANDWIDTH_SHRINK*100)}%以下に縮小 | ボラティリティが低下し、次の大きな動きの前兆となりうる状態 |
| **ブレイク気味** | 直近{BREAKOUT_LOOKBACK_DAYS}日中{BREAKOUT_MIN_DAYS}日以上がレンジ外 | レンジの上限/下限を明確に超えた状態（終値基準） |
| **高ボラ（荒い）** | ATR ÷ 平均価格 > {HIGH_VOLATILITY_THRESHOLD*100:.0f}% | 価格変動が大きく、安定した反復が弱い |

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
            picks = pd.read_csv(DAILY_PICKS_CSV, encoding="utf-8-sig", dtype={"code": str})
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
        "レンジ（波型）" in types
        and slope >= 0
        and (touch_high >= 3 or has_breakout)
    )


# ------------------------------------------------------------------
# 一覧画面
# ------------------------------------------------------------------
def show_list_view():
    st.title("JPX500 波形タイプ分類")

    # データ更新日時の表示
    dates = get_data_dates()
    if dates:
        data_date = dates.get("latest_data", "不明")
        batch_date = dates.get("batch_run", "不明")
        st.caption(f"株価データ: **{data_date}** 時点 ｜ バッチ実行: {batch_date}")

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

    # 規模区分
    st.sidebar.subheader("規模区分")
    size_options = ["TOPIX Core30", "TOPIX Large70", "TOPIX Mid400", "ETF"]
    selected_sizes = []
    for s in size_options:
        label = s.replace("TOPIX ", "")
        if st.sidebar.checkbox(label, value=True, key=f"sz_{s}"):
            selected_sizes.append(s)

    # 33業種区分
    st.sidebar.subheader("33業種区分")
    all_sectors = sorted(results["sector_33"].dropna().unique().tolist()) if "sector_33" in results.columns else []
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

    # --- サマリーカード ---
    st.markdown("### サマリー")
    cols = st.columns(len(WAVE_TYPES))
    for i, wt in enumerate(WAVE_TYPES):
        count = results["wave_types"].apply(
            lambda x, _wt=wt: _wt in str(x) if pd.notna(x) else False
        ).sum()
        cols[i].metric(wt, f"{count}銘柄")

    # 推奨銘柄数
    rec_count = results.apply(is_recommended, axis=1).sum()
    st.info(f"レンジ→上昇転換候補: **{rec_count}銘柄** （サイドバーの「推奨銘柄」で絞り込み可能）")

    # --- マーケット概況 ---
    st.markdown("---")
    st.markdown("### マーケット概況")
    mkt_col1, mkt_col2 = st.columns(2)
    with mkt_col1:
        show_nikkei225 = st.checkbox("日経225", value=True, key="mkt_nikkei")
    with mkt_col2:
        sector_list = sorted(results["sector_33"].dropna().unique().tolist()) if "sector_33" in results.columns else []
        selected_mkt_sector = st.selectbox(
            "33業種指数",
            options=["（なし）"] + sector_list,
            index=0,
            key="mkt_sector",
        )

    mkt_period_labels = {60: "60日", 120: "120日", 180: "180日", 260: "1年", 520: "2年", 0: "全期間"}
    mkt_period_sel = st.radio(
        "表示期間",
        list(mkt_period_labels.keys()),
        index=1,
        horizontal=True,
        format_func=lambda x: mkt_period_labels[x],
        key="mkt_window",
    )

    mkt_indices = []
    if show_nikkei225:
        nk_data = get_nikkei225()
        if nk_data is not None:
            mkt_indices.append({"name": "日経225", "data": nk_data, "color": "#1976d2"})
    if selected_mkt_sector and selected_mkt_sector != "（なし）":
        sec_data = compute_sector_index(selected_mkt_sector, results)
        if sec_data is not None:
            mkt_indices.append({"name": f"33業種: {selected_mkt_sector}", "data": sec_data, "color": "#E91E63"})

    if mkt_indices:
        fig_mkt = build_index_chart(mkt_indices, window=mkt_period_sel)
        st.plotly_chart(fig_mkt, use_container_width=True)
    else:
        st.caption("表示する指数を選択してください。")

    # --- 33業種サマリー ---
    st.markdown("---")
    st.markdown("### 33業種 PER・PBR・時価総額")
    st.caption("行をクリックすると銘柄一覧をその業種で絞り込みます")
    sector_stats = compute_sector_stats(results)
    if len(sector_stats) > 0:
        sector_display = sector_stats.rename(columns={
            "sector_33": "33業種",
            "count": "銘柄数",
            "per_median": "PER（中央値）",
            "pbr_median": "PBR（中央値）",
            "market_cap_total": "時価総額合計（億円）",
            "market_cap_count": "時価総額取得数",
            "market_cap_coverage_pct": "時価総額取得率（%）",
        })
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
        if "時価総額合計（億円）" in sector_display.columns and sector_display["時価総額合計（億円）"].isna().all():
            for col in ["時価総額合計（億円）", "時価総額取得数", "時価総額取得率（%）"]:
                if col in display_cols_sector:
                    display_cols_sector.remove(col)
        # 解除ボタンが押された直後はテーブル選択をリセット
        _sector_table_key = "sector_table"
        if st.session_state.get("_clear_sector_flag"):
            _sector_table_key = f"sector_table_{st.session_state.get('_sector_table_ver', 0)}"

        sector_event = st.dataframe(
            sector_display[[c for c in display_cols_sector if c in sector_display.columns]],
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
            selected_sector_names = [sector_stats.iloc[i]["sector_33"] for i in sector_event.selection.rows]
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
        label = "、".join(quick_sectors) if len(quick_sectors) <= 3 else f"{quick_sectors[0]} 他{len(quick_sectors)-1}業種"
        if st.button(f"業種絞り込み解除（現在: {label}）", key="btn_clear_sector"):
            st.session_state["quick_sector_filter"] = None
            # テーブルキーを変えて選択状態をリセット
            st.session_state["_clear_sector_flag"] = True
            st.session_state["_sector_table_ver"] = st.session_state.get("_sector_table_ver", 0) + 1
            st.rerun()

    # --- PER×PBR 散布図 ---
    st.markdown("---")
    st.markdown("### PER × PBR 散布図")
    perpbr_scope = st.radio(
        "対象",
        ["絞り込み後", "全銘柄"],
        index=0,
        horizontal=True,
        key="perpbr_scope",
    )
    base_df = filtered if perpbr_scope == "絞り込み後" else results
    if base_df is not None and len(base_df) > 0 and ("per" in base_df.columns) and ("pbr" in base_df.columns):
        scatter_df = base_df.copy()
        scatter_df["per"] = pd.to_numeric(scatter_df["per"], errors="coerce")
        scatter_df["pbr"] = pd.to_numeric(scatter_df["pbr"], errors="coerce")
        scatter_df = scatter_df.dropna(subset=["per", "pbr"])
        scatter_df = scatter_df[(scatter_df["per"] > 0) & (scatter_df["pbr"] > 0)]

        if len(scatter_df) == 0:
            st.caption("PER/PBRのデータがありません（欠損または0以下）。")
        else:
            color_col = "sector_33" if "sector_33" in scatter_df.columns else None
            fig_scatter = px.scatter(
                scatter_df,
                x="per",
                y="pbr",
                color=color_col,
                hover_name="name" if "name" in scatter_df.columns else None,
                hover_data={
                    "code": True if "code" in scatter_df.columns else False,
                    "size_category": True if "size_category" in scatter_df.columns else False,
                    "wave_types": True if "wave_types" in scatter_df.columns else False,
                    "per": ":.2f",
                    "pbr": ":.2f",
                },
            )
            fig_scatter.update_layout(
                height=520,
                xaxis_title="PER",
                yaxis_title="PBR",
                legend_title_text="33業種" if color_col else None,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"表示件数: {len(scatter_df)}")
    else:
        st.caption("PER/PBR列がないため散布図を表示できません。")

    # --- 本日の推奨銘柄 ---
    picks = load_daily_picks()
    if picks is not None and len(picks) > 0:
        st.markdown("---")
        st.markdown("### 本日の推奨銘柄（レンジ銘柄の直近タッチ）")
        st.caption(f"直近{DAILY_PICK_LOOKBACK}営業日以内にレンジ上限/下限付近にタッチした銘柄")

        low_picks = picks[picks["pick_type"].str.contains("下タッチ", na=False)]
        high_picks = picks[picks["pick_type"].str.contains("上タッチ", na=False)]

        col_low, col_high = st.columns(2)

        with col_low:
            st.markdown(f"#### 下タッチ - 買い候補 ({len(low_picks)}銘柄)")
            st.caption("レンジ下限付近まで下落 → 反発を狙える位置")
            if len(low_picks) > 0:
                pick_display = low_picks[["code", "name", "latest_close", "range_low", "range_high", "range_pct", "position_pct", "slope", "rsi", "rsi_signal"]].rename(columns={
                    "code": "コード", "name": "銘柄名", "latest_close": "直近終値",
                    "range_low": "下限", "range_high": "上限",
                    "range_pct": "レンジ幅%", "position_pct": "位置%", "slope": "傾き",
                    "rsi": "RSI", "rsi_signal": "RSIシグナル",
                })
                event_low = st.dataframe(
                    pick_display, use_container_width=True, hide_index=True,
                    on_select="rerun", selection_mode="single-row",
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
                pick_display = high_picks[["code", "name", "latest_close", "range_low", "range_high", "range_pct", "position_pct", "slope", "rsi", "rsi_signal"]].rename(columns={
                    "code": "コード", "name": "銘柄名", "latest_close": "直近終値",
                    "range_low": "下限", "range_high": "上限",
                    "range_pct": "レンジ幅%", "position_pct": "位置%", "slope": "傾き",
                    "rsi": "RSI", "rsi_signal": "RSIシグナル",
                })
                event_high = st.dataframe(
                    pick_display, use_container_width=True, hide_index=True,
                    on_select="rerun", selection_mode="single-row",
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
                lambda r: round(float(r["market_cap"]) / float(sector_mc_totals[r["sector_33"]]) * 100, 1)
                if pd.notna(r.get("market_cap")) and r.get("sector_33") in sector_mc_totals.index
                and sector_mc_totals[r["sector_33"]] > 0
                else None,
                axis=1,
            )

    display_cols = [
        "code", "name", "size_category", "sector_33", "wave_types",
        "range_pct", "touch_total", "slope", "atr", "breakout_days",
        "per", "pbr", "market_cap_oku", "sector_mc_pct",
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
    display_df = filtered[available_cols].rename(columns=display_names).reset_index(drop=True)

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
    wave_types_str = row["wave_types"] if row is not None and pd.notna(row["wave_types"]) else ""
    st.title(f"{code} {name}")

    # 波形タイプをタグ風に表示（results.csvの値 = デフォルト窓）
    type_tags = wave_types_str.replace("|", " | ") if wave_types_str else ""
    rec_label = " ★推奨" if row is not None and is_recommended(row) else ""
    sector_label = f" | {sector_33}" if sector_33 else ""
    st.markdown(f"**{size_cat}**{sector_label} | {type_tags}{rec_label}")

    # チャート設定（指標カードの前に配置 → 窓選択を先に取得）
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        chart_type = st.radio("チャートタイプ", ["ローソク足", "ライン"], horizontal=True)
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
        c2.metric("上タッチ", f"{int(row['touch_high'])}回" if pd.notna(row.get("touch_high")) else "-")
        c3.metric("下タッチ", f"{int(row['touch_low'])}回" if pd.notna(row.get("touch_low")) else "-")
        c4.metric("傾き", f"{row.get('slope', '-')}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("ATR", f"{row.get('atr', '-')}")
        c6.metric("レンジ外日数", f"{int(row['breakout_days'])}日" if pd.notna(row.get("breakout_days")) else "-")
        c7.metric("上限", f"¥{row['range_high']:,.0f}" if pd.notna(row.get("range_high")) else "-")
        c8.metric("下限", f"¥{row['range_low']:,.0f}" if pd.notna(row.get("range_low")) else "-")

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
            stock_mc = float(row["market_cap"]) if has_mc and pd.notna(row.get("market_cap")) else None
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
        fin_display = financials.rename(columns={
            "period": "期間",
            "revenue": "売上高（億円）",
            "op_margin": "営業利益率（%）",
            "is_forecast": "予測",
        })
        st.dataframe(fin_display, use_container_width=True, hide_index=True)
    else:
        st.caption("財務データを取得できませんでした。")

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
        ticker_options = results[results["ticker"] != ticker][["ticker", "code", "name", "sector_33"]].copy()
        ticker_options["label"] = ticker_options["code"] + " " + ticker_options["name"]

        # 同業種抽出ボタン
        comp_col1, comp_col2 = st.columns([1, 3])
        with comp_col1:
            if sector_33 and st.button(f"同業種を選択（{sector_33}）", key="btn_same_sector"):
                same_sector = ticker_options[ticker_options["sector_33"] == sector_33]["label"].tolist()
                st.session_state["comparison_stocks"] = same_sector[:5]
                st.rerun()

        selected_labels = st.multiselect(
            "比較銘柄を追加",
            options=ticker_options["label"].tolist(),
            max_selections=5,
            key="comparison_stocks",
        )
        if selected_labels:
            comparison_tickers = ticker_options[ticker_options["label"].isin(selected_labels)]["ticker"].tolist()

    # オーバーレイデータ構築
    overlays = []
    sector_index_df = None
    nikkei_df = None

    if show_sector and sector_33 and results is not None:
        sector_index_df = compute_sector_index(sector_33, results)
        if sector_index_df is not None:
            overlays.append({"name": f"33業種: {sector_33}", "data": sector_index_df, "color": "#FF5722"})

    if show_nikkei:
        nikkei_df = get_nikkei225()
        if nikkei_df is not None:
            overlays.append({"name": "日経225", "data": nikkei_df, "color": "#9C27B0"})

    # 決算発表予定日の取得
    earnings_df = load_earnings_dates()
    earnings_dates = get_earnings_dates_for_code(code, earnings_df) if earnings_df is not None else []

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
    if df is not None and (comparison_tickers or (overlays and comparison_tickers is not None)):
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
                comp_row = results[results["ticker"] == comp_ticker].iloc[0] if results is not None else None
                comp_label = f"{comp_row['code']} {comp_row['name']}" if comp_row is not None else comp_ticker
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
        col4.metric("取引コスト", f"{exec_cfg.get('transaction_cost_roundtrip', 0.003)*100:.1f}%")

        st.markdown("**パターン定義:**")
        patterns_cfg = strategy.get("patterns", {})
        for pat_name, pat_cfg in patterns_cfg.items():
            label = PATTERN_LABELS.get(pat_name, pat_name)
            notes = pat_cfg.get("notes", "")
            base_pts = strategy.get("scoring", {}).get("base_points", {}).get(pat_name, "?")
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
        stock_list_df = pd.read_csv(STOCK_LIST_CSV, encoding="utf-8-sig", dtype={"code": str})

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
            st.warning("株価キャッシュデータがありません。バッチ更新を実行してください。")
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
        count = ranking["matched_patterns"].apply(
            lambda x, _p=pat_key: _p in str(x)
        ).sum()
        pat_cols[i].metric(pat_label, f"{count}銘柄")

    # ランキングテーブル
    st.markdown(f"### 推奨ランキング ({len(filtered_ranking)}銘柄)")

    display_cols = [
        "code", "name", "size_category", "sector_33",
        "best_pattern", "best_score", "matched_patterns",
        "close", "rsi", "atr_pct", "volume_ratio",
        "vs_sma50", "vs_sma200", "turnover_rank_pct",
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
        st.error("バックテスト結果CSVが不足しています。先に batch/backtest.py を実行してください。")
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

    policies = [p for p in ["fixed_amount", "fixed_rate"] if p in set(summary.get("policy", []))]
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
    fig.add_trace(go.Bar(x=m["ym"], y=m["trades_closed"], name="trades_closed", opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=m["ym"], y=m["win_rate"] * 100.0, name="win_rate(%)", mode="lines+markers"), secondary_y=True)
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
    exit_cols = [c for c in ["rebalance_drop", "time_exit", "trailing_atr", "trend_exit"] if c in m.columns]
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
    fig3.add_trace(go.Scatter(x=m["ym"], y=m["month_return_pct"], name="month_return_pct", mode="lines+markers"))
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
        .agg(trade_count=("pnl", "size"), win_rate=("is_win", "mean"), avg_return_pct=("return_pct", "mean"))
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
    fig5.update_layout(height=max(380, 28 * len(ct.index)), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig5, use_container_width=True)

    # --- 負け要因（RSI） ---
    st.subheader("負け要因（RSI帯 × exit理由 など）")
    if enriched is None or enriched.empty:
        st.info("data/backtest_trades_enriched.csv が無いため、RSI要因の可視化はスキップします。")
    else:
        e = enriched[enriched["policy"] == policy].copy()
        for c in ["pnl", "return_pct", "rsi_signal"]:
            if c in e.columns:
                e[c] = pd.to_numeric(e[c], errors="coerce")
        loss = e[e.get("is_loss", False) == True].copy() if "is_loss" in e.columns else e[e["pnl"] < 0].copy()  # noqa: E712
        if loss.empty:
            st.write("負けトレードがありません。")
        else:
            if "rsi_bin" in loss.columns and "exit_reason" in loss.columns:
                ctl = pd.crosstab(loss["rsi_bin"].fillna("unknown"), loss["exit_reason"].fillna("unknown"))
                share = ctl.div(ctl.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)

                fig6 = go.Figure()
                for col in share.columns:
                    fig6.add_trace(go.Bar(x=share.index.astype(str), y=share[col], name=str(col)))
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

            if "rsi_signal" in loss.columns and "return_pct" in loss.columns and "exit_reason" in loss.columns:
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
        st.info("data/scenarios/summary_grid.csv が無いため、シナリオ比較はスキップします。")
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
    fig9.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    st.plotly_chart(fig9, use_container_width=True)

    top_n = st.slider("上位表示数", 5, 30, 12)
    metric = st.selectbox("ランキング指標", ["total_return_pct", "cagr", "max_drawdown", "trade_count"], index=0)
    ascending = metric in {"max_drawdown"}
    top = g.sort_values(metric, ascending=ascending).head(top_n)
    fig10 = px.bar(
        top,
        x="scenario",
        y=metric,
        title=f"上位{top_n}: {metric}",
    )
    fig10.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), xaxis_tickangle=45)
    st.plotly_chart(fig10, use_container_width=True)

    # equity curve compare
    st.subheader("エクイティカーブ比較（シナリオ別）")
    scenario_options = sorted(g["scenario"].dropna().unique().tolist())
    baseline = next((s for s in scenario_options if str(s).startswith("S0_baseline")), None)
    best = str(g.sort_values("cagr", ascending=False).iloc[0]["scenario"]) if len(g) else None
    default_sel = [x for x in [baseline, best] if x]
    selected = st.multiselect(
        "比較するシナリオ",
        options=scenario_options,
        default=default_sel,
    )
    include_nikkei = st.checkbox("日経225（Buy&Hold）も重ねる（元本1000万に補正）", value=True)
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
                    st.info("日経225データ（^N225）を取得できませんでした。ネットワークやキャッシュを確認してください。")
                else:
                    close_col = "Close" if "Close" in nikkei.columns else ("close" if "close" in nikkei.columns else None)
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
                            c0 = float(pd.to_numeric(nk[close_col].iloc[0], errors="coerce"))
                            if c0 and pd.notna(c0):
                                equity_nk = base * (pd.to_numeric(nk[close_col], errors="coerce") / c0)
                                add = pd.DataFrame({
                                    "date": nk.index,
                                    "equity": equity_nk,
                                    "scenario": "Nikkei225_BuyHold",
                                })
                                dfc = pd.concat([dfc, add], ignore_index=True)

            fig11 = px.line(dfc, x="date", y="equity", color="scenario", title="Equity curve")
            fig11.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig11, use_container_width=True)
        else:
            st.info("選択したシナリオのエクイティカーブCSVが見つかりませんでした。")


# ------------------------------------------------------------------
# 投資スタイル最適化タブ
# ------------------------------------------------------------------
import numpy as np  # noqa: E402  (already available via plotly deps)


_STYLE_OPTIMIZER_PATTERNS: dict[str, set[str]] = {
    "全(A+B+C+D)":   {"A_trend", "B_pullback", "C_breakout", "D_reversal"},
    "全(A+B+C+D+E)": {"A_trend", "B_pullback", "C_breakout", "D_reversal", "E_can_slim"},
    "A+B+C":         {"A_trend", "B_pullback", "C_breakout"},
    "A+B+C+E":       {"A_trend", "B_pullback", "C_breakout", "E_can_slim"},
    "A+B":           {"A_trend", "B_pullback"},
    "A+C":           {"A_trend", "C_breakout"},
    "B+C":           {"B_pullback", "C_breakout"},
    "Aのみ":         {"A_trend"},
    "Cのみ":         {"C_breakout"},
    "Eのみ":         {"E_can_slim"},
}

_STYLE_HOLDING_DAYS = {
    "短期（5〜20日）":  [5, 10, 15, 20],
    "中期（30〜60日）": [30, 60],
    "長期（120〜252日）": [120, 252],
}


@st.cache_resource(show_spinner="銘柄データを読み込み中（初回のみ時間がかかります）...")
def _load_optimizer_contexts():
    """コンテキストをキャッシュ（1アプリ起動につき1回だけ構築）"""
    base_dir = Path(__file__).resolve().parent
    strategy = load_strategy(base_dir / "config" / "strategy.yaml")
    stock_list_df = load_stock_list()
    bench = load_cached("^N225")
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
    for c in ["total_return_pct", "cagr", "max_drawdown", "sharpe", "win_rate",
              "trade_count", "beats_nikkei", "excess_return_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    nikkei_total = float(df["nikkei_total_return_pct"].dropna().iloc[0]) if "nikkei_total_return_pct" in df.columns else 0.0
    nikkei_cagr = float(df["nikkei_cagr"].dropna().iloc[0]) if "nikkei_cagr" in df.columns else 0.0

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
            st.metric("総リターン", f"{best.get('total_return_pct', 0):+.2f}%",
                      delta=f"vs 日経 {best.get('excess_return_pct', 0):+.2f}%")
            st.metric("CAGR", f"{best.get('cagr', 0):.4f}")
            st.metric("最大DD", f"{best.get('max_drawdown', 0):.4f}")
            st.metric("Sharpe", f"{best.get('sharpe', 0):.4f}")
            st.metric("勝率", f"{best.get('win_rate', 0):.1%}")

    # フィルタ付きランキング表
    st.subheader("📋 全シナリオ 成績一覧")
    filter_beats = st.checkbox("日経225を上回る設定のみ表示", value=True, key="opt_filter_beats")
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
        "style", "holding_days", "pattern_combo",
        "total_return_pct", "excess_return_pct",
        "cagr", "max_drawdown", "sharpe", "win_rate", "trade_count",
    ]
    display_cols = [c for c in display_cols if c in show.columns]
    rename_map = {
        "style": "スタイル", "holding_days": "保有日数", "pattern_combo": "パターン",
        "total_return_pct": "総リターン%", "excess_return_pct": "超過リターン%",
        "cagr": "CAGR", "max_drawdown": "最大DD", "sharpe": "Sharpe",
        "win_rate": "勝率", "trade_count": "取引数",
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

            scenario_options = [s for s in eq_df["scenario_id"].dropna().unique() if s != "Nikkei225_BuyHold"]
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
                    fig.add_trace(go.Scatter(
                        x=sub["date"],
                        y=sub["equity"],
                        name=scen,
                        line=dict(
                            width=3 if is_nikkei else 1.5,
                            dash="dash" if is_nikkei else "solid",
                            color="orange" if is_nikkei else None,
                        ),
                        mode="lines",
                    ))
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
            options=["短期（5〜20日）", "中期（30〜60日）", "長期（120〜252日）", "カスタム"],
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

        years = st.slider("バックテスト期間（年）", min_value=1, max_value=5, value=3, key="custom_years")

        run_btn = st.button("▶ シミュレーション実行", type="primary", key="custom_run_btn")

    with col_right:
        if not run_btn:
            st.info("左のパラメータを設定して「シミュレーション実行」ボタンを押してください。")
            return

        with st.spinner("シミュレーション実行中...（初回はコンテキスト構築に時間がかかります）"):
            contexts, strategy, trading_days, end_date, bench = _load_optimizer_contexts()

        if contexts is None:
            st.error("^N225 キャッシュが見つかりません。先に batch/update.py を実行してください。")
            return

        allowed_patterns = _STYLE_OPTIMIZER_PATTERNS[combo_choice]
        start_date = pd.Timestamp(end_date - pd.DateOffset(years=years))
        td_period = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]

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
            st.warning("有効な結果がありませんでした。パラメータや期間を見直してください。")
            return

        srow = summary_df.iloc[0]
        total_ret = float(srow.get("total_return_pct", 0))
        beats = total_ret > nikkei_total

        # 結果メトリクス
        badge = "✅ 日経225を上回りました！" if beats else "❌ 日経225に届きませんでした"
        st.markdown(f"### {badge}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("総リターン", f"{total_ret:+.2f}%", delta=f"vs 日経 {total_ret - nikkei_total:+.2f}%")
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
        fig.add_trace(go.Scatter(
            x=equity_df["date"],
            y=equity_df["equity"],
            name=f"ABCD戦略（{holding_days}日/{combo_choice}）",
            line=dict(color="royalblue", width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=nk.index,
            y=nk_equity,
            name="日経225 Buy&Hold",
            line=dict(color="orange", width=2, dash="dash"),
            mode="lines",
        ))
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

    bench_data = load_cached("^N225")
    if bench_data is None:
        st.error("^N225 キャッシュが見つかりません。先に batch/update.py を実行してください。")
        return

    _show_custom_simulation(base_dir, bench_data)


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

    tab1, tab2, tab3, tab4 = st.tabs(["波形分類", "ABCD戦略", "バックテスト", "投資スタイル最適化"])

    with tab1:
        show_list_view()

    with tab2:
        show_strategy_view()

    with tab3:
        show_backtest_view()

    with tab4:
        show_style_optimizer_view()


if __name__ == "__main__":
    main()
