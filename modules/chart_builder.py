"""Plotlyで3段または4段チャートを生成する（メイン+売買代金+RSI[+信用倍率]）"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import (
    BB_PERIOD,
    BB_STD,
    MA_PERIODS,
    MARGIN_RATIO_HIGH,
    MARGIN_RATIO_LOW,
    RSI_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    DEFAULT_WINDOW,
    TOUCH_THRESHOLD_PCT,
)


def _calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSIを計算する"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


INDEX_COLORS = ["#1976d2", "#E91E63", "#00BCD4", "#FF9800", "#8BC34A", "#9C27B0"]
COMPARISON_COLORS = ["#E91E63", "#00BCD4", "#FF9800", "#8BC34A", "#9C27B0", "#795548"]


def build_chart(
    df: pd.DataFrame,
    ticker: str,
    name: str,
    range_high: float | None = None,
    range_low: float | None = None,
    window: int = DEFAULT_WINDOW,
    chart_type: str = "candlestick",
    show_touch_points: bool = True,
    show_bb: bool = False,
    earnings_dates: list | None = None,
    overlays: list[dict] | None = None,
    margin_history: pd.DataFrame | None = None,
) -> go.Figure:
    """3段または4段チャート（メイン+売買代金+RSI[+信用倍率]）を生成

    Args:
        margin_history: observation_date, margin_ratio (+ source) を持つDataFrame。
            指定時は4段目に信用倍率の時系列を追加する。空/None時は3段で描画。
    """

    has_margin = (
        margin_history is not None
        and len(margin_history) >= 1
        and "margin_ratio" in margin_history.columns
    )

    if has_margin:
        rows = 4
        row_heights = [0.50, 0.16, 0.17, 0.17]
        chart_height = 880
    else:
        rows = 3
        row_heights = [0.6, 0.2, 0.2]
        chart_height = 750

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=None,
    )

    dates = df.index

    # === 上段: メインチャート ===
    if chart_type == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["Close"],
                mode="lines",
                name="終値",
                line=dict(color="#1976d2", width=1.5),
            ),
            row=1,
            col=1,
        )

    # 移動平均線
    ma_colors = {"MA13W": "#2196F3", "MA26W": "#F44336", "MA52W": "#4CAF50"}
    ma_names = {"MA13W": "MA13週(65)", "MA26W": "MA26週(130)", "MA52W": "MA52週(260)"}
    for ma_key, period in MA_PERIODS.items():
        if len(df) >= period:
            ma = df["Close"].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=ma,
                    mode="lines",
                    name=ma_names[ma_key],
                    line=dict(color=ma_colors[ma_key], width=1.2, dash="solid"),
                ),
                row=1,
                col=1,
            )

    # ボリンジャーバンド
    if show_bb and len(df) >= BB_PERIOD:
        bb_sma = df["Close"].rolling(BB_PERIOD).mean()
        bb_std = df["Close"].rolling(BB_PERIOD).std()
        bb_upper = bb_sma + BB_STD * bb_std
        bb_lower = bb_sma - BB_STD * bb_std

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bb_upper,
                mode="lines",
                name=f"BB上限({BB_PERIOD},{BB_STD}σ)",
                line=dict(color="rgba(156,39,176,0.4)", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bb_lower,
                mode="lines",
                name=f"BB下限({BB_PERIOD},{BB_STD}σ)",
                line=dict(color="rgba(156,39,176,0.4)", width=1),
                fill="tonexty",
                fillcolor="rgba(156,39,176,0.07)",
            ),
            row=1,
            col=1,
        )

    # 評価窓ハイライト（背景色）
    if len(df) >= window:
        window_start = dates[-window]
        fig.add_vrect(
            x0=window_start,
            x1=dates[-1],
            fillcolor="rgba(33, 150, 243, 0.08)",
            line_width=0,
            row=1,
            col=1,
        )

    # range_high / range_low 水平線（評価窓内のみ）
    if range_high is not None and range_low is not None and len(df) >= window:
        window_start = dates[-window]
        fig.add_trace(
            go.Scatter(
                x=[window_start, dates[-1]],
                y=[range_high, range_high],
                mode="lines",
                name=f"上限 ¥{range_high:,.0f}",
                line=dict(color="#FF9800", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[window_start, dates[-1]],
                y=[range_low, range_low],
                mode="lines",
                name=f"下限 ¥{range_low:,.0f}",
                line=dict(color="#FF9800", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # === タッチポイント表示 ===
    if (
        show_touch_points
        and range_high is not None
        and range_low is not None
        and len(df) >= window
    ):
        w = df.tail(window)
        close_w = w["Close"]
        midpoint = (range_high + range_low) / 2
        touch_band = midpoint * TOUCH_THRESHOLD_PCT / 100

        # 上限タッチ
        touch_high_mask = close_w >= (range_high - touch_band)
        if touch_high_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=w.index[touch_high_mask],
                    y=close_w[touch_high_mask],
                    mode="markers",
                    name="上限タッチ",
                    marker=dict(color="#F44336", size=8, symbol="triangle-down"),
                ),
                row=1,
                col=1,
            )

        # 下限タッチ
        touch_low_mask = close_w <= (range_low + touch_band)
        if touch_low_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=w.index[touch_low_mask],
                    y=close_w[touch_low_mask],
                    mode="markers",
                    name="下限タッチ",
                    marker=dict(color="#2196F3", size=8, symbol="triangle-up"),
                ),
                row=1,
                col=1,
            )

    # === 決算発表予定日の縦線 ===
    if earnings_dates:
        chart_start = dates[0]
        chart_end = dates[-1]
        for ed_str in earnings_dates:
            ed = pd.Timestamp(ed_str)
            if chart_start <= ed <= chart_end:
                fig.add_vline(
                    x=ed,
                    line_dash="dash",
                    line_color="#FF6F00",
                    line_width=1.5,
                    opacity=0.7,
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=ed,
                    y=1.02,
                    yref="y domain",
                    text="決算",
                    showarrow=False,
                    font=dict(size=9, color="#FF6F00"),
                    row=1,
                    col=1,
                )

    # === オーバーレイ ===
    if overlays and len(df) >= window:
        window_start = dates[-window]
        # 評価窓開始日の個別銘柄終値
        stock_at_start = float(df.loc[df.index >= window_start, "Close"].iloc[0])
        for ov in overlays:
            ov_data = ov["data"]
            ov_in_range = ov_data[ov_data.index >= window_start]
            if len(ov_in_range) == 0:
                continue
            ov_at_start = float(ov_in_range["Close"].iloc[0])
            if ov_at_start == 0:
                continue
            scale = stock_at_start / ov_at_start
            ov_scaled = ov_in_range["Close"] * scale
            fig.add_trace(
                go.Scatter(
                    x=ov_in_range.index,
                    y=ov_scaled,
                    mode="lines",
                    name=ov["name"],
                    line=dict(color=ov["color"], width=1.5, dash="dot"),
                ),
                row=1,
                col=1,
            )

    # === 中段: 売買代金 ===
    turnover = df["Close"] * df["Volume"]
    colors = [
        "#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=turnover,
            name="売買代金",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # === 下段: RSI ===
    rsi = _calc_rsi(df["Close"])
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rsi,
            mode="lines",
            name=f"RSI({RSI_PERIOD})",
            line=dict(color="#7B1FA2", width=1.5),
        ),
        row=3,
        col=1,
    )
    # 買われすぎ / 売られすぎライン
    fig.add_hline(
        y=RSI_OVERBOUGHT, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1
    )
    fig.add_hline(
        y=RSI_OVERSOLD, line_dash="dot", line_color="blue", opacity=0.5, row=3, col=1
    )
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

    # === 4段目: 信用倍率 (margin_history が渡された場合のみ) ===
    if has_margin:
        m = margin_history.copy()
        m["observation_date"] = pd.to_datetime(m["observation_date"])
        m = m.sort_values("observation_date")

        # 9999 (売残=0の上限値) は表示用に NaN にして線を切る
        m["margin_ratio_display"] = m["margin_ratio"].where(
            m["margin_ratio"] < 9999, other=None
        )

        # データソース別に色分け表示 (日次=青, 週次=橙, 株探長期=灰)
        sources = m["source"].unique() if "source" in m.columns else ["unknown"]
        source_styles = {
            "daily": {"color": "#1976D2", "name": "信用倍率(日次)"},
            "weekly": {"color": "#FF6F00", "name": "信用倍率(週次)"},
            "kabutan": {"color": "#8E8E8E", "name": "信用倍率(長期)"},
            "unknown": {"color": "#7B1FA2", "name": "信用倍率"},
        }
        for src in sources:
            sub = m if "source" not in m.columns else m[m["source"] == src]
            style = source_styles.get(src, source_styles["unknown"])
            fig.add_trace(
                go.Scatter(
                    x=sub["observation_date"],
                    y=sub["margin_ratio_display"],
                    mode="lines+markers",
                    name=style["name"],
                    line=dict(color=style["color"], width=1.8),
                    marker=dict(size=5),
                ),
                row=4,
                col=1,
            )

        # 閾値ライン
        fig.add_hline(
            y=MARGIN_RATIO_HIGH,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=4,
            col=1,
            annotation_text=f"買偏重({MARGIN_RATIO_HIGH})",
            annotation_position="right",
            annotation_font_size=10,
        )
        fig.add_hline(
            y=MARGIN_RATIO_LOW,
            line_dash="dash",
            line_color="blue",
            opacity=0.5,
            row=4,
            col=1,
            annotation_text=f"売偏重({MARGIN_RATIO_LOW})",
            annotation_position="right",
            annotation_font_size=10,
        )

    # === レイアウト ===
    fig.update_layout(
        title=f"{ticker} {name}",
        height=chart_height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        template="plotly_white",
    )

    fig.update_yaxes(title_text="価格", row=1, col=1)
    fig.update_yaxes(title_text="売買代金", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    if has_margin:
        fig.update_yaxes(title_text="信用倍率", row=4, col=1)
        fig.update_xaxes(title_text="日付", row=4, col=1)
    else:
        fig.update_xaxes(title_text="日付", row=3, col=1)

    # レスポンシブ
    fig.update_layout(autosize=True)

    return fig


def build_comparison_chart(
    main_name: str,
    main_df: pd.DataFrame,
    comparisons: list[dict],
    window: int = DEFAULT_WINDOW,
) -> go.Figure:
    """複数銘柄を評価窓の開始日でbase=100に正規化して比較するラインチャート

    Args:
        main_name: メイン銘柄の表示名
        main_df: メイン銘柄のOHLCVデータ
        comparisons: [{"name": str, "data": DataFrame}, ...]
        window: 評価窓の日数
    """
    fig = go.Figure()

    # 評価窓内のデータ
    main_w = main_df.tail(window).copy()
    if len(main_w) == 0:
        return fig

    window_start = main_w.index[0]
    main_start_price = float(main_w["Close"].iloc[0])

    if main_start_price == 0:
        return fig

    # メイン銘柄（base=100）
    main_normalized = main_w["Close"] / main_start_price * 100
    fig.add_trace(
        go.Scatter(
            x=main_w.index,
            y=main_normalized,
            mode="lines",
            name=main_name,
            line=dict(color="#1976d2", width=2),
        )
    )

    # 比較銘柄
    for i, comp in enumerate(comparisons):
        comp_data = comp["data"]
        comp_in_range = comp_data[comp_data.index >= window_start]
        if len(comp_in_range) == 0:
            continue
        comp_start_price = float(comp_in_range["Close"].iloc[0])
        if comp_start_price == 0:
            continue
        comp_normalized = comp_in_range["Close"] / comp_start_price * 100
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=comp_in_range.index,
                y=comp_normalized,
                mode="lines",
                name=comp["name"],
                line=dict(color=color, width=1.5),
            )
        )

    fig.update_layout(
        title="比較チャート（評価窓開始日=100に正規化）",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        template="plotly_white",
        yaxis_title="正規化価格（base=100）",
        xaxis_title="日付",
    )

    return fig


def build_financials_chart(fin_df: pd.DataFrame) -> go.Figure:
    """売上高(棒) + 営業利益率(線) + EPS(線)の三軸チャートを生成

    Args:
        fin_df: columns = [period, revenue, op_margin, eps, is_forecast]
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 売上高の棒グラフ（予測は色を変える）
    colors = [
        "rgba(100, 181, 246, 0.7)" if row["is_forecast"] else "rgba(30, 136, 229, 0.85)"
        for _, row in fin_df.iterrows()
    ]
    border_colors = [
        "rgba(100, 181, 246, 1)" if row["is_forecast"] else "rgba(30, 136, 229, 1)"
        for _, row in fin_df.iterrows()
    ]

    fig.add_trace(
        go.Bar(
            x=fin_df["period"],
            y=fin_df["revenue"],
            name="売上高（億円）",
            marker_color=colors,
            marker_line_color=border_colors,
            marker_line_width=1,
            text=[f"{v:,.0f}" for v in fin_df["revenue"]],
            textposition="outside",
        ),
        secondary_y=False,
    )

    # 営業利益率の折れ線（Noneを除外）
    margin_mask = fin_df["op_margin"].notna()
    if margin_mask.any():
        fig.add_trace(
            go.Scatter(
                x=fin_df.loc[margin_mask, "period"],
                y=fin_df.loc[margin_mask, "op_margin"],
                mode="lines+markers+text",
                name="営業利益率（%）",
                line=dict(color="#FF6F00", width=2.5),
                marker=dict(size=8, color="#FF6F00"),
                text=[f"{v:.1f}%" for v in fin_df.loc[margin_mask, "op_margin"]],
                textposition="top center",
                textfont=dict(size=11, color="#FF6F00"),
            ),
            secondary_y=True,
        )

    # EPS の折れ線（第3軸 yaxis3、実績は実線・予測は破線で繋ぐ）
    eps_mask = fin_df["eps"].notna() if "eps" in fin_df.columns else None
    if eps_mask is not None and eps_mask.any():
        eps_df = fin_df[eps_mask].reset_index(drop=True)
        # 実績と予測を分離して別trace（破線で予測を表現）
        actual = eps_df[~eps_df["is_forecast"]]
        forecast = eps_df[eps_df["is_forecast"]]

        fig.add_trace(
            go.Scatter(
                x=actual["period"],
                y=actual["eps"],
                mode="lines+markers+text",
                name="EPS（円/株）",
                line=dict(color="#2E7D32", width=2.5),
                marker=dict(size=8, color="#2E7D32", symbol="diamond"),
                text=[f"{v:.1f}" for v in actual["eps"]],
                textposition="bottom center",
                textfont=dict(size=11, color="#2E7D32"),
                yaxis="y3",
            )
        )
        if len(forecast) > 0:
            # 実績の最終点と予測を繋ぐ
            if len(actual) > 0:
                bridge_x = [actual["period"].iloc[-1]] + forecast["period"].tolist()
                bridge_y = [actual["eps"].iloc[-1]] + forecast["eps"].tolist()
            else:
                bridge_x = forecast["period"].tolist()
                bridge_y = forecast["eps"].tolist()
            fig.add_trace(
                go.Scatter(
                    x=bridge_x,
                    y=bridge_y,
                    mode="lines+markers+text",
                    name="EPS（予想）",
                    line=dict(color="#66BB6A", width=2, dash="dash"),
                    marker=dict(size=8, color="#66BB6A", symbol="diamond-open"),
                    text=[""] + [f"{v:.1f}" for v in forecast["eps"]],
                    textposition="bottom center",
                    textfont=dict(size=11, color="#66BB6A"),
                    yaxis="y3",
                )
            )

    # 予測部分に背景色
    forecast_periods = fin_df[fin_df["is_forecast"]]["period"].tolist()
    for fp in forecast_periods:
        fig.add_vrect(
            x0=fp,
            x1=fp,
            fillcolor="rgba(255, 235, 59, 0.15)",
            line_width=0,
        )

    # 第3軸 (EPS) のレイアウト: 右側さらに外、プロット領域を縮めて配置
    has_eps = eps_mask is not None and eps_mask.any()
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=70 if has_eps else 10, t=40, b=10),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        bargap=0.3,
        xaxis=dict(domain=[0, 0.93] if has_eps else [0, 1]),
        yaxis3=dict(
            title=dict(text="EPS（円/株）", font=dict(color="#2E7D32")),
            tickfont=dict(color="#2E7D32"),
            overlaying="y",
            side="right",
            position=1.0,
            anchor="free",
            showgrid=False,
        )
        if has_eps
        else None,
    )

    fig.update_yaxes(title_text="売上高（億円）", secondary_y=False)
    fig.update_yaxes(title_text="営業利益率（%）", secondary_y=True)

    # 予測がある場合、凡例注記
    if fin_df["is_forecast"].any():
        fig.add_annotation(
            text="※ 薄色/破線 = アナリスト予測（参考値）",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.12,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

    return fig


def build_index_chart(
    indices: list[dict],
    window: int = DEFAULT_WINDOW,
) -> go.Figure:
    """指数の重ね合わせラインチャート（評価窓開始日=100に正規化）

    Args:
        indices: [{"name": str, "data": DataFrame, "color": str}, ...]
            各DataFrameは "Close" 列を持つ
        window: 評価窓の日数（表示期間の計算に使用）
    """
    fig = go.Figure()

    for i, idx_info in enumerate(indices):
        data = idx_info["data"]
        name = idx_info["name"]
        color = idx_info.get("color", INDEX_COLORS[i % len(INDEX_COLORS)])

        if data is None or data.empty:
            continue

        # 表示期間: window=0なら全期間、それ以外は直近window日分
        display_data = data.copy() if window == 0 else data.tail(window).copy()
        if len(display_data) < 2:
            continue

        # 開始日=100に正規化
        start_price = float(display_data["Close"].iloc[0])
        if start_price == 0:
            continue
        normalized = display_data["Close"] / start_price * 100

        fig.add_trace(
            go.Scatter(
                x=display_data.index,
                y=normalized,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
            )
        )

    # base=100 の水平線
    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="gray",
        opacity=0.4,
    )

    fig.update_layout(
        title="マーケット指数（期間開始日=100に正規化）",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        yaxis_title="正規化価格（base=100）",
        xaxis_title="日付",
    )

    return fig


def build_flow_index_dual_chart(
    flow_cumulative: pd.Series,
    indices: list[dict],
    window: int = 0,
    flow_label: str = "海外投資家累積買い越し（億円）",
) -> go.Figure:
    """海外投資家の累積フローと株価指数を2軸で重ね表示する。

    Args:
        flow_cumulative: 累積買い越し額の週次Series (index=date, value=億円)
        indices: [{"name": str, "data": DataFrame(index=Date, "Close"), "color": str}]
        window: 表示窓（営業日換算）。0なら全期間
        flow_label: フロー軸ラベル
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # window は「営業日数」。weekly フローと daily 指数の単位差を解消するため、
    # window から start_date を求めて 両系列を同じ日付範囲で揃える。
    flow_display = flow_cumulative.copy()
    flow_display.index = pd.to_datetime(flow_display.index)
    start_date: pd.Timestamp | None = None
    if window and window > 0:
        end_ref = (
            flow_display.index.max() if not flow_display.empty else pd.Timestamp.today()
        )
        # 営業日換算: window 営業日 ≈ window * 7/5 暦日
        start_date = end_ref - pd.Timedelta(days=int(window * 7 / 5))
        if not flow_display.empty:
            flow_display = flow_display[flow_display.index >= start_date]

    if not flow_display.empty:
        fig.add_trace(
            go.Scatter(
                x=flow_display.index,
                y=flow_display.values,
                mode="lines",
                name=flow_label,
                line=dict(color="#1976d2", width=2),
                fill="tozeroy",
                fillcolor="rgba(25, 118, 210, 0.15)",
            ),
            secondary_y=False,
        )

    for i, idx_info in enumerate(indices):
        name = idx_info.get("name", f"index_{i}")
        data = idx_info.get("data")
        color = idx_info.get("color", INDEX_COLORS[i % len(INDEX_COLORS)])
        if data is None or data.empty or "Close" not in data.columns:
            continue
        d = data.copy()
        d.index = pd.to_datetime(d.index)
        if start_date is not None:
            d = d[d.index >= start_date]
        close = pd.to_numeric(d["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        normalized = close / float(close.iloc[0]) * 100.0
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=name,
                line=dict(color=color, width=2, dash="solid"),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        height=520,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="海外投資家フロー × 株価指数（左軸: 累積フロー / 右軸: 指数 base=100）",
    )
    fig.update_yaxes(title_text=flow_label, secondary_y=False)
    fig.update_yaxes(title_text="指数 (base=100)", secondary_y=True)
    fig.update_xaxes(title_text="日付")
    fig.add_hline(
        y=0, line_dash="dot", line_color="gray", opacity=0.3, secondary_y=False
    )
    fig.add_hline(
        y=100, line_dash="dot", line_color="gray", opacity=0.3, secondary_y=True
    )
    return fig
