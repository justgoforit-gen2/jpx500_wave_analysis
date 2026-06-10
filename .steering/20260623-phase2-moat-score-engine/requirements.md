---
phase: 2
title: MoatScoreEngine + Streamlit Moat/Ranking タブ
status: in_progress
---

# Phase 2 要件定義

## 背景

Phase 1 でベース整備（mindmap port 移行、policy_signals.json 初版）が完了。
Phase 2 はバフェット流 7軸スコアリングエンジンの実装とStreamlit への組み込みが目的。

## 必須Deliverables

### D1: range_breakout_detector.py 公開ラッパ
- `_evaluate(df, asof_idx)` の薄いラッパ `evaluate(code, ohlcv)` を追加
- 呼び出し元 (MoatScoreEngine) が `_` プレフィックスに依存しないようにする

### D2: MoatScoreEngine (`modules/moat_score.py`)
7軸スコア (各 0-10) と重み合計 100% の実装

| # | 軸 | 重み | 入力ソース |
|---|---|---|---|
| 1 | technical | 10% | `results.csv` の wave type → スコア変換 |
| 2 | fundamental | 25% | `capital_efficiency_screener.load_screening_result()` の CES スコア |
| 3 | foreign_flow | 10% | `foreign_flow_analyzer.compute_cumulative_flow()` 直近4週 net |
| 4 | growth | 15% | `naibu_client.fetch_jpx500_naibu_data()` の net_income CAGR 近似 |
| 5 | growth_sector | 10% | `policy_signals.json` sector_tags × naibu industry_name マッチング |
| 6 | moat_pp | 20% | naibu HTTPX `GET /api/pricing-power/companies/{edinet_code}` |
| 7 | policy | 10% | `policy_signals.json` sector_tags × naibu industry_name マッチング (strength ベース) |

**重要制約**:
- naibu API が落ちている場合: 軸 4 / 6 を `None`、`total_score = None`、`explanation.errors = ["naibu API unreachable"]`
- moat_scores.parquet への書き込みはバッチ専用、MoatScoreEngine は読み書きしない

### D3: api_server.py に 4 エンドポイント追加
- `GET /api/moat-score/{code}` — 単銘柄
- `GET /api/moat-score/ranking?top=20&sector=...` — ランキング
- `POST /api/moat-score/recompute` — バッチ用、X-Recompute-Token 必須
- `GET /api/foreign-flow/{code}` — Phase 3 jpx500-mcp 用

### D4: Streamlit 新タブ 2 本 (`app.py`)
- 「Moat Score」タブ: plotly レーダー(7軸) + 総合点ゲージ + 軸別説明 expander
- 「ランキング」タブ: st.dataframe + sector フィルタ + CSV出力 + 銘柄クリックで Moat タブへ
- recompute ボタンは Streamlit に**置かない**

### D5: テスト
- `tests/test_moat_score.py`: 7203 / 7974 / 6861 でスコア算出 (naibu 落ち fallback 含む)

## 非機能要件

- `/api/moat-score/ranking?top=10` レスポンスが **3.0 秒以内** (parquet 読み込みに依存)
- `recompute` エンドポイント: X-Recompute-Token ヘッダなし → 403

## スコープ外

- moat_scores.parquet の日次更新 (Phase 4)
- Streamlit「決算予定」「政府政策」「結論マインドマップ」タブ (Phase 4)
