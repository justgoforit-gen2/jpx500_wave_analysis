---
phase: 4
title: 日次バッチ統合 + Streamlit アラート
status: in_progress
---

# Phase 4 要件定義

## 背景

Phase 3 で MCP 6本が登録完了。
Phase 4 では MoatScore の日次自動更新と Streamlit 内アラート機能を実装する。

## 必須Deliverables

### D1: moat_scores バッチ計算 (`batch/update.py` への統合)
- `batch/update.py` の Step 9 として `compute_bulk_moat_scores()` を追加
- MoatScoreEngine.compute_bulk() を直接呼び (API 不要、独立実行可能)
- 結果を `data/moat_scores.parquet` に保存

### D2: Streamlit ランキングタブ「今朝の Top10」ハイライト枠
- `_tab_moat_ranking()` の先頭に追加
- moat_scores.parquet の最新 date の Top10 を表示
- 各銘柄行に「Moat Score タブで詳細を見る → コードをコピー」ガイド

### D3: Streamlit 「決算予定 × Moat Top50」セクション
- ランキングタブ内または新セクションとして追加
- jpx500 earnings CSV × moat_scores.parquet を join
- 直近 5 営業日の決算予定 × MoatScore 降順、Top10 は背景強調

### D4: Streamlit ヘッダ policy_signals.json 鮮度インジケータ
- 全タブ共通ヘッダ領域 (main() 内) に追加
- policy_signals.json の `updated_at` から経過日数を計算
- 7日超過: 赤字 + 「/policy-update を VSCode で実行してください」ガイダンス

### D5: daily_update.bat perf log フック
- 各ステップの所要時間を `logs/batch-perf.csv` に追記
- 日時 / ステップ名 / 所要秒 の CSV フォーマット

## スコープ外

- バッチ 18:00 起動スケジューラ設定 (ユーザーが Windows タスクスケジューラで設定)
- 1週間連続運用確認 (ユーザーが確認)
