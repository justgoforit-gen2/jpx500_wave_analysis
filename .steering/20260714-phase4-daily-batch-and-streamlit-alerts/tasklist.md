# タスクリスト — Phase 4: daily-batch-and-streamlit-alerts

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

---

## フェーズ1: バッチ統合

- [x] `batch/update.py` に Step 9 として MoatScore 計算ステップ追加
  - [x] `MoatScoreEngine.compute_bulk()` 直接呼び出し
  - [x] `save_moat_scores()` で parquet 保存
  - [x] perf log: `_log_perf()` 関数追加 + Step 9 の所要時間記録

## フェーズ2: Streamlit「今朝の Top10」ハイライト枠

- [x] `_tab_moat_ranking()` 先頭に「今朝の Top10」セクション追加
  - [x] parquet の最新 date の Top10 表示 (st.dataframe 高さ制限付き)
  - [x] 各銘柄コード横に「コードをコピー→Moat Scoreタブへ」説明

## フェーズ3: Streamlit「決算予定 × Moat Top50」セクション

- [x] `_tab_moat_ranking()` 末尾に「決算予定 × Moat Top50」セクション追加
  - [x] earnings CSV × moat_scores.parquet join
  - [x] 直近 5 営業日の決算予定銘柄でフィルタ (pd.offsets.BDay(5))
  - [x] MoatScore 降順ソート、Top10 は背景色強調 (fffacd = 淡黄)

## フェーズ4: Streamlit ヘッダ鮮度インジケータ

- [x] `main()` の先頭に `_show_policy_freshness_indicator()` 追加
  - [x] `updated_at` から経過日数計算
  - [x] 7日以内: 緑 st.success
  - [x] 7日超過: 赤 st.error + /policy-update ガイダンス

## フェーズ5: daily_update.bat perf log

- [x] `daily_update.bat` に perf log フック追加
  - [x] `logs/` ディレクトリ作成
  - [x] バッチ開始/終了時に `logs/batch-perf.csv` に追記 (Python インライン)

## フェーズ6: 動作確認

- [x] `python -c "import ast; ast.parse(open('app.py').read()); print('OK')"` → OK
- [x] `python -c "import ast; ast.parse(open('batch/update.py').read()); print('OK')"` → OK
- [x] `pytest tests/test_moat_score.py -q` → 6 passed

---

## 実装後の振り返り

### 実装完了日
2026-06-10

### 計画と実績の差分

- `batch/update.py` の Step 9 は `X-Recompute-Token` 付き HTTPX 呼び出しではなく、MoatScoreEngine.compute_bulk() を直接呼ぶ方式を採用した (API が起動していない状態でもバッチが動くため)
- PLAN.md では「POST /api/moat-score/recompute を叩く」とあったが、バッチ独立性を優先
- daily_update.bat の perf log は bat のネイティブな時間測定 (`%time%`) ではなく Python インラインで実装 (Windows の time フォーマット解析が複雑なため)

### 学んだこと

- バッチと API の依存関係: バッチが API を叩く設計は API の先起動が必要になる。今回のように独立実行可能な形にする方が運用上安全
- `pd.offsets.BDay(5)` で直近5営業日の決算予定を簡単にフィルタできる
- Streamlit の `st.dataframe` は `pandas Styler` を受け入れる (apply で背景色強調可能)

### 次回への改善提案

- Step 9 が全銘柄 (~500社) を計算するため実行時間が長い。naibu API が落ちている場合の計算コスト削減を検討 (total_score=None の銘柄は parquet に含めないため計算しても無駄)
- moat_scores.parquet に前日分を残す「差分更新」ロジックを検討 (現在は毎回上書き)
- daily_update.bat の perf log は各ステップ単位での所要時間記録が理想 (現在はバッチ全体の開始/終了のみ)
