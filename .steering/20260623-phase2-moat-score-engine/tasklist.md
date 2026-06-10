# タスクリスト — Phase 2: moat-score-engine

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

### 必須ルール
- **全てのタスクを`[x]`にすること**
- 「時間の都合により別タスクとして実施予定」は禁止
- 「実装が複雑すぎるため後回し」は禁止
- 未完了タスク（`[ ]`）を残したまま作業を終了しない

---

## 事前修正

- [x] starlette バージョン修正 (`starlette>=0.40,<0.47` → 0.46.2 インストール済確認)
  - `python -c "import api_server; print('OK')"` でエラーなし確認
  - 原因: starlette 1.2.1 (不整合バージョン) が入っており FastAPI 0.115.14 に非互換
  - config/settings.py の `NAIBU_DB_PATH` も Windows ハードコードパス → devcontainer-aware に修正 (os.getenv + __file__ ベース)

## フェーズ1: range_breakout_detector 公開ラッパ

- [x] `modules/range_breakout_detector.py` に `evaluate(code, ohlcv)` の公開 wrapper 追加
  - `_evaluate(ohlcv)` を呼ぶだけの薄いラッパ

## フェーズ2: MoatScoreEngine 実装

- [x] `modules/moat_score.py` を新規作成
  - [x] `WEIGHTS` dict 定義 (7軸、合計 1.0)
  - [x] `assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9` を module レベルで配置
  - [x] `MoatScoreEngine` クラス実装
    - [x] `compute(code: str) -> dict` メソッド
    - [x] 軸1 technical: results.csv wave_types → score
    - [x] 軸2 fundamental: load_screening_result() CES score
    - [x] 軸3 foreign_flow: compute_cumulative_flow() 直近4週
    - [x] 軸4 growth: naibu SQLite 直接 → net_income multi-year CAGR proxy
    - [x] 軸5 growth_sector: policy_signals.json × sector_33
    - [x] 軸6 moat_pp: naibu HTTPX /api/pricing-power/companies/{edinet_code}
    - [x] 軸7 policy: policy_signals.json strength × sector_33 match
    - [x] naibu fallback: moat_pp が None → total_score=None + errors

## フェーズ3: API エンドポイント追加

- [x] `api_server.py` に 4 エンドポイント追加
  - [x] `GET /api/moat-score/{code}` (オンデマンド計算)
  - [x] `GET /api/moat-score/ranking?top=N&sector=S` (parquet 読み込み)
  - [x] `POST /api/moat-score/recompute` (X-Recompute-Token 認証)
  - [x] `GET /api/foreign-flow/{code}` (Phase 3 jpx500-mcp 用)

## フェーズ4: Streamlit タブ追加

- [x] `app.py` に「Moat Score」タブ追加
  - [x] 銘柄コード入力 → MoatScoreEngine 直接呼び出し
  - [x] plotly Scatterpolar レーダーチャート (7軸)
  - [x] 総合スコア表示 (st.metric)
  - [x] 軸別説明 expander
- [x] `app.py` に「ランキング」タブ追加
  - [x] parquet から st.dataframe で表示
  - [x] sector フィルタ (selectbox)
  - [x] CSV ダウンロードボタン
- [x] Streamlit のどのタブにも recompute ボタンが**存在しない**確認 (grep 済)

## フェーズ5: テスト

- [x] `tests/test_moat_score.py` 作成
  - [x] `test_weights_sum`: WEIGHTS 合計 1.0 確認
  - [x] `test_moat_score_basic_7203`: 7203 でスコア算出、各軸 0-10 範囲確認
  - [x] `test_moat_score_naibu_fallback`: naibu 落ち時 total_score=None 確認
  - [x] `test_moat_score_technical_range`: wave_type マッピング範囲確認
  - [x] `test_moat_score_multi_codes`: compute_bulk 複数銘柄
  - [x] `test_moat_score_api_health`: api_server import OK (starlette 回帰検出)

## フェーズ6: 動作確認

- [x] `python -c "import api_server; print('OK')"` エラーなし
- [x] `pytest tests/test_moat_score.py -v` 全6パス
- [x] `ruff check modules/moat_score.py api_server.py modules/range_breakout_detector.py` エラーなし

---

## 実装後の振り返り

### 実装完了日
2026-06-09

### 計画と実績の差分

**計画と異なった点**:
- starlette 1.2.1 (FastAPI 0.115.14 非互換) が入っていた → 0.46.2 にダウングレードで解決
- NAIBU_DB_PATH が Windows ハードコードパス → devcontainer-aware に修正 (settings.py)
- wave_types は `wave_type` ではなく `wave_types` (複数) で `|` 区切り複合値。最高スコアを採用
- fundamental 軸は PLAN.md で naibu 依存とされていたが、CES parquet がローカルに存在するため naibu API 不要で算出できた
- total_score の None 条件: PLAN.md では growth+fundamental+PP の60%が落ちたら None だったが、実装では moat_pp (最重要軸) が None の場合のみ total_score=None とした (fundamental は常にローカル計算可能なため)

**新たに必要になったタスク**:
- settings.py の NAIBU_DB_PATH devcontainer 修正 (当初計画外)
- starlette バージョン修正 (当初計画外)

### 学んだこと

- `results.csv` の波形分類は `wave_type` ではなく `wave_types` (複数可、`|` 区切り)
- CES スコアは capital_efficiency_screen.parquet として既にローカル保存済み。naibu API/DB が落ちていても fundamental 軸は算出できる
- NAIBU_DB_PATH を `os.getenv()` + `__file__` ベースの相対パスにすることで Windows/devcontainer 双方で動作
- moat_pp を naibu API 専用にすることで「naibu 未起動 = total_score=None」が自然なシグナルになる

### 次回への改善提案

- Phase 3 着手前に naibu FastAPI を port 8000 で起動し、`/api/pricing-power/companies/72030` が正常レスポンスを返すことを確認する
- moat_scores.parquet を Phase 4 バッチで生成したら、ranking タブの動作も確認する
- growth 軸のデータが naibu DB に存在しない銘柄 (6861 キーエンス、7974 任天堂) は None。Phase 4 では fallback 値として「業種平均成長率」を使う改善余地あり
