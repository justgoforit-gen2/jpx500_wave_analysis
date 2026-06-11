# Glossary — 投資判断ハブ「バフェット流 Moat Score エンジン」

**プロジェクト**: jpx500_wave_analysis  
**バージョン**: 1.0  
**作成日**: 2026-06-09  
**ステータス**: 承認済み

このドキュメントはコードベース・ドキュメント・チャット内で使われる用語を統一的に定義する。  
新しい概念を導入する際はここに追記してから実装する。

---

## A

### ABCD ランキング

既存 Streamlit タブの名称。波形分析に基づく銘柄ランキング（A: 最強、D: 最弱）。  
`GET /api/abcd` エンドポイントで提供。MoatScore ランキングとは別物。

### api_server.py

`jpx500_wave_analysis/api_server.py`。FastAPI で jpx500 の REST API を提供するファイル。  
Phase 2 で moat-score 系エンドポイントを追加する。

---

## B

### 貸借対照表 (BS / Balance Sheet)

naibu-ryuho-app の `financial_metrics` テーブルに格納された資産・負債・純資産の財務データ。  
`modules/naibu_client.py::fetch_balance_sheet(jpx_code)` で最新本決算期 1 行を読み取り専用で取得する。  
銘柄詳細ビューに「BS・最新期」スナップショット（指標カード + 構成グラフ + 明細表）を表示する。  
主要フィールド: `total_assets`(総資産), `total_equity`(純資産), `total_liabilities`(総負債),  
`current_assets`, `current_liabilities`, `cash`, `short_term_debt`, `long_term_debt`,  
`retained_earnings_bs`(利益剰余金), `capital_stock`(資本金)。

### バッチ (Batch)

`jpx500_wave_analysis/batch/update.py` が実装する日次処理。毎晩 18:00 に `daily_update.bat` から起動。  
Phase 4 で全銘柄 MoatScore 計算ステップを追加する。

---

## C

### CES (Capital Efficiency Screener)

`modules/capital_efficiency_screener.py` が提供する資本効率スクリーナー。  
ROE・ROA・投下資本利益率などを算出し、ファンダ軸 (軸2) のスコア算出に使用する。

### compute_bulk()

`MoatScoreEngine.compute_bulk(codes)` — 複数銘柄のMoatScoreを一括計算するメソッド。  
バッチ処理で全 JPX500 銘柄 (最大598件) に対して実行する。

### compute_macro_resilience

`naibu-ryuho-app/scripts/utils/scoring.py` に存在するマクロ耐性スコア計算関数。  
jpx500 側から直接インポートせず、naibu API 経由で使う（薄ラッパ原則）。

### compute_pricing_power

`naibu-ryuho-app/scripts/utils/scoring.py` に存在する価格転嫁力スコア計算関数。  
jpx500 側からは `GET /api/pricing-power/{code}` (naibu :8000) 経由で取得する。

---

## D

### devcontainer (統合 devcontainer)

`dify_projects/.devcontainer/` に配置した統合 Docker 開発環境。  
Python 3.12 + Node.js で5アプリの実行環境を1コンテナに集約。forwardPorts: 8000/8001/8003/8501。

### docs/ (永続ドキュメント)

`jpx500_wave_analysis/docs/` に置く6本の永続ドキュメント。  
Phase 1〜4 の `/add-feature` 着手前に必ず読む。PLAN.md はフリーズ設計書であり、実装中に更新しない。

---

## E

### EDINET コード

naibu-ryuho-app が使う5桁の銘柄識別コード（例: `72030`）。  
jpx500 の4桁コード（例: `7203`）に末尾 "0" を付与したもの。変換責務は naibu 側。

### explanation (MoatScoreResult)

`MoatScoreResult.explanation: dict[str, str]`。各軸のスコア根拠テキスト。  
`explain_score` MCP ツールはこのフィールドを返す。固定テンプレ（軸名/数値/事実根拠の3点）で生成する。

---

## F

### FastMCP

MCP サーバーを Python で簡易に実装するフレームワーク。  
本プランでは `@mcp.tool()` デコレータで各ツールを定義し、stdio transport で VSCode Claude Code に接続する。

### fetch_balance_sheet

`modules/naibu_client.py::fetch_balance_sheet(jpx_code: str) -> dict | None`。  
4桁 jpx_code を受け取り naibu-ryuho-app の `financial_metrics` テーブルから  
`total_assets IS NOT NULL` な最新本決算期 BS を 1 行 dict で返す。  
DB 不在・該当なし時は `None`。金額は円単位 INTEGER。  
既存の `_connect()` (read-only URI) / `naibu_db_exists()` パターンを踏襲。

### foreign_flow_analyzer

`jpx500_wave_analysis/modules/foreign_flow_analyzer.py`。外国人投資家の業種別フローを分析するモジュール。  
`compute_sector_flow_correlation()` が外国人フロー軸 (軸3) のスコア算出に使われる。

---

## G

### GDP_distribution

`dify_projects/GDP_distribution/` に配置した MCP サーバー。GDP/業種統計データを提供する。  
Phase 1 から `.mcp.json` に登録。業種分析の補助データとして使う。

### GX (グリーン・トランスフォーメーション)

政府骨太政策のテーマのひとつ。脱炭素・再生可能エネルギー関連。  
`policy_signals.json` の `policy_id: "GX-2026"` として管理する。

---

## H

### Host Claude / devcontainer Claude

「Host Claude」= VSCode が `dify_projects/` を直接開いているときの Claude Code チャット。  
「devcontainer Claude」= `Reopen in Container` 後のコンテナ内 Claude Code チャット。  
本プランでは Host Claude が `.features/` を管理し、devcontainer Claude が実装を担う。

---

## J

### jpx500_list.csv

`jpx500_wave_analysis/data/jpx500_list.csv`。全598行 = JPX500構成498社 + ETF100行。  
書き換え禁止。naibu との結合時 ETF 行は自然に脱落（EDINET フィリングなし）。

### jpx500-mcp

Phase 3 で実装する MCP サーバー (`mcp_server/server.py`)。  
6ツール: `jpx500_get_wave` / `get_picks_today` / `get_abcd_ranking` / `get_prices` / `get_earnings` / `get_foreign_flow`。

---

## M

### .mcp.json

`jpx500_wave_analysis/.mcp.json`。VSCode Claude Code に登録する MCP サーバー設定ファイル。  
Phase 1: gdp/estat/mindmap の3本。Phase 3: +jpx500/naibu/moat_score で計6本。

### MCP (Model Context Protocol)

Claude が外部ツールを呼ぶための標準プロトコル。本プランでは stdio transport を使う。  
「薄ラッパ原則」: MCPツールは HTTPX 呼び出しのみ。業務ロジックは FastAPI 側に持つ。

### MoatScore (総合スコア)

7軸の重み付き加重平均 (0〜10)。バフェット流で moat/収益力に60%配分:  
`ファンダ25% + PP/産業障壁20% + 成長長期15% = 60%`。

### MoatScoreEngine

`jpx500_wave_analysis/modules/moat_score.py` の主要クラス。  
`compute(code)` で単銘柄、`compute_bulk(codes)` で一括計算。Phase 2 で実装。

### MoatScoreResult

MoatScoreEngine が返すデータクラス。7軸スコア + total_score + rank + explanation を持つ。  
`total_score = None` = naibu 未起動またはデータ不足。

### moat-score-mcp

Phase 3 で実装する MCP サーバー (`mcp_server/moat_score_server.py`)。  
4ツール: `compute_moat_score` / `rank_by_moat` / `explain_score` / `list_policy_signals`。

### moat_scores.parquet

`jpx500_wave_analysis/data/moat_scores.parquet`。バッチが毎晩書き込む全銘柄 MoatScore の計算結果。  
Phase 2 で新規作成。ランキング API はこのファイルを読み込む（高速化のため事前計算）。

---

## N

### naibu-mcp

Phase 3 で実装する MCP サーバー (`naibu-ryuho-app/mcp_server/server.py`)。  
6ツール: `naibu_get_company` / `get_financials` / `screen` / `pricing_power` / `activist_screen` / `industry_aggregate`。

### naibu-ryuho-app

`dify_projects/naibu-ryuho-app/`。財務・PP・スクリーナー API を提供する FastAPI アプリ (port 8000)。  
MoatScoreEngine から HTTPX で呼ぶ。直接インポートしない。

---

## P

### permissions.ask / permissions.allow

`.claude/settings.json` の MCP 権限設定。  
`allow`: 自動許可（読み取り系ツール）。`ask`: 都度確認（書き込み系・bulk 計算）。

### policy_signals.json

`jpx500_wave_analysis/data/policy_signals.json`。政府骨太政策シグナルを格納する JSON ファイル。  
スキーマ: `{policy_id, theme, sector_tags, strength: 0-3, valid_until, updated_at}`。  
`/policy-update` スラッシュコマンドで月次更新。7日超過で Streamlit が赤字警告を出す。

### /policy-update

`~/.claude/commands/policy-update.md` に定義するスラッシュコマンド。  
VSCode Claude Code CLI で手動実行し、`web_search-mcp` で骨太の方針ニュースを検索 → `policy_signals.json` 更新。

### PP (Pricing Power / 価格転嫁力)

企業がコスト上昇を製品価格に転嫁できる能力。産業障壁/PP 軸 (軸6・重み20%) の中核指標。  
naibu `scoring.py::compute_pricing_power` で計算。jpx500 からは `GET /api/pricing-power/{code}` で取得。

---

## R

### range_breakout_detector

`jpx500_wave_analysis/modules/range_breakout_detector.py`。  
`_evaluate` はプライベート関数。Phase 2 で `evaluate(code, ohlcv)` の公開 wrapper を1関数追加する（`_evaluate` は無変更）。

### RECOMPUTE_TOKEN

環境変数。`POST /api/moat-score/recompute` の認証に使う秘密トークン。  
バッチ (`daily_update.bat`) のみが知っている。Streamlit・MCP からは叩かない。

---

## S

### 銘柄詳細ビュー (show_detail_view)

`app.py::show_detail_view()` が描画する全画面オーバーレイビュー。  
`st.session_state["view"] == "detail"` のときタブ群の代わりに表示される。  
`_navigate_to_detail(code, name, ticker)` ヘルパで session_state をセットし `st.rerun()` で遷移する。  
表示内容: 波形チャート / 指標カード / 業種比較 / 業績(P/L) / **貸借対照表(BS)** / 信用取引残高 / 決算予定。  
「← 一覧に戻る」ボタンで `view="list"` に戻る。  
波形分類・ABCD戦略・ポートフォリオ・**ランキング**タブから遷移可能。

### 自己資本比率 (Equity Ratio)

`total_equity / total_assets × 100 [%]`。財務健全性の基本指標。  
銘柄詳細の BS セクションで `_format_oku` で整形して表示する。  
0 除算は「-」表示で安全にガード。

### securities_code

jpx500 内での銘柄コード。**4桁**の数字文字列（例: `"7203"`）。  
naibu 側の5桁 EDINET コード (`"72030"`) と区別する。

### steering (.steering/)

`jpx500_wave_analysis/.steering/{date}-{phase}/` に配置する Phase 別の作業ファイル群。  
3ファイル構成: `requirements.md` (要件) / `design.md` (設計) / `tasklist.md` (タスクリスト・正本)。  
`/add-feature [phase名]` で生成。全タスク `[x]` 完了で Phase 完了宣言可能。

---

## T

### 薄ラッパ原則 (Thin Wrapper Principle)

MCP サーバーには HTTPX 呼び出しだけを書き、業務ロジックを持たせない原則。  
これにより既存 FastAPI との2重管理を避け、テスト・デバッグを FastAPI 側に集中できる。

### trend_transition_detector

`jpx500_wave_analysis/modules/trend_transition_detector.py`。  
`detect_transitions()` でトレンド転換シグナルを検出。テクニカル軸 (軸1) のスコア計算に使用。

---

## U

### ui-hub

`dify_projects/ui-hub/`。本プランでは使わない（フォルダのみ残置）。  
VSCode Claude Code チャット + `.mcp.json` が代替として機能する。

---

## W

### wave_classifier

`jpx500_wave_analysis/modules/wave_classifier.py`。  
`classify(code)` / `classify_all()` で波形ラベルを返す。テクニカル軸 (軸1) のスコア算出に使用。  
wave_type ラベル → スコア変換表は `docs/functional-design.md` § 2.2 参照。

### wave_types

`results.csv` の列名。`|` 区切りで複数ラベルが入る（例: `"上昇トレンド|ブレイク気味"`）。

---

## X

### X-Recompute-Token

`POST /api/moat-score/recompute` のリクエストヘッダ。  
値は `env:RECOMPUTE_TOKEN` と一致する必要がある。不一致は 403 を返す。
