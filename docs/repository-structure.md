# Repository Structure — 投資判断ハブ「バフェット流 Moat Score エンジン」

**プロジェクト**: jpx500_wave_analysis (dify_projects 全体)  
**バージョン**: 1.0  
**作成日**: 2026-06-09  
**ステータス**: 承認済み

---

## 1. dify_projects 全体ディレクトリ構造

```
dify_projects/
├── .devcontainer/               # 統合 devcontainer 設定
│   ├── devcontainer.json
│   ├── Dockerfile
│   ├── postCreate.sh
│   └── NEXT_STEPS.md
├── .features/                   # クロスカット機能仕様書 (複数プロジェクト横断)
│   ├── _README.md
│   └── {YYYYMMDD}-{feature-slug}.md
├── .contracts/                  # MCP/API 契約定義
├── CLAUDE.md                    # 統合 devcontainer 使い方・ポート対応表
├── PLAN.md                      # バフェット流 Moat Score エンジン統合プラン (設計フリーズ)
│
├── jpx500_wave_analysis/        ★ 投資ハブ本体 (このリポジトリ)
├── naibu-ryuho-app/             # 財務・PP・スクリーナー
├── mindmap_and_mice/            # マインドマップ生成
├── GDP_distribution/            # GDP/業種統計 MCP (stdio)
├── estat_mcp_server/            # e-Stat 政府統計 MCP (stdio)
│
├── ai_procurement_os_ver2/      # スペック駆動開発の移植元 (独立 devcontainer、スコープ外)
├── ui-hub/                      # 本プランでは使わない (フォルダ残置)
└── (その他: career_*, job_search, rag-stack 等) # スコープ外
```

---

## 2. jpx500_wave_analysis 詳細構造

```
jpx500_wave_analysis/
│
├── CLAUDE.md                    # jpx500 プロジェクト固有ルール
├── INTEGRATION_PLAN.md          # naibu × jpx500 統合手順 (Step 1-7)
├── IMPLEMENTATION_NOTES.md      # 実装ログ (jpx500 側)
├── README.md
│
├── .claude/                     # Claude Code 設定 (ai_procurement_os_ver2 から移植)
│   ├── settings.json            # sandbox + permissions (MCP allow/ask/deny)
│   ├── agents/
│   │   ├── doc-reviewer.md
│   │   └── implementation-validator.md
│   ├── commands/
│   │   ├── setup-project.md     # /setup-project (docs 6本生成)
│   │   ├── add-feature.md       # /add-feature [phase名]
│   │   └── review-docs.md
│   └── skills/
│       ├── prd-writing/
│       ├── functional-design/
│       ├── architecture-design/
│       ├── repository-structure/
│       ├── development-guidelines/
│       ├── glossary-creation/
│       └── steering/
│
├── .mcp.json                    # VSCode Claude Code MCP サーバー登録
│                                # Phase 1: gdp/estat/mindmap
│                                # Phase 3: +jpx500/naibu/moat_score
│
├── .steering/                   # Phase 別 steering ファイル (正本)
│   ├── 20260609-phase0-devcontainer-and-steering/
│   │   ├── requirements.md
│   │   ├── design.md
│   │   └── tasklist.md
│   ├── 20260616-phase1-base-integration/    (Phase 1 完了後生成)
│   ├── 20260623-phase2-moat-score-engine/   (Phase 2 完了後生成)
│   ├── 20260707-phase3-mcp-vscode/          (Phase 3 完了後生成)
│   └── 20260714-phase4-daily-batch-and-streamlit-alerts/
│
├── docs/                        ★ 永続ドキュメント 6本 (Phase 0-3 で生成・承認)
│   ├── product-requirements.md  # PRD
│   ├── functional-design.md     # 機能設計書
│   ├── architecture.md          # アーキテクチャ
│   ├── repository-structure.md  # このファイル
│   ├── development-guidelines.md
│   └── glossary.md
│
├── app.py                       # Streamlit アプリ (既存5タブ + 新タブ追加)
├── api_server.py                # FastAPI サーバー (既存エンドポイント + moat-score 追加)
├── daily_update.bat             # Windows 日次バッチ起動スクリプト
│
├── modules/                     # 分析モジュール群
│   ├── wave_classifier.py       # テクニカル: 波形分類
│   ├── range_breakout_detector.py  # テクニカル: ブレイクアウト (Phase 2 で公開 wrapper 追加)
│   ├── trend_transition_detector.py # テクニカル: トレンド転換
│   ├── foreign_flow_analyzer.py    # 外国人フロー分析
│   ├── capital_efficiency_screener.py # CES (ファンダ軸)
│   ├── jpx_investor_flow_fetcher.py # 投資家別フロー取得
│   └── moat_score.py            # ★ Phase 2 新規: MoatScoreEngine
│
├── mcp_server/                  # ★ Phase 3 新規
│   ├── server.py                # jpx500-mcp (FastMCP stdio)
│   └── moat_score_server.py     # moat-score-mcp (FastMCP stdio)
│
├── batch/
│   └── update.py                # 日次バッチ (Phase 4 で moat-score recompute 追加)
│
├── data/
│   ├── jpx500_list.csv          # JPX500 銘柄リスト (598行: 498社 + ETF100)
│   ├── results.csv              # 波形分析結果 (正本、書き換え禁止)
│   ├── moat_scores.parquet      # ★ Phase 2 新規: MoatScore 計算結果
│   └── policy_signals.json      # ★ Phase 1 新規: 政府政策シグナル
│
├── config/
│   └── *.yaml / *.json          # 各種設定ファイル
│
├── tests/
│   ├── test_moat_score.py       # ★ Phase 2 新規
│   ├── test_moat_score_naibu_down.py  # ★ Phase 2 新規
│   └── e2e_moat_score.py        # ★ Phase 2 新規 (Playwright)
│
└── e2e_*.py                     # 既存 E2E スクリプト群 (無変更)
```

---

## 3. naibu-ryuho-app 関連構造 (参照用)

```
naibu-ryuho-app/
├── api/
│   ├── main.py                  # FastAPI :8000
│   ├── routers/
│   │   ├── stocks.py            # 財務詳細 API
│   │   ├── screener.py          # スクリーナー API
│   │   └── pricing_power.py     # PP スコア API
│   └── jpx_client.py            # HTTPX クライアント
├── scripts/
│   ├── utils/
│   │   └── scoring.py           # compute_pricing_power / compute_macro_resilience
│   └── 09_sync_jpx500_membership.py  # JPX500 500社同期 (Phase 1 で実行)
└── mcp_server/
    └── server.py                # ★ Phase 3 新規: naibu-mcp (FastMCP stdio)
```

---

## 4. ファイル命名規則

| 種類 | 命名パターン | 例 |
|---|---|---|
| Steering ディレクトリ | `YYYYMMDD-{phase}-{slug}/` | `20260609-phase0-devcontainer-and-steering/` |
| .features 仕様書 | `YYYYMMDD-{feature-slug}.md` | `20260511-mail-text-suggestion-only.md` |
| テストファイル | `test_{module}.py` / `e2e_{feature}.py` | `test_moat_score.py` |
| MCP サーバー | `mcp_server/server.py` | 各アプリの `mcp_server/` に配置 |

---

## 5. 変更禁止ファイル・ディレクトリ

以下は `data/` 正本またはパスがハードコードされているため、移動・名前変更禁止:

| パス | 理由 |
|---|---|
| `jpx500_wave_analysis/data/results.csv` | 波形分析の正本。api_server.py がパス固定で参照 |
| `jpx500_wave_analysis/data/jpx500_list.csv` | 銘柄マスタ。複数モジュールが参照 |
| `naibu-ryuho-app/*.db` | SQLite DB。naibu 側でパス固定 |
| 各アプリの `.venv/` | 失敗時の戻り先として保持 |

---

## 6. 新規ファイルの配置ルール

| ファイル種別 | 配置先 | 命名 |
|---|---|---|
| 分析モジュール | `jpx500_wave_analysis/modules/` | `{機能名}.py` |
| MCP サーバー | `{アプリ}/mcp_server/server.py` | `server.py` 固定 |
| テスト | `jpx500_wave_analysis/tests/` | `test_{module}.py` |
| E2E テスト | `jpx500_wave_analysis/` 直下 | `e2e_{feature}.py` |
| データ出力 | `jpx500_wave_analysis/data/` | `{用途}.parquet` or `.json` |
| Steering | `{アプリ}/.steering/{date}-{phase}/` | `requirements.md` / `design.md` / `tasklist.md` |
| 永続ドキュメント | `jpx500_wave_analysis/docs/` | 固定6本 (命名変更不可) |
