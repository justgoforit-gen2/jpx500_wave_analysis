# 要求内容 — Phase 0: devcontainer + steering 体制構築

## 概要

5アプリを1コンテナに集約する統合 devcontainer を構築し、スペック駆動開発のための
`.claude/` 設定・永続ドキュメント 6本・`.mcp.json` 初版を揃える。

## 背景

jpx500_wave_analysis を投資ハブの中核とする「バフェット流 Moat Score エンジン」を
Phase 1〜4 で実装する前提として、再現可能な開発環境と仕様ドキュメント基盤が必要。

## 実装対象の機能

### 1. 統合 devcontainer

- Python 3.12 + Node.js (LTS) + Claude Code を1コンテナに集約
- forwardPorts: 8000 / 8001 / 8003 / 8501
- postCreate.sh で全アプリの依存インストール自動化

### 2. スペック駆動開発体制 (.claude/ 設定)

- skills: steering / prd-writing / functional-design / architecture-design / repository-structure / development-guidelines / glossary-creation
- commands: setup-project / add-feature / review-docs / pre-push-check / status
- agents: doc-reviewer / implementation-validator

### 3. 永続ドキュメント 6本 (docs/)

- product-requirements.md / functional-design.md / architecture.md
- repository-structure.md / development-guidelines.md / glossary.md

### 4. .mcp.json 初版 (gdp + estat の 2 本)

- gdp: `GDP_distribution/mcp_server/server.py`
- estat: `estat_mcp_server/server.py` (ESTAT_API_KEY は server.py が .env から自動読み込み)
- mindmap は Phase 1 でポート移行完了後に追加

## 受け入れ条件

### devcontainer
- [ ] `devcontainer.json` でコンテナ起動・Claude Code が使える
- [ ] forwardPorts: 8000/8001/8003/8501 が正しく設定されている

### docs/
- [ ] 6本すべてのステータスが「承認済み」になっている

### .mcp.json
- [ ] gdp / estat の 2 本が登録済み
- [ ] `python /workspaces/.../GDP_distribution/mcp_server/server.py` が起動できる
- [ ] `python /workspaces/.../estat_mcp_server/server.py` が起動できる

## スコープ外

- mindmap MCP (Phase 1 で実装)
- jpx500-mcp / naibu-mcp / moat-score-mcp (Phase 3 で実装)
- MoatScoreEngine 実装 (Phase 2)
- policy_signals.json 初版 (Phase 1)

## 参照ドキュメント

- `docs/product-requirements.md` — PRD §9 フェーズ概要
- `docs/repository-structure.md` — .steering/ ディレクトリ構造
- `docs/development-guidelines.md` — スペック駆動3原則
