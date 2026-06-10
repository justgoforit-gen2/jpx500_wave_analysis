# タスクリスト — Phase 0: devcontainer + steering 体制構築

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

### 必須ルール
- **全てのタスクを`[x]`にすること**
- 「時間の都合により別タスクとして実施予定」は禁止
- 「実装が複雑すぎるため後回し」は禁止
- 未完了タスク（`[ ]`）を残したまま作業を終了しない

---

## フェーズ1: devcontainer + .claude/ 体制構築

- [x] 統合 devcontainer 作成
  - [x] `.devcontainer/devcontainer.json` (Python 3.12 + Node.js + Claude Code feature)
  - [x] `.devcontainer/postCreate.sh` (依存インストール)
  - [x] `.devcontainer/NEXT_STEPS.md` (起動後の手順)
  - [x] forwardPorts: 8000/8001/8003/8501 設定確認

- [x] .claude/ スペック駆動開発体制 設定確認
  - [x] skills/: steering / prd-writing / functional-design / architecture-design / repository-structure / development-guidelines / glossary-creation
  - [x] commands/: setup-project / add-feature / review-docs / pre-push-check / status
  - [x] agents/: doc-reviewer / implementation-validator
  - [x] settings.json (sandbox + permissions)

## フェーズ2: 永続ドキュメント 6本 生成・承認

- [x] `docs/product-requirements.md` 生成 (PLAN.md ソース)
- [x] `docs/functional-design.md` 生成
- [x] `docs/architecture.md` 生成
- [x] `docs/repository-structure.md` 生成
- [x] `docs/development-guidelines.md` 生成
- [x] `docs/glossary.md` 生成
- [x] docs/ 6本のステータスを「承認待ち」→「承認済み」に更新

## フェーズ3: .mcp.json 初版作成

- [x] `.mcp.json` 作成 (gdp / estat の 2 本)
  - [x] gdp: `GDP_distribution/mcp_server/server.py` パス確認・登録
  - [x] estat: `estat_mcp_server/server.py` パス確認・登録 (ESTAT_API_KEY は .env 自己読み込みで不要)

## フェーズ4: 動作確認

- [x] GDP MCP 起動テスト (`python .../GDP_distribution/mcp_server/server.py`) — 絶対パスで正常起動確認
- [x] estat MCP 起動テスト (`python .../estat_mcp_server/server.py`) — 絶対パスで正常起動確認

---

## 実装後の振り返り

### 実装完了日
2026-06-09

### 計画と実績の差分

**計画と異なった点**:
- `/setup-project` `/add-feature` コマンドが Host Claude Code セッションでは認識されないため、
  コマンド定義を読んで直接実行する方式に切り替えた

**新たに必要になったタスク**:
- (Phase 0 完了後に記入)

### 学んだこと

- Host Claude Code (dify_projects 直下) から jpx500 のカスタムコマンドを呼ぶには
  コマンド定義を Read してステップを直接実行するワークアラウンドが必要

### 次回への改善提案
- Phase 1 以降は `cd jpx500_wave_analysis && claude` で devcontainer Claude を起動するか、
  または今回同様にコマンド定義を直接読んで実行する
