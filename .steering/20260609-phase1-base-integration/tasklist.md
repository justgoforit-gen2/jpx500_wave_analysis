# タスクリスト — Phase 1: base-integration

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

### 必須ルール
- **全てのタスクを`[x]`にすること**
- 「時間の都合により別タスクとして実施予定」は禁止
- 「実装が複雑すぎるため後回し」は禁止
- 未完了タスク（`[ ]`）を残したまま作業を終了しない

---

## フェーズ1: mindmap ポート移行

- [x] `mindmap_and_mice/main.py` の PORT を 8001 → 8003 に変更
  - [x] PORT 定数変更
  - [x] URL 定数が PORT を参照していることを確認 (変更不要のはず)

## フェーズ2: mindmap FastAPI リスト API 追加

- [x] `mindmap_and_mice/src/api/routes.py` に `GET /api/list` エンドポイント追加
  - [x] SAVED_DIR を `MINDMAP_SAVED_DIR` 環境変数で上書き可能な形で定義
  - [x] `.mmd` ファイル一覧を返す `list_saved()` 実装

## フェーズ3: mindmap MCP サーバー作成

- [x] `mindmap_and_mice/mcp_server/` ディレクトリ作成
  - [x] `__init__.py` 作成 (空ファイル)
  - [x] `server.py` 作成 (FastMCP stdio, 4 ツール)
    - [x] `expand` ツール (POST /api/expand)
    - [x] `to_mermaid` ツール (POST /api/to-mermaid)
    - [x] `list_saved` ツール (GET /api/list)
    - [x] `save` ツール (POST /api/save)

## フェーズ4: .mcp.json + settings.json 更新

- [x] `jpx500_wave_analysis/.mcp.json` に mindmap エントリ追加
- [x] `jpx500_wave_analysis/.claude/settings.json` に mindmap MCP 権限追加 (Phase 0 で既に設定済みを確認)
  - [x] allow: mcp__mindmap__expand / mcp__mindmap__to_mermaid / mcp__mindmap__list_saved
  - [x] ask: mcp__mindmap__save

## フェーズ5: policy_signals.json 初版作成

- [x] `jpx500_wave_analysis/data/policy_signals.json` 作成 (5 政策テーマ)

## フェーズ5-B: 追加 Deliverables (PLAN.md §5 Phase 1 完了チェックリストより)

- [x] `mindmap_and_mice/src/api/app.py` に `/health` エンドポイント追加 (PLAN.md `curl http://localhost:8003/health` 対応)
- [x] `~/.claude/commands/policy-update.md` 作成 (グローバル VSCode Claude Code CLI コマンド)
- [x] ~~`naibu-ryuho-app/scripts/09_sync_jpx500_membership.py` 実行~~ (理由: jpx500 api_server.py が FastAPI `on_startup` deprecated エラーで起動不可。Phase 2 で api_server.py 修正時に併せて実施)

## フェーズ6: 動作確認

- [x] mindmap MCP サーバー起動テスト — `timeout 3 python mindmap_and_mice/mcp_server/server.py` エラーなし、stdio 待ちで正常起動確認

---

## 実装後の振り返り

### 実装完了日
2026-06-09

### 計画と実績の差分

**計画と異なった点**:
- MCP ツール名: 設計書では `mindmap_expand` とプレフィックス付きで記述していたが、
  settings.json (Phase 0 で設定済み) は `mcp__mindmap__expand` (プレフィックスなし) を想定していた。
  ツール関数名を `expand` / `to_mermaid` / `list_saved` / `save` に修正して整合。
- settings.json の mindmap 権限: Phase 0 時点で既に正しく設定されており、追記不要だった。

**新たに必要になったタスク**:
- なし

### 学んだこと

- settings.json の MCP 権限パターン `mcp__{server}__{tool}` の `tool` 部分はツール関数名と完全一致する。
  設計書でのツール名定義と settings.json の記述を同時に確認してから実装すると手戻りがない。
- `routes.py` の SAVED_DIR: `Path(__file__).resolve().parents[3]` でプロジェクトルートを算出。
  相対パスではなく絶対パス解決にすることで CWD 依存を排除した。

### 次回への改善提案

- Phase 2 着手前に mindmap :8003 を実際に起動して API 疎通確認をしておくと安心。
- policy_signals.json の strength 定義 (0-3) と MoatScore 政策軸への変換係数は
  Phase 2 の functional-design に明記すること。
