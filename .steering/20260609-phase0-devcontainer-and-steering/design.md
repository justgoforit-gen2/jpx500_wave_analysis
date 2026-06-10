# 設計書 — Phase 0: devcontainer + steering 体制構築

## アーキテクチャ概要

Phase 0 は実装フェーズではなく「環境構築フェーズ」。
コード変更は最小限で、インフラ・設定ファイル・ドキュメントのみを扱う。

```
dify_projects/
├── .devcontainer/devcontainer.json  ← コンテナ定義
├── .devcontainer/postCreate.sh      ← 依存インストール自動化
│
└── jpx500_wave_analysis/
    ├── .claude/
    │   ├── settings.json          ← sandbox + MCP permissions
    │   ├── skills/steering/       ← steering スキル (tasklist管理)
    │   ├── commands/add-feature.md← /add-feature コマンド
    │   └── agents/                ← doc-reviewer / implementation-validator
    ├── .mcp.json                  ← VSCode Claude Code MCP 登録 (Phase 0: gdp/estat)
    └── docs/                      ← 永続ドキュメント 6本
```

## コンポーネント設計

### 1. devcontainer.json

**責務**:
- Python 3.12 + Node.js (LTS) + Claude Code の feature 指定
- forwardPorts で 4 ポートを外部公開
- VSCode 拡張 (ruff / pylance / debugpy / rest-client) を自動インストール
- postCreateCommand で postCreate.sh を実行

**実装の要点**:
- `remoteUser: "vscode"` / `containerUser: "vscode"` で権限統一
- `source=claude-code-config` volume マウントで設定を永続化
- `source=naibu-ryuho-raw` volume マウントで大容量 EDINET raw データを分離

### 2. postCreate.sh

**責務**:
- 各アプリの `requirements.txt` を devcontainer 共有 Python 環境にインストール
- git safe.directory 設定 (CLAUDE.md グローバルルール参照)

### 3. .mcp.json

**責務**:
- VSCode Claude Code が stdio で MCP サーバーを起動する設定を保持
- Phase 0: gdp / estat の 2 本
- Phase 1: mindmap を追加して 3 本
- Phase 3: jpx500 / naibu / moat_score を追加して 6 本

**実装の要点**:
- コマンドは絶対パス (`/workspaces/...`) で記述 (devcontainer 内固定パス)
- estat は server.py が `load_dotenv(_PACKAGE_DIR / ".env")` で自己完結するため env 不要
- gdp も同様に API Key 不要

```json
{
  "mcpServers": {
    "gdp": {
      "command": "python",
      "args": ["/workspaces/dify_projects/GDP_distribution/mcp_server/server.py"]
    },
    "estat": {
      "command": "python",
      "args": ["/workspaces/dify_projects/estat_mcp_server/server.py"]
    }
  }
}
```

### 4. docs/ 6本

**責務**:
- 実装前に必ず読む「設計フリーズ文書」
- PLAN.md を単一ソースに生成。実装中は変更しない

**ステータス遷移**:
- `承認待ち` → ユーザー確認 → `承認済み`

## データフロー

### .mcp.json → MCP ツール呼び出し
```
1. VSCode Claude Code が .mcp.json を読み込む
2. "gdp" / "estat" の各サーバーを stdio で起動
3. ユーザーが「GDP 製造業の売上推移を調べて」と質問
4. Claude が gdp_get_timeseries ツールを呼び出す
5. server.py が _tools.py に委譲して JSON を返す
```

## テスト戦略

Phase 0 にユニットテストは不要。受け入れ確認は手動:

```bash
# GDP MCP 起動確認
python /workspaces/dify_projects/GDP_distribution/mcp_server/server.py &
# → "GDP_distribution MCP Server" ログが出て待機状態になれば OK
kill %1

# estat MCP 起動確認
python /workspaces/dify_projects/estat_mcp_server/server.py &
# → ログが出て待機状態になれば OK
kill %1
```

## ディレクトリ構造 (Phase 0 完了時点)

```
jpx500_wave_analysis/
├── .mcp.json                  ★ 新規 (gdp/estat)
├── docs/
│   ├── product-requirements.md  ★ 承認済み
│   ├── functional-design.md     ★ 承認済み
│   ├── architecture.md          ★ 承認済み
│   ├── repository-structure.md  ★ 承認済み
│   ├── development-guidelines.md★ 承認済み
│   └── glossary.md              ★ 承認済み
└── .steering/
    └── 20260609-phase0-devcontainer-and-steering/
        ├── requirements.md  ★ 新規
        ├── design.md        ★ 新規
        └── tasklist.md      ★ 新規
```

## 実装の順序

1. `.steering/` 3ファイル作成 (このファイル含む)
2. `.mcp.json` 初版作成 (gdp/estat)
3. docs/ の「承認待ち」→「承認済み」ステータス更新
4. MCP 起動テスト

## セキュリティ考慮事項

- `.mcp.json` に API Key を埋め込まない (estat は server.py が .env から自己読み込み)
- `web_search-mcp` は登録しない (PRD §6 セキュリティ要件)

## 将来の拡張性

Phase 1 で mindmap を追加:
```json
"mindmap": {
  "command": "python",
  "args": ["/workspaces/dify_projects/mindmap_and_mice/mcp_server/server.py"]
}
```
(mcp_server/server.py は Phase 1 で新規作成)
