# 要件定義 — Phase 1: base-integration

**プロジェクト**: jpx500_wave_analysis  
**フェーズ**: phase1-base-integration  
**作成日**: 2026-06-09  
**ステータス**: 作業中

---

## 背景と目的

Phase 0 で統合 devcontainer と docs/ 6本・.mcp.json (gdp/estat) が整った。
Phase 1 では「mindmap_and_mice を MCP 経由で VSCode Claude Code から呼べる状態」と
「policy_signals.json (政策シグナルの正本ファイル) の初期化」を完了させる。

これにより、VSCode チャットから `mcp__mindmap__expand` 等を直接呼べるようになり、
モートスコアの政策軸 (Phase 2) の準備も整う。

---

## スコープ

### In Scope

1. **mindmap ポート移行**: `mindmap_and_mice/main.py` の PORT を 8001 → 8003 に変更
   - 理由: jpx500 FastAPI が 8001 を占有するため競合回避
   - URL 定数・uvicorn 起動引数も合わせて更新

2. **mindmap FastAPI リスト API**: `GET /api/list` エンドポイントを追加
   - saved ディレクトリの .mmd ファイル一覧を返す
   - 薄ラッパ原則: MCP ツールが HTTPX でこれを呼ぶ

3. **mindmap MCP サーバー作成**: `mindmap_and_mice/mcp_server/server.py`
   - FastMCP + httpx で mindmap FastAPI を薄ラッパ
   - 4ツール: mindmap_expand / mindmap_to_mermaid / mindmap_save / mindmap_list_saved
   - PORT 変数を環境変数 `MINDMAP_PORT` (デフォルト 8003) で上書き可能に

4. **.mcp.json 更新**: mindmap を 3本目として追加
   - 絶対パス: `/workspaces/dify_projects/mindmap_and_mice/mcp_server/server.py`

5. **policy_signals.json 初版**: `jpx500_wave_analysis/data/policy_signals.json`
   - スキーマ: `{policy_id, theme, sector_tags, strength: 0-3, valid_until, updated_at}`
   - 5政策テーマ: GX / DX / 防衛 / 少子化対策 / インフラ老朽化

### Out of Scope

- Phase 2 以降: MoatScoreEngine 実装
- Phase 3 以降: jpx500-mcp / naibu-mcp / moat-score-mcp
- naibu-ryuho-app の内部変更
- Streamlit タブの追加

---

## 受け入れ条件

1. `mindmap_and_mice/main.py` を実行すると PORT 8003 で起動する
2. `python mindmap_and_mice/mcp_server/server.py` が stdio MCP として起動する
3. `.mcp.json` に mindmap エントリが存在し、パスが正しい
4. `jpx500_wave_analysis/data/policy_signals.json` が存在し、5件のエントリを持つ
5. Phase 1 tasklist.md が 100% `[x]`
