# タスクリスト — Phase 3: mcp-vscode

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

---

## フェーズ1: jpx500-mcp 作成

- [x] `jpx500_wave_analysis/mcp_server/__init__.py` 作成 (空)
- [x] `jpx500_wave_analysis/mcp_server/server.py` 作成
  - [x] `get_wave(code)` ツール
  - [x] `get_picks_today()` ツール
  - [x] `get_abcd_ranking()` ツール
  - [x] `get_prices(code, days)` ツール
  - [x] `get_earnings()` ツール
  - [x] `get_foreign_flow(code)` ツール

## フェーズ2: naibu-mcp 作成

- [x] `naibu-ryuho-app/mcp_server/__init__.py` 作成 (空)
- [x] `naibu-ryuho-app/mcp_server/server.py` 作成
  - [x] `get_company(edinet_code)` ツール
  - [x] `get_financials(edinet_code)` ツール
  - [x] `screen(...)` ツール
  - [x] `pricing_power(edinet_code)` ツール
  - [x] `activist_screen()` ツール
  - [x] `industry_aggregate()` ツール

## フェーズ3: moat-score-mcp 作成

- [x] `jpx500_wave_analysis/mcp_server/moat_score_server.py` 作成
  - [x] `compute_moat_score(code)` ツール
  - [x] `rank_by_moat(top, sector)` ツール
  - [x] `explain_score(code)` ツール
  - [x] `list_policy_signals()` ツール

## フェーズ4: .mcp.json 更新

- [x] `jpx500_wave_analysis/.mcp.json` に jpx500 / naibu / moat_score の 3 サーバー追記 (合計 6 サーバー登録)

## フェーズ5: 起動テスト

- [x] jpx500-mcp: `timeout 4 python mcp_server/server.py` → exit 0 (stdin 待ち正常)
- [x] naibu-mcp: `timeout 4 python naibu-ryuho-app/mcp_server/server.py` → exit 0
- [x] moat-score-mcp: `timeout 4 python mcp_server/moat_score_server.py` → exit 0

## フェーズ6: settings.json 手動追加指示

- [x] ~~`mcp__naibu__activist_screen` と `mcp__naibu__industry_aggregate` の allow 追加~~ (理由: settings.json の自己変更はセキュリティゲートでブロック。ユーザーに手動追加を依頼)

---

## 実装後の振り返り

### 実装完了日
2026-06-10

### 計画と実績の差分

- settings.json の `mcp__naibu__activist_screen` / `mcp__naibu__industry_aggregate` の allow 追加がセキュリティゲートでブロックされた。ユーザーに手動追加を依頼する
- moat-score-mcp の `compute_moat_score` と `rank_by_moat` は HTTPX/直接呼び出しを併用。rank_by_moat は parquet 存在前提

### 学んだこと

- FastMCP stdio サーバーは `__name__ == "__main__"` ブロックに `mcp.run()` を置くだけで動作する
- `sys.path.insert` で jpx500_wave_analysis をパスに追加しないと moat_score_server.py が modules を import できない
- naibu-mcp は naibu FastAPI (:8000) が起動していない場合 HTTPX 接続エラーになるが、MCP サーバー自体の起動は成功する

### 次回への改善提案

- VSCode を再起動して MCP 6本が補完候補に出ることを確認すること
- naibu API `/api/industries` の実際のパラメータを確認 (industry_aggregate ツールの引数を調整する可能性あり)
- settings.json に `mcp__naibu__activist_screen` と `mcp__naibu__industry_aggregate` を手動で追加すること
