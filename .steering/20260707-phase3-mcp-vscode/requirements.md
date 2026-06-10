---
phase: 3
title: MCP 3本実装と VSCode Claude Code 登録
status: in_progress
---

# Phase 3 要件定義

## 背景

Phase 2 で MoatScoreEngine と API エンドポイントが完成。
Phase 3 では jpx500 / naibu / moat-score の 3 MCP サーバーを実装し、
VSCode Claude Code チャットからこのチャット画面で直接呼べるようにする。

## 必須Deliverables

### D1: jpx500-mcp (`jpx500_wave_analysis/mcp_server/server.py`)
FastMCP stdio サーバー。jpx500 API (:8001) への薄いラッパ。

| ツール名 | 呼び先 |
|---|---|
| `get_wave` | GET /api/wave/{code} |
| `get_picks_today` | GET /api/picks/today |
| `get_abcd_ranking` | GET /api/abcd-ranking |
| `get_prices` | GET /api/prices/{code}?days=N |
| `get_earnings` | GET /api/earnings |
| `get_foreign_flow` | GET /api/foreign-flow/{code} |

### D2: naibu-mcp (`naibu-ryuho-app/mcp_server/server.py`)
FastMCP stdio サーバー。naibu API (:8000) への薄いラッパ。

| ツール名 | 呼び先 |
|---|---|
| `get_company` | GET /api/companies/{edinet_code} |
| `get_financials` | GET /api/stocks/{edinet_code} |
| `screen` | GET /api/screener |
| `pricing_power` | GET /api/pricing-power/companies/{edinet_code} |
| `activist_screen` | GET /api/activist-screen |
| `industry_aggregate` | GET /api/industries |

### D3: moat-score-mcp (`jpx500_wave_analysis/mcp_server/moat_score_server.py`)
FastMCP stdio サーバー。jpx500 moat-score エンドポイントへの薄いラッパ。

| ツール名 | 呼び先 |
|---|---|
| `compute_moat_score` | MoatScoreEngine.compute() 直接呼び出し (API 不要) |
| `rank_by_moat` | GET /api/moat-score/ranking?top=N |
| `explain_score` | MoatScoreEngine.compute() の explanation フィールド |
| `list_policy_signals` | policy_signals.json を直接読み込み |

### D4: .mcp.json 更新
jpx500 / naibu / moat_score の 3 サーバーを追記 (合計 6 サーバー)

### D5: settings.json 権限確認
- allow 済: mcp__jpx500__get_*, mcp__naibu__get_*, mcp__naibu__screen, mcp__naibu__pricing_power, mcp__moat_score__rank_by_moat, mcp__moat_score__explain_score, mcp__moat_score__list_policy_signals
- ask 済: mcp__moat_score__compute_moat_score
- **手動追加必要**: mcp__naibu__activist_screen, mcp__naibu__industry_aggregate

## スコープ外

- VSCode 再起動後の MCP 疎通確認 (ユーザーが手動で確認)
- 全 MCP スタンドアロン起動テスト (python <path> で stdin 待ち確認)
