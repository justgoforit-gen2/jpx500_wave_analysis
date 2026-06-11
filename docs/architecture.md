# Architecture Document — 投資判断ハブ「バフェット流 Moat Score エンジン」

**プロジェクト**: jpx500_wave_analysis  
**バージョン**: 1.0  
**作成日**: 2026-06-09  
**ステータス**: 承認済み

---

## 1. システム全体構成

```
[Streamlit :8501]                    [VSCode Claude Code チャット]
ビジュアル分析・6タブ+               対話的横断分析・MCP 直接呼出
       │                                         │ stdio (.mcp.json 設定)
       ↓ HTTPX                                   ↓
 ┌────────────────┐    ┌──────────────────────────────────────────────┐
 │  jpx500 :8001  │    │ MCP layer (6本):                              │
 │  +/moat-score  │←──┤  gdp / estat / mindmap                        │
 └───┬────────────┘    │  jpx500 / naibu / moat_score                  │
     │ HTTPX           └──────────────┬───────────────────────────────┘
     ↓                                │ HTTPX
 [naibu :8000]                        ↓
     │                [jpx500:8001] [naibu:8000] [mindmap:8003]
     ↓
 [data.db / naibu.db / moat_scores.parquet]
```

---

## 2. コンポーネント詳細

### 2.1 jpx500_wave_analysis (投資ハブ本体)

| コンポーネント | 技術 | 役割 |
|---|---|---|
| `app.py` | Streamlit | 全タブ UI |
| `api_server.py` | FastAPI + uvicorn :8001 | 波形・MoatScore・外国人フロー API |
| `modules/moat_score.py` | Python | MoatScoreEngine (7軸計算) |
| `modules/wave_classifier.py` | Python | テクニカル軸: 波形分類 |
| `modules/range_breakout_detector.py` | Python | テクニカル軸: ブレイクアウト検出 |
| `modules/trend_transition_detector.py` | Python | テクニカル軸: トレンド転換検出 |
| `modules/foreign_flow_analyzer.py` | Python | 外国人フロー分析 |
| `modules/capital_efficiency_screener.py` | Python | ファンダ軸: 資本効率スクリーナー |
| `mcp_server/server.py` | FastMCP stdio | jpx500-mcp (Phase 3) |
| `mcp_server/moat_score_server.py` | FastMCP stdio | moat-score-mcp (Phase 3) |
| `batch/update.py` | Python | 日次バッチ |
| `data/moat_scores.parquet` | Parquet | MoatScore 計算結果 (バッチ書込) |
| `data/policy_signals.json` | JSON | 政府政策シグナル (手動更新) |

### 2.2 naibu-ryuho-app (財務・PP・スクリーナー)

| コンポーネント | 技術 | 役割 |
|---|---|---|
| `api/main.py` | FastAPI + uvicorn :8000 | 財務・PP API |
| `api/routers/stocks.py` | FastAPI Router | 財務詳細 |
| `api/routers/screener.py` | FastAPI Router | スクリーナー |
| `api/routers/pricing_power.py` | FastAPI Router | PP スコア API |
| `scripts/utils/scoring.py` | Python | `compute_pricing_power` / `compute_macro_resilience` |
| `mcp_server/server.py` | FastMCP stdio | naibu-mcp (Phase 3) |

### 2.3 mindmap_and_mice (マインドマップ生成)

| コンポーネント | 技術 | 役割 |
|---|---|---|
| `main.py` | FastAPI :8003 | マインドマップ API (Phase 1 で 8001→8003 移行) |
| `mcp_server/server.py` | FastMCP stdio | mindmap-mcp (Phase 1) |

### 2.4 GDP_distribution / estat_mcp_server

既存 MCP サーバー (stdio)。Phase 1 から `.mcp.json` に登録して VSCode から呼ぶ。

---

## 3. データフロー

### 3.1 MoatScore 算出フロー (オンデマンド)

```
VSCode チャット or Streamlit
  → GET /api/moat-score/{code} (jpx500 :8001)
    → MoatScoreEngine.compute(code)
      ├── wave_classifier.classify(code)      [local]
      ├── range_breakout_detector.evaluate()  [local]
      ├── trend_transition_detector()         [local]
      ├── GET /companies/{code}/financials    [naibu :8000 via HTTPX]
      ├── GET /api/pricing-power/{code}       [naibu :8000 via HTTPX]
      ├── foreign_flow_analyzer()             [local]
      └── policy_signals.json 読み込み        [local file]
    → MoatScoreResult (JSON)
```

### 3.2 日次バッチフロー

```
daily_update.bat (18:00)
  → batch/update.py
    → 既存更新ステップ (CSV更新 etc.)
    → POST /api/moat-score/recompute (X-Recompute-Token ヘッダ付き)
      → MoatScoreEngine.compute_bulk(all_codes)
        → (3.1 フローを全銘柄分実行)
      → moat_scores.parquet 上書き
```

### 3.3 policy_signals.json 更新フロー

```
ユーザー (VSCode で月曜朝)
  → /policy-update スラッシュコマンド
    → mcp__web_search__ でニュース検索 (骨太の方針 / GX / 防衛 等)
    → Claude が固定スキーマで JSON 抽出
    → data/policy_signals.json 上書き
```

---

## 4. インターフェース定義

### 4.1 jpx500 FastAPI ↔ naibu FastAPI

| 呼び出し元 | エンドポイント | 用途 |
|---|---|---|
| MoatScoreEngine (軸2/4) | `GET http://localhost:8000/companies/{code}/financials` | ROE/ROA/CF/CAGR |
| MoatScoreEngine (軸6) | `GET http://localhost:8000/api/pricing-power/{code}` | PP スコア |

naibu コード形式: 5桁 EDINET (`{code}0`)。変換は naibu 側で実施。

**銘柄詳細 BS (直読み)**:  
銘柄詳細ビューの貸借対照表は naibu FastAPI を経由せず `modules/naibu_client.fetch_balance_sheet()` が  
SQLite `financial_metrics` を **read-only URI** で直接参照する。  
naibu API が `financial_metrics` の BS フィールドを公開していないためこの方式を採用。  
既存の `fetch_jpx500_naibu_data()` と同じ `_connect()` 読み取り専用パターンを踏襲し、書込みは行わない。

### 4.2 MCP ↔ FastAPI (薄ラッパ原則)

```python
# MCP ツール = HTTPX call のみ。ロジックは FastAPI 側に持つ。
@mcp.tool()
async def compute_moat_score(code: str) -> dict:
    r = await httpx.AsyncClient().get(
        f"http://localhost:8001/api/moat-score/{code}", timeout=10
    )
    return r.json()
```

### 4.3 moat_scores.parquet スキーマ

```
securities_code: str (4桁)
date: str (YYYY-MM-DD)
axis_technical: float64
axis_fundamental: float64
axis_foreign_flow: float64
axis_growth: float64
axis_growth_sector: float64
axis_moat_pp: float64
axis_policy: float64
total_score: float64
rank: int64
explanation: str (JSON 文字列)
```

---

## 5. セキュリティアーキテクチャ

### 5.1 書込操作の多段保護

```
[バッチ経路]    POST /recompute ──────── X-Recompute-Token 必須 (env var)
[MCP経路]      mcp__mindmap__save ────── permissions.ask (settings.json)
               mcp__moat_score__compute ─ permissions.ask (bulk のみ)
[Streamlit]    GET のみ ─────────────── recompute ボタン非設置
```

### 5.2 untrusted データ隔離

```
web_search-mcp
  ↓ (登録しない)
.mcp.json ←── gdp / estat / mindmap / jpx500 / naibu / moat_score のみ

/policy-update (VSCode で手動実行)
  ↓ ユーザーが目視レビュー
policy_signals.json ←── 唯一の untrusted → trusted 変換点
```

### 5.3 既存正本データの保護

各アプリの `data/`, `*.db`, `*.parquet` はアプリの所有のまま。  
MoatScore の書き込み先は `jpx500_wave_analysis/data/moat_scores.parquet` のみ (新規ファイル)。

---

## 6. 非機能アーキテクチャ

### 6.1 パフォーマンス

- ランキング API はバッチ計算済み parquet を読み込み (全銘柄オンデマンド計算を避ける)
- 単銘柄 Moat Score のみオンデマンド計算を許容
- naibu へのHTTPS 接続は `httpx.AsyncClient` で非同期化

### 6.2 障害許容

- naibu 未起動: `total_score=None` を返し、Streamlit は「naibu 起動してください」と表示
- mindmap 未起動: マインドマップタブに接続エラーメッセージを表示
- 各 MCP は独立 stdio プロセスのため、1 サーバ失敗が他に影響しない

### 6.3 開発環境

- devcontainer: Python 3.12 + Node.js (Claude Code 用)
- 統合 venv: `/workspaces/.venv`
- forwardPorts: 8000, 8001, 8003, 8501

---

## 7. 依存関係グラフ

```
app.py (Streamlit)
  └── api_server.py (HTTPX)
        └── modules/moat_score.py
              ├── modules/wave_classifier.py
              ├── modules/range_breakout_detector.py
              ├── modules/trend_transition_detector.py
              ├── modules/foreign_flow_analyzer.py
              ├── modules/capital_efficiency_screener.py
              ├── data/policy_signals.json
              └── naibu :8000 (HTTPX)

mcp_server/moat_score_server.py
  └── api_server.py (HTTPX :8001)

mcp_server/server.py (jpx500-mcp)
  └── api_server.py (HTTPX :8001)

naibu-ryuho-app/mcp_server/server.py
  └── naibu :8000 (HTTPX)
```

---

## 8. アーキテクチャ原則

1. **MCP は薄ラッパに徹する** — 業務ロジックは FastAPI/モジュールに持つ
2. **既存正本データに書き込まない** — MoatScore 結果は `moat_scores.parquet` に分離
3. **untrusted データ流入経路を限定** — web_search-mcp は常時登録しない
4. **ポート衝突を作らない** — mindmap は Phase 1 で 8003 に移行
5. **Streamlit 既存タブのロジックは無変更** — 新タブは追加のみ
6. **naibu 依存 60% を守る** — ファンダ/成長/PP は naibu HTTPX 経由
