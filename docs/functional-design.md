# Functional Design — 投資判断ハブ「バフェット流 Moat Score エンジン」

**プロジェクト**: jpx500_wave_analysis  
**バージョン**: 1.0  
**作成日**: 2026-06-09  
**ステータス**: 承認済み

---

## 1. 機能一覧

| 機能 ID | 機能名 | Phase | 優先度 |
|---|---|---|---|
| F-01 | MoatScoreEngine (7軸計算) | 2 | Must |
| F-02 | FastAPI moat-score エンドポイント群 | 2 | Must |
| F-03 | Streamlit「Moat Score」タブ | 2 | Must |
| F-04 | Streamlit「ランキング」タブ | 2 | Must |
| F-05 | Streamlit「外国人フロー」タブ | 2 | Should |
| F-06 | Streamlit「政府政策」タブ | 2 | Should |
| F-07 | Streamlit「結論マインドマップ」タブ | 2 | Should |
| F-08 | mindmap 8001→8003 移行 + mindmap-mcp | 1 | Must |
| F-09 | jpx500-mcp 実装 | 3 | Must |
| F-10 | naibu-mcp 実装 | 3 | Must |
| F-11 | moat-score-mcp 実装 | 3 | Must |
| F-12 | `.mcp.json` 登録 (6サーバ) | 1-3 | Must |
| F-13 | `policy_signals.json` 初版 + `/policy-update` | 1 | Must |
| F-14 | 日次バッチ MoatScore 計算 | 4 | Must |
| F-15 | Streamlit「決算予定」タブ + Top10 枠 + 鮮度インジケータ | 4 | Should |

---

## 2. F-01: MoatScoreEngine

### 2.1 クラス設計

```python
# jpx500_wave_analysis/modules/moat_score.py

class MoatScoreResult:
    securities_code: str      # 4桁 (jpx500 形式)
    date: str                  # YYYY-MM-DD
    axis_technical: float | None
    axis_fundamental: float | None
    axis_foreign_flow: float | None
    axis_growth: float | None
    axis_growth_sector: float | None
    axis_moat_pp: float | None
    axis_policy: float | None
    total_score: float | None  # None = naibu 未起動 or データ不足
    rank: int | None
    explanation: dict[str, str]  # axis -> reason_text

class MoatScoreEngine:
    WEIGHTS = {
        "technical": 0.10,
        "fundamental": 0.25,
        "foreign_flow": 0.10,
        "growth": 0.15,
        "growth_sector": 0.10,
        "moat_pp": 0.20,
        "policy": 0.10,
    }  # sum == 1.00

    def compute(self, code: str) -> MoatScoreResult: ...
    def compute_bulk(self, codes: list[str]) -> list[MoatScoreResult]: ...
```

### 2.2 軸別算出ロジック

#### 軸 1: テクニカル (10%)

```
score = mean([
    wave_score,        # classify() → ラベルを 0-10 にマッピング
    breakout_score,    # evaluate(code, ohlcv) → 0-10
    trend_score,       # detect_transitions() → 直近シグナル強度 0-10
])
```

波形ラベル → スコアマッピング表:

| wave_type | score |
|---|---|
| 上昇トレンド | 9 |
| ブレイク気配 | 7 |
| 保ち合い | 5 |
| 下落トレンド | 2 |
| その他 | 4 |

#### 軸 2: ファンダ (25%)

naibu `GET /companies/{code}/financials` + `capital_efficiency_screener` から:
- ROE ≥ 15% → +3 pt
- ROA ≥ 8% → +2 pt
- 自己資本比率 ≥ 40% → +2 pt
- 有利子負債/EBITDA ≤ 3 → +3 pt
満点 10 pt、各 None は 0 pt として計算

#### 軸 3: 外国人フロー (10%)

`foreign_flow_analyzer.compute_sector_flow_correlation()` の直近4週累積を 0-10 に正規化  
(業種全体の最大累積を 10 とした線形スケール)

#### 軸 4: 成長長期 (15%)

naibu 財務データから:
- 売上 CAGR(5y) ≥ 10% → 10; ≥ 5% → 7; ≥ 0% → 4; < 0% → 1
- 営業益 CAGR(5y): 同上スケール
- 両者の平均

#### 軸 5: 成長分野 (10%)

`policy_signals.json` の該当銘柄業種タグ `strength` (0-3) を `strength * 3.33` で 0-10 換算  
業種タグが見つからない場合: 5 (中立)

#### 軸 6: 産業障壁/PP (20%)

naibu `GET /api/pricing-power/{code}` の `pp_score` (0-10) をそのまま使用  
naibu 未起動: `None`

#### 軸 7: 政府骨太政策 (10%)

`data/policy_signals.json` から銘柄の業種に一致する政策テーマの `strength` を 0-10 換算  
軸 5 と同一のファイルだが、政策テーマ (policy_id) 単位で取得する点が異なる

### 2.3 フォールバック仕様

```
naibu 依存軸 (fundamental/growth/moat_pp) が 1 つでも None
→ total_score = None
→ explanation.errors = ["naibu API unreachable"]
```

naibu 到達チェック: `GET /health` を 3 秒でタイムアウト

---

## 3. F-02: FastAPI エンドポイント

### `GET /api/moat-score/{code}`

レスポンス: `MoatScoreResult` (JSON)

```json
{
  "securities_code": "7203",
  "date": "2026-06-09",
  "axis_technical": 7.5,
  "axis_fundamental": 8.2,
  "axis_foreign_flow": 6.0,
  "axis_growth": 7.0,
  "axis_growth_sector": 5.0,
  "axis_moat_pp": 9.0,
  "axis_policy": 4.0,
  "total_score": 7.30,
  "rank": 12,
  "explanation": {
    "technical": "上昇トレンド継続。ブレイクアウト圏内。",
    "fundamental": "ROE 18%, ROA 9%, 自己資本比率 42%。高収益体質。",
    "moat_pp": "PP スコア 9.0 — 強いブランドと価格転嫁力。"
  }
}
```

### `GET /api/moat-score/ranking`

クエリパラメータ: `top` (int, default=20), `sector` (str, optional)

レスポンス: `list[MoatScoreResult]` (total_score 降順)

parquet キャッシュを読み込み (バッチが書き込んだもの)、キャッシュ未存在時はオンデマンド計算

### `POST /api/moat-score/recompute`

ヘッダ: `X-Recompute-Token: <env:RECOMPUTE_TOKEN>`  
ヘッダ欠如または不一致 → 403  
処理: 全 JPX500 銘柄を `compute_bulk()` → `moat_scores.parquet` 上書き  
レスポンス: `{"computed": N, "elapsed_sec": N}`

### `GET /api/foreign-flow/{code}`

`foreign_flow_analyzer` + `jpx_investor_flow_fetcher` の結果を JSON で返す  
既存 e2e スクリプトで使われているデータを API 化したもの

---

## 4. F-03〜F-07: Streamlit タブ

### Streamlit タブ追加方針

`app.py` の `st.sidebar.radio` で既存 5 タブの後に追記。既存タブのロジックは無変更。

### F-03: Moat Score タブ

1. 銘柄コード入力 (テキストボックス)
2. `GET /api/moat-score/{code}` 呼び出し
3. plotly レーダーチャート (7軸、0-10 スケール)
4. 総合点ゲージ (plotly.graph_objects.Indicator)
5. 軸別説明 (st.expander 7個)
6. `total_score = None` の場合: 「naibu API が応答しません。起動してから再試行してください。」

### F-04: ランキングタブ

1. `GET /api/moat-score/ranking?top=50` 呼び出し
2. sector フィルタ (st.selectbox)
3. `st.dataframe` (total_score 降順、列: rank/code/名称/total_score/各軸)
4. CSV ダウンロードボタン
5. 行クリック → Moat Score タブへ遷移 (st.session_state でコード引き継ぎ)
6. Phase 4 追加: 冒頭に「今朝の Top10」ハイライト枠 (parquet の最新 date × top 10)

### F-05: 外国人フロー タブ

- `foreign_flow_analyzer.compute_sector_flow_correlation()` の結果を業種×週次ヒートマップで表示
- 既存 `e2e_foreign_flow_visual.py` の plotly 実装を関数化して流用

### F-06: 政府政策タブ

- `data/policy_signals.json` を読み込み、政策テーマカードを表示
- カードクリック → 関連業種銘柄を MoatScore 順に一覧表示

### F-07: 結論マインドマップタブ

- mindmap API (`:8003/api/expand`) に銘柄コードと分析結果を送信
- `/api/to-mermaid` で Mermaid 記法に変換
- `streamlit-mermaid` でレンダリング
- 「保存」ボタン → `POST /api/save` (mindmap_and_mice の MCP ツール `save` 経由)

---

## 5. F-08: mindmap 8001 → 8003 移行

`mindmap_and_mice/main.py` の `PORT = 8001` を `PORT = 8003` に変更。  
`mindmap_and_mice/mcp_server/server.py` 新規作成 (FastMCP stdio ラッパ)。

ツール仕様:

| ツール名 | 引数 | 返値 |
|---|---|---|
| `mindmap_expand` | `topic: str, context: str` | `{nodes: [...], edges: [...]}` |
| `to_mermaid` | `graph: dict` | `{mermaid: str}` |
| `save` | `code: str, mermaid: str` | `{path: str}` |
| `list_saved` | `query: str = ""` | `[{name, path, created_at}]` |

---

## 6. F-09〜F-11: MCP サーバー設計

### 薄ラッパ原則

業務ロジックは MCP に書かず、既存 FastAPI/モジュールに HTTPX で委譲する。

```python
# 例: jpx500-mcp
@mcp.tool()
async def get_picks_today(top: int = 10) -> list[dict]:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"http://localhost:8001/api/abcd?top={top}", timeout=10)
        r.raise_for_status()
        return r.json()
```

### F-09: jpx500-mcp ツール一覧

| ツール | 委譲先 API |
|---|---|
| `jpx500_get_wave` | `GET /api/wave/{code}` |
| `get_picks_today` | `GET /api/abcd?top=N` |
| `get_abcd_ranking` | `GET /api/abcd` |
| `get_prices` | `GET /api/prices/{code}` |
| `get_earnings` | `GET /api/earnings/{code}` |
| `get_foreign_flow` | `GET /api/foreign-flow/{code}` |

### F-10: naibu-mcp ツール一覧

| ツール | 委譲先 API |
|---|---|
| `naibu_get_company` | `GET /companies/{code}` |
| `get_financials` | `GET /companies/{code}/financials` |
| `screen` | `GET /api/screen` |
| `pricing_power` | `GET /api/pricing-power/{code}` |
| `activist_screen` | `GET /api/activist` |
| `industry_aggregate` | `GET /api/industry` |

### F-11: moat-score-mcp ツール一覧

| ツール | 委譲先 API |
|---|---|
| `compute_moat_score` | `GET /api/moat-score/{code}` |
| `rank_by_moat` | `GET /api/moat-score/ranking` |
| `explain_score` | `GET /api/moat-score/{code}` の explanation フィールド |
| `list_policy_signals` | `data/policy_signals.json` を直読み |

---

## 7. F-13: policy_signals.json 仕様

```json
[
  {
    "policy_id": "GX-2026",
    "theme": "GX・脱炭素",
    "sector_tags": ["電気機器", "化学", "電力・ガス"],
    "strength": 3,
    "valid_until": "2027-03-31",
    "updated_at": "2026-06-09"
  }
]
```

初版は 5 政策テーマ手動作成。`/policy-update` スラッシュコマンドで月次更新。

---

## 8. F-14: 日次バッチ

`batch/update.py` に `compute_bulk()` ステップ追加:

```python
# Phase 4 追加部分 (既存の daily update ロジックの後に追記)
def step_moat_score_recompute():
    token = os.environ["RECOMPUTE_TOKEN"]
    r = httpx.post(
        "http://localhost:8001/api/moat-score/recompute",
        headers={"X-Recompute-Token": token},
        timeout=1200,
    )
    r.raise_for_status()
    log(f"MoatScore recompute: {r.json()}")
```

---

## 9. 画面遷移

```
Streamlit サイドバー選択
├── ポートフォリオ (既存)
├── 波形分析 (既存)
├── ABCD ランキング (既存)
├── バックテスト (既存)
├── 最適化 (既存)
├── Moat Score ──→ [銘柄選択] ──→ レーダー + 軸説明
├── ランキング ──→ [行クリック] ──→ Moat Score タブ
├── 外国人フロー
├── 政府政策 ──→ [テーマクリック] ──→ 関連銘柄一覧
├── 結論マインドマップ ──→ [保存ボタン] ──→ mindmap_and_mice/saved/
└── 決算予定 (Phase 4)
```

---

## 10. 受け入れ条件 (機能レベル)

- [ ] `MoatScoreEngine.compute("7203")` が 3 秒以内に `MoatScoreResult` を返す
- [ ] 重み合計 `sum(WEIGHTS.values()) == 1.00` を assert で保証
- [ ] naibu 停止時に `total_score = None` + `errors = ["naibu API unreachable"]` が返る
- [ ] Streamlit Moat タブでレーダーチャートが表示される
- [ ] ランキングタブで行クリック → Moat タブに銘柄コードが引き継がれる
- [ ] Streamlit から `POST /api/moat-score/recompute` が呼ばれない (grep で確認)
- [ ] mindmap の保存ボタンが permission prompt を出して承認後に `.mmd` が生成される
