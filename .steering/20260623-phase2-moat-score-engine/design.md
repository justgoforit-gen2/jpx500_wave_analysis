---
phase: 2
title: MoatScoreEngine 設計書
---

# Phase 2 詳細設計

## 1. evaluate() 公開ラッパ (`range_breakout_detector.py`)

```python
def evaluate(code: str, ohlcv: pd.DataFrame) -> dict | None:
    """_evaluate の公開ラッパ。MoatScoreEngine から呼ぶ用。"""
    return _evaluate(ohlcv)
```

## 2. MoatScoreEngine クラス設計

### ファイル: `modules/moat_score.py`

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from config.settings import DATA_DIR, NAIBU_API_BASE_URL, NAIBU_FETCH_TIMEOUT_SEC
from modules.capital_efficiency_screener import load_screening_result
from modules.foreign_flow_analyzer import compute_cumulative_flow, load_foreign_flow
from modules.naibu_client import naibu_health_check, to_edinet_securities_code
from modules.range_breakout_detector import evaluate as rb_evaluate

POLICY_SIGNALS_PATH = DATA_DIR / "policy_signals.json"
MOAT_SCORES_PARQUET = DATA_DIR / "moat_scores.parquet"

WEIGHTS = {
    "technical": 0.10,
    "fundamental": 0.25,
    "foreign_flow": 0.10,
    "growth": 0.15,
    "growth_sector": 0.10,
    "moat_pp": 0.20,
    "policy": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

class MoatScoreEngine:
    def compute(self, code: str) -> dict[str, Any]:
        ...
```

### 各軸の計算ロジック

**軸1 technical (10%)**
- `results.csv` から code の wave_type を取得
- wave_type → スコアマッピング (例: `stage2_breakout`=9, `uptrend`=7, `range`=5, `downtrend`=2, `unknown`=4)
- データなし → 5.0 (中央値)

**軸2 fundamental (25%)**
- `load_screening_result()` を呼んで CES DataFrame を取得
- code に対応する行の `total_score` (0-10) を直接使用
- データなし → None (フォールバックしない: CES は jpx500 内部データ)

**軸3 foreign_flow (10%)**
- `load_foreign_flow()` → `compute_cumulative_flow()` で直近4週 (20営業日) net 累積
- 累積が正 → 分位数で 5-10 にスケール、負 → 0-5
- parquet なし → 5.0

**軸4 growth (15%)**
- `naibu_client.fetch_jpx500_naibu_data()` で net_income を使う（SQLite直読み）
- 2期分のデータから成長率を計算（`net_income_fy` で期比較）
- naibu DB なし → None

**軸5 growth_sector (10%) / 軸7 policy (10%)**
- `policy_signals.json` をロードし `sector_tags` リスト取得
- naibu_client の industry_name と照合 (部分一致)
- マッチした政策の strength 合計 / 最大値でスコア化
- JSON なし → 5.0

**軸6 moat_pp (20%)**
- naibu ヘルスチェック → OK なら HTTPX で `/api/pricing-power/companies/{edinet_code}`
- `pricing_power` フィールド (0.0-1.0) を × 10 でスケール
- naibu 落ち → None

### fallback ロジック
```python
naibu_axes = {"growth": ..., "moat_pp": ...}
if any(v is None for v in naibu_axes.values()):
    errors = ["naibu API unreachable"]
    total_score = None
else:
    total_score = sum(score * WEIGHTS[k] for k, score in all_scores.items())
```

### 出力スキーマ
```python
{
    "securities_code": "72030",  # 5桁
    "code": "7203",              # 4桁
    "date": "2026-06-23",
    "axis_technical": 7.5,
    "axis_fundamental": 8.2,
    "axis_foreign_flow": 6.0,
    "axis_growth": 7.0,          # None if naibu down
    "axis_growth_sector": 5.0,
    "axis_moat_pp": 9.0,         # None if naibu down
    "axis_policy": 4.0,
    "total_score": 7.30,         # None if naibu down
    "rank": None,                # バッチで後付け
    "explanation": {
        "technical": "Stage2 breakout検出 (range_breakout)",
        "fundamental": "CES total_score=8.2",
        "foreign_flow": "直近4週累積フロー: +1200億円",
        "growth": "net_income 前期比 +12%",
        "growth_sector": "DX-2026/GX-2026 セクタータグ一致",
        "moat_pp": "pricing_power=0.85",
        "policy": "GX-2026 strength=3 一致",
        "errors": []
    }
}
```

## 3. API エンドポイント設計 (`api_server.py`)

### GET /api/moat-score/{code}
```python
@app.get("/api/moat-score/{code}", tags=["moat"])
def moat_score_one(code: str) -> dict:
    # parquet があれば最新行を返す
    # なければ MoatScoreEngine.compute(code) をオンデマンド実行
```

### GET /api/moat-score/ranking
```python
@app.get("/api/moat-score/ranking", tags=["moat"])
def moat_score_ranking(
    top: int = Query(20, ge=1, le=500),
    sector: str | None = Query(None),
) -> list[dict]:
    # moat_scores.parquet を読み込んで top N 返却
    # parquet なし → 503
```

### POST /api/moat-score/recompute
```python
@app.post("/api/moat-score/recompute", tags=["moat"])
def moat_score_recompute(
    request: Request,
    codes: list[str] | None = None,
) -> dict:
    token = request.headers.get("X-Recompute-Token")
    if token != os.getenv("RECOMPUTE_TOKEN", ""):
        raise HTTPException(status_code=403, detail="X-Recompute-Token required")
    # バッチ実行 → moat_scores.parquet 更新
```

### GET /api/foreign-flow/{code}
```python
@app.get("/api/foreign-flow/{code}", tags=["flow"])
def foreign_flow(code: str) -> dict:
    # foreign_flow_analyzer 使って直近4週 net 返却
```

## 4. Streamlit タブ設計 (`app.py`)

既存の `st.sidebar.radio` に 2 タブを追加:

```python
elif selected == "Moat Score":
    _tab_moat_score()

elif selected == "ランキング":
    _tab_moat_ranking()
```

### _tab_moat_score()
- 銘柄コード入力 → `GET /api/moat-score/{code}` 呼び出し
- plotly `go.Scatterpolar` でレーダーチャート (7軸)
- st.metric で総合スコア表示
- st.expander で軸別説明

### _tab_moat_ranking()
- `GET /api/moat-score/ranking?top=20` 呼び出し
- st.dataframe (ソート可能)
- sector フィルタ (selectbox)
- CSV ダウンロードボタン
- 行クリック → st.session_state でMoat Scoreタブに銘柄コードをセット

## 5. テスト設計 (`tests/test_moat_score.py`)

```python
def test_moat_score_7203():
    engine = MoatScoreEngine()
    result = engine.compute("7203")
    assert 0 <= result["axis_technical"] <= 10
    assert result["axis_fundamental"] is not None

def test_moat_score_naibu_fallback(monkeypatch):
    monkeypatch.setattr("modules.naibu_client.naibu_health_check", lambda: False)
    engine = MoatScoreEngine()
    result = engine.compute("7203")
    assert result["total_score"] is None
    assert "naibu API unreachable" in result["explanation"]["errors"]

def test_weights_sum():
    from modules.moat_score import WEIGHTS
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
```
