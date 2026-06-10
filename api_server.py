"""JPX500 波形分析データの薄いHTTP公開API。

naibu-ryuho-app から HTTP で叩かれ、波形分類・推奨銘柄・株価OHLCV等を返す。
データの正本は本リポジトリ内の data/*.csv と data/cache/*.parquet。

起動:
    uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload

エンドポイント:
    GET /api/health
    GET /api/jpx500-list                          -- JPX500 構成銘柄
    GET /api/wave/{code}                          -- 1銘柄の最新波形分類
    GET /api/wave/bulk?codes=1332,6857            -- 複数銘柄まとめ取得
    GET /api/picks/today                          -- 本日の推奨銘柄
    GET /api/abcd-ranking                         -- ABCD戦略ランキング
    GET /api/prices/{code}?days=60                -- OHLCV履歴
    GET /api/earnings                             -- 決算予定日
    GET /api/moat-score/{code}                    -- 1銘柄 MoatScore
    GET /api/moat-score/ranking?top=N&sector=X   -- MoatScore ランキング
    POST /api/moat-score/recompute                -- バッチ再計算 (X-Recompute-Token 必須)
    GET /api/foreign-flow/{code}                  -- 外国人フロー (Phase 3 jpx500-mcp 用)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    DAILY_PICKS_CSV,
    DATA_DIR,
    EARNINGS_COMBINED_CSV,
    RESULTS_CSV,
    STOCK_LIST_CSV,
)
from modules.data_fetcher import load_cached

ABCD_RANKING_CSV = DATA_DIR / "abcd_ranking.csv"

app = FastAPI(
    title="JPX500 波形分析 API",
    version="0.1.0",
    description="naibu-ryuho-app から消費される、jpx500分析データのHTTP公開API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://host.docker.internal:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"source not found: {path.name}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})


def _records(df: pd.DataFrame) -> list[dict]:
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.get("/api/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jpx500-list", tags=["universe"])
def jpx500_list() -> list[dict]:
    return _records(_read_csv(STOCK_LIST_CSV))


@app.get("/api/wave/bulk", tags=["wave"])
def wave_bulk(
    codes: str = Query(..., description="comma-separated codes, e.g. 1332,6857"),
) -> list[dict]:
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    if not code_list:
        return []
    df = _read_csv(RESULTS_CSV)
    return _records(df[df["code"].isin(code_list)])


@app.get("/api/wave/{code}", tags=["wave"])
def wave_one(code: str) -> dict:
    df = _read_csv(RESULTS_CSV)
    row = df[df["code"] == code]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"code {code} not found in results")
    return _records(row)[0]


@app.get("/api/picks/today", tags=["picks"])
def picks_today() -> list[dict]:
    return _records(_read_csv(DAILY_PICKS_CSV))


@app.get("/api/abcd-ranking", tags=["abcd"])
def abcd_ranking() -> list[dict]:
    return _records(_read_csv(ABCD_RANKING_CSV))


@app.get("/api/prices/{code}", tags=["prices"])
def prices(
    code: str,
    days: int = Query(60, ge=1, le=2000),
    ticker: str | None = Query(
        None, description="override ticker, defaults to {code}.T"
    ),
) -> list[dict]:
    t = ticker or f"{code}.T"
    df = load_cached(t)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"price cache not found for {t}")
    df = df.tail(days).reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    return _records(df)


@app.get("/api/earnings", tags=["earnings"])
def earnings() -> list[dict]:
    return _records(_read_csv(EARNINGS_COMBINED_CSV))


# ── MoatScore エンドポイント ──────────────────────────────────────────────────


@app.get("/api/moat-score/ranking", tags=["moat"])
def moat_score_ranking(
    top: int = Query(20, ge=1, le=500),
    sector: str | None = Query(None, description="業種フィルタ (部分一致)"),
) -> list[dict]:
    """moat_scores.parquet から上位 N 件を返す。parquet がなければ 503。"""
    from modules.moat_score import load_moat_scores

    df = load_moat_scores()
    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail="moat_scores.parquet not found. Run /api/moat-score/recompute first.",
        )
    if sector:
        df = df[df["code"].isin(_sector_codes(sector))]
    df = df.sort_values("total_score", ascending=False).head(top)
    return _records(df)


@app.get("/api/moat-score/{code}", tags=["moat"])
def moat_score_one(code: str) -> dict:
    """1銘柄の MoatScore をオンデマンド計算して返す。"""
    from modules.moat_score import MoatScoreEngine, load_moat_scores

    df = load_moat_scores()
    if df is not None and not df.empty:
        row = df[df["code"] == code]
        if not row.empty:
            return _records(row.tail(1))[0]
    engine = MoatScoreEngine()
    return engine.compute(code)


@app.post("/api/moat-score/recompute", tags=["moat"])
def moat_score_recompute(request: Request, codes: list[str] | None = None) -> dict:
    """全銘柄 or 指定銘柄の MoatScore を再計算して parquet に保存する。
    X-Recompute-Token ヘッダが必須 (環境変数 RECOMPUTE_TOKEN と照合)。
    """
    token = request.headers.get("X-Recompute-Token", "")
    expected = os.getenv("RECOMPUTE_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="X-Recompute-Token required")

    from modules.moat_score import MoatScoreEngine, save_moat_scores

    if codes is None:
        stock_df = _read_csv(STOCK_LIST_CSV)
        codes = list(stock_df["code"].dropna().astype(str).unique())

    engine = MoatScoreEngine()
    results = engine.compute_bulk(codes)
    save_moat_scores(results)
    scored = sum(1 for r in results if r["total_score"] is not None)
    return {"status": "ok", "total": len(results), "scored": scored}


@app.get("/api/foreign-flow/{code}", tags=["flow"])
def foreign_flow_code(code: str) -> dict:
    """銘柄コードに対応する外国人フロー情報を返す (Phase 3 jpx500-mcp 用)。
    外国人フローは市場全体のデータのため、コードは metadata として返す。
    """
    from modules.foreign_flow_analyzer import (
        compute_cumulative_flow,
        load_foreign_flow,
    )

    ff_df = load_foreign_flow()
    if ff_df is None or ff_df.empty:
        raise HTTPException(status_code=503, detail="foreign_flow.parquet not found")
    cum = compute_cumulative_flow(ff_df)
    recent_net = float(cum.iloc[-1]) - float(cum.iloc[-min(21, len(cum))])
    return {
        "code": code,
        "market": "TSE Prime",
        "cumulative_flow_4w_oku": round(recent_net, 2),
        "cumulative_total_oku": round(float(cum.iloc[-1]), 2),
        "data_points": len(cum),
    }


def _sector_codes(sector_keyword: str) -> list[str]:
    """results.csv から sector_33 に keyword が含まれる銘柄コードを返す。"""
    if not RESULTS_CSV.exists():
        return []
    df = pd.read_csv(RESULTS_CSV, dtype={"code": str})
    if "sector_33" not in df.columns:
        return list(df["code"].unique())
    return list(
        df[df["sector_33"].str.contains(sector_keyword, na=False)]["code"].unique()
    )
