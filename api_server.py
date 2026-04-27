"""JPX500 波形分析データの薄いHTTP公開API。

naibu-ryuho-app から HTTP で叩かれ、波形分類・推奨銘柄・株価OHLCV等を返す。
データの正本は本リポジトリ内の data/*.csv と data/cache/*.parquet。

起動:
    uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload

エンドポイント:
    GET /api/health
    GET /api/jpx500-list                  -- JPX500 構成銘柄
    GET /api/wave/{code}                  -- 1銘柄の最新波形分類
    GET /api/wave/bulk?codes=1332,6857    -- 複数銘柄まとめ取得
    GET /api/picks/today                  -- 本日の推奨銘柄
    GET /api/abcd-ranking                 -- ABCD戦略ランキング
    GET /api/prices/{code}?days=60        -- OHLCV履歴
    GET /api/earnings                     -- 決算予定日
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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
