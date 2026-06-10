"""moat-score-mcp — MoatScore エンジンへの薄いラッパ MCP サーバー。

MoatScoreEngine と policy_signals.json を直接参照する。
rank_by_moat のみ jpx500 API (:8001) を呼ぶ。

起動: python /workspaces/dify_projects/jpx500_wave_analysis/mcp_server/moat_score_server.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mcp = FastMCP("moat_score")

PORT = int(os.getenv("JPX500_PORT", "8001"))
BASE_URL = f"http://localhost:{PORT}/api"
_TIMEOUT = 30


@mcp.tool()
async def compute_moat_score(code: str) -> dict:
    """1銘柄の MoatScore を 7軸で算出して返す。MoatScoreEngine を直接呼ぶ。
    naibu API が落ちている場合 total_score は None。
    code は 4桁証券コード (例: '7203')。
    """
    from modules.moat_score import MoatScoreEngine

    engine = MoatScoreEngine()
    return engine.compute(code)


@mcp.tool()
async def rank_by_moat(top: int = 20, sector: Optional[str] = None) -> list:
    """MoatScore 上位 N 社を返す。GET /api/moat-score/ranking を呼ぶ。
    parquet がない場合は 503 エラー。
    """
    params: dict = {"top": top}
    if sector is not None:
        params["sector"] = sector
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{BASE_URL}/moat-score/ranking", params=params)
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def explain_score(code: str) -> dict:
    """1銘柄の MoatScore 軸別説明を返す。MoatScoreEngine の explanation フィールドを返す。"""
    from modules.moat_score import MoatScoreEngine

    engine = MoatScoreEngine()
    result = engine.compute(code)
    return {
        "code": result["code"],
        "total_score": result["total_score"],
        "explanation": result["explanation"],
    }


@mcp.tool()
async def list_policy_signals() -> list:
    """policy_signals.json の全政策シグナル一覧を返す。"""
    from modules.moat_score import POLICY_SIGNALS_PATH

    if not POLICY_SIGNALS_PATH.exists():
        return []
    with POLICY_SIGNALS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("signals", [])


if __name__ == "__main__":
    mcp.run()
