"""jpx500-mcp — JPX500 波形分析 API への薄いラッパ MCP サーバー。

jpx500_wave_analysis FastAPI (:8001) を HTTPX 経由で呼ぶだけ。業務ロジックは持たない。

起動: python /workspaces/dify_projects/jpx500_wave_analysis/mcp_server/server.py
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("jpx500")

PORT = int(os.getenv("JPX500_PORT", "8001"))
BASE_URL = f"http://localhost:{PORT}/api"
_TIMEOUT = 30


@mcp.tool()
async def get_wave(code: str) -> dict:
    """1銘柄の最新波形分類を返す。GET /api/wave/{code} を呼ぶ。"""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{BASE_URL}/wave/{code}")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def get_picks_today() -> list:
    """本日の推奨銘柄を返す。GET /api/picks/today を呼ぶ。"""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{BASE_URL}/picks/today")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def get_abcd_ranking() -> list:
    """ABCD戦略ランキングを返す。GET /api/abcd-ranking を呼ぶ。"""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{BASE_URL}/abcd-ranking")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def get_prices(code: str, days: int = 60) -> list:
    """銘柄の OHLCV 履歴を返す。GET /api/prices/{code}?days=N を呼ぶ。"""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{BASE_URL}/prices/{code}", params={"days": days})
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def get_earnings() -> list:
    """決算予定日一覧を返す。GET /api/earnings を呼ぶ。"""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{BASE_URL}/earnings")
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def get_foreign_flow(code: str) -> dict:
    """外国人投資家フロー情報を返す。GET /api/foreign-flow/{code} を呼ぶ。"""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(f"{BASE_URL}/foreign-flow/{code}")
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run()
