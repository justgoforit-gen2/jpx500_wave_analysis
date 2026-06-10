# 設計書 — Phase 1: base-integration

**プロジェクト**: jpx500_wave_analysis  
**フェーズ**: phase1-base-integration  
**作成日**: 2026-06-09

---

## 1. mindmap ポート移行設計

### 変更ファイル: `mindmap_and_mice/main.py`

```python
# Before
PORT = 8001
URL = f"http://{HOST}:{PORT}"

# After
PORT = 8003
URL = f"http://{HOST}:{PORT}"
```

影響範囲: main.py のみ。src/ 内部は HOST/PORT を参照していない (uvicorn.run() に渡すだけ)。

---

## 2. FastAPI リスト API 設計

### 変更ファイル: `mindmap_and_mice/src/api/routes.py`

```python
SAVED_DIR = Path("/workspaces/dify_projects/mindmap_and_mice/saved")

@router.get("/list")
def list_saved() -> dict:
    if not SAVED_DIR.exists():
        return {"files": []}
    files = sorted(str(p) for p in SAVED_DIR.glob("*.mmd"))
    return {"files": files}
```

設計判断:
- saved ディレクトリは `/workspaces/dify_projects/mindmap_and_mice/saved/` に固定
- 環境変数 `MINDMAP_SAVED_DIR` で上書き可能にする
- 薄ラッパ原則: MCP は GET http://localhost:8003/api/list を呼ぶだけ

---

## 3. MCP サーバー設計

### 新規ファイル: `mindmap_and_mice/mcp_server/server.py`

```
mindmap_and_mice/
└── mcp_server/
    ├── __init__.py   (空)
    └── server.py     (FastMCP stdio)
```

#### 4 ツール定義

| ツール名 | HTTP 呼び出し | 権限 |
|---|---|---|
| `mindmap_expand` | POST http://localhost:{PORT}/api/expand | allow (読み取り相当) |
| `mindmap_to_mermaid` | POST http://localhost:{PORT}/api/to-mermaid | allow |
| `mindmap_list_saved` | GET http://localhost:{PORT}/api/list | allow |
| `mindmap_save` | POST http://localhost:{PORT}/api/save | ask (書き込み) |

#### FastMCP 実装パターン (薄ラッパ)

```python
import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mindmap")
PORT = int(os.getenv("MINDMAP_PORT", "8003"))
BASE_URL = f"http://localhost:{PORT}/api"

@mcp.tool()
async def mindmap_expand(parent_label: str, existing_siblings: list[str] | None = None, path_from_root: list[str] | None = None) -> dict:
    """ノードを展開して子ノード候補を返す"""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{BASE_URL}/expand", json={
            "parent_label": parent_label,
            "existing_siblings": existing_siblings or [],
            "path_from_root": path_from_root or [],
        })
        r.raise_for_status()
        return r.json()
```

#### settings.json 権限追記 (Phase 1 で追加)

```json
"permissions": {
  "allow": [
    "mcp__mindmap__mindmap_expand",
    "mcp__mindmap__mindmap_to_mermaid",
    "mcp__mindmap__mindmap_list_saved"
  ],
  "ask": [
    "mcp__mindmap__mindmap_save"
  ]
}
```

---

## 4. .mcp.json 更新設計

```json
{
  "mcpServers": {
    "gdp": { ... },
    "estat": { ... },
    "mindmap": {
      "command": "python",
      "args": ["/workspaces/dify_projects/mindmap_and_mice/mcp_server/server.py"]
    }
  }
}
```

---

## 5. policy_signals.json 設計

### 配置先: `jpx500_wave_analysis/data/policy_signals.json`

```json
{
  "version": "1.0",
  "updated_at": "2026-06-09",
  "signals": [
    {
      "policy_id": "GX-2026",
      "theme": "グリーン・トランスフォーメーション (GX)",
      "sector_tags": ["電力", "エネルギー", "自動車", "鉄鋼", "化学"],
      "strength": 3,
      "valid_until": "2027-03-31",
      "updated_at": "2026-06-09"
    },
    {
      "policy_id": "DX-2026",
      "theme": "デジタル・トランスフォーメーション (DX)",
      "sector_tags": ["情報通信", "電子部品", "システム"],
      "strength": 3,
      "valid_until": "2027-03-31",
      "updated_at": "2026-06-09"
    },
    {
      "policy_id": "DEF-2026",
      "theme": "防衛・安全保障強化",
      "sector_tags": ["防衛", "航空", "電子機器", "造船"],
      "strength": 3,
      "valid_until": "2030-03-31",
      "updated_at": "2026-06-09"
    },
    {
      "policy_id": "CHILD-2026",
      "theme": "少子化対策・子育て支援",
      "sector_tags": ["教育", "医療", "住宅", "流通"],
      "strength": 2,
      "valid_until": "2027-03-31",
      "updated_at": "2026-06-09"
    },
    {
      "policy_id": "INFRA-2026",
      "theme": "インフラ老朽化対策・国土強靭化",
      "sector_tags": ["建設", "セメント", "鉄鋼", "機械"],
      "strength": 2,
      "valid_until": "2030-03-31",
      "updated_at": "2026-06-09"
    }
  ]
}
```

strength 定義: 0=無効, 1=弱, 2=中, 3=強 (MoatScore 政策軸の倍率に使用)
