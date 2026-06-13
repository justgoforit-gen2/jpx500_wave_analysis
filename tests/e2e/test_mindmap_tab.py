"""F-07 結論マインドマップタブ — E2E テスト。

テスト構成:
  - ユニットテスト: _build_mindmap_from_moat() の出力構造を検証
  - 統合テスト: mindmap API の context フィールド拡張を検証
  - Playwright テスト: Streamlit UI でタブが表示されることを検証
"""
from __future__ import annotations

import sys
import os

import httpx
import pytest

# jpx500_wave_analysis をパスに追加
_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

MINDMAP_API = "http://localhost:8003/api"

# ── ユニットテスト ──────────────────────────────────────────────────


def _make_dummy_result(total_score: float | None = 7.5) -> dict:
    return {
        "code": "7203",
        "securities_code": "72030",
        "date": "2026-06-11",
        "total_score": total_score,
        "axis_technical": 7.5,
        "axis_fundamental": 8.0,
        "axis_foreign_flow": 6.0,
        "axis_growth": 5.0,
        "axis_growth_sector": 7.0,
        "axis_moat_pp": 8.5,
        "axis_policy": 6.5,
        "explanation": {
            "technical": "wave_types score=7.5",
            "fundamental": "CES score=8.0",
            "foreign_flow": "flow=6.0",
            "growth": "cagr=5.0",
            "growth_sector": "sector=7.0",
            "moat_pp": "pp=8.5",
            "policy": "policy=6.5",
            "errors": "",
        },
    }


def test_build_mindmap_structure():
    """_build_mindmap_from_moat が正しい MindMap JSON を返す。"""
    # app モジュールのインポートを回避して関数だけテスト
    # app.py が Streamlit 依存をトップレベルで import するため
    # 関数を単独抽出してテスト
    import importlib.util
    import types

    # app.py から _build_mindmap_from_moat だけを文字列 exec で取り出す
    app_path = os.path.join(_APP_DIR, "app.py")
    with open(app_path, encoding="utf-8") as f:
        source = f.read()

    # 関数定義のみ抽出
    start = source.find("def _build_mindmap_from_moat(")
    end = source.find("\ndef _tab_mindmap_conclusion(")
    assert start != -1 and end != -1

    func_source = source[start:end]
    ns: dict = {}
    exec(func_source, ns)  # noqa: S102
    _build = ns["_build_mindmap_from_moat"]

    result = _make_dummy_result(total_score=7.5)
    mindmap = _build("7203", result)

    assert mindmap["title"] == "7203 投資結論マップ"
    root = mindmap["root"]
    assert root["id"] == "root"
    assert "7203" in root["label"]
    assert "7.5" in root["label"]
    assert len(root["children"]) == 7

    axis_ids = [c["id"] for c in root["children"]]
    for expected in ["ax_technical", "ax_fundamental", "ax_foreign_flow",
                     "ax_growth", "ax_growth_sector", "ax_moat_pp", "ax_policy"]:
        assert expected in axis_ids

    # スコアがラベルに含まれること
    tech_node = next(c for c in root["children"] if c["id"] == "ax_technical")
    assert "7.5" in tech_node["label"]


def test_build_mindmap_none_total_score():
    """total_score が None の場合でも MindMap が生成される (naibu 未起動時)。"""
    app_path = os.path.join(_APP_DIR, "app.py")
    with open(app_path, encoding="utf-8") as f:
        source = f.read()
    start = source.find("def _build_mindmap_from_moat(")
    end = source.find("\ndef _tab_mindmap_conclusion(")
    func_source = source[start:end]
    ns: dict = {}
    exec(func_source, ns)  # noqa: S102
    _build = ns["_build_mindmap_from_moat"]

    result = _make_dummy_result(total_score=None)
    result["axis_moat_pp"] = None

    mindmap = _build("7203", result)
    assert "N/A" in mindmap["root"]["label"]
    pp_node = next(c for c in mindmap["root"]["children"] if c["id"] == "ax_moat_pp")
    assert "N/A" in pp_node["label"]


# ── 統合テスト (mindmap API) ────────────────────────────────────────


def _mindmap_api_available() -> bool:
    try:
        httpx.get(f"{MINDMAP_API}/list", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mindmap_api_available(), reason="mindmap API (port 8003) 未起動")
def test_expand_api_accepts_context():
    """context フィールド付きの expand リクエストが正常に処理される。"""
    r = httpx.post(
        f"{MINDMAP_API}/expand",
        json={
            "parent_label": "7203",
            "context": "銘柄: 7203, technical=7.5, fundamental=8.0, total=7.5",
        },
        timeout=30,
    )
    assert r.status_code == 200
    data = r.json()
    assert "children" in data
    assert len(data["children"]) >= 1


@pytest.mark.skipif(not _mindmap_api_available(), reason="mindmap API (port 8003) 未起動")
def test_to_mermaid_from_moat_json():
    """_build_mindmap_from_moat の出力を to-mermaid API に渡すと Mermaid 文字列が返る。"""
    # ビルド関数を抽出
    app_path = os.path.join(_APP_DIR, "app.py")
    with open(app_path, encoding="utf-8") as f:
        source = f.read()
    start = source.find("def _build_mindmap_from_moat(")
    end = source.find("\ndef _tab_mindmap_conclusion(")
    ns: dict = {}
    exec(source[start:end], ns)  # noqa: S102
    _build = ns["_build_mindmap_from_moat"]

    mindmap_json = _build("7203", _make_dummy_result())
    r = httpx.post(
        f"{MINDMAP_API}/to-mermaid",
        json={"mindmap": mindmap_json},
        timeout=10,
    )
    assert r.status_code == 200
    mermaid_str = r.json()["mermaid"]
    assert "mindmap" in mermaid_str.lower() or "7203" in mermaid_str


# ── Playwright E2E テスト ──────────────────────────────────────────


def test_mindmap_tab_visible(streamlit_url: str):
    """「結論マップ」タブが Streamlit UI に表示される。"""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(streamlit_url, timeout=30000)
        page.wait_for_load_state("networkidle", timeout=30000)

        # Streamlit のタブボタンを探す
        tab_locator = page.get_by_role("tab", name="結論マップ")
        tab_locator.wait_for(timeout=20000)
        assert tab_locator.is_visible()

        browser.close()


def test_mindmap_tab_shows_input(streamlit_url: str):
    """「結論マップ」タブをクリックすると「マップ生成」ボタンが表示される。

    注意: jpx500 app.py は重いため、タブ内容のレンダリングに最大 60 秒かかる場合がある。
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(streamlit_url, timeout=30000)
        page.wait_for_selector('[data-baseweb="tab"]', timeout=30000)

        page.get_by_role("tab", name="結論マップ").click()

        # app.py のロードが重いため最大 65 秒待機
        gen_btn = page.get_by_role("button", name="マップ生成")
        gen_btn.wait_for(state="visible", timeout=65000)
        assert gen_btn.is_visible()

        browser.close()


def test_mindmap_api_down_shows_error(streamlit_url: str):
    """mindmap API が 8003 で起動していない場合、「マップ生成」後にエラーが表示される。"""
    if _mindmap_api_available():
        pytest.skip("mindmap API が起動中のため「API 未起動」テストをスキップ")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(streamlit_url, timeout=30000)
        page.wait_for_load_state("networkidle", timeout=30000)

        page.get_by_role("tab", name="結論マップ").click()
        page.wait_for_timeout(2000)

        page.get_by_role("button", name="マップ生成").click()
        page.wait_for_timeout(10000)

        error_text = page.get_by_text("接続できません")
        assert error_text.is_visible(timeout=15000)

        browser.close()
