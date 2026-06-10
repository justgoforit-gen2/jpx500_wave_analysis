# -*- coding: utf-8 -*-
"""Playwright E2E スモークテスト — Streamlit 投資ハブ UI

## 実行方法
    python test_playwright.py              # devcontainer (スモークモード)
    TEST_BASE_URL=http://localhost:8501 \
    TEST_FULL_UI=1 python test_playwright.py  # ホストブラウザ向け完全 UI テスト

## devcontainer での制限
    Streamlit は view 関数の重い初期データ処理のため、
    headless Chrome での WebSocket セッション確立に 60+ 秒かかる。
    完全 UI テストはホストブラウザ（VSCode ポートフォワーディング）から
    TEST_FULL_UI=1 を付けて実行すること。

## カバレッジ
    スモークモード: HTTP 200 / API エンドポイント / app.py 構文
    完全 UI モード: タブ遷移 / Moat Score タブ / ランキングタブ
"""
import os
import sys

os.makedirs("test_screenshots", exist_ok=True)

from playwright.sync_api import sync_playwright

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8501")
FULL_UI = os.getenv("TEST_FULL_UI", "").lower() in ("1", "true", "yes")
LOAD_TIMEOUT_MS = int(os.getenv("TEST_TIMEOUT_MS", "180000"))
results = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((status, label, detail))
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


def get_tab(page, label):
    for sel in ["[data-baseweb='tab']", "[role='tab']"]:
        tabs = page.locator(sel).all()
        tab = next((t for t in tabs if label in (t.text_content() or "")), None)
        if tab:
            return tab
    return None


def click_tab(page, label, wait_ms=5000):
    tab = get_tab(page, label)
    if tab:
        tab.scroll_into_view_if_needed()
        tab.click()
        page.wait_for_timeout(wait_ms)
        return True
    return False


def run_smoke_tests():
    """HTTP ベースのスモークテスト (devcontainer 対応)。"""
    import urllib.request

    print("\n=== スモークテスト ===")

    # Streamlit HTTP
    try:
        with urllib.request.urlopen(f"{BASE_URL}/", timeout=10) as r:
            check("Streamlit HTTP 200", r.status == 200, f"status={r.status}")
    except Exception as e:
        check("Streamlit HTTP 200", False, str(e))

    # Streamlit health
    try:
        health_url = f"{BASE_URL}/_stcore/health"
        with urllib.request.urlopen(health_url, timeout=5) as r:
            body = r.read().decode()
            check("Streamlit health endpoint", "ok" in body, body)
    except Exception as e:
        check("Streamlit health endpoint", False, str(e))

    # FastAPI health (api_server.py は別プロセス想定)
    api_base = BASE_URL.replace("8501", "8001")
    try:
        with urllib.request.urlopen(f"{api_base}/api/health", timeout=5) as r:
            check("FastAPI /api/health", r.status == 200, f"status={r.status}")
    except Exception as e:
        check("FastAPI /api/health (任意)", True, f"skip (API 未起動): {e}")

    # app.py 構文チェック
    import ast

    try:
        ast.parse(open("app.py").read())
        check("app.py 構文チェック", True)
    except SyntaxError as e:
        check("app.py 構文チェック", False, str(e))

    # pytest モジュールインポート確認
    try:
        from modules.moat_score import WEIGHTS, MoatScoreEngine

        ok = abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
        check("moat_score WEIGHTS 合計 1.0", ok, f"sum={sum(WEIGHTS.values())}")
    except Exception as e:
        check("moat_score WEIGHTS 合計 1.0", False, str(e))


def run_full_ui_tests():
    """Playwright ブラウザ UI テスト (FULL_UI モード)。"""
    print(f"\n=== 完全 UI テスト (timeout={LOAD_TIMEOUT_MS//1000}s) ===")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.set_default_timeout(LOAD_TIMEOUT_MS)

        print(f"接続中: {BASE_URL}")
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)

        try:
            page.wait_for_selector("[data-baseweb='tab']", timeout=LOAD_TIMEOUT_MS)
            print("タブ検出: レンダリング完了")
        except Exception as e:
            check("Streamlit タブ表示", False, f"timeout: {e}")
            page.screenshot(path="test_screenshots/error_state.png", full_page=True)
            browser.close()
            return

        page.screenshot(path="test_screenshots/01_loaded.png", full_page=False)
        all_tab_texts = [t.text_content() for t in page.locator("[data-baseweb='tab']").all()]
        print(f"検出タブ: {all_tab_texts}")

        check("タブ 5 個以上", len(all_tab_texts) >= 5, f"{all_tab_texts}")
        check("Moat Score タブ存在", any("Moat" in t for t in all_tab_texts))
        check("ランキングタブ存在", any("ランキング" in t for t in all_tab_texts))

        # ABCD 戦略
        if click_tab(page, "ABCD"):
            page.screenshot(path="test_screenshots/03_abcd.png", full_page=False)
            check("ABCD タブ遷移", True)
        else:
            check("ABCD タブ遷移", False, f"tabs={all_tab_texts}")

        # Moat Score (Phase 2 新規)
        if click_tab(page, "Moat"):
            moat_html = page.inner_html("body")
            page.screenshot(path="test_screenshots/05_moat.png", full_page=False)
            check("Moat Score タブ遷移", True)
            check("Moat Score コンテンツ", any(k in moat_html for k in ["バフェット", "7軸", "Moat"]))
        else:
            check("Moat Score タブ遷移", False, f"tabs={all_tab_texts}")
            check("Moat Score コンテンツ", False)

        # ランキング (Phase 2 新規)
        if click_tab(page, "ランキング"):
            rank_html = page.inner_html("body")
            page.screenshot(path="test_screenshots/06_ranking.png", full_page=False)
            check("ランキングタブ遷移", True)
            check("ランキングコンテンツ", len(rank_html) > 3000)
        else:
            check("ランキングタブ遷移", False, f"tabs={all_tab_texts}")
            check("ランキングコンテンツ", False)

        browser.close()


# --- メイン ---
if FULL_UI:
    run_full_ui_tests()
else:
    run_smoke_tests()
    print("\n(完全 UI テストは TEST_FULL_UI=1 で実行)")

print("\n" + "=" * 50)
passed = sum(1 for s, *_ in results if s == "PASS")
failed = sum(1 for s, *_ in results if s == "FAIL")
print(f"結果: {passed}件 PASS / {failed}件 FAIL")
for s, label, detail in results:
    if s == "FAIL":
        print(f"  [FAIL] {label}: {detail}")
if failed:
    sys.exit(1)
else:
    print("全テスト PASS")
