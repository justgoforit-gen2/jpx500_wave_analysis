# -*- coding: utf-8 -*-
"""Playwright E2E — Moat Score の naibu Pricing Power 軸が UI で算出されるか検証。

回帰対象バグ:
  _compute_pp_from_naibu が securities_code を edinet エンドポイントに渡し常に404 →
  「naibu API 未起動」と誤表示。/api/stocks/{code} + scores.pricing_power に修正。

実行:
  TEST_BASE_URL=http://localhost:8510 python test_moat_pp_e2e.py
"""
import os
import sys

from playwright.sync_api import sync_playwright

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8510")
CODE = os.getenv("TEST_CODE", "7272")
TIMEOUT_MS = int(os.getenv("TEST_TIMEOUT_MS", "120000"))
results = []


def check(label, ok, detail=""):
    results.append(("PASS" if ok else "FAIL", label, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


def get_tab(page, label):
    for sel in ["[data-baseweb='tab']", "[role='tab']"]:
        for t in page.locator(sel).all():
            if label in (t.text_content() or ""):
                return t
    return None


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.set_default_timeout(TIMEOUT_MS)

    print(f"接続中: {BASE_URL}")
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    page.wait_for_selector("[data-baseweb='tab']", timeout=TIMEOUT_MS)

    tab = get_tab(page, "Moat")
    check("Moat Score タブ存在", tab is not None)
    if not tab:
        browser.close()
        sys.exit(1)
    tab.scroll_into_view_if_needed()
    tab.click()
    # Moat タブ描画完了まで、銘柄コード入力欄の出現を明示的に待つ (cold render 対策)
    page.wait_for_selector('input[aria-label*="銘柄コード"]', state="visible", timeout=TIMEOUT_MS)

    # 銘柄コード入力 — Streamlit は全タブを隠しDOMで描画するため、aria-label と可視で厳密指定
    code_input = page.locator('input[aria-label*="銘柄コード"]:visible').first
    check("銘柄コード入力欄(可視)存在", code_input.count() > 0)
    code_input.click()
    code_input.fill("")
    code_input.fill(CODE)
    page.wait_for_timeout(500)

    # スコア算出ボタン (可視のもの)
    btn = page.get_by_role("button", name="スコア算出").locator("visible=true")
    check("スコア算出ボタン存在", btn.count() > 0)
    btn.first.click()
    page.wait_for_timeout(8000)  # naibu API 往復 + レンダリング

    body = page.inner_html("body")
    page.screenshot(path="test_screenshots/moat_pp_e2e.png", full_page=True)

    # 1. 「naibu API 未起動」警告が消えていること
    has_warning = "naibu API 未起動" in body
    check(f"[{CODE}] naibu未起動の警告が出ない", not has_warning,
          "警告が残存" if has_warning else "警告なし")

    # 2. Pricing Power 軸ラベルが表示されている
    check("PP 軸ラベル(PP/Pricing)が画面に存在", ("PP" in body or "Pricing" in body or "pricing_power" in body))

    # 3. レーダーチャートが描画されている (plotly svg)
    has_radar = page.locator("svg.main-svg, .js-plotly-plot").count() > 0
    check("レーダーチャート描画", has_radar)

    # 4. エラーバナー (st.warning の黄色) が naibu 由来でない
    check(f"[{CODE}] スコア算出完了", not has_warning and has_radar)

    browser.close()

print("\n" + "=" * 50)
passed = sum(1 for s, *_ in results if s == "PASS")
failed = sum(1 for s, *_ in results if s == "FAIL")
print(f"結果: {passed}件 PASS / {failed}件 FAIL")
for s, label, detail in results:
    if s == "FAIL":
        print(f"  [FAIL] {label}: {detail}")
sys.exit(1 if failed else 0)
