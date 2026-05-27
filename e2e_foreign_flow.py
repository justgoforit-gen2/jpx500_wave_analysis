"""海外投資家フロー × 指数 連動分析 のE2Eテスト + スクリーンショット視覚検証"""
import os
import sys
from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = "http://localhost:8511"
results: list[tuple[str, str, str]] = []


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    results.append((status, label, detail))
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1400})

    print("--- Loading app ---")
    page.goto(BASE_URL)
    try:
        page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
    except Exception as e:
        print(f"[WARN] {e}")
    page.wait_for_timeout(4000)

    body_text = page.locator("body").inner_text()
    if "ImportError" in body_text or "ModuleNotFoundError" in body_text:
        print(f"[ERROR] app error:\n{body_text[:800]}")
        browser.close()
        sys.exit(1)

    # 新セクションを見つけてスクロール
    target = page.locator("text=海外投資家フロー × 指数 連動分析").first
    try:
        target.scroll_into_view_if_needed(timeout=15000)
    except Exception:
        for y in [1000, 2000, 3000, 4000, 5000]:
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(600)
    page.wait_for_timeout(3000)

    full_html = page.inner_html("body")

    # 基本要素の存在確認
    check("セクション見出し『海外投資家フロー × 指数 連動分析』", "海外投資家フロー × 指数 連動分析" in full_html)
    check("市場ラジオ TSE Prime", "TSE Prime" in full_html)
    check("市場ラジオ TSE Standard", "TSE Standard" in full_html)
    check("市場ラジオ TSE Growth", "TSE Growth" in full_html)
    check("市場ラジオ Tokyo & Nagoya", "Tokyo & Nagoya" in full_html)
    check("比較対象 multiselect", "比較対象" in full_html)
    check("日経225 オプション", "日経225" in full_html)
    check("TOPIX (1306.T) オプション", "TOPIX (1306.T)" in full_html)
    check("表示形式『累積フロー vs 指数』", "累積フロー vs 指数" in full_html)
    check("表示形式『週次フロー × リターン散布図』", "週次フロー × リターン散布図" in full_html)
    check("表示形式『業種別相関バー』", "業種別相関バー" in full_html)

    # 初期表示（2軸チャート）のスクリーンショット
    bbox = target.bounding_box()
    if bbox:
        page.screenshot(
            path="test_screenshots/foreign_flow_dual_axis.png",
            clip={"x": 100, "y": max(bbox["y"] - 30, 0), "width": 1400, "height": 1000},
        )
    # Plotlyグラフ存在確認
    plot_count_dual = page.locator(".js-plotly-plot").count()
    check("Plotlyグラフ（2軸チャート）", plot_count_dual >= 1, f"plot_count={plot_count_dual}")

    # 表示形式「週次散布図」に切替
    scatter_radio = page.locator("text=週次フロー × リターン散布図").last
    try:
        scatter_radio.click()
        page.wait_for_timeout(4000)
        target.scroll_into_view_if_needed()
        page.wait_for_timeout(2000)
        bbox2 = target.bounding_box()
        if bbox2:
            page.screenshot(
                path="test_screenshots/foreign_flow_scatter.png",
                clip={"x": 100, "y": max(bbox2["y"] - 30, 0), "width": 1400, "height": 1000},
            )
        html2 = page.inner_html("body")
        # 散布図はデータ不足なら「データが少なすぎます」のキャプションあり
        has_scatter = (
            "週次リターン" in html2
            or "データが少なすぎます" in html2
        )
        check("週次散布図表示 or データ不足メッセージ", has_scatter)
    except Exception as e:
        check("週次散布図クリック", False, str(e)[:80])

    # 表示形式「業種別相関バー」に切替
    bar_radio = page.locator("text=業種別相関バー").last
    try:
        bar_radio.click()
        page.wait_for_timeout(5000)
        target.scroll_into_view_if_needed()
        page.wait_for_timeout(2000)
        bbox3 = target.bounding_box()
        if bbox3:
            page.screenshot(
                path="test_screenshots/foreign_flow_sector_bar.png",
                clip={"x": 100, "y": max(bbox3["y"] - 30, 0), "width": 1400, "height": 1200},
            )
        html3 = page.inner_html("body")
        has_bar_or_msg = (
            "ラグ" in html3 or "業種別相関" in html3 or "計算できません" in html3
        )
        check("業種別相関バー表示 or エラーメッセージ", has_bar_or_msg)
    except Exception as e:
        check("業種別相関バークリック", False, str(e)[:80])

    browser.close()

print("\n" + "=" * 50)
passed = sum(1 for s, *_ in results if s == "PASS")
failed = sum(1 for s, *_ in results if s == "FAIL")
print(f"Result: {passed} PASS / {failed} FAIL")
for s, label, detail in results:
    if s == "FAIL":
        print(f"  [FAIL] {label}: {detail}")
if failed:
    sys.exit(1)
print("All E2E tests PASSED")
