"""ROEサイズモードのスクリーンショット撮影"""
import os
from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = "http://localhost:8511"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1400})
    page.goto(BASE_URL)
    try:
        page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(3500)

    target = page.locator("text=PER × PBR 時系列アニメーション").first
    target.scroll_into_view_if_needed(timeout=10000)
    page.wait_for_timeout(2500)

    # 個別銘柄モード + 全銘柄スコープに切替
    try:
        page.locator("text=全銘柄").last.click()
        page.wait_for_timeout(2500)
    except Exception:
        pass

    # 「点サイズ」のROEラジオをクリック
    try:
        # horizontal radioなのでlabel直接クリック
        roe_label = page.locator("text=ROE").first
        roe_label.click()
        page.wait_for_timeout(4000)
        target.scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
    except Exception as e:
        print("ROE radio click error:", e)

    html = page.inner_html("body")
    print("'点サイズ' label present:", "点サイズ" in html)
    print("'ROE' radio option present:", "ROE" in html)

    bbox = target.bounding_box()
    if bbox:
        page.screenshot(
            path="test_screenshots/pp_size_roe.png",
            clip={
                "x": 200,
                "y": max(bbox["y"] - 20, 0),
                "width": 1300,
                "height": 1100,
            },
        )
    browser.close()
print("done")
