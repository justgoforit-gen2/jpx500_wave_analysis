"""点サイズ（時価総額比例）動作確認スクリーンショット"""
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
    page.wait_for_timeout(3000)

    # 「対象=全銘柄」に切替（点が多くサイズ比較しやすい）
    full_radio = page.locator("text=全銘柄").last
    try:
        full_radio.click()
        page.wait_for_timeout(3500)
        target.scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
    except Exception as e:
        print("radio click error:", e)

    bbox = target.bounding_box()
    if bbox:
        page.screenshot(
            path="test_screenshots/pp_size_proportional.png",
            clip={
                "x": 200,
                "y": max(bbox["y"] - 20, 0),
                "width": 1300,
                "height": 1100,
            },
        )

    browser.close()
print("done")
