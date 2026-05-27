"""表示単位=業種加重平均モードのスクリーンショット撮影"""
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

    # 「業種加重平均」ラジオをクリック
    weighted = page.locator("text=業種加重平均").first
    try:
        weighted.click()
        page.wait_for_timeout(4500)
        target.scroll_into_view_if_needed()
        page.wait_for_timeout(2500)
    except Exception as e:
        print("click error:", e)

    html = page.inner_html("body")
    print("caption '単位: 業種加重平均' present:", "単位: 業種加重平均" in html)

    bbox = target.bounding_box()
    if bbox:
        page.screenshot(
            path="test_screenshots/pp_unit_weighted.png",
            clip={
                "x": 200,
                "y": max(bbox["y"] - 20, 0),
                "width": 1300,
                "height": 1100,
            },
        )
    browser.close()
print("done")
