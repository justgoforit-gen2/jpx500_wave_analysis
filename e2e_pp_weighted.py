"""加重平均切替機能のスクリーンショット撮影"""
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
    page.wait_for_timeout(3000)

    target = page.locator("text=PER × PBR 時系列アニメーション").first
    target.scroll_into_view_if_needed(timeout=10000)
    page.wait_for_timeout(2000)

    # expander「業種別 PER/PBR 中央値の時系列推移」を展開
    expander_btn = page.locator("text=業種別 PER/PBR").first
    try:
        expander_btn.click()
        page.wait_for_timeout(2500)
    except Exception as e:
        print("expander click error:", e)

    # 「集計方法」ラジオが出ているか確認
    html = page.inner_html("body")
    has_agg = "集計方法" in html and "時価総額加重平均" in html
    print("集計方法ラジオ表示:", has_agg)

    # 中央値スクリーンショット
    expander_loc = page.locator("text=業種別 PER/PBR").first
    bbox = expander_loc.bounding_box()
    if bbox:
        page.screenshot(
            path="test_screenshots/pp_agg_median.png",
            clip={
                "x": 200,
                "y": max(bbox["y"] - 20, 0),
                "width": 1300,
                "height": 700,
            },
        )

    # 「時価総額加重平均」をクリック
    # st.radioのラベル「時価総額加重平均」をクリック
    weighted_label = page.locator("text=時価総額加重平均").first
    try:
        weighted_label.click()
        page.wait_for_timeout(3500)
        # スクロールしておく
        target.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        expander_loc.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)

        # 加重平均スクリーンショット
        bbox2 = expander_loc.bounding_box()
        if bbox2:
            page.screenshot(
                path="test_screenshots/pp_agg_weighted.png",
                clip={
                    "x": 200,
                    "y": max(bbox2["y"] - 20, 0),
                    "width": 1300,
                    "height": 700,
                },
            )

        # 加重平均切替後のHTML確認
        html2 = page.inner_html("body")
        print("title '時価総額加重' 表示:", "時価総額加重" in html2)
    except Exception as e:
        print("weighted click error:", e)

    browser.close()
print("done")
