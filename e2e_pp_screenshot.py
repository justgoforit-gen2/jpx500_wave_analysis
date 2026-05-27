"""PER×PBR時系列セクションのスクリーンショット撮影用スクリプト"""
import os
from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = "http://localhost:8511"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1200})
    page.goto(BASE_URL)
    try:
        page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(3000)

    # PER×PBR時系列セクションまでスクロール
    # 見出しを探して scrollIntoView する
    target = page.locator("text=PER × PBR 時系列アニメーション").first
    try:
        target.scroll_into_view_if_needed(timeout=10000)
    except Exception:
        # 見つからない場合、下方向に順次スクロール
        for y in [1500, 3000, 4500, 6000, 7500]:
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(800)

    page.wait_for_timeout(2500)

    # ヘッダの真上から800px分撮影
    # 現在のスクロール位置を保存し、見出しから800pxまでを撮る
    try:
        bbox = target.bounding_box()
        if bbox:
            # 見出しから少し上(80px)を起点に、widthx900の範囲で撮る
            page.screenshot(
                path="test_screenshots/pp_animation_section.png",
                clip={"x": 0, "y": max(bbox["y"] - 30, 0), "width": 1500, "height": 900},
            )
            print(f"saved: y={bbox['y']:.0f}")
        else:
            page.screenshot(path="test_screenshots/pp_animation_section.png", full_page=False)
    except Exception as e:
        page.screenshot(path="test_screenshots/pp_animation_section.png", full_page=False)
        print(f"fallback fullpage: {e}")

    browser.close()

print("done")
