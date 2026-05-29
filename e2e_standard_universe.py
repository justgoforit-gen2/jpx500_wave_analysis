"""TSE Standard ユニバース追加の Playwright E2E + 画像検証スクリプト。

撮影パーツ:
  1) サイドバーの「市場」フィルタ + 拡張された「規模区分」
  2) CES スクリーナーの サイズフィルタ (Standard Top100/Top400 追加確認)
  3) 海外フロー × サイズ別相関 (TSE Standard 選択時)
"""

import os

from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = "http://localhost:8512"


def shoot_clip(page, filename, clip):
    page.screenshot(path=f"test_screenshots/{filename}", clip=clip)
    print(f"saved: {filename}")


def shoot_full(page, filename):
    page.screenshot(path=f"test_screenshots/{filename}", full_page=True)
    print(f"saved (fullpage): {filename}")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1800, "height": 1500})
    page.goto(BASE_URL, wait_until="networkidle")
    page.wait_for_timeout(6000)

    # 1) サイドバー (市場 + 規模区分)
    shoot_clip(
        page,
        "standard_sidebar_filters.png",
        clip={"x": 0, "y": 0, "width": 420, "height": 1500},
    )

    # 2) CES スクリーナー
    ces_anchor = page.locator("text=資本効率改善期待スクリーナー").first
    try:
        ces_anchor.scroll_into_view_if_needed(timeout=10000)
        page.wait_for_timeout(3000)
        bbox = ces_anchor.bounding_box()
        if bbox:
            shoot_clip(
                page,
                "ces_with_standard_sizes.png",
                clip={
                    "x": 200,
                    "y": max(bbox["y"] - 20, 0),
                    "width": 1500,
                    "height": 700,
                },
            )
    except Exception as e:
        print(f"CES anchor not found: {e}")
        shoot_full(page, "ces_with_standard_sizes_full.png")

    # 3) 海外フロー × サイズ別相関 (TSE Standard を選択)
    try:
        ff_anchor = page.locator("text=海外投資家フロー").first
        ff_anchor.scroll_into_view_if_needed(timeout=10000)
        page.wait_for_timeout(2000)
        # 市場ラジオで TSE Standard を選択
        page.get_by_text("TSE Standard", exact=True).first.click()
        page.wait_for_timeout(2000)
        # 表示形式: サイズ別相関バー
        page.get_by_text("サイズ別相関バー", exact=True).first.click()
        page.wait_for_timeout(3000)
        bbox = ff_anchor.bounding_box()
        if bbox:
            shoot_clip(
                page,
                "ff_standard_size_correlation.png",
                clip={
                    "x": 200,
                    "y": max(bbox["y"] - 20, 0),
                    "width": 1500,
                    "height": 900,
                },
            )
        else:
            shoot_full(page, "ff_standard_size_correlation_full.png")
    except Exception as e:
        print(f"Foreign flow Standard view shoot failed: {e}")

    browser.close()

print("done")
