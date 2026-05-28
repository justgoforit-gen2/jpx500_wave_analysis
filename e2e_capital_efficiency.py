"""資本効率改善期待スクリーナーのPlaywright E2E + 画像検証スクリプト。

3パーツ (ランキングテーブル / PBR×ROE散布図 / 上位10スコア内訳バー) を順次撮影する。
"""

import os

from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = "http://localhost:8511"


def shoot(page, target_locator, filename, height=1100, width=1600):
    try:
        target_locator.scroll_into_view_if_needed(timeout=10000)
    except Exception:
        pass
    try:
        target_locator.evaluate(
            "el => el.scrollIntoView({behavior: 'instant', block: 'start'})"
        )
    except Exception:
        pass
    page.wait_for_timeout(3000)
    bbox = target_locator.bounding_box()
    if bbox is None:
        page.screenshot(path=f"test_screenshots/{filename}", full_page=True)
        print(f"fullpage fallback: {filename}")
        return
    page.screenshot(
        path=f"test_screenshots/{filename}",
        clip={
            "x": 200,
            "y": max(bbox["y"] - 20, 0),
            "width": width,
            "height": height,
        },
    )
    print(f"saved: {filename} (y={bbox['y']:.0f})")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    # 列が4増えたので横幅を広げる
    page = browser.new_page(viewport={"width": 1800, "height": 1500})
    page.goto(BASE_URL, wait_until="networkidle")
    page.wait_for_timeout(6000)

    target = page.locator("text=資本効率改善期待スクリーナー").first

    # 1) ランキングテーブル (default score >= 7)
    shoot(page, target, "ces_v11_table.png", height=900)

    # 2) 散布図含め
    shoot(page, target, "ces_v11_scatter.png", height=1500)

    # 3) スコア内訳バー (5軸) まで
    page.wait_for_timeout(3000)
    shoot(page, target, "ces_v11_full.png", height=2200)

    browser.close()

print("done")
