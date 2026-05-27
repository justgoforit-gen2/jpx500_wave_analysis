"""海外投資家フロー × 指数 視覚検証用のスクリーンショット撮影スクリプト。

各表示形式をクリックして個別にスクショ。撮影領域はターゲット見出しから1100pxまで。
"""
import os
from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = "http://localhost:8511"


def shoot(page, target_locator, filename, height=1100):
    """ターゲット要素を強制的に画面トップに合わせてからクリップ撮影。"""
    try:
        target_locator.scroll_into_view_if_needed(timeout=10000)
    except Exception:
        pass
    # 強制的にターゲットを画面トップに（scrollIntoViewはbottomに揃うことがある）
    try:
        target_locator.evaluate(
            "el => el.scrollIntoView({behavior: 'instant', block: 'start'})"
        )
    except Exception:
        pass
    page.wait_for_timeout(3500)
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
            "width": 1300,
            "height": height,
        },
    )
    print(f"saved: {filename} (y={bbox['y']:.0f})")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1500, "height": 1300})
    page.goto(BASE_URL)
    try:
        page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(2000)
    # フル再読み込みで hot-reload キャッシュをクリア
    page.reload(wait_until="networkidle")
    try:
        page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(5000)

    target = page.locator("text=海外投資家フロー × 指数 連動分析").first

    # 1) 初期表示（累積フロー × 指数 2軸）
    shoot(page, target, "ff_view1_dual_axis.png", height=1100)

    # 2) 「週次フロー × リターン散布図」に切替
    try:
        scatter_label = page.get_by_text("週次フロー × リターン散布図", exact=True).first
        scatter_label.click()
        page.wait_for_timeout(8000)  # Plotlyの再描画を十分に待つ
        # Plotlyグラフのsvg/canvas待ち
        try:
            page.wait_for_selector(".js-plotly-plot .main-svg", timeout=10000)
        except Exception:
            pass
        page.wait_for_timeout(2000)
        shoot(page, target, "ff_view2_scatter.png", height=1200)
    except Exception as e:
        print("scatter click error:", e)

    # 3) 「業種別相関バー」に切替
    try:
        bar_label = page.get_by_text("業種別相関バー", exact=True).first
        bar_label.click()
        page.wait_for_timeout(10000)  # 33業種計算 + plotly描画は重い
        try:
            page.wait_for_selector(".js-plotly-plot .main-svg", timeout=10000)
        except Exception:
            pass
        page.wait_for_timeout(2000)
        shoot(page, target, "ff_view3_sector_bar.png", height=1300)
    except Exception as e:
        print("bar click error:", e)

    browser.close()

print("done")
