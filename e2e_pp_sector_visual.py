"""業種フィルタ機能の動作スクリーンショット撮影（業種=輸送用機器選択後の表示）"""
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

    # 「PER × PBR 時系列アニメーション」までスクロール
    target = page.locator("text=PER × PBR 時系列アニメーション").first
    target.scroll_into_view_if_needed(timeout=10000)
    page.wait_for_timeout(2500)

    # 業種selectboxを見つけてクリック → 「輸送用機器」をtype選択
    selects = page.locator("[data-baseweb='select']").all()
    sector_select = None
    for s in selects:
        try:
            if not s.is_visible():
                continue
            if "（業種で絞らない）" in (s.inner_text() or "") or "業種で絞らない" in (
                s.inner_text() or ""
            ):
                sector_select = s
                break
        except Exception:
            continue

    if sector_select:
        bb = sector_select.bounding_box()
        page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.wait_for_timeout(1500)
        page.keyboard.type("輸送用機器", delay=80)
        page.wait_for_timeout(1500)
        # 表示中のオプションをクリック
        for sel_str in ["[data-baseweb='menu'] li", "[role='option']"]:
            items = page.locator(sel_str).all()
            clicked = False
            for it in items:
                try:
                    if it.text_content() and "輸送用機器" in it.text_content():
                        it.click()
                        clicked = True
                        break
                except Exception:
                    continue
            if clicked:
                break
        page.wait_for_timeout(4500)

    # 業種selectbox を含む領域から下に伸ばしてキャプチャ
    bbox = target.bounding_box()
    if bbox:
        # 見出しから1100px分を撮る（コントロール + グラフ + キャプション含む）
        page.screenshot(
            path="test_screenshots/pp_sector_transport_full.png",
            clip={
                "x": 200,
                "y": max(bbox["y"] - 20, 0),
                "width": 1300,
                "height": 1200,
            },
        )

    # 全ページ HTML 取得して「業種=輸送用機器」キャプションの有無確認
    html = page.inner_html("body")
    print("caption '業種=輸送用機器' present:", "業種=輸送用機器" in html)
    print("legend or color split present (name):", '凡例' in html or "name" in html)

    browser.close()
print("done")
