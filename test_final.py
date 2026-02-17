"""最終検証: 決算日縦線のチャート表示を確認"""
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path(__file__).parent / "test_screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)
BASE_URL = "http://localhost:8501"


def test():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1200})

        # メインページ読み込み
        print("1. Loading main page...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(5000)
        page.screenshot(path=str(SCREENSHOT_DIR / "final_01_main.png"), full_page=True)

        # 下タッチテーブルの行をクリックして詳細画面へ遷移
        print("2. Clicking stock row...")
        dfs = page.locator('[data-testid="stDataFrame"]')
        if dfs.count() >= 1:
            target_df = dfs.nth(0)
            target_df.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            canvas = target_df.locator("canvas")
            if canvas.count() > 0:
                bbox = canvas.first.bounding_box()
                if bbox:
                    page.mouse.click(bbox["x"] + 30, bbox["y"] + 55)
                    page.wait_for_timeout(5000)

        body = page.inner_text("body")
        if "一覧に戻る" in body:
            print("   Detail view loaded!")
            page.screenshot(path=str(SCREENSHOT_DIR / "final_02_detail.png"), full_page=True)
            print("   Screenshot: final_02_detail.png")

            # チャート拡大スクリーンショット
            charts = page.locator(".js-plotly-plot")
            if charts.count() > 0:
                charts.first.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)
                chart_bbox = charts.first.bounding_box()
                if chart_bbox:
                    page.screenshot(
                        path=str(SCREENSHOT_DIR / "final_03_chart.png"),
                        clip={
                            "x": max(0, chart_bbox["x"]),
                            "y": max(0, chart_bbox["y"]),
                            "width": min(1920, chart_bbox["width"]),
                            "height": min(900, chart_bbox["height"]),
                        }
                    )
                    print("   Screenshot: final_03_chart.png (chart zoom)")

            # 指標の見方expanderを開く
            expanders = page.locator('[data-testid="stExpander"]')
            if expanders.count() > 0:
                expanders.first.click()
                page.wait_for_timeout(1000)
                page.screenshot(path=str(SCREENSHOT_DIR / "final_04_help.png"), full_page=True)

                body2 = page.inner_text("body")
                checks = {
                    "RSIシグナルの見方": "RSIシグナルの見方" in body2,
                    "オレンジ破線（縦）": "オレンジ破線（縦）" in body2,
                    "決算発表予定日": "決算発表予定日" in body2,
                }
                print("   Help content:")
                for k, v in checks.items():
                    print(f"     {k}: {'OK' if v else 'MISSING'}")
        else:
            print("   Could not navigate to detail view")
            page.screenshot(path=str(SCREENSHOT_DIR / "final_02_nodetail.png"), full_page=True)

        print("\nDone!")
        browser.close()


if __name__ == "__main__":
    test()
