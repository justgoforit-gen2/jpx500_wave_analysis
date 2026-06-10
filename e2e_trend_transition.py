"""トレンド転換セクションの E2E スモークテスト。

波形分類タブを開いて「トレンド転換検出」エクスパンダーを展開し、
検出テーブルが表示されることを確認する。
"""

import os
import sys

from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = os.environ.get("APP_URL", "http://localhost:8520")


def run() -> int:
    failures: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1400})
        page.goto(BASE_URL)
        try:
            page.wait_for_selector(
                "[data-testid='stMainBlockContainer']", timeout=60000
            )
        except Exception:
            pass
        page.wait_for_timeout(3000)

        # 波形分類タブをクリック
        try:
            page.locator(
                "button[role='tab']", has_text="波形分類"
            ).first.click(timeout=10000)
            page.wait_for_timeout(2500)
            print("[OK] 波形分類タブをクリック")
        except Exception as e:
            failures.append(f"波形分類タブ未検出: {e}")

        # トレンド転換セクションのエクスパンダーを展開
        try:
            expander = page.locator("text=トレンド転換検出").first
            expander.wait_for(timeout=10000)
            print("[OK] トレンド転換セクション検出")
            expander.click()
            page.wait_for_timeout(2500)
        except Exception as e:
            failures.append(f"エクスパンダー未検出/クリック失敗: {e}")

        # 検出テーブル(可視の stDataFrame)の存在確認
        # トレンド転換セクションへスクロールして可視化
        try:
            expander.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(500)
            page.locator(
                "[data-testid='stDataFrame']:visible"
            ).first.wait_for(timeout=15000)
            print("[OK] 検出テーブル表示")
        except Exception as e:
            failures.append(f"テーブル未表示: {e}")

        # スクリーンショット
        page.screenshot(
            path="test_screenshots/trend_transition_expanded.png",
            full_page=True,
        )
        print("[OK] スクリーンショット保存")

        browser.close()

    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n=== ALL PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(run())
