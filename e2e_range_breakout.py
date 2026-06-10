"""Stage 2 ブレイクアウトセクションの E2E スモークテスト。"""

import os
import sys

from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = os.environ.get("APP_URL", "http://localhost:8522")


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

        # Stage 2 セクションのエクスパンダーを展開
        try:
            expander = page.locator("text=Stage 2 ブレイクアウト").first
            expander.wait_for(timeout=10000)
            print("[OK] Stage 2 セクション検出")
            expander.click()
            page.wait_for_timeout(2500)
        except Exception as e:
            failures.append(f"エクスパンダー未検出/クリック失敗: {e}")

        # 検出テーブル(可視の stDataFrame)の存在確認
        try:
            page.locator(
                "[data-testid='stDataFrame']:visible"
            ).first.wait_for(timeout=15000)
            print("[OK] 検出テーブル表示")
        except Exception as e:
            failures.append(f"テーブル未表示: {e}")

        # 該当キャプションの存在確認
        try:
            page.locator("text=/該当: \\d+ 銘柄/").first.wait_for(timeout=5000)
            print("[OK] 該当銘柄キャプション表示")
        except Exception as e:
            failures.append(f"キャプション未表示: {e}")

        page.screenshot(
            path="test_screenshots/range_breakout_expanded.png",
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
