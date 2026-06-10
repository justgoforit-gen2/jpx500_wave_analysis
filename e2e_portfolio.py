"""ポートフォリオタブの E2E スモークテスト。

サイドバーから初期ポートフォリオを生成し、サマリ・シグナル・
保有テーブル・監視テーブルが表示されることを確認する。
"""

import os
import sys
from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)
BASE_URL = os.environ.get("APP_URL", "http://localhost:8511")


def run() -> int:
    failures = []
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

        # 1) ポートフォリオタブが先頭にあるか
        portfolio_tab = page.locator("button[role='tab']", has_text="ポートフォリオ").first
        try:
            portfolio_tab.wait_for(timeout=10000)
            portfolio_tab.click()
            page.wait_for_timeout(1500)
            print("[OK] ポートフォリオタブをクリック")
        except Exception as e:
            failures.append(f"ポートフォリオタブ未検出: {e}")

        # 2) ダッシュボードタイトル
        title = page.locator("text=ポートフォリオ・ダッシュボード").first
        try:
            title.wait_for(timeout=10000)
            print("[OK] ダッシュボードタイトル表示")
        except Exception as e:
            failures.append(f"タイトル未表示: {e}")

        # 3) 初期表示(空の場合)スクリーンショット
        page.screenshot(
            path="test_screenshots/portfolio_initial.png", full_page=True
        )
        print("[OK] portfolio_initial.png 保存")

        # 4) サイドバーの「初期ポートフォリオ生成」を試行
        try:
            page.locator("text=初期ポートフォリオ生成").first.click(timeout=5000)
            page.wait_for_timeout(800)
            page.locator("button", has_text="初期投入を実行").first.click(timeout=5000)
            page.wait_for_timeout(8000)  # 初期化(API 呼出)に時間がかかる場合あり
            print("[OK] 初期投入実行ボタン押下")
        except Exception as e:
            print(f"[INFO] 初期投入はスキップ(既にデータあり or キャッシュ不足): {e}")

        # 5) サマリー要素の確認
        try:
            page.locator("text=総資産").first.wait_for(timeout=5000)
            print("[OK] 総資産メトリック確認")
        except Exception as e:
            failures.append(f"総資産メトリック未表示: {e}")

        # 6) シグナルセクションの確認
        try:
            page.locator("text=本日のシグナル").first.wait_for(timeout=5000)
            print("[OK] シグナルセクション確認")
        except Exception as e:
            failures.append(f"シグナルセクション未表示: {e}")

        # 7) 保有テーブル or "保有なし" の確認
        try:
            page.locator("text=保有銘柄").first.wait_for(timeout=5000)
            print("[OK] 保有銘柄セクション確認")
        except Exception as e:
            failures.append(f"保有銘柄セクション未表示: {e}")

        # 8) 監視銘柄テーブルの確認
        try:
            page.locator("text=監視銘柄").first.wait_for(timeout=5000)
            print("[OK] 監視銘柄セクション確認")
        except Exception as e:
            failures.append(f"監視銘柄セクション未表示: {e}")

        # 最終スクリーンショット
        page.screenshot(
            path="test_screenshots/portfolio_dashboard.png", full_page=True
        )
        print("[OK] portfolio_dashboard.png 保存")

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
