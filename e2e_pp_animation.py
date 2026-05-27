"""PER×PBR 時系列アニメーションUI のE2Eテスト（Playwright）。

事前条件:
  - Streamlit が http://localhost:8511 で起動済み
  - data/per_pbr_history.parquet が存在（小規模でも可）
"""
import os
import sys
from playwright.sync_api import sync_playwright

os.makedirs("test_screenshots", exist_ok=True)

BASE_URL = "http://localhost:8511"
results: list[tuple[str, str, str]] = []


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    results.append((status, label, detail))
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})

    print("--- Loading app ---")
    page.goto(BASE_URL)
    try:
        page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
    except Exception as e:
        print(f"[WARN] {e}")
        page.wait_for_timeout(15000)

    body_text = page.locator("body").inner_text()
    if "ImportError" in body_text or "ModuleNotFoundError" in body_text:
        print(f"[ERROR] app error:\n{body_text[:800]}")
        browser.close()
        sys.exit(1)

    # スクロールで遅延描画分のロードを進める
    for y in [600, 1500, 2500, 3500, 4500, 5500, 6500, 7500]:
        page.evaluate(f"window.scrollTo(0, {y})")
        page.wait_for_timeout(800)

    page.wait_for_timeout(2000)
    full_html = page.inner_html("body")
    page.screenshot(path="test_screenshots/pp_full.png", full_page=True)

    check(
        "アニメ散布図 見出し",
        "PER × PBR 時系列アニメーション" in full_html,
    )
    check(
        "既存スナップショット散布図と併存",
        full_html.count("PER × PBR") >= 2,
        f"count={full_html.count('PER × PBR')}",
    )

    # 期間ラジオボタン
    check("期間ラジオ「直近1年」", "直近1年" in full_html)
    check("期間ラジオ「直近2年」", "直近2年" in full_html)
    check("期間ラジオ「直近3年」", "直近3年" in full_html)

    # 各種コントロール
    check("PER上限スライダー", "PER上限" in full_html)
    check("PBR上限スライダー", "PBR上限" in full_html)
    check("再生速度コントロール", "再生速度" in full_html)
    check("対数軸チェックボックス", "PER軸を対数表示" in full_html)
    check("赤字含めるチェックボックス", "赤字銘柄も含める" in full_html)
    check("業種別中央値expander", "業種別" in full_html and "中央値" in full_html)

    # Plotly chart が描画されているか（または「データなし」メッセージか）
    has_plot = page.locator(".js-plotly-plot").count() > 0
    has_no_data_msg = "PER/PBR履歴データがまだありません" in full_html
    check(
        "Plotlyグラフ存在 or データなしメッセージ",
        has_plot or has_no_data_msg,
        f"plot_count={page.locator('.js-plotly-plot').count()}, no_data={has_no_data_msg}",
    )

    browser.close()

print("\n" + "=" * 50)
passed = sum(1 for s, *_ in results if s == "PASS")
failed = sum(1 for s, *_ in results if s == "FAIL")
print(f"Result: {passed} PASS / {failed} FAIL")
for s, label, detail in results:
    if s == "FAIL":
        print(f"  [FAIL] {label}: {detail}")
if failed:
    sys.exit(1)
else:
    print("All E2E tests PASSED")
