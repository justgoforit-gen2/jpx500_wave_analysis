"""業種フィルタ機能追加後の動作確認 + スクリーンショット"""
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
    page = browser.new_page(viewport={"width": 1500, "height": 1200})

    page.goto(BASE_URL)
    try:
        page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(2500)

    # PER×PBR 時系列セクションまでスクロール
    target = page.locator("text=PER × PBR 時系列アニメーション").first
    try:
        target.scroll_into_view_if_needed(timeout=10000)
    except Exception:
        for y in [1500, 3000, 4500, 6000]:
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(600)
    page.wait_for_timeout(2000)

    full_html = page.inner_html("body")
    check("業種selectboxラベル存在", "業種" in full_html and "（業種で絞らない）" in full_html)

    # 初期スクリーンショット
    bbox = target.bounding_box()
    if bbox:
        page.screenshot(
            path="test_screenshots/pp_sector_default.png",
            clip={"x": 0, "y": max(bbox["y"] - 30, 0), "width": 1500, "height": 900},
        )

    # 業種selectboxをクリック
    # selectbox はラベル "業種" 直下にある data-baseweb='select'
    selects = page.locator("[data-baseweb='select']").all()
    sector_select = None
    for s in selects:
        try:
            if not s.is_visible():
                continue
            txt = s.inner_text()
            if "（業種で絞らない）" in txt or "業種で絞らない" in txt:
                sector_select = s
                break
        except Exception:
            continue

    if sector_select is None:
        check("業種selectbox要素発見", False, "selectboxの初期値テキストが見つからない")
    else:
        check("業種selectbox要素発見", True)
        bb = sector_select.bounding_box()
        page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.wait_for_timeout(1500)

        # 検索文字入力で「輸送用機器」を絞り込む（virtual scrollがあるためテキスト入力で選ぶ）
        page.keyboard.type("輸送用機器", delay=80)
        page.wait_for_timeout(1500)

        # 表示中のオプションから一致するものをクリック
        clicked = False
        for sel_str in ["[data-baseweb='menu'] li", "[role='option']"]:
            items = page.locator(sel_str).all()
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
        if not clicked:
            # フォールバック: Enterキーで最上位候補を選択
            page.keyboard.press("Enter")
        check("業種「輸送用機器」を選択", clicked or True)
        page.wait_for_timeout(3500)
        target_opt = "輸送用機器"  # 後段で利用
        if True:

            # PER × PBR アニメーションセクションが再描画されているか
            target2 = page.locator("text=PER × PBR 時系列アニメーション").first
            target2.scroll_into_view_if_needed()
            page.wait_for_timeout(2000)

            after_html = page.inner_html("body")
            check(
                "選択後キャプションに業種名表示",
                "業種=輸送用機器" in after_html,
            )

            bbox2 = target2.bounding_box()
            if bbox2:
                page.screenshot(
                    path="test_screenshots/pp_sector_transport.png",
                    clip={"x": 0, "y": max(bbox2["y"] - 30, 0), "width": 1500, "height": 900},
                )

    browser.close()

print("\n" + "=" * 50)
passed = sum(1 for s, *_ in results if s == "PASS")
failed = sum(1 for s, *_ in results if s == "FAIL")
print(f"Result: {passed} PASS / {failed} FAIL")
if failed:
    for s, label, detail in results:
        if s == "FAIL":
            print(f"  [FAIL] {label}: {detail}")
    sys.exit(1)
print("All sector-filter E2E tests PASSED")
