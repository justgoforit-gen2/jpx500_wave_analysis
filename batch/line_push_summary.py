"""LINE Messaging API へ日次更新サマリをPush送信する。

想定: GitHub Actions などのCIで batch/update.py 実行後に呼び出す。
必要な環境変数:
  - LINE_CHANNEL_ACCESS_TOKEN: チャネルアクセストークン（長期）
  - LINE_TO: 送信先（ユーザーID / グループID / ルームID）
任意:
  - UPDATE_EXIT_CODE: update.py の終了コード
  - STREAMLIT_APP_URL: アプリURL（メッセージ末尾に追記）
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def _base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_results_summary(results_csv: Path) -> dict[str, Any]:
    if not results_csv.exists():
        return {"results_total": None, "data_shortage": None}

    total = 0
    data_shortage = 0

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            wave_types = (row.get("wave_types") or "").strip()
            if wave_types == "データ不足":
                data_shortage += 1

    return {"results_total": total, "data_shortage": data_shortage}


def _read_picks_summary(picks_csv: Path, max_lines: int = 5) -> dict[str, Any]:
    if not picks_csv.exists():
        return {
            "picks_total": None,
            "picks_low_touch": None,
            "picks_high_touch": None,
            "picks_preview": [],
        }

    total = 0
    low_touch = 0
    high_touch = 0
    preview: list[str] = []

    with picks_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            pick_type = row.get("pick_type") or ""
            if "下タッチ" in pick_type:
                low_touch += 1
            if "上タッチ" in pick_type:
                high_touch += 1

            if len(preview) < max_lines:
                code = (row.get("code") or "").strip()
                name = (row.get("name") or "").strip()
                latest_close = (row.get("latest_close") or "").strip()
                position_pct = (row.get("position_pct") or "").strip()

                # 値が空でも崩れにくい表示
                price_txt = f"¥{latest_close}" if latest_close else ""
                pos_txt = f"位置{position_pct}%" if position_pct else ""
                detail = " ".join(x for x in [pick_type, price_txt, pos_txt] if x)
                preview.append(f"- {code} {name} {detail}".strip())

    return {
        "picks_total": total,
        "picks_low_touch": low_touch,
        "picks_high_touch": high_touch,
        "picks_preview": preview,
    }


def build_message() -> str:
    base_dir = _base_dir()
    results_csv = base_dir / "data" / "results.csv"
    picks_csv = base_dir / "data" / "daily_picks.csv"

    update_exit_code = os.getenv("UPDATE_EXIT_CODE")
    app_url = os.getenv("STREAMLIT_APP_URL")

    # 日付はJST固定（ActionsのTZに依存させない）
    jst = dt.timezone(dt.timedelta(hours=9))
    today = dt.datetime.now(tz=jst).strftime("%Y-%m-%d")

    results = _read_results_summary(results_csv)
    picks = _read_picks_summary(picks_csv)

    status = (
        "OK"
        if (update_exit_code in (None, "0", 0))
        else f"NG (exit {update_exit_code})"
    )

    lines: list[str] = []
    lines.append(f"【JPX500 日次更新】{today}")
    lines.append(f"更新: {status}")

    if results["results_total"] is not None:
        total = results["results_total"]
        shortage = results["data_shortage"]
        ok = total - shortage if (total is not None and shortage is not None) else None
        if ok is not None and shortage is not None:
            lines.append(f"分類: {ok}/{total}（データ不足: {shortage}）")

    if picks["picks_total"] is not None:
        lines.append(
            f"本日推奨: {picks['picks_total']}（下タッチ: {picks['picks_low_touch']} / 上タッチ: {picks['picks_high_touch']}）"
        )

    if picks.get("picks_preview"):
        lines.append("上位:")
        lines.extend(picks["picks_preview"])

    if app_url:
        lines.append("")
        lines.append(f"App: {app_url}")

    return "\n".join(lines)


def push_line_message(channel_access_token: str, to: str, text: str) -> None:
    url = "https://api.line.me/v2/bot/message/push"
    body = {
        "to": to,
        "messages": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }

    req = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {channel_access_token}",
        },
        method="POST",
    )

    # bandit B310: URL は LINE Messaging API への固定文字列でユーザー入力経路なし、安全。
    with urlopen(req, timeout=30) as resp:  # nosec B310
        # 成功時は204 No Content
        if resp.status not in (200, 201, 202, 204):
            raise RuntimeError(f"LINE push failed: HTTP {resp.status}")


def main() -> int:
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    to = os.getenv("LINE_TO")

    if not token:
        print("ERROR: LINE_CHANNEL_ACCESS_TOKEN is required", file=sys.stderr)
        return 2
    if not to:
        print("ERROR: LINE_TO is required", file=sys.stderr)
        return 2

    msg = build_message()
    push_line_message(token, to, msg)
    print("LINE push sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
