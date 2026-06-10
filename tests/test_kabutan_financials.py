"""株探(kabutan.jp)業績ページパーサーのテスト。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from modules.kabutan_financials_fetcher import (
    _parse_announce_date,
    _parse_number,
    _parse_period_cell,
    _ticker_to_code4,
    fetch_kabutan_financials,
    get_latest_actual_period,
    parse_finance_html,
)


def test_ticker_to_code4():
    assert _ticker_to_code4("6146.T") == "6146"
    assert _ticker_to_code4("9984.T") == "9984"


def test_parse_period_cell_actual():
    """通常の期セル (2026.03 形式)。"""
    assert _parse_period_cell("&nbsp;　&nbsp;2026.03&nbsp;&nbsp;") == (
        "2026年3月期",
        False,
    )
    assert _parse_period_cell("2025.03") == ("2025年3月期", False)


def test_parse_period_cell_forecast():
    """予想セル (予 接頭辞付き)。"""
    assert _parse_period_cell("&nbsp;予&nbsp;2027.03&nbsp;&nbsp;") == (
        "2027年3月期",
        True,
    )
    assert _parse_period_cell("予 2027.03") == ("2027年3月期", True)


def test_parse_period_cell_invalid():
    assert _parse_period_cell("") is None
    assert _parse_period_cell("業績") is None
    assert _parse_period_cell("---") is None


def test_parse_number():
    assert _parse_number("4,368,890") == 4368890.0
    assert _parse_number("42.42") == 42.42
    assert _parse_number("") is None
    assert _parse_number("－") is None
    assert _parse_number("-") is None
    assert _parse_number("‐") is None


def test_parse_announce_date():
    assert _parse_announce_date("26/04/22") == date(2026, 4, 22)
    assert _parse_announce_date("25/04/17") == date(2025, 4, 17)
    assert _parse_announce_date("") is None
    assert _parse_announce_date("－") is None


def test_parse_finance_html_disco_actual():
    """DISCO実HTMLから2026年3月期実績が正しく抽出される。

    キャッシュHTMLが無ければスキップ。
    """
    cache = Path("data/kabutan_finance/6146.html")
    if not cache.exists():
        pytest.skip("DISCO finance cache not available")

    html = cache.read_text(encoding="utf-8")
    df = parse_finance_html(html)
    assert len(df) >= 3

    # 2026年3月期実績を確認 (Web発表値: 売上4,368.89億, 営業利益率42.3%)
    target = df[df["period"] == "2026年3月期"]
    assert len(target) == 1
    row = target.iloc[0]
    assert row["is_forecast"] == False
    assert abs(row["revenue"] - 4368.9) < 1.0
    assert abs(row["op_margin"] - 42.34) < 0.2
    assert row["announce_date"] == date(2026, 4, 22)


def test_get_latest_actual_period_returns_newest_actual():
    df = pd.DataFrame(
        [
            {"period": "2024年3月期", "is_forecast": False, "revenue": 100, "op_margin": 10, "eps": 5, "announce_date": None},
            {"period": "2025年3月期", "is_forecast": False, "revenue": 110, "op_margin": 11, "eps": 6, "announce_date": None},
            {"period": "2026年3月期", "is_forecast": False, "revenue": 120, "op_margin": 12, "eps": 7, "announce_date": None},
            {"period": "2027年3月期", "is_forecast": True, "revenue": None, "op_margin": None, "eps": None, "announce_date": None},
        ]
    )
    assert get_latest_actual_period(df) == "2026年3月期"


def test_get_latest_actual_period_no_actual():
    """予想行しかない場合は None。"""
    df = pd.DataFrame(
        [{"period": "2027年3月期", "is_forecast": True, "revenue": None, "op_margin": None, "eps": None, "announce_date": None}]
    )
    assert get_latest_actual_period(df) is None


def test_fetch_kabutan_financials_disco_live():
    """実フェッチ: DISCO 2026年3月期実績が含まれる (キャッシュ使用)。"""
    df = fetch_kabutan_financials("6146.T")
    if df is None or len(df) == 0:
        pytest.skip("kabutan fetch returned None (offline or blocked)")
    # 2026/3期実績が含まれている
    assert "2026年3月期" in df["period"].tolist()
    actual_2026 = df[(df["period"] == "2026年3月期") & (df["is_forecast"] == False)]
    assert len(actual_2026) == 1
    assert actual_2026.iloc[0]["revenue"] > 4000  # 4,368.9億


def test_fetch_financials_uses_kabutan_first():
    """fetch_financials が株探優先になっている (2026/3期実績取得)。"""
    from modules.data_fetcher import fetch_financials

    df = fetch_financials("6146.T")
    if df is None:
        pytest.skip("fetch_financials returned None")
    # 株探経由なら 2026年3月期 が含まれているはず
    # (yfinanceなら 2025年3月期 が最新で 2026年3月期 はNaN/欠落)
    actual_periods = df[df["is_forecast"] == False]["period"].tolist()
    assert "2026年3月期" in actual_periods, (
        f"株探優先なら2026年3月期実績が含まれるはず。実際: {actual_periods}"
    )
