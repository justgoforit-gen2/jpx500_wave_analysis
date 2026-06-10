"""株探(kabutan.jp)信用残スクレイパーの単体テスト。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from modules.kabutan_margin_fetcher import (
    _parse_number,
    _parse_yy_date,
    _ticker_to_code4,
    parse_kabutan_html,
)


def test_ticker_to_code4():
    assert _ticker_to_code4("9984.T") == "9984"
    assert _ticker_to_code4("4661.T") == "4661"


def test_parse_yy_date_valid():
    assert _parse_yy_date("26/05/29") == date(2026, 5, 29)
    assert _parse_yy_date("23/07/28") == date(2023, 7, 28)


def test_parse_yy_date_invalid():
    assert _parse_yy_date("") is None
    assert _parse_yy_date("2026/05/29") is None  # 4桁年は対象外
    assert _parse_yy_date("abc") is None
    assert _parse_yy_date("26/02/30") is None  # 不正日付


def test_parse_number_valid():
    assert _parse_number("4,629,500") == 4629500.0
    assert _parse_number("5.77") == 5.77
    assert _parse_number("-1.23") == -1.23
    assert _parse_number("+0.53") == 0.53


def test_parse_number_invalid():
    assert _parse_number("") is None
    assert _parse_number("－") is None  # 全角ハイフン (株探の欠損値)
    assert _parse_number("-") is None
    assert _parse_number("abc") is None


def test_parse_kabutan_html_real_sbg_snippet():
    """株探SBGページから抽出した実HTMLスニペットをパースできる。

    実データ確認: 2026-05-29 SBG 信用倍率 5.77 (JPX 週次データと一致)。
    """
    # 株探のテーブル構造を模した最小HTML
    html = """
    <table>
        <tr><th>日付</th><th>終値</th><th>前週比%</th><th>売買単価</th>
            <th>売買高(株)</th><th>売残高</th><th>買残高</th><th>信用倍率</th></tr>
        <tr><td>26/05/29</td><td>7,491</td><td>+10.86</td><td>7,384</td>
            <td>466,599,700</td><td>4,629,500</td><td>26,718,500</td><td>5.77</td></tr>
        <tr><td>26/05/22</td><td>6,757</td><td>+17.62</td><td>5,866</td>
            <td>350,080,400</td><td>5,518,100</td><td>24,229,200</td><td>4.39</td></tr>
        <tr><td>24/01/05</td><td>1,520</td><td>-3.37</td><td>1,516</td>
            <td>62,240,400</td><td>－</td><td>－</td><td>－</td></tr>
    </table>
    """
    df = parse_kabutan_html(html, "9984.T")
    assert isinstance(df, pd.DataFrame)
    # 信用残データのある2行のみ取り込まれる (3行目は全欠損のため除外)
    assert len(df) == 2
    assert set(df.columns) == {
        "ticker", "observation_date", "close",
        "sell_balance", "buy_balance", "margin_ratio",
    }
    # 5/29
    r0 = df[df["observation_date"] == date(2026, 5, 29)].iloc[0]
    assert r0["sell_balance"] == 4629500.0
    assert r0["buy_balance"] == 26718500.0
    assert r0["margin_ratio"] == 5.77
    assert r0["ticker"] == "9984.T"
    # 5/22
    r1 = df[df["observation_date"] == date(2026, 5, 22)].iloc[0]
    assert r1["margin_ratio"] == 4.39


def test_parse_kabutan_html_empty():
    """テーブル無しのHTMLは空DataFrameを返す。"""
    df = parse_kabutan_html("<html><body>nothing here</body></html>", "9984.T")
    assert len(df) == 0


def test_parse_kabutan_html_excludes_current_week_ohlcv_row():
    """直近週は信用残未確定で OHLCV モード(別レイアウト)になる。これを除外。

    実データ例 (2026/6/5 SBG):
        日付, 終値, 高値(9,074), 安値, 始値, 前週比, 前週比%, 売買高
        → cell[2] が "9,074" (価格) なので margin row ではないと判定すべき。
    """
    html = """
    <table>
        <tr><td>26/06/05</td><td>7,498</td><td>9,074</td><td>7,105</td>
            <td>7,528</td><td>+37</td><td>+0.49</td><td>436,928,100</td></tr>
        <tr><td>26/05/29</td><td>7,491</td><td>+10.86</td><td>7,384</td>
            <td>466,599,700</td><td>4,629,500</td><td>26,718,500</td><td>5.77</td></tr>
    </table>
    """
    df = parse_kabutan_html(html, "9984.T")
    # 直近週は除外され、margin row だけ拾われる
    assert len(df) == 1
    assert df.iloc[0]["observation_date"] == date(2026, 5, 29)
    assert df.iloc[0]["margin_ratio"] == 5.77


def test_parse_kabutan_html_real_cached_page():
    """実キャッシュページからのパースが SBG の既知値と一致するか。

    キャッシュが無ければスキップ。
    """
    import pytest
    from pathlib import Path

    cache = Path("data/kabutan_margin/9984_p1.html")
    if not cache.exists():
        pytest.skip("kabutan cache not yet generated")

    html = cache.read_text(encoding="utf-8")
    df = parse_kabutan_html(html, "9984.T")
    assert len(df) >= 20  # 30週分くらい入っているはず

    # 5/29 の SBG 値を確認 (JPX 週次データと一致するはず)
    target = df[df["observation_date"] == date(2026, 5, 29)]
    assert len(target) == 1, "2026-05-29 SBG row should exist"
    row = target.iloc[0]
    assert row["sell_balance"] == 4629500.0
    assert row["buy_balance"] == 26718500.0
    assert abs(row["margin_ratio"] - 5.77) < 0.01
