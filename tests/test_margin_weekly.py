"""週次信用残PDFパース モジュールの単体テスト。

実PDFを使う統合テストは別ファイル (test_margin_weekly_integration.py) に分離。
ここでは行レベルのパース正確性のみ検証する。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from modules.margin_fetcher import (
    _normalize_weekly_numbers,
    _parse_weekly_row,
    load_margin_weekly_history,
    load_margin_weekly_latest,
    parse_weekly_pdf,
)


def test_normalize_numbers_simple():
    """カンマ区切り + ▲負数 の基本パターン。"""
    text = "1,000 ▲ 200 3,000 4,000 5,000 6,000 7,000 8,000 9,000 10,000 11,000 12,000"
    nums = _normalize_weekly_numbers(text)
    assert nums == [1000, -200, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]


def test_normalize_numbers_stray_space_inside_number():
    """PDFテキスト抽出時のカンマ前後への不正空白挿入を吸収。"""
    # "1, 974,100" → 1974100
    text = "1, 974,100 2,000 3,000 4,000 5,000 6,000 7,000 8,000 9,000 10,000 11,000 12,000"
    nums = _normalize_weekly_numbers(text)
    assert nums is not None
    assert nums[0] == 1974100


def test_normalize_numbers_sbg_actual():
    """SBG (9984) の実データ行をパースできる。

    2026/5/29 申込み現在 の実PDFから抽出した行。
    """
    text = (
        "4,629,500 ▲ 888,600 26,718,500 2,489,300 1,356,500 ▲ 140 ,400 "
        "3,273,000 ▲ 748,200 10,278,100 515,200 16,440,400 1, 974,100"
    )
    nums = _normalize_weekly_numbers(text)
    assert nums == [
        4629500,
        -888600,
        26718500,
        2489300,
        1356500,
        -140400,
        3273000,
        -748200,
        10278100,
        515200,
        16440400,
        1974100,
    ]


def test_normalize_numbers_wrong_count_returns_none():
    """12個に満たない場合は None。"""
    text = "1,000 2,000 3,000"
    assert _normalize_weekly_numbers(text) is None


def test_normalize_numbers_double_space_after_marker():
    """▲ と数字の間に複数空白が入るパターン (2026/5/15 SBG)。

    "▲  1,231,100" → -1,231,100 として扱う。
    """
    text = (
        "5,853,200 ▲ 279,900 27,810,500 ▲ 2,085,200 1,427,900 43, 500 "
        "4,425,300 ▲ 323,400 10,463,000 ▲ 854,100 17,347,500 ▲  1,231,100"
    )
    nums = _normalize_weekly_numbers(text)
    assert nums is not None
    assert nums[-1] == -1231100, f"last expected -1,231,100, got {nums[-1]}"


def test_normalize_numbers_space_inside_digit_group():
    """数字の途中(カンマ無し位置)に空白が挿入されるパターン (2026/5/1 SBG)。

    "▲ 2 31,300" → -231,300 として扱う。
    "18,578,6 00" → 18,578,600 として扱う。
    """
    text = (
        "6,133,100 ▲ 2,071,400 29,895,700 1,252,200 1,384,400 ▲ 2 31,300 "
        "4,748,700 ▲ 1,840,100 11,317,100 475,200 18,578,6 00 777,000"
    )
    nums = _normalize_weekly_numbers(text)
    assert nums is not None
    # 一般信用売 前週比 (index 5) = -231,300
    assert nums[5] == -231300, f"expected -231,300, got {nums[5]}"
    # 制度信用買 残高 (index 10) = 18,578,600
    assert nums[10] == 18578600, f"expected 18,578,600, got {nums[10]}"


def test_normalize_numbers_4_24_sbg():
    """2026/4/24 SBG 行 (パターンD混在)。"""
    text = (
        "8,204,500 1,847,900 28,643,500 ▲ 8,612,100 1,615,700 240 ,400 "
        "6,588,800 1,607,500 10,841,900 ▲ 2,007,900 17,801,6 00 ▲ 6,604,200"
    )
    nums = _normalize_weekly_numbers(text)
    assert nums is not None
    assert nums[0] == 8204500  # 売残
    assert nums[2] == 28643500  # 買残
    assert nums[5] == 240400  # 一般信用売 前週比 = +240,400
    assert nums[10] == 17801600  # 制度信用買 残高


def test_normalize_numbers_all_zeros():
    """全ゼロケース (退化テスト - マージロジックが暴走しないこと)。"""
    text = "0 0 0 0 0 0 0 0 0 0 0 0"
    nums = _normalize_weekly_numbers(text)
    assert nums == [0] * 12


def test_parse_weekly_row_sbg():
    """SBG行を最初から最後までパース。code5/ISIN/信用倍率を確認。"""
    line = (
        "B ソフトバンクグループ　普通株式 99840 JP3436100006 "
        "4,629,500 ▲ 888,600 26,718,500 2,489,300 1,356,500 ▲ 140 ,400 "
        "3,273,000 ▲ 748,200 10,278,100 515,200 16,440,400 1, 974,100"
    )
    row = _parse_weekly_row(line)
    assert row is not None
    assert row["code5"] == "99840"
    assert row["isin"] == "JP3436100006"
    assert "ソフトバンクグループ" in row["name"]
    assert row["sell_balance"] == 4629500
    assert row["buy_balance"] == 26718500
    assert row["sell_change"] == -888600
    assert row["buy_change"] == 2489300
    # 信用倍率 = 買残/売残 ≒ 5.77
    assert abs(row["margin_ratio"] - 26718500 / 4629500) < 1e-9


def test_parse_weekly_row_negative_sell():
    """売残ゼロ時は margin_ratio = 9999 (上限)。"""
    line = (
        "B テスト銘柄　普通株式 12340 JP3000000000 "
        "0 0 100,000 0 0 0 0 0 50,000 0 50,000 0"
    )
    row = _parse_weekly_row(line)
    assert row is not None
    assert row["margin_ratio"] == 9999.0


def test_parse_weekly_row_zero_zero():
    """売残・買残ともゼロは margin_ratio = 0。"""
    line = (
        "B テスト銘柄　普通株式 12340 JP3000000000 "
        "0 0 0 0 0 0 0 0 0 0 0 0"
    )
    row = _parse_weekly_row(line)
    assert row is not None
    assert row["margin_ratio"] == 0.0


def test_parse_weekly_row_invalid_returns_none():
    """ヘッダー行など、ISIN/code5 を含まない行は None。"""
    assert _parse_weekly_row("") is None
    assert _parse_weekly_row("売残高 買残高 一般信用 制度信用") is None
    assert _parse_weekly_row("プライム Prime 1483 銘柄") is None


def test_parse_weekly_pdf_returns_dataframe_with_required_columns():
    """実PDF (キャッシュ済み) をパースして DataFrame カラムが揃うことを確認。"""
    pdf_path = "data/jpx_margin_weekly/syumatsu2026052900.pdf"
    from pathlib import Path

    p = Path(pdf_path)
    if not p.exists():
        # キャッシュにない場合は事前ダウンロードが必要 (E2Eで実行)
        import pytest

        pytest.skip(f"Sample PDF not cached: {pdf_path}")

    content = p.read_bytes()
    df = parse_weekly_pdf(content, date(2026, 5, 29))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000  # プライム1400銘柄+α
    required = {
        "code5", "isin", "name", "ticker", "code4",
        "sell_balance", "sell_change", "buy_balance", "buy_change",
        "margin_ratio", "observation_date",
    }
    assert required.issubset(set(df.columns))

    # SBG が含まれていること
    sbg = df[df["code5"] == "99840"]
    assert len(sbg) == 1, f"SBG should appear exactly once, got {len(sbg)}"
    assert sbg.iloc[0]["sell_balance"] == 4629500
    assert sbg.iloc[0]["buy_balance"] == 26718500
    assert abs(sbg.iloc[0]["margin_ratio"] - 26718500 / 4629500) < 1e-9


def test_e2e_weekly_data_persisted_and_queryable():
    """E2E スモーク: parquet 保存後の読み込み API が機能すること。

    update_margin_weekly_history を別途実行済みであることを前提とする
    (CIで毎回ダウンロードはしないため、parquet が無ければ skip)。
    """
    import pytest

    latest = load_margin_weekly_latest()
    if latest is None or len(latest) == 0:
        pytest.skip(
            "margin_weekly_latest.parquet が未生成。update_margin_weekly_history() を先に実行してください。"
        )

    # 必須カラム
    required = {
        "ticker", "code5", "name", "sell_balance", "buy_balance",
        "margin_ratio", "observation_date",
    }
    assert required.issubset(set(latest.columns))

    # プライム + standard 規模のカバレッジ (週次は全市場対象)
    assert len(latest) > 2000, f"too few tickers covered: {len(latest)}"

    # 信用倍率の妥当性
    finite_ratios = latest[(latest["margin_ratio"] > 0) & (latest["margin_ratio"] < 9999)]
    assert len(finite_ratios) > 1000

    # SBG カバー確認 (これが今回の実装目的)
    sbg_latest = latest[latest["ticker"] == "9984.T"]
    assert len(sbg_latest) == 1, "SBG (9984.T) must be in weekly latest"
    r = sbg_latest.iloc[0]
    assert r["buy_balance"] > 0
    assert r["sell_balance"] > 0
    assert 0.1 < r["margin_ratio"] < 100  # 極端でない値

    # 履歴側
    hist = load_margin_weekly_history("9984.T")
    assert hist is not None and len(hist) >= 1
    assert (hist["observation_date"] == hist["observation_date"]).all()
    assert all(hist["buy_balance"] > 0)
