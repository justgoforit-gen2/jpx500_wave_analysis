"""ユニットテスト: asset_value_tab の信頼性判別ガード関数"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# asset_value_tab はトップレベルにある
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset_value_tab import _classify_filer, _land_sanity


class TestClassifyFiler:
    def test_bank_by_ordinance_code(self):
        assert _classify_filer("030", "なんでもいい") == "bank"

    def test_insurance_by_ordinance_code(self):
        assert _classify_filer("040", "なんでもいい") == "insurance"

    def test_bank_by_keyword_financial_group(self):
        assert _classify_filer("010", "株式会社京都フィナンシャルグループ") == "bank"

    def test_bank_by_keyword_bank(self):
        assert _classify_filer("010", "三菱UFJ銀行") == "bank"

    def test_bank_by_keyword_shinkin(self):
        assert _classify_filer("010", "京都中央信用金庫") == "bank"

    def test_insurance_by_keyword(self):
        assert _classify_filer("010", "東京海上日動火災保険") == "insurance"

    def test_insurance_by_keyword_seimei(self):
        assert _classify_filer("010", "日本生命保険相互会社") == "insurance"

    def test_general_manufacturer(self):
        assert _classify_filer("010", "日産自動車株式会社") == "general"

    def test_ordinance_code_takes_priority_over_keyword(self):
        # 府令コードが 030 なら keyword は無視
        assert _classify_filer("030", "日産自動車株式会社") == "bank"


class TestLandSanity:
    def _make_df(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"土地簿価_百万円": values})

    def test_reliable_ratio_within_5(self):
        df = self._make_df([100.0, 200.0])  # sum=300
        bs_yen = 300 * 1_000_000  # ratio=1.0
        ok, ratio = _land_sanity(df, bs_yen)
        assert ok is True
        assert ratio == pytest.approx(1.0)

    def test_unreliable_ratio_over_5(self):
        df = self._make_df([3000.0])  # sum=3000
        bs_yen = 100 * 1_000_000  # ratio=30.0
        ok, ratio = _land_sanity(df, bs_yen)
        assert ok is False
        assert ratio == pytest.approx(30.0)

    def test_none_properties_df(self):
        ok, ratio = _land_sanity(None, 100_000_000)
        assert ok is True
        assert ratio is None

    def test_empty_df(self):
        ok, ratio = _land_sanity(pd.DataFrame(), 100_000_000)
        assert ok is True
        assert ratio is None

    def test_none_bs_land(self):
        df = self._make_df([100.0])
        ok, ratio = _land_sanity(df, None)
        assert ok is True
        assert ratio is None

    def test_zero_bs_land(self):
        df = self._make_df([100.0])
        ok, ratio = _land_sanity(df, 0)
        assert ok is True
        assert ratio is None

    def test_zero_parsed_sum(self):
        df = self._make_df([0.0])
        ok, ratio = _land_sanity(df, 100_000_000)
        assert ok is True
        assert ratio is None

    def test_nan_values_ignored(self):
        import numpy as np
        df = self._make_df([float("nan"), 200.0])  # sum=200
        bs_yen = 200 * 1_000_000  # ratio=1.0
        ok, ratio = _land_sanity(df, bs_yen)
        assert ok is True
