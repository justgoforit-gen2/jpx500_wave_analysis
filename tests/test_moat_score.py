"""MoatScoreEngine のユニットテスト。"""

from __future__ import annotations

from unittest.mock import patch


def test_weights_sum():
    from modules.moat_score import WEIGHTS
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_moat_score_basic_7203():
    from modules.moat_score import MoatScoreEngine
    engine = MoatScoreEngine()
    result = engine.compute("7203")

    assert result["code"] == "7203"
    assert result["securities_code"] == "72030"
    assert "date" in result

    # 各軸は 0-10 の範囲 or None
    for axis in ["axis_technical", "axis_fundamental", "axis_foreign_flow",
                 "axis_growth_sector", "axis_policy"]:
        val = result[axis]
        assert val is None or 0.0 <= val <= 10.0, f"{axis}={val} out of range"

    # technical と fundamental は常に値を持つ (ローカルデータから計算)
    assert result["axis_technical"] is not None
    assert result["axis_fundamental"] is not None


def test_moat_score_naibu_fallback():
    """naibu API が落ちているときの fallback テスト。"""
    from modules.moat_score import MoatScoreEngine

    def mock_pp(*args, **kwargs):
        return None

    with patch("modules.moat_score._compute_pp_from_naibu", mock_pp):
        engine = MoatScoreEngine()
        result = engine.compute("7203")

    assert result["total_score"] is None
    errors = result["explanation"].get("errors", "")
    assert "naibu API unreachable" in errors


def test_moat_score_technical_range():
    """wave_type マッピングが 0-10 範囲に収まること。"""
    from modules.moat_score import _WAVE_TYPE_SCORES
    for k, v in _WAVE_TYPE_SCORES.items():
        assert 0.0 <= v <= 10.0, f"wave_type {k}={v} out of range"


def test_moat_score_multi_codes():
    """複数銘柄を compute_bulk で算出できること。"""
    from modules.moat_score import MoatScoreEngine
    engine = MoatScoreEngine()
    results = engine.compute_bulk(["7203", "6861"])
    assert len(results) == 2
    for r in results:
        assert "code" in r
        assert "explanation" in r


def test_moat_score_api_health():
    """api_server が import できること (starlette バージョン問題の回帰検出)。"""
    import api_server  # noqa: F401
    assert api_server.app is not None
