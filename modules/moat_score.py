"""バフェット流 Moat Score エンジン — 7軸スコアリング。

7軸 (各 0-10) を重み付きで合算し、銘柄の経済的堀の強さを定量化する。

重み構成:
    technical:     10%  波形 / ブレイクアウト状態
    fundamental:   25%  CES 資本効率スコア (ローカル parquet)
    foreign_flow:  10%  海外投資家フロー直近4週累積
    growth:        15%  naibu SQLite 多年度 net_income 成長率
    growth_sector: 10%  policy_signals.json × 業種タグ
    moat_pp:       20%  naibu HTTPX /api/pricing-power/companies/{edinet_code}
    policy:        10%  policy_signals.json strength × 業種タグ合致

naibu API 落ち時:
    moat_pp → None, total_score → None, explanation.errors に追記

naibu SQLite 落ち時:
    growth → None (軽量 fallback: 市場平均で 5.0 を返す代わりに None)
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from config.settings import (
    DATA_DIR,
    NAIBU_API_BASE_URL,
    NAIBU_DB_PATH,
    NAIBU_FETCH_TIMEOUT_SEC,
    RESULTS_CSV,
)
from modules.capital_efficiency_screener import load_screening_result
from modules.foreign_flow_analyzer import compute_cumulative_flow, load_foreign_flow
from modules.naibu_client import naibu_db_exists, to_edinet_securities_code

logger = logging.getLogger(__name__)

POLICY_SIGNALS_PATH = DATA_DIR / "policy_signals.json"
MOAT_SCORES_PARQUET = DATA_DIR / "moat_scores.parquet"

WEIGHTS: dict[str, float] = {
    "technical": 0.10,
    "fundamental": 0.25,
    "foreign_flow": 0.10,
    "growth": 0.15,
    "growth_sector": 0.10,
    "moat_pp": 0.20,
    "policy": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# wave_types (日本語ラベル) → テクニカルスコア (0-10)
_WAVE_TYPE_SCORES: dict[str, float] = {
    "上昇トレンド": 7.5,
    "ブレイク気味": 8.5,
    "収束（スクイーズ）": 6.0,
    "レンジ（波型）": 5.0,
    "高ボラ（荒い）": 4.0,
    "下降トレンド": 2.0,
    "データ不足": 4.0,
    "未分類": 4.5,
}
_WAVE_DEFAULT = 4.5


def _load_results() -> pd.DataFrame | None:
    if not RESULTS_CSV.exists():
        return None
    return pd.read_csv(RESULTS_CSV, encoding="utf-8-sig", dtype={"code": str})


def _load_policy_signals() -> list[dict]:
    if not POLICY_SIGNALS_PATH.exists():
        return []
    with POLICY_SIGNALS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("signals", [])


def _compute_technical(code: str, results_df: pd.DataFrame | None) -> float:
    if results_df is None or results_df.empty:
        return _WAVE_DEFAULT
    row = results_df[results_df["code"] == code]
    if row.empty:
        return _WAVE_DEFAULT
    wave_types_str = str(row.iloc[0].get("wave_types", ""))
    if not wave_types_str or wave_types_str == "nan":
        return _WAVE_DEFAULT
    types = [t.strip() for t in wave_types_str.split("|") if t.strip()]
    scores = [_WAVE_TYPE_SCORES.get(t, _WAVE_DEFAULT) for t in types]
    return float(max(scores)) if scores else _WAVE_DEFAULT


def _compute_fundamental(code: str) -> float | None:
    ces_df = load_screening_result()
    if ces_df is None or ces_df.empty:
        return None
    row = ces_df[ces_df["code"] == code]
    if row.empty:
        return None
    score = row.iloc[0].get("score")
    if score is None or pd.isna(score):
        return None
    return float(min(max(float(score), 0.0), 10.0))


def _compute_foreign_flow() -> float:
    ff_df = load_foreign_flow()
    if ff_df is None or ff_df.empty:
        return 5.0
    cum = compute_cumulative_flow(ff_df)
    if cum is None or cum.empty or len(cum) < 2:
        return 5.0
    recent_4w = float(cum.iloc[-1]) - float(cum.iloc[-min(21, len(cum))])
    if recent_4w >= 0:
        return float(
            min(5.0 + recent_4w / max(abs(float(cum.iloc[-1])), 1e-9) * 5.0, 10.0)
        )
    else:
        return float(
            max(5.0 + recent_4w / max(abs(float(cum.iloc[-1])), 1e-9) * 5.0, 0.0)
        )


def _compute_growth_from_db(code: str) -> float | None:
    if not naibu_db_exists():
        return None
    import sqlite3

    edinet_code5 = to_edinet_securities_code(code)
    uri = f"file:{Path(NAIBU_DB_PATH).as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT co.edinet_code FROM companies co WHERE co.securities_code=? LIMIT 1",
            (edinet_code5,),
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return None
        edinet_code = row[0]
        cur.execute(
            """SELECT net_income FROM financial_metrics
               WHERE edinet_code=? AND net_income IS NOT NULL
               ORDER BY fiscal_year_end DESC LIMIT 3""",
            (edinet_code,),
        )
        rows = cur.fetchall()
        conn.close()
        if len(rows) < 2:
            return None
        latest = float(rows[0][0])
        oldest = float(rows[-1][0])
        if oldest <= 0:
            return None
        cagr = (latest / oldest) ** (1.0 / max(len(rows) - 1, 1)) - 1.0
        # cagr → score: 20%+ → 10, 10% → 8, 0% → 5, -10% → 2, -20%以下 → 0
        score = 5.0 + cagr * 25.0
        return float(min(max(score, 0.0), 10.0))
    except Exception as e:
        logger.warning(f"growth DB error for {code}: {e}")
        return None


def _compute_sector_and_policy(
    code: str,
    results_df: pd.DataFrame | None,
    signals: list[dict],
) -> tuple[float, float]:
    if results_df is None or results_df.empty or not signals:
        return 5.0, 5.0
    row = results_df[results_df["code"] == code]
    industry = ""
    if not row.empty:
        industry = str(row.iloc[0].get("sector_33", ""))

    max_strength = 3.0
    matched_strengths: list[int] = []
    for sig in signals:
        tags = sig.get("sector_tags", [])
        strength = int(sig.get("strength", 0))
        if any(tag in industry for tag in tags):
            matched_strengths.append(strength)

    if not matched_strengths:
        return 5.0, 5.0

    avg_strength = sum(matched_strengths) / len(matched_strengths)
    sector_score = float(avg_strength / max_strength * 10.0)
    policy_score = float(max(matched_strengths) / max_strength * 10.0)
    return sector_score, policy_score


def _compute_pp_from_naibu(code: str) -> float | None:
    edinet_code5 = to_edinet_securities_code(code)
    url = f"{NAIBU_API_BASE_URL}/api/pricing-power/companies/{edinet_code5}"
    try:
        with httpx.Client(timeout=NAIBU_FETCH_TIMEOUT_SEC) as client:
            r = client.get(url)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            pp = data.get("pricing_power")
            if pp is None:
                return None
            return float(min(max(float(pp) * 10.0, 0.0), 10.0))
    except Exception as e:
        logger.warning(f"naibu PP API unreachable for {code}: {e}")
        return None


class MoatScoreEngine:
    """7軸バフェット流 MoatScore 算出エンジン。"""

    def compute(self, code: str) -> dict[str, Any]:
        today = date.today().isoformat()
        errors: list[str] = []
        explanation: dict[str, str] = {}

        results_df = _load_results()
        signals = _load_policy_signals()

        # --- 軸1: テクニカル ---
        ax_tech = _compute_technical(code, results_df)
        explanation["technical"] = f"wave_types score={ax_tech:.1f}"

        # --- 軸2: ファンダメンタル ---
        ax_fund = _compute_fundamental(code)
        if ax_fund is None:
            ax_fund = 5.0
            explanation["fundamental"] = "CES データなし, デフォルト5.0"
        else:
            explanation["fundamental"] = f"CES total_score={ax_fund:.1f}"

        # --- 軸3: 外国人フロー ---
        ax_flow = _compute_foreign_flow()
        explanation["foreign_flow"] = f"直近4週累積フロー score={ax_flow:.1f}"

        # --- 軸4: 成長 ---
        ax_growth = _compute_growth_from_db(code)
        if ax_growth is None:
            errors.append("naibu DB unreachable or data missing for growth")
            explanation["growth"] = "naibu DB データなし"
        else:
            explanation["growth"] = f"net_income CAGR proxy={ax_growth:.1f}"

        # --- 軸5/7: 成長分野 / 政府政策 ---
        ax_sector, ax_policy = _compute_sector_and_policy(code, results_df, signals)
        explanation["growth_sector"] = f"policy sector match score={ax_sector:.1f}"
        explanation["policy"] = f"policy strength score={ax_policy:.1f}"

        # --- 軸6: Pricing Power (naibu HTTPX) ---
        ax_pp = _compute_pp_from_naibu(code)
        if ax_pp is None:
            errors.append("naibu API unreachable")
            explanation["moat_pp"] = "naibu API 未起動またはデータなし"
        else:
            explanation["moat_pp"] = f"pricing_power score={ax_pp:.1f}"

        explanation["errors"] = ", ".join(errors) if errors else ""

        # --- 合計スコア ---
        if ax_pp is None:
            total_score = None
        else:
            growth_val = ax_growth if ax_growth is not None else 5.0
            total_score = round(
                ax_tech * WEIGHTS["technical"]
                + ax_fund * WEIGHTS["fundamental"]
                + ax_flow * WEIGHTS["foreign_flow"]
                + growth_val * WEIGHTS["growth"]
                + ax_sector * WEIGHTS["growth_sector"]
                + ax_pp * WEIGHTS["moat_pp"]
                + ax_policy * WEIGHTS["policy"],
                2,
            )

        return {
            "securities_code": to_edinet_securities_code(code),
            "code": code,
            "date": today,
            "axis_technical": ax_tech,
            "axis_fundamental": ax_fund,
            "axis_foreign_flow": ax_flow,
            "axis_growth": ax_growth,
            "axis_growth_sector": ax_sector,
            "axis_moat_pp": ax_pp,
            "axis_policy": ax_policy,
            "total_score": total_score,
            "rank": None,
            "explanation": explanation,
        }

    def compute_bulk(self, codes: list[str]) -> list[dict[str, Any]]:
        results = []
        for code in codes:
            try:
                results.append(self.compute(code))
            except Exception as e:
                logger.error(f"MoatScoreEngine.compute_bulk error for {code}: {e}")
        scored = [r for r in results if r["total_score"] is not None]
        scored.sort(key=lambda r: r["total_score"], reverse=True)
        for i, r in enumerate(scored, 1):
            r["rank"] = i
        return results


def load_moat_scores() -> pd.DataFrame | None:
    if not MOAT_SCORES_PARQUET.exists():
        return None
    return pd.read_parquet(MOAT_SCORES_PARQUET)


def save_moat_scores(records: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k != "explanation"}
            for r in records
            if r["total_score"] is not None
        ]
    )
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(MOAT_SCORES_PARQUET, index=False)
    logger.info(f"moat_scores.parquet saved: {len(df)} rows")
