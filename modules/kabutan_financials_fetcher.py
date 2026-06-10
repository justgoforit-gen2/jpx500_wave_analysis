"""株探(kabutan.jp)の業績ページから通期業績推移をスクレイピングする。

yfinanceの日本企業データは最新年度の反映が数ヶ月遅延するため、
最新年度の実績を確実に取り込みたい場合は株探から補完する。

URL: https://kabutan.jp/stock/finance?code={4桁}

ページ構造:
    通期業績推移 (連結) テーブル:
        日付YY.MM 形式の決算期 + 売上/営業益/経常益/最終益/EPS/配当/発表日
        最後の行は "予" 接頭辞付きで会社予想 (値が "－" の場合あり)

出力フォーマット:
    DataFrame columns = period, revenue, op_margin, eps, is_forecast, announce_date
        - period: "2026年3月期" 形式
        - revenue: 億円
        - op_margin: 営業利益率 (%)
        - eps: 1株当たり利益 (円)
        - is_forecast: 予想行ならTrue
        - announce_date: 発表日 (YYYY-MM-DD, 欠損可)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    KABUTAN_FETCH_SLEEP_SEC,
    KABUTAN_FETCH_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

KABUTAN_FINANCE_URL = "https://kabutan.jp/stock/finance?code={code4}"
KABUTAN_FINANCE_CACHE_DIR = Path("data/kabutan_finance")
KABUTAN_FINANCE_CACHE_MAX_AGE_HOURS = 24

_TR_PATTERN = re.compile(r"<tr[^>]*>(.+?)</tr>", re.DOTALL)
_TD_PATTERN = re.compile(r"<t[hd][^>]*>(.+?)</t[hd]>", re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_NBSP = "　"  # 全角空白
# 期マッチ: "2026.03" 形式 (前後に &nbsp; が付くことがある)
_PERIOD_PATTERN = re.compile(r"(\d{4})\.(\d{2})")
# 発表日: 24/04/25
_ANNOUNCE_DATE_PATTERN = re.compile(r"^(\d{2})/(\d{2})/(\d{2})$")


def _ticker_to_code4(ticker: str) -> str:
    return ticker.split(".")[0]


def _cache_path(code4: str) -> Path:
    return KABUTAN_FINANCE_CACHE_DIR / f"{code4}.html"


def _parse_number(s: str) -> float | None:
    """ "393,313" → 393313.0, "－"/"-" → None"""
    s = s.strip().replace(_NBSP, "").replace(" ", "")
    if s in ("", "－", "-", "−", "‐"):
        return None
    s = s.replace(",", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_announce_date(s: str) -> date | None:
    """'26/04/22' → date(2026, 4, 22)"""
    s = s.strip().replace(_NBSP, "").replace(" ", "")
    m = _ANNOUNCE_DATE_PATTERN.match(s)
    if not m:
        return None
    yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None


def _clean_cell(s: str) -> str:
    return _TAG_PATTERN.sub("", s).replace("&nbsp;", "").replace(_NBSP, "").strip()


def _parse_period_cell(cell: str) -> tuple[str, bool] | None:
    """期セルから (period_label, is_forecast) を返す。

    例:
        '2026.03'    → ('2026年3月期', False)
        '予 2027.03' → ('2027年3月期', True)
        'その他'      → None
    """
    cleaned = _clean_cell(cell)
    is_forecast = cleaned.startswith("予") or "予" in cleaned[:2]
    m = _PERIOD_PATTERN.search(cleaned)
    if not m:
        return None
    yyyy = int(m.group(1))
    mm = int(m.group(2))
    return (f"{yyyy}年{mm}月期", is_forecast)


def fetch_finance_html(code4: str, force: bool = False) -> str | None:
    """株探の業績ページHTMLを取得 (24hキャッシュ)。"""
    cache = _cache_path(code4)
    if not force and cache.exists():
        age_hours = (pd.Timestamp.now().timestamp() - cache.stat().st_mtime) / 3600
        if age_hours < KABUTAN_FINANCE_CACHE_MAX_AGE_HOURS:
            return cache.read_text(encoding="utf-8")

    url = KABUTAN_FINANCE_URL.format(code4=code4)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=KABUTAN_FETCH_TIMEOUT_SEC,
        )
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            logger.warning(f"kabutan finance {code4}: HTTP {resp.status_code}")
            return None
        KABUTAN_FINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        time.sleep(KABUTAN_FETCH_SLEEP_SEC)
        return resp.text
    except Exception as e:
        logger.warning(f"kabutan finance {code4}: {e}")
        return None


def parse_finance_html(html: str) -> pd.DataFrame:
    """株探業績ページのHTMLから通期業績推移をパース。

    対象テーブル行の形式:
        [期, 売上高(百万), 営業益(百万), 経常益(百万), 最終益(百万), EPS(円), 配当(円), 発表日]

    Returns:
        DataFrame columns = period, revenue, op_margin, eps, is_forecast, announce_date
            revenue: 億円換算、op_marginは営業益/売上*100
    """
    rows: list[dict] = []
    seen_periods: set[str] = set()

    for tr_match in _TR_PATTERN.finditer(html):
        tr_body = tr_match.group(1)
        cells_raw = _TD_PATTERN.findall(tr_body)
        if len(cells_raw) < 7:
            continue
        cells = [_clean_cell(c) for c in cells_raw]

        # 1列目が期のパターンかチェック
        parsed_period = _parse_period_cell(cells_raw[0])
        if parsed_period is None:
            continue
        period_label, is_forecast = parsed_period

        # 同じ期が複数テーブルに出てくる→最初の出現を採用
        if period_label in seen_periods:
            continue

        # 期待: cells[1]=売上, [2]=営業益, [3]=経常益, [4]=最終益, [5]=EPS, [6]=配当, [7]=発表日(あれば)
        revenue_mil = _parse_number(cells[1])
        op_income_mil = _parse_number(cells[2])
        eps = _parse_number(cells[5])
        announce_d = _parse_announce_date(cells[7]) if len(cells) > 7 else None

        # 売上が無ければスキップ (予想で実発表前=値ナシの行など)
        if revenue_mil is None:
            # 予想行で売上が無い場合、レコードは作るが値はNaN
            if is_forecast:
                rows.append(
                    {
                        "period": period_label,
                        "revenue": None,
                        "op_margin": None,
                        "eps": eps,
                        "is_forecast": True,
                        "announce_date": announce_d,
                    }
                )
                seen_periods.add(period_label)
            continue

        # 営業利益率
        op_margin = (
            (op_income_mil / revenue_mil * 100)
            if op_income_mil is not None and revenue_mil > 0
            else None
        )

        rows.append(
            {
                "period": period_label,
                "revenue": round(revenue_mil / 100.0, 1),  # 百万→億
                "op_margin": round(op_margin, 2) if op_margin is not None else None,
                "eps": round(eps, 2) if eps is not None else None,
                "is_forecast": is_forecast,
                "announce_date": announce_d,
            }
        )
        seen_periods.add(period_label)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 期を年月で昇順に並べる
    df["_sort_key"] = df["period"].str.extract(r"(\d{4})年(\d{1,2})月")[0].astype(
        int
    ) * 100 + df["period"].str.extract(r"(\d{4})年(\d{1,2})月")[1].astype(int)
    df = (
        df.sort_values(["_sort_key", "is_forecast"])
        .drop(columns=["_sort_key"])
        .reset_index(drop=True)
    )
    return df


def fetch_kabutan_financials(ticker: str, force: bool = False) -> pd.DataFrame | None:
    """株探から業績データを取得。失敗時 None。"""
    code4 = _ticker_to_code4(ticker)
    html = fetch_finance_html(code4, force=force)
    if html is None:
        return None
    df = parse_finance_html(html)
    if df.empty:
        return None
    return df


def get_latest_actual_period(df: pd.DataFrame) -> str | None:
    """最新の実績(is_forecast=False)期ラベルを返す。"""
    actuals = df[~df["is_forecast"]]
    if len(actuals) == 0:
        return None
    return str(actuals.iloc[-1]["period"])
