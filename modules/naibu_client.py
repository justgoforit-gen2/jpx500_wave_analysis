"""naibu-ryuho-app の SQLite 直読みクライアント。

naibu の FastAPI (port 8000) は `financial_metrics` の全列を公開していないため、
JPX500 スクリーニングでは naibu の data.db を read-only で直接参照する。

責務:
- securities_code 変換 (4桁 -> 5桁)
- 内部留保 (retained_earnings) の最新値取得
- naibu が持つ範囲の財務 (B/S, P/L, CF) 取得 (欠損多数のため補助情報)
- 業種マッピング (companies.industry_name)
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    NAIBU_API_BASE_URL,
    NAIBU_DB_PATH,
    NAIBU_FETCH_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)


def to_edinet_securities_code(jpx_code: str) -> str:
    """jpx500 の 4桁コードを naibu の 5桁 securities_code に変換 ('7203' -> '72030')。

    既に5桁ならそのまま返す。
    """
    s = str(jpx_code).strip()
    if len(s) == 4:
        return s + "0"
    return s


def naibu_health_check() -> bool:
    """naibu FastAPI の /api/health が 200 を返すか。"""
    try:
        r = requests.get(
            f"{NAIBU_API_BASE_URL}/api/health", timeout=NAIBU_FETCH_TIMEOUT_SEC
        )
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"naibu health check 失敗: {e}")
        return False


def naibu_db_exists() -> bool:
    """naibu の SQLite DB が物理存在するか。"""
    return Path(NAIBU_DB_PATH).exists()


@contextmanager
def _connect():
    """read-only で SQLite 接続。"""
    uri = f"file:{Path(NAIBU_DB_PATH).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        yield conn
    finally:
        conn.close()


def fetch_jpx500_naibu_data() -> pd.DataFrame:
    """JPX500 全銘柄について naibu の財務情報を一括取得。

    アプローチ:
    1. jpx500_membership × companies × financial_metrics を結合
    2. 各 edinet_code について 各列ごとに「最新の非NULL値」を採用
       (Toyota の operating_cf のように完全欠損な企業も存在するため
        フィールドごとに古いFYに遡って拾う)

    Returns:
        DataFrame[code (4桁), edinet_code, name, industry_name,
                  fiscal_year, retained_earnings,
                  total_assets, total_equity, cash, short_term_debt,
                  long_term_debt, operating_cf, net_income,
                  is_consolidated]
        欠損は NaN。
    """
    if not naibu_db_exists():
        logger.warning(f"naibu DB not found at {NAIBU_DB_PATH}")
        return pd.DataFrame()

    fields = [
        "total_assets",
        "total_equity",
        "cash",
        "short_term_debt",
        "long_term_debt",
        "operating_cf",
        "net_income",
    ]
    # 1) 基本情報 + 最新FYの retained_earnings
    base_q = """
        SELECT
            jm.securities_code AS sec5,
            co.edinet_code,
            co.name,
            co.industry_name,
            re.fiscal_year,
            re.amount AS retained_earnings,
            re.is_consolidated
        FROM jpx500_membership jm
        JOIN companies co ON co.securities_code = jm.securities_code
        LEFT JOIN retained_earnings re ON re.edinet_code = co.edinet_code
        WHERE re.fiscal_year_end = (
            SELECT MAX(fiscal_year_end) FROM retained_earnings
            WHERE edinet_code = co.edinet_code
        )
    """
    with _connect() as conn:
        base_df = pd.read_sql_query(base_q, conn)

        # 2) フィールドごとに最新の非NULL値を取得
        # NOTE: f-string SQL composition is safe — `field` is from a fixed
        # internal whitelist (`fields`), never user input. (bandit: nosec B608)
        for field in fields:
            sub_q = f"""
                SELECT
                    fm.edinet_code,
                    fm.{field} AS {field},
                    fm.fiscal_year AS {field}_fy
                FROM financial_metrics fm
                WHERE fm.{field} IS NOT NULL
                  AND fm.fiscal_year_end = (
                      SELECT MAX(fiscal_year_end) FROM financial_metrics
                      WHERE edinet_code = fm.edinet_code AND {field} IS NOT NULL
                  )
            """  # nosec B608  # field is from internal whitelist `fields`
            sub_df = pd.read_sql_query(sub_q, conn)
            base_df = base_df.merge(sub_df, on="edinet_code", how="left")

    # 5桁 → 4桁 (末尾の "0" を取る) で jpx500 と整合
    base_df["code"] = base_df["sec5"].str[:4]
    base_df = base_df.drop(columns=["sec5"])

    cols_order = [
        "code",
        "edinet_code",
        "name",
        "industry_name",
        "fiscal_year",
        "retained_earnings",
        "is_consolidated",
    ] + fields
    other = [c for c in base_df.columns if c not in cols_order]
    return base_df[cols_order + other].reset_index(drop=True)


def fetch_universe_naibu_data(
    securities_codes_4digit: list[str] | None = None,
) -> pd.DataFrame:
    """JPX500 / TSE Standard どちらでも使える汎用 naibu フェッチャ。

    Args:
        securities_codes_4digit: 4桁コードリスト。None なら従来通り
            `jpx500_membership` 経由で JPX500 全銘柄取得 (=fetch_jpx500_naibu_data)。
            list 指定時は `companies` テーブルに直接 JOIN し、
            JPX500 に含まれない Standard 銘柄も拾える。

    Returns: fetch_jpx500_naibu_data と同じスキーマ。
    """
    if securities_codes_4digit is None:
        return fetch_jpx500_naibu_data()

    if not naibu_db_exists():
        logger.warning(f"naibu DB not found at {NAIBU_DB_PATH}")
        return pd.DataFrame()

    codes5 = sorted(
        {to_edinet_securities_code(c) for c in securities_codes_4digit if c}
    )
    if not codes5:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(codes5))
    fields = [
        "total_assets",
        "total_equity",
        "cash",
        "short_term_debt",
        "long_term_debt",
        "operating_cf",
        "net_income",
    ]
    # `companies` に直接 JOIN (jpx500_membership に依存しない)
    # NOTE: placeholders は SQL injection 安全 (内部生成のみ)。 (bandit: nosec B608)
    base_q = f"""
        SELECT
            co.securities_code AS sec5,
            co.edinet_code,
            co.name,
            co.industry_name,
            re.fiscal_year,
            re.amount AS retained_earnings,
            re.is_consolidated
        FROM companies co
        LEFT JOIN retained_earnings re
            ON re.edinet_code = co.edinet_code
            AND re.fiscal_year_end = (
                SELECT MAX(fiscal_year_end) FROM retained_earnings
                WHERE edinet_code = co.edinet_code
            )
        WHERE co.securities_code IN ({placeholders})
    """  # nosec B608  # placeholders are parameter markers, not user input
    with _connect() as conn:
        base_df = pd.read_sql_query(base_q, conn, params=codes5)

        for field in fields:
            sub_q = f"""
                SELECT
                    fm.edinet_code,
                    fm.{field} AS {field},
                    fm.fiscal_year AS {field}_fy
                FROM financial_metrics fm
                WHERE fm.{field} IS NOT NULL
                  AND fm.fiscal_year_end = (
                      SELECT MAX(fiscal_year_end) FROM financial_metrics
                      WHERE edinet_code = fm.edinet_code AND {field} IS NOT NULL
                  )
            """  # nosec B608  # field is from internal whitelist `fields`
            sub_df = pd.read_sql_query(sub_q, conn)
            base_df = base_df.merge(sub_df, on="edinet_code", how="left")

    base_df["code"] = base_df["sec5"].str[:4]
    base_df = base_df.drop(columns=["sec5"])

    cols_order = [
        "code",
        "edinet_code",
        "name",
        "industry_name",
        "fiscal_year",
        "retained_earnings",
        "is_consolidated",
    ] + fields
    other = [c for c in base_df.columns if c not in cols_order]
    return base_df[cols_order + other].reset_index(drop=True)


_BS_FIELDS = [
    "current_assets",
    "total_assets",
    "current_liabilities",
    "total_liabilities",
    "total_equity",
    "cash",
    "short_term_debt",
    "long_term_debt",
    "retained_earnings_bs",
    "capital_stock",
]


def fetch_balance_sheet(jpx_code: str) -> dict | None:
    """4桁コードの最新本決算期 BS 項目を 1 件の dict で返す。

    financial_metrics テーブルから total_assets が NULL でない最新
    fiscal_year_end を採用する。DB 不在・該当なしは None を返す。
    """
    if not naibu_db_exists():
        logger.warning(f"naibu DB not found at {NAIBU_DB_PATH}")
        return None

    sec5 = to_edinet_securities_code(jpx_code)
    fields_sql = ", ".join(f"fm.{f}" for f in _BS_FIELDS)

    # _BS_FIELDS は内部固定ホワイトリスト (ユーザー入力なし)  # nosec B608
    query = f"""
        SELECT
            fm.fiscal_year,
            fm.fiscal_year_end,
            fm.is_consolidated,
            {fields_sql}
        FROM financial_metrics fm
        JOIN companies co ON co.edinet_code = fm.edinet_code
        WHERE co.securities_code = ?
          AND fm.total_assets IS NOT NULL
        ORDER BY fm.fiscal_year_end DESC
        LIMIT 1
    """  # nosec B608
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(query, (sec5,)).fetchone()
        if row is None:
            return None
        result: dict = {
            "fiscal_year": row["fiscal_year"],
            "fiscal_year_end": str(row["fiscal_year_end"]),
            "is_consolidated": bool(row["is_consolidated"]),
        }
        for f in _BS_FIELDS:
            val = row[f]
            result[f] = int(val) if val is not None else None
        return result
    except Exception as e:
        logger.warning(f"fetch_balance_sheet({jpx_code}): {e}")
        return None
