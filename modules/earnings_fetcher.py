"""決算発表予定日をJPX公式Excelから取得・キャッシュするモジュール"""
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    EARNINGS_CACHE_DIR,
    EARNINGS_COMBINED_CSV,
    EARNINGS_CACHE_MAX_AGE_HOURS,
    EARNINGS_XLSX_URLS,
)

logger = logging.getLogger(__name__)

_JPX_INDEX_URL = "https://www.jpx.co.jp/listing/event-schedules/financial-announcement/index.html"
_JPX_BASE_URL = "https://www.jpx.co.jp"


def _discover_jpx_urls() -> list[str]:
    """JPXページをクロールして現在有効な kessan*.xlsx のURLを返す。
    取得失敗時は空リストを返す（呼び出し元で EARNINGS_XLSX_URLS にフォールバック）。
    """
    try:
        resp = requests.get(_JPX_INDEX_URL, timeout=15)
        resp.raise_for_status()
        hrefs = re.findall(r'href="([^"]*kessan[^"]*\.xlsx)"', resp.text)
        urls = []
        for h in hrefs:
            url = h if h.startswith("http") else _JPX_BASE_URL + h
            if url not in urls:
                urls.append(url)
        if urls:
            logger.info(f"JPX URL自動検出: {urls}")
        return urls
    except Exception as e:
        logger.warning(f"JPX URL自動検出失敗: {e}")
        return []


def _download_xlsx(url: str, dest: Path) -> bool:
    """ExcelファイルをURLからダウンロードする"""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info(f"ダウンロード完了: {url} -> {dest.name}")
        return True
    except Exception as e:
        logger.warning(f"ダウンロード失敗: {url} - {e}")
        return False


def _parse_earnings_xlsx(filepath: Path) -> pd.DataFrame:
    """JPX決算発表Excelをパースして (code, earnings_date) DataFrameを返す。

    JPXのExcelはヘッダ行が固定位置ではないため、
    「コード」列を含む行を自動検出してヘッダとする。
    """
    try:
        # まずヘッダなしで読み込み、ヘッダ行を探す
        raw = pd.read_excel(filepath, header=None, engine="openpyxl")
    except Exception as e:
        logger.warning(f"Excel読み込み失敗: {filepath} - {e}")
        return pd.DataFrame(columns=["code", "earnings_date"])

    # 「コード」を含む行をヘッダ行として検出
    header_row = None
    for idx, row in raw.iterrows():
        row_str = row.astype(str).str.strip()
        if row_str.str.contains("コード").any():
            header_row = idx
            break

    if header_row is None:
        logger.warning(f"ヘッダ行が見つかりません: {filepath}")
        return pd.DataFrame(columns=["code", "earnings_date"])

    # ヘッダ行以降を再読み込み
    df = pd.read_excel(filepath, header=header_row, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()

    # 「コード」列と日付列を特定
    code_col = None
    date_col = None
    for col in df.columns:
        col_lower = col.strip()
        if "コード" in col_lower and code_col is None:
            code_col = col
        if ("発表日" in col_lower or "決算発表" in col_lower or "予定日" in col_lower) and date_col is None:
            date_col = col

    if code_col is None:
        logger.warning(f"コード列が見つかりません: {filepath}, columns={list(df.columns)}")
        return pd.DataFrame(columns=["code", "earnings_date"])

    # 日付列が見つからない場合、日付っぽい列を探す
    if date_col is None:
        for col in df.columns:
            if col != code_col:
                sample = df[col].dropna().head(10)
                try:
                    pd.to_datetime(sample)
                    date_col = col
                    break
                except (ValueError, TypeError):
                    continue

    if date_col is None:
        logger.warning(f"日付列が見つかりません: {filepath}, columns={list(df.columns)}")
        return pd.DataFrame(columns=["code", "earnings_date"])

    result = df[[code_col, date_col]].copy()
    result.columns = ["code", "earnings_date"]

    # コードを文字列に正規化（4桁ゼロ埋め）
    result["code"] = pd.to_numeric(result["code"], errors="coerce")
    result = result.dropna(subset=["code"])
    result["code"] = result["code"].astype(int).astype(str).str.zfill(4)

    # 日付を変換
    result["earnings_date"] = pd.to_datetime(result["earnings_date"], errors="coerce")
    result = result.dropna(subset=["earnings_date"])
    result["earnings_date"] = result["earnings_date"].dt.strftime("%Y-%m-%d")

    logger.info(f"パース完了: {filepath.name} -> {len(result)}件")
    return result


def fetch_earnings_data(force: bool = False) -> pd.DataFrame:
    """全Excelを取得・パースし、CSVキャッシュに保存する。

    force=True: キャッシュの有無にかかわらず再取得
    """
    EARNINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # キャッシュ有効性チェック
    if not force and EARNINGS_COMBINED_CSV.exists():
        mtime = datetime.fromtimestamp(EARNINGS_COMBINED_CSV.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=EARNINGS_CACHE_MAX_AGE_HOURS):
            logger.info("決算日キャッシュは有効期間内。スキップ")
            return pd.read_csv(EARNINGS_COMBINED_CSV, dtype={"code": str})

    # 動的URL検出を優先し、失敗時は設定値にフォールバック
    urls = _discover_jpx_urls()
    if not urls:
        logger.warning("JPX URL自動検出できず。設定ファイルのURLを使用します")
        urls = EARNINGS_XLSX_URLS

    all_dfs = []
    for i, url in enumerate(urls):
        dest = EARNINGS_CACHE_DIR / f"earnings_{i+1}.xlsx"
        if _download_xlsx(url, dest):
            df = _parse_earnings_xlsx(dest)
            if len(df) > 0:
                all_dfs.append(df)

    if not all_dfs:
        logger.warning("決算日データの取得に失敗しました")
        return pd.DataFrame(columns=["code", "earnings_date"])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["code", "earnings_date"])
    combined = combined.sort_values(["code", "earnings_date"]).reset_index(drop=True)
    combined.to_csv(EARNINGS_COMBINED_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"決算日データ保存: {len(combined)}件 -> {EARNINGS_COMBINED_CSV}")
    return combined


def load_earnings_dates() -> pd.DataFrame | None:
    """キャッシュCSVから決算日データを読み込む"""
    if EARNINGS_COMBINED_CSV.exists():
        return pd.read_csv(EARNINGS_COMBINED_CSV, dtype={"code": str})
    return None


def get_earnings_dates_for_code(code: str, earnings_df: pd.DataFrame | None) -> list[str]:
    """指定銘柄の決算日リストを返す"""
    if earnings_df is None or len(earnings_df) == 0:
        return []
    code = str(code).zfill(4)
    matches = earnings_df[earnings_df["code"] == code]
    return matches["earnings_date"].tolist()
