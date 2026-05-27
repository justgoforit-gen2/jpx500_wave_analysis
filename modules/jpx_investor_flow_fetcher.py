"""JPX「投資部門別取引状況」の取得・パース・統合モジュール。

JPX公式ページから週次のExcel(*.xls)を自動取得し、4市場
(TSE Prime / Standard / Growth / Tokyo & Nagoya) の海外投資家フローを
data/foreign_flow.parquet に集約する。
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    JPX_FETCH_SLEEP_SEC,
    JPX_FETCH_TIMEOUT_SEC,
    JPX_INVESTOR_FLOW_FALLBACK_URLS,
    JPX_INVESTOR_FLOW_LOOKBACK_YEARS,
    JPX_INVESTOR_FLOW_MARKETS,
    JPX_INVESTOR_FLOW_PARQUET,
    JPX_INVESTOR_TYPE_ARCHIVE_MAX_PAGES,
    JPX_INVESTOR_TYPE_ARCHIVE_URL_TEMPLATE,
    JPX_INVESTOR_TYPE_CACHE_DIR,
    JPX_INVESTOR_TYPE_PAGE_URL,
)

logger = logging.getLogger(__name__)

_JPX_BASE_URL = "https://www.jpx.co.jp"
_FNAME_RE = re.compile(r"stock_val_1_(\d{2})(\d{2})(\d{2})\.xls", re.IGNORECASE)
_YEAR_RE = re.compile(r"(\d{4})\s*年")
_DATE_RANGE_RE = re.compile(r"(\d{1,2})/(\d{1,2})\s*[-~〜−–—]\s*(\d{1,2})/(\d{1,2})")


def _extract_xls_links(html: str) -> list[tuple[str, str]]:
    """1ページのHTMLから stock_val_*.xls リンクを抽出。"""
    hrefs = re.findall(r'href="([^"]*stock_val_[^"]*\.xls)"', html)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for h in hrefs:
        url = h if h.startswith("http") else _JPX_BASE_URL + h
        basename = Path(url).name
        if basename in seen:
            continue
        seen.add(basename)
        results.append((url, basename))
    return results


def _fetch_page_xls(url: str) -> list[tuple[str, str]]:
    """指定URLを取得して xls リンク一覧を返す（失敗時は空）。"""
    try:
        resp = requests.get(url, timeout=JPX_FETCH_TIMEOUT_SEC)
        resp.raise_for_status()
        return _extract_xls_links(resp.text)
    except Exception as e:
        logger.warning(f"ページ取得失敗 {url}: {e}")
        return []


def _discover_jpx_flow_urls(
    lookback_years: int = JPX_INVESTOR_FLOW_LOOKBACK_YEARS,
) -> list[tuple[str, str]]:
    """JPX投資部門別ページ + 年次アーカイブ群から stock_val_*.xls URLを収集。

    - index.html: 直近5週程度
    - 00-00-archives-{n:02d}.html: -01=前年, -02=2年前, ... 各52週

    Returns:
        [(absolute_url, basename), ...] basenameで重複除去済み
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(items: list[tuple[str, str]]) -> None:
        for u, n in items:
            if n in seen:
                continue
            seen.add(n)
            results.append((u, n))

    # 1) インデックス（直近）
    _add(_fetch_page_xls(JPX_INVESTOR_TYPE_PAGE_URL))

    # 2) 年次アーカイブを必要分クロール
    # -00=現年, -01=前年, -02=2年前, ... 1ページ=約1年。
    # lookback_years 年分 + 現年(-00) + 余裕1ページ。
    pages_to_fetch = min(
        max(lookback_years + 1, 1), JPX_INVESTOR_TYPE_ARCHIVE_MAX_PAGES
    )
    for n in range(0, pages_to_fetch + 1):
        url = JPX_INVESTOR_TYPE_ARCHIVE_URL_TEMPLATE.format(n=n)
        items = _fetch_page_xls(url)
        if not items:
            # 連続失敗なら以降も無いとみなして打ち切り
            logger.info(f"アーカイブ -{n:02d} 取得0件、以降スキップ")
            if n > 0:
                break
            # n=0 が空でも index に -01〜は存在し得るので継続
            continue
        _add(items)
        time.sleep(JPX_FETCH_SLEEP_SEC)

    if results:
        logger.info(f"JPX投資部門別URL検出: {len(results)}件")
    else:
        logger.warning("JPX投資部門別URL検出ゼロ、フォールバック適用")
        for url in JPX_INVESTOR_FLOW_FALLBACK_URLS:
            results.append((url, Path(url).name))
    return results


def _filename_to_date(basename: str) -> pd.Timestamp | None:
    """ファイル名 stock_val_1_YYMMWW.xls から週末日を推定（パース失敗フォールバック用）。

    YY=年下2桁(20YY)、MM=月、WW=月内週番号。週末は月内W週目の最後の金曜。
    """
    m = _FNAME_RE.search(basename)
    if not m:
        return None
    yy, mm, ww = int(m.group(1)), int(m.group(2)), int(m.group(3))
    year = 2000 + yy
    # 月のWW週目の金曜を推定
    try:
        first = pd.Timestamp(year=year, month=mm, day=1)
        # 月内の金曜日リスト
        fridays = pd.date_range(first, first + pd.offsets.MonthEnd(), freq="W-FRI")
        if ww - 1 < len(fridays):
            return fridays[ww - 1]
        return None
    except Exception:
        return None


def _download_xls(url: str, dest: Path) -> bool:
    """xlsをDLしてdestに保存。"""
    try:
        resp = requests.get(url, timeout=JPX_FETCH_TIMEOUT_SEC)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.info(f"DL完了: {url} -> {dest.name}")
        return True
    except Exception as e:
        logger.warning(f"DL失敗 {url}: {e}")
        return False


def _extract_week_end_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """シート上方の '2026年5月2週 2026/5 week2 (5/11 - 5/15)' から週末を抽出。

    年は '\\d{4}年' から、月日範囲は最後の '5/11 - 5/15' から取得し、終端日を週末とする。
    """
    year: int | None = None
    last_date_range: tuple[int, int, int, int] | None = None
    for ridx in range(min(15, len(df))):
        for val in df.iloc[ridx].fillna("").values:
            s = str(val)
            if not s.strip():
                continue
            if year is None:
                ym = _YEAR_RE.search(s)
                if ym:
                    year = int(ym.group(1))
            dm = _DATE_RANGE_RE.search(s)
            if dm:
                last_date_range = (
                    int(dm.group(1)),
                    int(dm.group(2)),
                    int(dm.group(3)),
                    int(dm.group(4)),
                )
    if year is not None and last_date_range is not None:
        _sm, _sd, em, ed = last_date_range
        try:
            return pd.Timestamp(year=year, month=em, day=ed)
        except Exception:
            return None
    return None


def _extract_week_label(df: pd.DataFrame) -> str:
    """週ラベル文字列をシート上方から抽出（ログ・表示用）。"""
    for ridx in range(min(15, len(df))):
        for val in df.iloc[ridx].fillna("").values:
            s = str(val).strip()
            if "week" in s.lower() or "週" in s:
                return s[:80]
    return ""


def _parse_value(v) -> int | None:
    """セル値を整数化（NaN/空/不正値はNone）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _find_foreigners_rows(df: pd.DataFrame) -> tuple[int, int, int] | None:
    """Sales/Purchases/Total の行番号を返す。

    Foreignersキーが含まれる行が Purchases 行（実証済み）。
    その1行手前が Sales、1行後が Total。
    """
    for ridx in range(len(df)):
        row_str = " | ".join(str(v) for v in df.iloc[ridx].fillna("").values)
        if "Foreigners" in row_str:
            # Purchases行 = ridx, Sales行 = ridx-1, Total行 = ridx+1
            if ridx >= 1 and ridx + 1 < len(df):
                return (ridx - 1, ridx, ridx + 1)
            return None
    return None


def _parse_investor_flow_xls(
    filepath: Path,
    markets: tuple[str, ...] = JPX_INVESTOR_FLOW_MARKETS,
) -> pd.DataFrame:
    """1ファイルから4市場分のForeignersフローを抽出。"""
    try:
        xls = pd.ExcelFile(filepath)
    except Exception as e:
        logger.warning(f"xlsオープン失敗 {filepath.name}: {e}")
        return pd.DataFrame()

    rows = []
    for market in markets:
        if market not in xls.sheet_names:
            logger.warning(f"{filepath.name}: シート '{market}' が見つからない")
            continue
        df = pd.read_excel(xls, sheet_name=market, header=None)

        week_end = _extract_week_end_date(df)
        if week_end is None:
            # ファイル名からフォールバック
            week_end = _filename_to_date(filepath.name)
        if week_end is None:
            logger.warning(f"{filepath.name}/{market}: 週末日付抽出失敗")
            continue
        week_label = _extract_week_label(df)

        rows_idx = _find_foreigners_rows(df)
        if rows_idx is None:
            logger.warning(f"{filepath.name}/{market}: Foreigners行検出失敗")
            continue

        sales_row_idx, purch_row_idx, total_row_idx = rows_idx
        # 今週分の値は col 8 (Value), col 9 (Ratio), col 10 (Balance)
        # 4市場すべてで同じレイアウト（実証済み）
        sales_value = _parse_value(df.iat[sales_row_idx, 8])
        purchase_value = _parse_value(df.iat[purch_row_idx, 8])
        total_value = _parse_value(df.iat[total_row_idx, 8])
        # Balanceは Purchases 行の col 10 にある（差引=買い-売り）
        balance_value = _parse_value(df.iat[purch_row_idx, 10])
        foreigner_ratio = _parse_float(df.iat[total_row_idx, 9])

        # net_value: Balanceがあればそれを使う、なければ purchase - sales
        if balance_value is not None:
            net_value = balance_value
        elif purchase_value is not None and sales_value is not None:
            net_value = purchase_value - sales_value
        else:
            net_value = None

        if sales_value is None or purchase_value is None:
            logger.warning(f"{filepath.name}/{market}: 値抽出失敗")
            continue

        rows.append(
            {
                "date": pd.Timestamp(week_end),
                "week_label": week_label,
                "market": market,
                "sales_value": sales_value,
                "purchase_value": purchase_value,
                "net_value": net_value,
                "total_value": total_value if total_value is not None else 0,
                "foreigner_ratio_pct": foreigner_ratio
                if foreigner_ratio is not None
                else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_all_investor_flow(
    force: bool = False,
    lookback_years: int = JPX_INVESTOR_FLOW_LOOKBACK_YEARS,
    progress_callback=None,
) -> pd.DataFrame:
    """全URL列挙→差分DL→パース→統合parquet出力。

    Args:
        force: True なら既存parquetを無視して全件再構築
        lookback_years: 取得対象の過去年数（カットオフ）
        progress_callback: (i, total, basename) を受けるコールバック

    Returns:
        統合 DataFrame
    """
    cache_dir = Path(JPX_INVESTOR_TYPE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing: pd.DataFrame | None = None
    existing_dates: set[pd.Timestamp] = set()
    if not force and JPX_INVESTOR_FLOW_PARQUET.exists():
        try:
            existing = pd.read_parquet(JPX_INVESTOR_FLOW_PARQUET)
            existing["date"] = pd.to_datetime(existing["date"])
            existing_dates = set(existing["date"].unique())
            logger.info(
                f"既存parquet読込: {len(existing)}件 / 既存週数={len(existing_dates)}"
            )
        except Exception as e:
            logger.warning(f"既存parquet読込失敗: {e}")
            existing = None

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(
        days=lookback_years * 365 + 30
    )

    urls: list[tuple[str | None, str]] = list(_discover_jpx_flow_urls(lookback_years))
    # キャッシュ済みの .xls もマージ対象に
    local_xls: list[tuple[str | None, str]] = [
        (None, p.name) for p in sorted(cache_dir.glob("*.xls"))
    ]
    local_names = {n for _, n in local_xls}
    web_names = {n for _, n in urls}
    # ローカルにあってWebにないものは local-only として保持
    for tup in local_xls:
        if tup[1] not in web_names:
            urls.append(tup)

    new_rows: list[pd.DataFrame] = []
    total = len(urls)
    for i, (url, basename) in enumerate(urls):
        if progress_callback:
            progress_callback(i, total, basename)

        # 日付推定（ファイル名から）
        guessed_date = _filename_to_date(basename)
        if guessed_date is not None and guessed_date < cutoff:
            continue  # 取得範囲外
        # 差分更新: 既にparquetに該当週があり、かつローカルxlsも存在するならスキップ
        if (
            not force
            and guessed_date is not None
            and guessed_date in existing_dates
            and basename in local_names
        ):
            continue

        dest = cache_dir / basename
        if not dest.exists() or dest.stat().st_size == 0:
            if url is None:
                # ローカルのみエントリーで実体不在
                continue
            ok = _download_xls(url, dest)
            if not ok:
                continue
            time.sleep(JPX_FETCH_SLEEP_SEC)

        df = _parse_investor_flow_xls(dest)
        if not df.empty:
            new_rows.append(df)

    if not new_rows and existing is not None:
        return existing

    if new_rows:
        new_df = pd.concat(new_rows, ignore_index=True)
        if existing is not None:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
    elif existing is not None:
        combined = existing
    else:
        return pd.DataFrame()

    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.drop_duplicates(subset=["date", "market"], keep="last")
        .sort_values(["market", "date"])
        .reset_index(drop=True)
    )
    combined = combined[combined["date"] >= cutoff].reset_index(drop=True)

    JPX_INVESTOR_FLOW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(JPX_INVESTOR_FLOW_PARQUET, index=False)
    logger.info(
        f"投資部門別フロー保存: {len(combined)}件 "
        f"({combined['date'].min().date()} ~ {combined['date'].max().date()})"
    )
    return combined


def load_investor_flow(market: str | None = None) -> pd.DataFrame:
    """parquetを読込（小さいヘルパ）。market指定で絞り込み。"""
    if not JPX_INVESTOR_FLOW_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(JPX_INVESTOR_FLOW_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    if market:
        df = df[df["market"] == market]
    return df.sort_values("date").reset_index(drop=True)
