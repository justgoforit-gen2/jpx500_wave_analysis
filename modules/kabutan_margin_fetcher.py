"""株探(kabutan.jp) の週次信用残時系列をスクレイピングして履歴を構築する。

JPX公式は直近5週分のPDFしか公開していないため、3年以上の時系列を可視化
したい場合は別ソースから補完する必要がある。株探の銘柄ページ
(/stock/kabuka?code=NNNN&ashi=shin) は週次の終値・売残高・買残高・
信用倍率をパブリックに公開しており、page= パラメータで過去にページング可能。

URL形式:
    https://kabutan.jp/stock/kabuka?code={4桁}&ashi=shin&page={1..N}

各ページに表形式で約30週分が含まれる。page=5 で約3年遡れる。

カラム順 (信用残テーブル):
    日付, 終値, 前週比%, 売買単価, 売買高(株), 売残高, 買残高, 信用倍率

注意点:
- 株主優待やIPO週など信用残が「－」の週がある → NaN として扱う
- 直近行は当週の値動きしか持たない (信用残まだ確定せず) → 信用残部分が空欄
- 取得間に sleep を入れて節度ある取得とする
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
    KABUTAN_DEFAULT_MAX_PAGES,
    KABUTAN_FETCH_SLEEP_SEC,
    KABUTAN_FETCH_TIMEOUT_SEC,
    KABUTAN_MARGIN_CACHE_DIR,
    KABUTAN_MARGIN_HISTORY_PARQUET,
    KABUTAN_MARGIN_URL_TEMPLATE,
)

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_DATE_PATTERN = re.compile(r"^(\d\d)/(\d\d)/(\d\d)$")
_TR_PATTERN = re.compile(r"<tr[^>]*>(.+?)</tr>", re.DOTALL)
_TD_PATTERN = re.compile(r"<t[hd][^>]*>(.+?)</t[hd]>", re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")


def _ticker_to_code4(ticker: str) -> str:
    """'9984.T' → '9984' に変換。"""
    return ticker.split(".")[0]


def _build_url(code4: str, page: int) -> str:
    return KABUTAN_MARGIN_URL_TEMPLATE.format(code4=code4, page=page)


def _cache_path(code4: str, page: int) -> Path:
    return KABUTAN_MARGIN_CACHE_DIR / f"{code4}_p{page}.html"


def _parse_yy_date(s: str) -> date | None:
    """'YY/MM/DD' → date オブジェクト。20YY と仮定。"""
    m = _DATE_PATTERN.match(s)
    if not m:
        return None
    yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None


def _parse_number(s: str) -> float | None:
    """'4,629,500' → 4629500.0, '－' / '-' / '' → None。"""
    s = s.strip()
    if s in ("", "－", "-", "−"):
        return None
    s = s.replace(",", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def fetch_page_html(code4: str, page: int) -> str | None:
    """株探のページHTMLを取得 (キャッシュあり)。エラー時 None。"""
    cache = _cache_path(code4, page)
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    url = _build_url(code4, page)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=KABUTAN_FETCH_TIMEOUT_SEC,
        )
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            logger.warning(f"kabutan fetch {code4} p{page}: HTTP {resp.status_code}")
            return None
        KABUTAN_MARGIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        return resp.text
    except Exception as e:
        logger.warning(f"kabutan fetch {code4} p{page}: {e}")
        return None


def parse_kabutan_html(html: str, ticker: str) -> pd.DataFrame:
    """株探のHTMLから週次信用残テーブルを抽出する。

    Returns:
        observation_date, close, sell_balance, buy_balance, margin_ratio
        を持つ DataFrame。各行は1週分 (日付は週の最終営業日 = 金曜)。
    """
    rows: list[dict] = []
    for tr_match in _TR_PATTERN.finditer(html):
        tr_body = tr_match.group(1)
        cells_raw = _TD_PATTERN.findall(tr_body)
        cells = [_TAG_PATTERN.sub("", c).strip() for c in cells_raw]

        if not cells:
            continue
        obs = _parse_yy_date(cells[0])
        if obs is None:
            continue

        # 信用残テーブルは 8 セル: 日付, 終値, 前週比%, 売買単価, 売買高, 売残, 買残, 信用倍率
        # ただし直近週は信用残未確定で、同じ8セルでも OHLCV 表示になる:
        #   日付, 終値, 高値, 安値, 始値, 前週比, 前週比%, 売買高
        # → cell[2] が大きな価格(>1000) になっている場合は OHLCV モード
        if len(cells) < 8:
            continue

        # cell[2] (前週比%) は通常 -50% ~ +50% 程度の小数。
        # OHLCVモードでは高値(価格)が入るため絶対値が大きくなる、または
        # 銘柄名/コードを含んだ別形式の行 → 信用残行ではないと判定
        cell2_val = _parse_number(cells[2])
        if cell2_val is not None and abs(cell2_val) > 500:
            continue  # 直近週(信用残未確定行)

        close = _parse_number(cells[1])
        sell_balance = _parse_number(cells[5])
        buy_balance = _parse_number(cells[6])
        margin_ratio = _parse_number(cells[7])

        if sell_balance is None and buy_balance is None and margin_ratio is None:
            continue

        # 信用残は常に非負整数の株数。負値や極小値が来たらパース失敗とみなす
        if buy_balance is not None and buy_balance < 0:
            continue
        if sell_balance is not None and sell_balance < 0:
            continue

        rows.append(
            {
                "ticker": ticker,
                "observation_date": obs,
                "close": close,
                "sell_balance": sell_balance,
                "buy_balance": buy_balance,
                "margin_ratio": margin_ratio,
            }
        )

    return pd.DataFrame(rows)


def fetch_kabutan_history(
    ticker: str,
    max_pages: int = KABUTAN_DEFAULT_MAX_PAGES,
    sleep_sec: float = KABUTAN_FETCH_SLEEP_SEC,
) -> pd.DataFrame:
    """株探の信用残時系列を max_pages 分取得して縦結合する。

    Args:
        ticker: '9984.T' 形式
        max_pages: 取得ページ数 (1ページ約30週、5ページで約3年)
        sleep_sec: ページ間のスリープ秒数

    Returns:
        重複排除済み (observation_date) の DataFrame、観測日昇順。
    """
    code4 = _ticker_to_code4(ticker)
    frames: list[pd.DataFrame] = []
    fetched = 0
    for page in range(1, max_pages + 1):
        html = fetch_page_html(code4, page)
        if html is None:
            logger.warning(f"kabutan {ticker} p{page}: 取得失敗、以降スキップ")
            break
        parsed = parse_kabutan_html(html, ticker)
        if len(parsed) == 0:
            logger.info(f"kabutan {ticker} p{page}: 行なし、以降終了")
            break
        frames.append(parsed)
        fetched += 1
        # 取得間のスリープ (キャッシュヒット時はスキップ)
        if not _cache_path(code4, page).exists():
            time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["observation_date"], keep="first")
    combined["observation_date"] = pd.to_datetime(combined["observation_date"]).dt.date
    combined = combined.sort_values("observation_date").reset_index(drop=True)
    logger.info(
        f"kabutan {ticker}: {fetched}ページ取得, {len(combined)}週分 "
        f"({combined['observation_date'].min()} ~ {combined['observation_date'].max()})"
    )
    return combined


def update_kabutan_history(
    tickers: list[str],
    max_pages: int = KABUTAN_DEFAULT_MAX_PAGES,
    sleep_sec: float = KABUTAN_FETCH_SLEEP_SEC,
) -> pd.DataFrame:
    """複数銘柄を順次取得してparquetに保存する。

    既存parquetがあればマージして重複排除。
    """
    new_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = fetch_kabutan_history(ticker, max_pages=max_pages, sleep_sec=sleep_sec)
        if len(df) > 0:
            new_frames.append(df)

    if not new_frames:
        logger.warning("kabutan: 取得結果ゼロ")
        return pd.DataFrame()

    new_data = pd.concat(new_frames, ignore_index=True)

    if KABUTAN_MARGIN_HISTORY_PARQUET.exists():
        existing = pd.read_parquet(KABUTAN_MARGIN_HISTORY_PARQUET)
        merged = pd.concat([existing, new_data], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["ticker", "observation_date"], keep="last"
        )
    else:
        merged = new_data

    merged = merged.sort_values(["ticker", "observation_date"]).reset_index(drop=True)
    merged.to_parquet(KABUTAN_MARGIN_HISTORY_PARQUET, index=False)
    logger.info(f"kabutan履歴更新: 累積{len(merged):,}行")
    return merged


def load_kabutan_history(ticker: str | None = None) -> pd.DataFrame | None:
    """株探履歴を読み込む。ticker指定で該当銘柄のみ。"""
    if not KABUTAN_MARGIN_HISTORY_PARQUET.exists():
        return None
    df = pd.read_parquet(KABUTAN_MARGIN_HISTORY_PARQUET)
    if ticker is not None:
        df = df[df["ticker"] == ticker].copy()
    return df.sort_values("observation_date").reset_index(drop=True)
