"""JPX「銘柄別信用取引残高」(mtdailyk) の取得・パース・累積モジュール。

JPX公式から日次のExcel (mtdailyk{YYYYMMDD}00.xls) を取得し、
個別銘柄ごとの信用買残/売残/上場株式数比/前日比 を時系列として
data/margin_history.parquet に累積する。

各日次ファイルは「その日に申込のあった銘柄のみ」が掲載される仕様のため、
過去N日分を累積することで主要銘柄のスナップショットを構築する。

最新スナップショット (各銘柄の直近既知残高) は margin_latest.parquet。
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    JPX_FETCH_SLEEP_SEC,
    JPX_FETCH_TIMEOUT_SEC,
    JPX_MARGIN_CACHE_DIR,
    JPX_MARGIN_FILE_URL_TEMPLATE,
    JPX_MARGIN_HISTORY_PARQUET,
    JPX_MARGIN_LATEST_PARQUET,
    JPX_MARGIN_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# mtdailyk Excel の列マッピング (header=None で読んだときのインデックス)
# col0: 'B' 区分 (制度信用区分の頭文字)
# col1: 貸借区分 ('貸' / '制' / '一' など)
# col2: 補助記号 (任意)
# col3: 銘柄名
# col4: 市場 ('プライム'/'スタンダード'/'グロース'/'立会外')
# col5: 単位 ('株')
# col6: 5文字コード (4桁+'0' or 新形式英数字)
# col7: ISINコード (JP...)
# col8: 売残高 (株数, 合計)
# col9: 売残 前日比
# col10: 売残 上場株式数比 (%)
# col11: 買残高 (株数, 合計)
# col12: 買残 前日比
# col13: 買残 上場株式数比 (%)
# col14: 売買比率 (売残/買残*100)
# col15-22: 一般/制度内訳 (詳細分析用)
_COL_MAP = {
    "name": 3,
    "market": 4,
    "code5": 6,
    "isin": 7,
    "sell_balance": 8,
    "sell_change": 9,
    "sell_pct_listed": 10,
    "buy_balance": 11,
    "buy_change": 12,
    "buy_pct_listed": 13,
    "sell_buy_ratio_pct": 14,
}
_HEADER_ROWS = 7  # データはR7以降


def _ensure_cache_dir() -> None:
    JPX_MARGIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _build_url(target_date: date) -> str:
    """指定日のJPX信用残ファイルURLを構築する。"""
    return JPX_MARGIN_FILE_URL_TEMPLATE.format(date=target_date.strftime("%Y%m%d"))


def _cache_path(target_date: date) -> Path:
    return JPX_MARGIN_CACHE_DIR / f"mtdailyk{target_date.strftime('%Y%m%d')}00.xls"


def _fetch_excel(target_date: date) -> bytes | None:
    """指定日のExcelファイルを取得。404や祝日休日は None を返す。"""
    cache = _cache_path(target_date)
    if cache.exists():
        return cache.read_bytes()

    url = _build_url(target_date)
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=JPX_FETCH_TIMEOUT_SEC,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        _ensure_cache_dir()
        cache.write_bytes(resp.content)
        time.sleep(JPX_FETCH_SLEEP_SEC)
        return resp.content
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) == 404:
            return None
        logger.warning(f"信用残取得失敗 {target_date}: {e}")
        return None
    except Exception as e:
        logger.warning(f"信用残取得失敗 {target_date}: {e}")
        return None


def parse_margin_excel(content: bytes, observation_date: date) -> pd.DataFrame:
    """mtdailyk Excelをパースして DataFrame に変換する。"""
    df = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
    if len(df) <= _HEADER_ROWS:
        return pd.DataFrame()

    body = df.iloc[_HEADER_ROWS:].copy()
    out = pd.DataFrame(
        {
            "observation_date": observation_date,
            "code5": body.iloc[:, _COL_MAP["code5"]].astype(str),
            "isin": body.iloc[:, _COL_MAP["isin"]].astype(str),
            "name": body.iloc[:, _COL_MAP["name"]].astype(str),
            "market": body.iloc[:, _COL_MAP["market"]].astype(str),
            "sell_balance": pd.to_numeric(
                body.iloc[:, _COL_MAP["sell_balance"]], errors="coerce"
            ),
            "sell_change": pd.to_numeric(
                body.iloc[:, _COL_MAP["sell_change"]], errors="coerce"
            ),
            "sell_pct_listed": pd.to_numeric(
                body.iloc[:, _COL_MAP["sell_pct_listed"]], errors="coerce"
            ),
            "buy_balance": pd.to_numeric(
                body.iloc[:, _COL_MAP["buy_balance"]], errors="coerce"
            ),
            "buy_change": pd.to_numeric(
                body.iloc[:, _COL_MAP["buy_change"]], errors="coerce"
            ),
            "buy_pct_listed": pd.to_numeric(
                body.iloc[:, _COL_MAP["buy_pct_listed"]], errors="coerce"
            ),
            "sell_buy_ratio_pct": pd.to_numeric(
                body.iloc[:, _COL_MAP["sell_buy_ratio_pct"]], errors="coerce"
            ),
        }
    )

    # 有効行: コードが5文字英数字のもの
    valid = out["code5"].str.match(r"^[0-9A-Z]{5}$", na=False)
    out = out[valid].copy()

    # ticker (yfinance形式 NNNN.T) を生成
    # 旧4桁コード: 末尾'0'を落とす, 新コード: 末尾の'0'を落とす同様処理
    out["code4"] = out["code5"].str[:4]
    out["ticker"] = out["code4"] + ".T"

    # 信用倍率 (買残÷売残)
    # sell_balance==0 (売り建てなし) のとき倍率は無限大 → 999 (上限) に丸める
    out["margin_ratio"] = out.apply(
        lambda r: (
            min(r["buy_balance"] / r["sell_balance"], 9999.0)
            if r["sell_balance"] > 0
            else (9999.0 if r["buy_balance"] > 0 else 0.0)
        ),
        axis=1,
    )

    out["observation_date"] = pd.to_datetime(out["observation_date"]).dt.date
    return out.reset_index(drop=True)


def fetch_range(
    end_date: date | None = None,
    lookback_days: int = JPX_MARGIN_LOOKBACK_DAYS,
    sleep_sec: float = JPX_FETCH_SLEEP_SEC,
) -> pd.DataFrame:
    """end_date から過去 lookback_days 営業日分を取得して縦結合する。

    各営業日の Excel を取得 → パース → 全件統合。
    祝日・休日は404になるので自動スキップ。
    """
    if end_date is None:
        end_date = date.today()

    frames: list[pd.DataFrame] = []
    success = 0
    skipped = 0

    for delta in range(lookback_days):
        d = end_date - timedelta(days=delta)
        # 土日は明らかにファイルが無いのでスキップ
        if d.weekday() >= 5:
            continue

        content = _fetch_excel(d)
        if content is None:
            skipped += 1
            continue

        try:
            parsed = parse_margin_excel(content, d)
            if len(parsed) > 0:
                frames.append(parsed)
                success += 1
        except Exception as e:
            logger.warning(f"パース失敗 {d}: {e}")
            skipped += 1

        time.sleep(sleep_sec)

    logger.info(f"信用残取得: 成功={success}日, スキップ={skipped}日")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def update_margin_history(
    end_date: date | None = None,
    lookback_days: int = JPX_MARGIN_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日次データを取得して margin_history / margin_latest parquet を更新する。

    Returns:
        (history_df, latest_df): 累積履歴と銘柄別最新スナップショット
    """
    new_data = fetch_range(end_date=end_date, lookback_days=lookback_days)
    if new_data.empty:
        logger.warning("信用残データが取得できませんでした")
        return pd.DataFrame(), pd.DataFrame()

    # 既存履歴とマージ
    if JPX_MARGIN_HISTORY_PARQUET.exists():
        existing = pd.read_parquet(JPX_MARGIN_HISTORY_PARQUET)
        merged = pd.concat([existing, new_data], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["observation_date", "ticker"], keep="last"
        )
    else:
        merged = new_data

    merged = merged.sort_values(["ticker", "observation_date"]).reset_index(drop=True)
    merged.to_parquet(JPX_MARGIN_HISTORY_PARQUET, index=False)

    # 最新スナップショット
    latest = (
        merged.sort_values("observation_date")
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest.to_parquet(JPX_MARGIN_LATEST_PARQUET, index=False)

    logger.info(
        f"信用残履歴更新: 累積{len(merged):,}行, 最新スナップショット{len(latest):,}銘柄"
    )
    return merged, latest


def load_margin_latest() -> pd.DataFrame | None:
    """銘柄別最新スナップショットを読み込む。"""
    if not JPX_MARGIN_LATEST_PARQUET.exists():
        return None
    return pd.read_parquet(JPX_MARGIN_LATEST_PARQUET)


def load_margin_history(ticker: str | None = None) -> pd.DataFrame | None:
    """累積履歴を読み込む。tickerを指定すれば該当銘柄のみ。"""
    if not JPX_MARGIN_HISTORY_PARQUET.exists():
        return None
    df = pd.read_parquet(JPX_MARGIN_HISTORY_PARQUET)
    if ticker is not None:
        df = df[df["ticker"] == ticker].copy()
    return df.sort_values("observation_date").reset_index(drop=True)


def compute_deadline_calendar(
    ticker: str,
    deadline_days: int = 180,
) -> pd.DataFrame | None:
    """買残の週次増加分から6ヶ月後の期日売り予想カレンダーを構築する。

    各観測日の buy_change が正の場合 → その日に新規建てされた信用買い
    → deadline_days 日後に期日到来 → 予想売り圧

    Returns:
        DataFrame with columns: deadline_date, expected_selling_shares
    """
    hist = load_margin_history(ticker)
    if hist is None or len(hist) < 2:
        return None

    hist = hist.sort_values("observation_date").reset_index(drop=True)
    hist["observation_date"] = pd.to_datetime(hist["observation_date"])

    # buy_change が正の日が新規建てが優勢な日
    new_positions = hist[hist["buy_change"] > 0].copy()
    if len(new_positions) == 0:
        return None

    new_positions["deadline_date"] = new_positions["observation_date"] + pd.Timedelta(
        days=deadline_days
    )
    new_positions["expected_selling_shares"] = new_positions["buy_change"]

    calendar = new_positions[["deadline_date", "expected_selling_shares"]].copy()
    calendar = (
        calendar.groupby(
            pd.Grouper(key="deadline_date", freq="W-FRI"),
            as_index=False,
        )["expected_selling_shares"]
        .sum()
        .sort_values("deadline_date")
        .reset_index(drop=True)
    )

    # 未来分のみ返す
    calendar = calendar[calendar["deadline_date"] >= pd.Timestamp.today()]
    return calendar.reset_index(drop=True)


def attach_margin_metrics(
    results_df: pd.DataFrame,
    avg_volume_lookup: dict[str, float] | None = None,
) -> pd.DataFrame:
    """results_df (波形分析結果) に信用指標カラムを追加する。

    Args:
        results_df: ticker列を持つDataFrame (results.csv相当)
        avg_volume_lookup: {ticker: 20日平均出来高} の辞書。指定時は出来高比も計算。

    Returns:
        margin_ratio, margin_buy_pct_listed, margin_sell_pct_listed,
        margin_buy_voldays, margin_observation_date を追加した DataFrame
    """
    latest = load_margin_latest()
    if latest is None or len(latest) == 0:
        results_df = results_df.copy()
        for c in [
            "margin_ratio",
            "margin_buy_pct_listed",
            "margin_sell_pct_listed",
            "margin_buy_voldays",
            "margin_observation_date",
        ]:
            results_df[c] = None
        return results_df

    keep_cols = [
        "ticker",
        "margin_ratio",
        "buy_pct_listed",
        "sell_pct_listed",
        "buy_balance",
        "observation_date",
    ]
    snap = latest[keep_cols].rename(
        columns={
            "buy_pct_listed": "margin_buy_pct_listed",
            "sell_pct_listed": "margin_sell_pct_listed",
            "observation_date": "margin_observation_date",
        }
    )

    out = results_df.merge(snap, on="ticker", how="left")

    if avg_volume_lookup is not None:
        out["margin_buy_voldays"] = out.apply(
            lambda r: (
                r["buy_balance"] / avg_volume_lookup.get(r["ticker"], float("nan"))
                if pd.notna(r.get("buy_balance"))
                and avg_volume_lookup.get(r["ticker"], 0) > 0
                else None
            ),
            axis=1,
        )
    else:
        out["margin_buy_voldays"] = None

    out = out.drop(columns=["buy_balance"])
    return out
