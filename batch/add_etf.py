"""ETF時価総額上位100銘柄をjpx500_list.csvに追加するスクリプト"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import STOCK_LIST_CSV

XLSX_PATH = PROJECT_ROOT / "data" / "earnings" / "jpstocklist20260212.xlsx"
TOP_N = 100

# Excel列名（Sheet1: ヘッダー0行目）
COL_DATE = "日付"
COL_CODE = "コード"
COL_NAME = "銘柄名"
COL_MARKET = "市場・商品区分"


def load_etf_list() -> pd.DataFrame:
    """Excelから ETF・ETN 銘柄を抽出"""
    df = pd.read_excel(XLSX_PATH, sheet_name="Sheet1", header=0)
    etf = df[df[COL_MARKET] == "ETF・ETN"].copy()
    etf[COL_CODE] = etf[COL_CODE].astype(str)
    print(f"ETF・ETN銘柄数: {len(etf)}")
    return etf[[COL_CODE, COL_NAME]].reset_index(drop=True)


def fetch_market_caps(etf_df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo FinanceからETF各銘柄の時価総額を取得"""
    results = []
    total = len(etf_df)

    for i, row in etf_df.iterrows():
        code = row[COL_CODE]
        name = row[COL_NAME]
        ticker = f"{code}.T"

        try:
            info = yf.Ticker(ticker).info
            market_cap = info.get("totalAssets") or info.get("marketCap") or 0
        except Exception as e:
            print(f"  [{i+1}/{total}] {ticker} 取得失敗: {e}")
            market_cap = 0

        results.append({
            "code": code,
            "name": name,
            "ticker": ticker,
            "market_cap": market_cap,
        })

        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  時価総額取得中... {i+1}/{total}")

        time.sleep(0.3)

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("ETF上位100銘柄をjpx500_list.csvに追加")
    print("=" * 60)

    # Step 1: ETF銘柄リスト取得
    etf_df = load_etf_list()

    # Step 2: 時価総額取得
    print("\nYahoo Financeから時価総額を取得中...")
    caps_df = fetch_market_caps(etf_df)

    # 時価総額が取得できた銘柄のみ
    valid = caps_df[caps_df["market_cap"] > 0].copy()
    print(f"\n時価総額取得成功: {len(valid)}/{len(caps_df)}")

    # Step 3: 上位100銘柄を選択
    valid = valid.sort_values("market_cap", ascending=False)
    top100 = valid.head(TOP_N).copy()
    print(f"上位{TOP_N}銘柄を選択")

    for i, row in top100.iterrows():
        cap_b = row["market_cap"] / 1e9
        print(f"  {row['code']} {row['name'][:20]:<20s} 時価総額: {cap_b:>10,.1f}B円")

    # Step 4: jpx500_list.csv に追記（重複チェック付き）
    existing = pd.read_csv(STOCK_LIST_CSV, dtype={"code": str})
    existing_codes = set(existing["code"].astype(str))

    new_rows = []
    for _, row in top100.iterrows():
        if row["code"] not in existing_codes:
            new_rows.append({
                "code": row["code"],
                "name": row["name"],
                "size_category": "ETF",
                "sector_33": "ETF・ETN",
                "sector_17": "ETF・ETN",
                "ticker": row["ticker"],
            })

    if not new_rows:
        print("\n追加対象なし（全て既存）")
        return

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(STOCK_LIST_CSV, index=False)
    print(f"\n{len(new_rows)}銘柄を追加しました（合計: {len(combined)}銘柄）")
    print(f"出力: {STOCK_LIST_CSV}")


if __name__ == "__main__":
    main()
