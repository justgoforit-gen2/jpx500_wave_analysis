"""日次バッチ更新スクリプト: データ取得 → 波形分類 → results.csv出力"""
import logging
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CACHE_DIR, DAILY_PICKS_CSV, DEFAULT_WINDOW, RESULTS_CSV
from modules.data_fetcher import fetch_all, load_stock_list
from modules.earnings_fetcher import fetch_earnings_data
from modules.strategy_engine import generate_ranking
from modules.strategy_loader import load_strategy
from modules.wave_classifier import classify_all, generate_daily_picks

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parent.parent / "data" / "update.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("JPX500 バッチ更新開始")
    logger.info("=" * 60)

    # キャッシュディレクトリ作成
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: データ取得
    stocks = load_stock_list()
    total = len(stocks)
    logger.info(f"対象銘柄数: {total}")

    start_time = time.time()

    def progress(i, total, ticker):
        if i % 50 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            logger.info(f"  取得中... {i+1}/{total} ({ticker}) [{elapsed:.0f}秒経過]")

    failures = fetch_all(progress_callback=progress)
    elapsed = time.time() - start_time
    logger.info(f"データ取得完了: {elapsed:.0f}秒, 失敗: {len(failures)}銘柄")

    if failures:
        logger.warning("失敗銘柄一覧:")
        for ticker, reason in failures.items():
            logger.warning(f"  {ticker}: {reason}")

    # Step 2: 波形分類
    logger.info(f"波形分類開始 (窓: {DEFAULT_WINDOW}日)")
    result_df = classify_all(window=DEFAULT_WINDOW)
    logger.info(f"波形分類完了: {len(result_df)}銘柄 → {RESULTS_CSV}")

    # サマリー
    valid = result_df[result_df["wave_types"] != "データ不足"]
    logger.info(f"分類成功: {len(valid)}銘柄, データ不足: {len(result_df) - len(valid)}銘柄")

    type_counts = {}
    for types_str in valid["wave_types"]:
        for t in str(types_str).split("|"):
            type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {t}: {c}銘柄")

    # Step 3: 本日の推奨銘柄
    logger.info("本日の推奨銘柄を生成中...")
    picks_df = generate_daily_picks(window=DEFAULT_WINDOW)
    logger.info(f"本日の推奨銘柄: {len(picks_df)}銘柄 → {DAILY_PICKS_CSV}")

    if len(picks_df) > 0:
        low_touch = picks_df[picks_df["pick_type"].str.contains("下タッチ")]
        high_touch = picks_df[picks_df["pick_type"].str.contains("上タッチ")]
        logger.info(f"  下タッチ（買い候補）: {len(low_touch)}銘柄")
        logger.info(f"  上タッチ（利確/ブレイク監視）: {len(high_touch)}銘柄")
        for _, p in picks_df.iterrows():
            logger.info(f"    {p['code']} {p['name']} | {p['pick_type']} | ¥{p['latest_close']:,.0f} (位置{p['position_pct']}%)")

    # Step 3.5: ABCD戦略ランキング（Cloud表示用CSV）
    logger.info("ABCD戦略ランキングを生成中...")
    try:
        strategy = load_strategy()
        stock_list_df = load_stock_list()
        ranking_df = generate_ranking(
            stock_list_df,
            load_cached_fn=lambda t: __import__("modules.data_fetcher", fromlist=["load_cached"]).load_cached(t),
            strategy=strategy,
            max_positions=None,
        )

        out_path = Path(__file__).resolve().parent.parent / "data" / "abcd_ranking.csv"
        if ranking_df is None or len(ranking_df) == 0:
            # 空でもファイルは出しておく（Cloud側の確認用）
            ranking_df = ranking_df if ranking_df is not None else None
            logger.info("ABCD戦略: シグナルなし（ランキング空）")
            # ヘッダだけでも残す
            import pandas as pd
            pd.DataFrame().to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            ranking_df.to_csv(out_path, index=True, index_label="rank", encoding="utf-8-sig")
            logger.info(f"ABCD戦略ランキング: {len(ranking_df)}銘柄 → {out_path}")
    except Exception as e:
        logger.warning(f"ABCD戦略ランキング生成に失敗: {e}")

    # Step 4: 決算発表予定日データ取得
    logger.info("決算発表予定日データを取得中...")
    try:
        earnings_df = fetch_earnings_data(force=True)
        logger.info(f"決算発表予定日データ: {len(earnings_df)}件")
    except Exception as e:
        logger.warning(f"決算発表予定日データの取得に失敗: {e}")

    logger.info("バッチ更新完了")


if __name__ == "__main__":
    main()
