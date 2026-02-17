"""分類のみ再実行（データ取得スキップ）: market_capを含むresults.csvを再生成"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time

from config.settings import DEFAULT_WINDOW, RESULTS_CSV, DAILY_PICKS_CSV
from modules.wave_classifier import classify_all, generate_daily_picks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== 分類のみ再実行（データ取得スキップ）===")

    start = time.time()
    result_df = classify_all(window=DEFAULT_WINDOW)
    elapsed = time.time() - start
    logger.info(f"分類完了: {len(result_df)}銘柄, {elapsed:.0f}秒 → {RESULTS_CSV}")

    # market_cap列の確認
    if "market_cap" in result_df.columns:
        mc_count = result_df["market_cap"].notna().sum()
        logger.info(f"market_cap取得成功: {mc_count}/{len(result_df)}銘柄")
    else:
        logger.warning("market_cap列がありません")

    # 本日の推奨銘柄も再生成
    picks_df = generate_daily_picks(window=DEFAULT_WINDOW)
    logger.info(f"推奨銘柄: {len(picks_df)}銘柄 → {DAILY_PICKS_CSV}")

    logger.info("完了")


if __name__ == "__main__":
    main()
