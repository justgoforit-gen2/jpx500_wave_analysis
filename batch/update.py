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
from modules.per_pbr_history_fetcher import update_per_pbr_history
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

    # Step 0: JPX ユニバース更新 (data_j.xls 取得 + Standard Top 400 再生成)
    # 7日以内はスキップ (jpx_universe_fetcher.update_universe 内で判定)
    logger.info("ユニバース更新を確認中 (data_j.xls + Standard Top 400)...")
    try:
        from modules.jpx_universe_fetcher import update_universe

        update_universe()
    except Exception as e:
        logger.warning(f"ユニバース更新失敗 (既存リストで継続): {e}")

    # Step 1: データ取得
    stocks = load_stock_list()
    total = len(stocks)
    logger.info(f"対象銘柄数: {total}")

    start_time = time.time()

    def progress(i, total, ticker):
        if i % 50 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            logger.info(f"  取得中... {i + 1}/{total} ({ticker}) [{elapsed:.0f}秒経過]")

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
    logger.info(
        f"分類成功: {len(valid)}銘柄, データ不足: {len(result_df) - len(valid)}銘柄"
    )

    type_counts = {}
    for types_str in valid["wave_types"]:
        for t in str(types_str).split("|"):
            type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {t}: {c}銘柄")

    # Step 2.5: PER/PBR履歴更新（週次スナップショット）
    logger.info("PER/PBR履歴を更新中...")
    pp_start = time.time()

    def pp_progress(i, tot, ticker):
        if i % 50 == 0 or i == tot - 1:
            elapsed = time.time() - pp_start
            logger.info(
                f"  PER/PBR履歴... {i + 1}/{tot} ({ticker}) [{elapsed:.0f}秒経過]"
            )

    try:
        pp_failures = update_per_pbr_history(progress_callback=pp_progress)
        elapsed = time.time() - pp_start
        logger.info(
            f"PER/PBR履歴更新完了: {elapsed:.0f}秒, 失敗 {len(pp_failures)}銘柄"
        )
    except Exception as e:
        logger.warning(f"PER/PBR履歴更新に失敗: {e}")

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
            logger.info(
                f"    {p['code']} {p['name']} | {p['pick_type']} | ¥{p['latest_close']:,.0f} (位置{p['position_pct']}%)"
            )

    # Step 3.5: ABCD戦略ランキング（Cloud表示用CSV）
    logger.info("ABCD戦略ランキングを生成中...")
    try:
        strategy = load_strategy()
        stock_list_df = load_stock_list()
        ranking_df = generate_ranking(
            stock_list_df,
            load_cached_fn=lambda t: __import__(
                "modules.data_fetcher", fromlist=["load_cached"]
            ).load_cached(t),
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
            ranking_df.to_csv(
                out_path, index=True, index_label="rank", encoding="utf-8-sig"
            )
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

    # Step 4.5: JPX投資部門別取引データ（海外投資家フロー）
    logger.info("JPX投資部門別データを取得中...")
    try:
        from modules.jpx_investor_flow_fetcher import fetch_all_investor_flow

        flow_df = fetch_all_investor_flow(force=False)
        logger.info(f"投資部門別フロー: {len(flow_df)}件")
    except Exception as e:
        logger.warning(f"JPX投資部門別データ取得に失敗: {e}")

    # Step 4.6: JPX 銘柄別信用取引残高 (mtdailyk 日次)
    # JPXは直近1営業日分のみ公開のため、毎日実行して履歴を累積する
    logger.info("JPX信用取引銘柄別残高を更新中...")
    try:
        from modules.margin_fetcher import (
            attach_margin_metrics,
            update_margin_history,
        )
        from modules.data_fetcher import load_cached
        import pandas as pd

        margin_history, margin_latest = update_margin_history(lookback_days=5)
        logger.info(
            f"信用残: 累積{len(margin_history):,}行 / 最新スナップショット{len(margin_latest):,}銘柄"
        )

        # results.csv に信用指標を合流 (Prime市場優先)
        if len(margin_latest) > 0:
            results_df = pd.read_csv(
                RESULTS_CSV, encoding="utf-8-sig", dtype={"code": str}
            )

            # 20日平均出来高を辞書化
            vol_lookup: dict[str, float] = {}
            for tk in results_df["ticker"].dropna().unique():
                try:
                    df = load_cached(tk)
                    if df is not None and len(df) >= 20:
                        vol_lookup[tk] = float(df["Volume"].tail(20).mean())
                except Exception:
                    pass

            enriched = attach_margin_metrics(results_df, avg_volume_lookup=vol_lookup)
            enriched.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
            covered = enriched["margin_ratio"].notna().sum()
            prime_total = (enriched["market"] == "TSE Prime").sum()
            prime_covered = (
                (enriched["market"] == "TSE Prime") & (enriched["margin_ratio"].notna())
            ).sum()
            logger.info(
                f"信用指標合流: 全体カバー{covered}/{len(enriched)}銘柄, "
                f"Prime {prime_covered}/{prime_total}銘柄"
            )
    except Exception as e:
        logger.warning(f"信用残データ更新に失敗: {e}")

    # Step 5: 資本効率改善期待スクリーニング
    logger.info("資本効率改善期待スクリーニングを実行中...")
    try:
        import pandas as pd
        from modules.capital_efficiency_screener import run_screening
        from modules.naibu_client import naibu_db_exists

        if not naibu_db_exists():
            logger.warning("naibu DB が見つからないためスクリーニングをスキップ")
        else:
            results_df = pd.read_csv(
                RESULTS_CSV, encoding="utf-8-sig", dtype={"code": str}
            )
            ces_df = run_screening(results_df, use_yf_cache=False)
            high_score = (ces_df["score"] >= 6).sum()
            logger.info(
                f"資本効率スクリーニング: {len(ces_df)}件 / 強い候補(score>=6): {high_score}件"
            )
    except Exception as e:
        logger.warning(f"資本効率スクリーニングに失敗: {e}")

    # Step 5.5: 下降→上昇トレンド転換検出
    logger.info("トレンド転換検出を実行中...")
    try:
        from modules.trend_transition_detector import update_trend_transition_csv

        tt_df = update_trend_transition_csv()
        logger.info(f"転換検出: {len(tt_df)} 銘柄")
    except Exception as e:
        logger.warning(f"トレンド転換検出に失敗: {e}")

    # Step 5.6: Stage 2 ブレイクアウト検出(レンジ → 全MA上抜け+RSI回復)
    logger.info("Stage2 ブレイクアウト検出を実行中...")
    try:
        from modules.range_breakout_detector import update_range_breakout_csv

        rb_df = update_range_breakout_csv()
        logger.info(f"Stage2 ブレイクアウト: {len(rb_df)} 銘柄")
    except Exception as e:
        logger.warning(f"Stage2 ブレイクアウト検出に失敗: {e}")

    # Step 6: 拡張ユニバース(海外ETF/個別株)取得
    logger.info("拡張ユニバース取得を開始...")
    try:
        from modules.extended_fetcher import (
            compute_extended_indicators,
            fetch_extended_all,
        )

        ext_start = time.time()

        def ext_progress(i, tot, ticker):
            if i % 5 == 0 or i == tot - 1:
                elapsed = time.time() - ext_start
                logger.info(
                    f"  拡張取得... {i + 1}/{tot} ({ticker}) [{elapsed:.0f}秒経過]"
                )

        ext_failures = fetch_extended_all(progress_callback=ext_progress)
        logger.info(f"拡張取得完了: 失敗 {len(ext_failures)}件")
        ext_df = compute_extended_indicators()
        logger.info(f"拡張指標出力: {len(ext_df)}件")
    except Exception as e:
        logger.warning(f"拡張ユニバース処理に失敗: {e}")

    # Step 7: ポートフォリオ評価額スナップショット
    logger.info("ポートフォリオ評価額を更新中...")
    try:
        from modules.portfolio_manager import update_portfolio_history

        update_portfolio_history()
    except Exception as e:
        logger.warning(f"ポートフォリオ評価額更新に失敗: {e}")

    # Step 8: シグナル生成
    logger.info("売買シグナルを生成中...")
    try:
        import pandas as pd

        from config.settings import WATCHLIST_CSV
        from modules.portfolio_manager import load_portfolio
        from modules.signal_engine import (
            compute_signals_for_holdings,
            compute_signals_for_watchlist,
            log_signals,
        )

        portfolio = load_portfolio()
        sig_h = compute_signals_for_holdings(portfolio)

        if Path(WATCHLIST_CSV).exists():
            watchlist = pd.read_csv(
                WATCHLIST_CSV, encoding="utf-8-sig", dtype={"code": str}
            )
        else:
            watchlist = pd.DataFrame()
        sig_w = compute_signals_for_watchlist(watchlist)

        all_signals = pd.concat([sig_h, sig_w], ignore_index=True)
        log_signals(all_signals)
        logger.info(f"シグナル: 保有 {len(sig_h)} 件 / 監視 {len(sig_w)} 件")
    except Exception as e:
        logger.warning(f"シグナル生成に失敗: {e}")

    # Step 9: MoatScore 全銘柄計算 → moat_scores.parquet 更新
    logger.info("MoatScore 全銘柄計算を開始...")
    _step9_start = time.time()
    try:
        from modules.moat_score import MoatScoreEngine, save_moat_scores

        stocks = load_stock_list()
        all_codes = list(stocks["code"].dropna().astype(str).unique())
        engine = MoatScoreEngine()
        moat_results = engine.compute_bulk(all_codes)
        save_moat_scores(moat_results)
        scored = sum(1 for r in moat_results if r["total_score"] is not None)
        logger.info(f"MoatScore 完了: {scored}/{len(all_codes)} 銘柄スコア算出")
        _log_perf("moat_score_bulk", time.time() - _step9_start)
    except Exception as e:
        logger.warning(f"MoatScore 計算失敗 (継続): {e}")

    logger.info("バッチ更新完了")


def _log_perf(step: str, elapsed_sec: float) -> None:
    """バッチステップのパフォーマンスを logs/batch-perf.csv に追記する。"""
    import csv
    from datetime import datetime

    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    perf_csv = logs_dir / "batch-perf.csv"
    write_header = not perf_csv.exists()
    with perf_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "step", "elapsed_sec"])
        writer.writerow(
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, round(elapsed_sec, 1)]
        )


if __name__ == "__main__":
    main()
