"""設定値（窓長、閾値、分類境界など）"""

import os
from pathlib import Path

# --- パス ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
STOCK_LIST_CSV = DATA_DIR / "jpx500_list.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
DAILY_PICKS_CSV = DATA_DIR / "daily_picks.csv"
PER_PBR_HISTORY_PARQUET = DATA_DIR / "per_pbr_history.parquet"
PER_PBR_FAILURES_CSV = DATA_DIR / "per_pbr_failures.csv"

# --- PER/PBR履歴 ---
PER_PBR_SAMPLING_RULE = "W-FRI"  # 金曜終値で週次サンプリング
PER_PBR_LOOKBACK_YEARS = 3
PER_PBR_REPORT_LAG_DAYS = 45  # 発表日不明時の保守的フォールバック
PER_PBR_MIN_QUARTERS = 4  # TTM算出に必要な最小四半期数
PER_PBR_FETCH_RETRY = 3
PER_PBR_FETCH_RETRY_DELAY_SEC = 3
PER_PBR_DEFAULT_PER_CAP = 100
PER_PBR_DEFAULT_PBR_CAP = 10

# --- JPX投資部門別取引（海外投資家フロー） ---
JPX_INVESTOR_TYPE_PAGE_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html"
)
# 年次バックナンバーアーカイブ。 -01=直近年, -02=前年, ... の連番で過去年が並ぶ。
JPX_INVESTOR_TYPE_ARCHIVE_URL_TEMPLATE = "https://www.jpx.co.jp/markets/statistics-equities/investor-type/00-00-archives-{n:02d}.html"
JPX_INVESTOR_TYPE_ARCHIVE_MAX_PAGES = 6  # -01 ~ -06 (約6年遡れる)
JPX_INVESTOR_TYPE_CACHE_DIR = DATA_DIR / "jpx_investor_type"
JPX_INVESTOR_FLOW_PARQUET = DATA_DIR / "foreign_flow.parquet"
JPX_INVESTOR_FLOW_MARKETS = (
    "TSE Prime",
    "TSE Standard",
    "TSE Growth",
    "Tokyo & Nagoya",
)
JPX_INVESTOR_FLOW_LOOKBACK_YEARS = 3
JPX_INVESTOR_FLOW_FALLBACK_URLS: list[str] = []
JPX_FETCH_TIMEOUT_SEC = 30
JPX_FETCH_SLEEP_SEC = 1

# --- 資本効率改善期待スクリーナー (CES = Capital Efficiency Screener) ---
NAIBU_DB_PATH = Path(
    os.getenv(
        "NAIBU_DB_PATH",
        str(Path(__file__).resolve().parents[2] / "naibu-ryuho-app" / "data.db"),
    )
)
NAIBU_API_BASE_URL = "http://localhost:8000"
NAIBU_FETCH_TIMEOUT_SEC = 10
CAPITAL_EFFICIENCY_PARQUET = DATA_DIR / "capital_efficiency_screen.parquet"
CAPITAL_EFFICIENCY_RAW_PARQUET = DATA_DIR / "capital_efficiency_raw.parquet"

# ハードフィルタ閾値
CES_MIN_EQUITY_RATIO = 0.50  # 自己資本比率 >= 50%
# スコア帯閾値 (PBR)
CES_PBR_TIERS = ((0.7, 3), (1.0, 2), (1.2, 1))
# スコア帯閾値 (ネットキャッシュ/時価)
CES_NETCASH_TIERS = ((0.5, 3), (0.3, 2), (0.0, 1))
# スコア帯閾値 (ROE bin: 3-8%帯がスイートスポット)
CES_ROE_SWEET_LO = 3.0
CES_ROE_SWEET_HI = 8.0
CES_ROE_SECONDARY_LO = 0.0
CES_ROE_SECONDARY_HI = 10.0
# スコア帯閾値 (配当性向)
CES_PAYOUT_TIERS = ((0.20, 2), (0.30, 1))
CES_DIVIDEND_YIELD_FALLBACK = 2.0  # %, 配当性向欠損時の代替閾値

# --- v1.1: 株主構造KPI 閾値 ---
CES_INSIDER_THRESHOLD = 50.0  # %, インサイダー保有>=これ -> 外圧効かないと判定
CES_INSTITUTION_HIGH = 30.0  # %, 機関投資家>=これ -> アクティビスト動きやすい (2点)
CES_INSTITUTION_MID = 20.0  # %, 機関投資家>=これ -> 中間 (1点)
CES_TOTAL_SCORE_MAX = (
    12  # PBR(3) + ネットキャッシュ(3) + ROE(2) + 還元(2) + 株主構造(2)
)

# --- ユニバース拡張 (TSE Standard Top 400) ---
JPX_DATA_J_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
JPX_DATA_J_CACHE = CACHE_DIR / "jpx_data_j.xls"
STANDARD_LIST_CSV = DATA_DIR / "standard_list.csv"
STANDARD_TOP_N = 400  # スタンダード市場で取り込む上位銘柄数
STANDARD_TOP100_THRESHOLD = 100  # Top100 (1-100) と Top400 (101-400) の境界
JPX_UNIVERSE_REFRESH_DAYS = 7  # data_j.xls の再取得間隔
STANDARD_MARKET_LABEL = "TSE Standard"
PRIME_MARKET_LABEL = "TSE Prime"

# --- 決算発表日 ---
EARNINGS_CACHE_DIR = DATA_DIR / "earnings"
EARNINGS_COMBINED_CSV = DATA_DIR / "earnings" / "earnings_dates.csv"
EARNINGS_XLSX_URLS = [
    "https://www.jpx.co.jp/listing/event-schedules/financial-announcement/tvdivq0000001ofb-att/kessan12_0206.xlsx",
    "https://www.jpx.co.jp/listing/event-schedules/financial-announcement/tvdivq0000001ofb-att/kessan01_0220.xlsx",
]
EARNINGS_CACHE_MAX_AGE_HOURS = 24

# --- データ取得 ---
DATA_PERIOD_YEARS = 3
YFINANCE_INTERVAL = "1d"
FETCH_BATCH_SIZE = 10  # 同時取得銘柄数
FETCH_RETRY_COUNT = 3
FETCH_RETRY_DELAY_SEC = 5

# --- 評価窓 ---
DEFAULT_WINDOW = 120  # 営業日
WINDOW_OPTIONS = [60, 120, 180]

# --- 移動平均線（週ベース→営業日換算） ---
MA_PERIODS = {
    "MA13W": 65,  # 13週
    "MA26W": 130,  # 26週
    "MA52W": 260,  # 52週
}

# --- RSI ---
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- ATR ---
ATR_PERIOD = 14

# --- レンジ判定 ---
RANGE_PERCENTILE_HIGH = 95  # 上限パーセンタイル
RANGE_PERCENTILE_LOW = 5  # 下限パーセンタイル
TOUCH_THRESHOLD_PCT = 1.0  # タッチ判定: range端から±1%以内

# --- 本日の推奨: 直近N日で判定 ---
DAILY_PICK_LOOKBACK = 3  # 直近N営業日以内にタッチしていれば「本日の推奨」

# --- 波形分類閾値 ---
# slope: 正規化傾き（評価窓内の終値に対する線形回帰傾き / 平均価格）
SLOPE_THRESHOLD = 0.0003  # これ以上なら上昇/下降トレンド

# レンジ（波型）: 傾き小さい & タッチ多い
RANGE_MIN_TOUCHES = 4  # 上下合計タッチ回数の最低値

# 収束（スクイーズ）: バンド幅の縮小率
SQUEEZE_BANDWIDTH_SHRINK = 0.3  # 後半のbandwidthが前半の70%以下

# ブレイク気味: 直近でレンジ外に出た日数
BREAKOUT_LOOKBACK_DAYS = 10
BREAKOUT_MIN_DAYS = 2  # 直近N日中これ以上レンジ外なら

# 高ボラ: ATR / 価格の比率
HIGH_VOLATILITY_THRESHOLD = 0.03  # 3%以上

# ボリンジャーバンド
BB_PERIOD = 20
BB_STD = 2.0

# --- レンジ幅フィルタ（UI用） ---
RANGE_PCT_SMALL = 5.0  # 小波: ≤5%
RANGE_PCT_LARGE = 15.0  # 大波: ≥15%

# --- 波形タイプ定義 ---
WAVE_TYPES = [
    "レンジ（波型）",
    "上昇トレンド",
    "下降トレンド",
    "収束（スクイーズ）",
    "ブレイク気味",
    "高ボラ（荒い）",
]

# --- JPX 信用取引銘柄別残高 (mtdailyk = 日々公表銘柄のみ・日次) ---
JPX_MARGIN_PAGE_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/margin/index.html"
)
JPX_MARGIN_FILE_URL_TEMPLATE = (
    "https://www.jpx.co.jp/markets/statistics-equities/margin/"
    "tvdivq0000001r92-att/mtdailyk{date}00.xls"
)
JPX_MARGIN_CACHE_DIR = DATA_DIR / "jpx_margin"
JPX_MARGIN_HISTORY_PARQUET = DATA_DIR / "margin_history.parquet"
JPX_MARGIN_LATEST_PARQUET = DATA_DIR / "margin_latest.parquet"
JPX_MARGIN_LOOKBACK_DAYS = 90  # 過去90営業日分を累積（初回バックフィル）
JPX_MARGIN_DEADLINE_DAYS = 180  # 制度信用は約6ヶ月で期日

# --- JPX 銘柄別信用取引週末残高 (syumatsu = 全プライム銘柄・週次 PDF) ---
# 日々公表銘柄(mtdailyk)に乗らない優良大型株(SBG等)はこちらでしかカバーできない。
# 各週金曜時点の残高を翌週水曜頃にPDFで公表。
JPX_MARGIN_WEEKLY_PAGE_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/margin/05.html"
)
JPX_MARGIN_WEEKLY_FILE_URL_TEMPLATE = (
    "https://www.jpx.co.jp/markets/statistics-equities/margin/"
    "tvdivq0000001rnl-att/syumatsu{date}00.pdf"
)
JPX_MARGIN_WEEKLY_CACHE_DIR = DATA_DIR / "jpx_margin_weekly"
JPX_MARGIN_WEEKLY_HISTORY_PARQUET = DATA_DIR / "margin_weekly_history.parquet"
JPX_MARGIN_WEEKLY_LATEST_PARQUET = DATA_DIR / "margin_weekly_latest.parquet"
JPX_MARGIN_WEEKLY_LOOKBACK_WEEKS = 12  # 過去12週分を累積

# --- 株探 週次信用残バックフィル (3年以上のヒストリ取得用) ---
# JPXは直近5週しかアーカイブを公開しないため、株探の公開ページから
# 過去の週次信用残データを取得して履歴を補完する。
KABUTAN_MARGIN_URL_TEMPLATE = (
    "https://kabutan.jp/stock/kabuka?code={code4}&ashi=shin&page={page}"
)
KABUTAN_MARGIN_CACHE_DIR = DATA_DIR / "kabutan_margin"
KABUTAN_MARGIN_HISTORY_PARQUET = DATA_DIR / "margin_kabutan_history.parquet"
KABUTAN_FETCH_SLEEP_SEC = 2.0  # ページ取得間のスリープ (サイト負荷配慮)
KABUTAN_FETCH_TIMEOUT_SEC = 20
KABUTAN_DEFAULT_MAX_PAGES = 5  # 約3年分相当

# 信用過熱度 判定閾値
MARGIN_RATIO_HIGH = 5.0  # 信用倍率 >= 5: 買い偏重・戻り売り重い
MARGIN_RATIO_LOW = 1.0  # 信用倍率 < 1: 売り偏重・踏み上げ余地
MARGIN_BUY_PCT_WARN = 5.0  # 買残/上場株式数 >= 5%: 警戒
MARGIN_BUY_PCT_DANGER = 10.0  # 買残/上場株式数 >= 10%: 危険
MARGIN_VOLDAYS_HEAVY = 20.0  # 買残/平均出来高 >= 20日分: 上値しこり重い

# --- ポートフォリオ管理 ---
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"
PORTFOLIO_TRADES_CSV = DATA_DIR / "portfolio_trades.csv"
PORTFOLIO_HISTORY_PARQUET = DATA_DIR / "portfolio_history.parquet"
PORTFOLIO_INITIAL_CSV = DATA_DIR / "portfolio_initial.csv"
WATCHLIST_CSV = DATA_DIR / "watchlist.csv"
EXTENDED_UNIVERSE_CSV = DATA_DIR / "extended_universe.csv"
EXTENDED_RESULTS_CSV = DATA_DIR / "extended_results.csv"
SIGNAL_LOG_PARQUET = DATA_DIR / "signal_log.parquet"

INITIAL_CAPITAL_JPY = 30_000_000

# --- シグナル閾値 ---
SIGNAL_LOSS_CUT_PCT = 0.10  # avg_cost から -10% で損切シグナル
SIGNAL_LOSS_WARNING_PCT = 0.07  # -7% で警告
SIGNAL_TAKE_PROFIT_PCT = 0.30  # +30% で利確シグナル
SIGNAL_RSI_BUY_THRESHOLD = 40  # RSI < 40 で押し目買い候補
SIGNAL_BREAKOUT_LOOKBACK_DAYS = 20  # 直近N日高値ブレイクで上抜けシグナル
SIGNAL_BREAKOUT_VOLUME_RATIO = 1.5  # 出来高がN倍で確認

# --- トレンド転換検出(下降→上昇) ---
TREND_TRANSITION_CSV = DATA_DIR / "trend_transition.csv"
# 直近 N 日と、その前 N 日 を別々に slope 計算して比較
TT_WINDOW_DAYS = 25  # 直近窓 = 25営業日(約5週)
# 過去窓 slope の上限(これより下=確かに下降していた)
TT_PAST_SLOPE_MAX = -0.0005
# 直近窓 slope の下限(これより上=確かに上昇に転じている)
TT_RECENT_SLOPE_MIN = 0.001
# 直近25日の安値からの最低反発率
TT_MIN_REBOUND_PCT = 5.0

# --- Stage 2 ブレイクアウト検出(レンジ → 全MA上抜け+RSI回復) ---
# 5214 日本電気硝子の 2025/7 ブレイクアウト型(初動)を機械的に拾う
RANGE_BREAKOUT_CSV = DATA_DIR / "range_breakout.csv"
# ベース期: 過去 N 日(うち直近 M 日は除外)で「ほぼレンジ」だったか
RB_BASE_LOOKBACK_DAYS = 100  # 過去ベース期の長さ
RB_BASE_EXCLUDE_RECENT_DAYS = 20  # 直近 N 日はベース判定から除外
RB_BASE_MAX_ABS_SLOPE = 0.0015  # ベース期 slope の絶対値上限(やや下げのレンジもOK)
RB_BASE_MAX_RANGE_PCT = 30.0  # ベース期の range幅(%)上限
# MA 構造
RB_MA_TIGHT_THRESHOLD = 0.06  # (max MA - min MA)/mean MA がこれ以下 = MA束ね中
RB_BREAKOUT_MAX_ABOVE_MA = 0.15  # close が max(MA) より +15% 以内 = 走り始め
# RSI 回復
RB_RSI_LOW_LOOKBACK_DAYS = 30  # 直近 N 日に RSI < lo を経験
RB_RSI_LOW_THRESHOLD = 40  # RSI 過去最低の閾値
RB_RSI_NOW_MIN = 40  # 現 RSI 最低
RB_RSI_NOW_MAX = 70  # 現 RSI 最高(過熱除外)
