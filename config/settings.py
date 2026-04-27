"""設定値（窓長、閾値、分類境界など）"""

from pathlib import Path

# --- パス ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
STOCK_LIST_CSV = DATA_DIR / "jpx500_list.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
DAILY_PICKS_CSV = DATA_DIR / "daily_picks.csv"

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
