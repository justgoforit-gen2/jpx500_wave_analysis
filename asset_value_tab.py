"""資産バリュータブ: EDINET 有報から B/S・有価証券・土地の含み損益を算出して表示する。

Stage 2〜7 の分析ロジックを Streamlit UI に統合したモジュール。
EdinetClient は naibu-ryuho-app/scripts/utils/ を動的インポートして流用。
"""
from __future__ import annotations

import re
import sys
import time
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

# naibu-ryuho-app の scripts/ をインポートパスに追加（EdinetClient 流用）
_NAIBU_SCRIPTS = Path(__file__).parent.parent / "naibu-ryuho-app" / "scripts"
if str(_NAIBU_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_NAIBU_SCRIPTS))

# ダウンロードキャッシュは naibu-ryuho-app/scripts/raw/ と共用
_RAW_DIR = _NAIBU_SCRIPTS / "raw"
_RAW_DIR.mkdir(parents=True, exist_ok=True)

# 政策保有株式 XBRL タグ（SpecifiedInvestment）
_SPECIFIED_TAGS = {
    "name": "jpcrp_cor:NameOfSecuritiesDetailsOfSpecifiedInvestmentEquitySecuritiesHeldForPurposesOtherThanPureInvestmentReportingCompany",
    "shares": "jpcrp_cor:NumberOfSharesHeldDetailsOfSpecifiedInvestmentEquitySecuritiesHeldForPurposesOtherThanPureInvestmentReportingCompany",
    "book_value": "jpcrp_cor:BookValueDetailsOfSpecifiedInvestmentEquitySecuritiesHeldForPurposesOtherThanPureInvestmentReportingCompany",
}
_DEEMED_TAGS = {
    "name": "jpcrp_cor:NameOfSecuritiesDetailsOfDeemedHoldingsOfEquitySecuritiesHeldForPurposesOtherThanPureInvestmentReportingCompany",
    "shares": "jpcrp_cor:NumberOfSharesHeldDetailsOfDeemedHoldingsOfEquitySecuritiesHeldForPurposesOtherThanPureInvestmentReportingCompany",
    "book_value": "jpcrp_cor:BookValueDetailsOfDeemedHoldingsOfEquitySecuritiesHeldForPurposesOtherThanPureInvestmentReportingCompany",
}

# 日産固定ティッカーマップ（銘柄名 → Yahoo Finance ティッカー）
_DEFAULT_TICKER_MAP: dict[str, str] = {
    "スターフライヤー": "9206.T",
    "㈱スターフライヤー": "9206.T",
    "ミツバ": "7280.T",
    "㈱ミツバ": "7280.T",
}


# ─────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────

def _get_client(raw_dir: Path | None = None):
    from utils.edinet_client import EdinetClient  # noqa: PLC0415
    return EdinetClient(raw_dir=raw_dir or _RAW_DIR)


def _to_int(s: str | None) -> int | None:
    if not s:
        return None
    cleaned = re.sub(r"[^0-9]", "", str(s))
    return int(cleaned) if cleaned else None


def _extract_zip(zip_path: Path) -> Path:
    extract_to = zip_path.parent / zip_path.stem
    if not extract_to.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_to)
    return extract_to


def _ixbrl_files(zip_path: Path) -> list[Path]:
    extract_to = _extract_zip(zip_path)
    return sorted((extract_to / "XBRL" / "PublicDoc").glob("*_ixbrl.htm"))


# ─────────────────────────────────────────────
# Stage 1: EDINET doc_id 検索
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def search_doc_id(edinet_code: str) -> dict | None:
    """最新有価証券報告書の doc_id を検索（キャッシュ 1h）。

    検索順:
    1. 直近 90 日（最新提出を先に確認）
    2. 有報提出が集中する 6〜8 月（過去 4 年分、月末から逆順）
    3. フォールバック: 過去 730 日を逆順に補完
    """
    from datetime import date, timedelta  # noqa: PLC0415
    client = _get_client()
    today = date.today()
    candidates: list[date] = []
    seen: set[str] = set()

    def _add(d: date) -> None:
        k = d.isoformat()
        if k not in seen and d <= today:
            candidates.append(d)
            seen.add(k)

    # ① 直近 90 日（逆順）
    for i in range(90):
        _add(today - timedelta(days=i))

    # ② 有報提出月（6〜8 月）を過去 4 年分、月末から逆順で優先検索
    for year_offset in range(1, 5):
        year = today.year - year_offset
        for month in [6, 7, 8]:
            for day in range(31, 0, -1):
                try:
                    _add(date(year, month, day))
                except ValueError:
                    pass

    # ③ フォールバック: 過去 730 日の残り日付を補完
    for i in range(730):
        _add(today - timedelta(days=i))

    for target in candidates:
        data = client.list_documents(target.isoformat(), type_=2)
        for r in (data.get("results") or []):
            if r.get("edinetCode") == edinet_code and r.get("docTypeCode") == "120":
                return {
                    "doc_id": r["docID"],
                    "filer_name": r.get("filerName", ""),
                    "period_end": r.get("periodEnd", ""),
                    "submit_date": r.get("submitDateTime", "")[:10],
                    "ordinance_code": r.get("ordinanceCode", "010"),
                }
    return None


# ─────────────────────────────────────────────
# Stage 2: B/S 簿価取得
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_bs_values(doc_id: str) -> dict:
    """投資有価証券・土地・現金の B/S 簿価を返す（キャッシュ 24h）。"""
    warnings.filterwarnings("ignore")
    client = _get_client()
    zip_path = client.download_document(doc_id, type_=1)
    extract_to = _extract_zip(zip_path)

    xbrl_list = list((extract_to / "XBRL" / "PublicDoc").glob("*.xbrl"))
    if not xbrl_list:
        return {}

    from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser  # noqa: PLC0415
    edi = EdinetXbrlParser().parse_file(str(xbrl_list[0]))

    def _pick(element_id: str) -> tuple[int | None, bool]:
        for ctx, is_con in [
            ("CurrentYearInstant", True),
            ("CurrentYearInstant_NonConsolidatedMember", False),
        ]:
            try:
                data = edi.get_data_by_context_ref(element_id, ctx)
            except Exception:
                continue
            if data is None:
                continue
            val = data.get_value()
            if val is None or val == "":
                continue
            try:
                return int(float(val)), is_con
            except (TypeError, ValueError):
                continue
        return None, False

    inv_sec, inv_con = _pick("jppfs_cor:InvestmentSecurities")
    if inv_sec is None:
        inv_sec, inv_con = _pick("jpigp_cor:OtherInvestmentsNCLIFRS")
    land, land_con = _pick("jppfs_cor:Land")
    cash, cash_con = _pick("jppfs_cor:CashAndCashEquivalents")
    if cash is None:
        cash, cash_con = _pick("jpigp_cor:CashAndCashEquivalentsIFRS")
    total_assets, _ = _pick("jppfs_cor:Assets")
    if total_assets is None:
        total_assets, _ = _pick("jpigp_cor:TotalAssetsIFRS")

    return {
        "investment_securities": inv_sec,
        "investment_securities_consolidated": inv_con,
        "land": land,
        "land_consolidated": land_con,
        "cash": cash,
        "cash_consolidated": cash_con,
        "total_assets": total_assets,
    }


# ─────────────────────────────────────────────
# Stage 3: 政策保有株式リスト
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_holdings(doc_id: str) -> pd.DataFrame:
    """政策保有株式（特定投資株式・みなし保有）を返す（キャッシュ 24h）。"""
    from bs4 import BeautifulSoup  # noqa: PLC0415
    client = _get_client()
    zip_path = client.download_document(doc_id, type_=1)

    def _parse_rows(soup: "BeautifulSoup", tags: dict) -> dict[str, dict]:
        rows: dict[str, dict] = defaultdict(dict)
        for field, tag_name in tags.items():
            for tag in soup.find_all(attrs={"name": tag_name}):
                ctx = tag.get("contextref", "")
                if "CurrentYear" not in ctx:
                    continue
                m = re.search(r"(Row\d+)", ctx)
                row_key = m.group(1) if m else ctx
                rows[row_key][field] = tag.get_text(strip=True)
        return dict(rows)

    all_rows = []
    for htm in _ixbrl_files(zip_path):
        content = htm.read_text(errors="replace")
        if "SpecifiedInvestment" not in content and "DeemedHoldings" not in content:
            continue
        soup = BeautifulSoup(content, "lxml")
        for kind, tags in [("特定投資株式", _SPECIFIED_TAGS), ("みなし保有", _DEEMED_TAGS)]:
            for _, data in sorted(_parse_rows(soup, tags).items()):
                name = data.get("name", "")
                if name:
                    all_rows.append({
                        "種類": kind,
                        "銘柄名": name,
                        "株数": _to_int(data.get("shares")),
                        "簿価_百万円": _to_int(data.get("book_value")),
                    })
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────
# Stage 4: 設備の状況（土地リスト）
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_properties(doc_id: str) -> pd.DataFrame:
    """設備の状況から土地面積・簿価・所在地を返す（キャッシュ 24h）。"""
    from bs4 import BeautifulSoup  # noqa: PLC0415
    client = _get_client()
    zip_path = client.download_document(doc_id, type_=1)

    for htm in _ixbrl_files(zip_path):
        content = htm.read_text(errors="replace")
        if "MajorFacilitiesTextBlock" not in content and "設備の状況" not in content:
            continue
        soup = BeautifulSoup(content, "lxml")
        for table in soup.find_all("table"):
            text = table.get_text()
            if "土地" not in text or "所在地" not in text or "面積" not in text:
                continue
            rows = table.find_all("tr")
            data_start = 0
            for i, row in enumerate(rows):
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if "面積" in " ".join(cells) and "金額" in " ".join(cells):
                    data_start = i + 1
                    break
            records = []
            current_facility = None
            for row in rows[data_start:]:
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if not cells or len(cells) < 3:
                    continue
                if len(cells) >= 10:
                    facility, location = cells[0], cells[1]
                    area = _to_int(cells[3])
                    land_val = _to_int(cells[4])
                    current_facility = facility
                elif len(cells) == 9 and current_facility:
                    facility = current_facility + "（続き）"
                    location = cells[0]
                    area = _to_int(cells[2])
                    land_val = _to_int(cells[3])
                else:
                    continue
                if area is None and land_val is None:
                    continue
                records.append({
                    "事業所名": facility,
                    "所在地": location,
                    "土地面積_m2": area,
                    "土地簿価_百万円": land_val,
                })
            if records:
                return pd.DataFrame(records)
    return pd.DataFrame()


# ─────────────────────────────────────────────
# Stage 5: 有価証券時価換算
# ─────────────────────────────────────────────

def mark_to_market_securities(
    holdings_df: pd.DataFrame,
    ticker_map: dict[str, str],
) -> pd.DataFrame:
    """yfinance で株価取得し含み損益を算出する（キャッシュなし・毎回最新）。"""
    import yfinance as yf  # noqa: PLC0415

    rows = []
    for _, row in holdings_df.iterrows():
        name = str(row.get("銘柄名", ""))
        shares = row.get("株数")
        book_val = row.get("簿価_百万円")
        ticker = ticker_map.get(name)
        price = None
        market_val = None
        unrealized = None
        if ticker and shares:
            try:
                hist = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
                if not hist.empty:
                    price = hist.iloc[-1]
                    market_val = round(price * int(shares) / 1_000_000)
                    if book_val is not None:
                        unrealized = market_val - int(book_val)
            except Exception:
                pass
            time.sleep(0.3)
        rows.append({
            "種類": row.get("種類", ""),
            "銘柄名": name,
            "株数": shares,
            "簿価_百万円": book_val,
            "ティッカー": ticker or "—",
            "株価_円": int(price) if price is not None else None,
            "時価_百万円": market_val,
            "含み損益_百万円": unrealized,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Stage 7: 土地時価換算
# ─────────────────────────────────────────────

def mark_to_market_land(
    properties_df: pd.DataFrame,
    multiplier: float,
) -> pd.DataFrame:
    """簿価 × multiplier で土地時価を算出する。"""
    rows = []
    for _, row in properties_df.iterrows():
        book_val = row.get("土地簿価_百万円")
        market_val = round(book_val * multiplier) if book_val is not None else None
        unrealized = (market_val - book_val) if (market_val is not None and book_val is not None) else None
        rows.append({
            "事業所名": row.get("事業所名", ""),
            "所在地": row.get("所在地", ""),
            "土地面積_m2": row.get("土地面積_m2"),
            "土地簿価_百万円": book_val,
            "土地時価_百万円": market_val,
            "含み損益_百万円": unrealized,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 信頼性判別ガード
# ─────────────────────────────────────────────

_BANK_KW = ["銀行", "フィナンシャルグループ", "信用金庫", "信用組合"]
_INS_KW  = ["保険", "生命", "損保", "共済"]


def _classify_filer(ordinance_code: str, filer_name: str) -> str:
    """'bank' / 'insurance' / 'general' を返す。"""
    if ordinance_code == "030":
        return "bank"
    if ordinance_code == "040":
        return "insurance"
    if any(k in filer_name for k in _BANK_KW):
        return "bank"
    if any(k in filer_name for k in _INS_KW):
        return "insurance"
    return "general"


def _land_sanity(
    properties_df: pd.DataFrame | None,
    bs_land_yen: int | None,
) -> tuple[bool, float | None]:
    """設備の状況テーブルの土地簿価合計が B/S と整合するか確認。
    Returns (is_reliable, ratio). ratio > 5 → 信頼できない。
    """
    if properties_df is None or properties_df.empty or bs_land_yen is None or bs_land_yen == 0:
        return True, None
    bs_land_m = bs_land_yen // 1_000_000
    if bs_land_m <= 0:
        return True, None
    parsed_sum = float(properties_df["土地簿価_百万円"].dropna().sum())
    if parsed_sum <= 0:
        return True, None
    ratio = parsed_sum / bs_land_m
    return ratio <= 5.0, ratio


def _render_bs_land_fallback(land: int | None, key_suffix: str) -> tuple[int, int]:
    """B/S 土地簿価 × 倍率スライダーで土地含み損益の概算を表示する。
    Returns (est_market_m, est_unrealized_m) in 百万円 units.
    """
    bs_land_m = (land // 1_000_000) if land else 0
    multiplier = st.slider(
        "土地時価 倍率（簿価 × N）",
        min_value=1.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
        key=f"av_land_multiplier_{key_suffix}",
        help="設備の状況テーブルが使用できないため B/S 連結簿価 × 倍率で概算します。",
    )
    est_market = int(bs_land_m * multiplier)
    est_unrealized = est_market - bs_land_m
    c1, c2, c3 = st.columns(3)
    c1.metric("土地時価（概算）", f"{est_market:,} 百万円")
    c2.metric("B/S 土地簿価（連結）", f"{bs_land_m:,} 百万円")
    c3.metric(
        "含み損益（概算）",
        f"+{est_unrealized:,} 百万円" if est_unrealized >= 0 else f"{est_unrealized:,} 百万円",
    )
    st.caption("※ 設備の状況テーブルは非標準フォーマットのため使用せず、B/S 連結簿価を直接使用")
    return est_market, est_unrealized


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

def _fmt_m(v: int | None) -> str:
    if v is None:
        return "N/A"
    return f"{v // 1_000_000:,} 百万円"


def _colored_delta(v: int | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    color = "#2ecc71" if v >= 0 else "#e74c3c"
    return f'<span style="color:{color}; font-weight:bold">{sign}{v:,} 百万円</span>'


def render_tab() -> None:
    """資産バリュータブの Streamlit UI を描画する。"""
    st.header("資産バリュー分析")
    st.caption(
        "EDINET 有報（XBRL）から投資有価証券・土地の含み損益を算出します。"
        " EDINET_API_KEY が .env に設定済みである必要があります。"
    )

    # ── 入力フォーム ──────────────────────────────────────────────
    col_code, col_btn = st.columns([3, 1])
    with col_code:
        edinet_code = st.text_input(
            "EDINET コード",
            value=st.session_state.get("av_edinet_code", "E02142"),
            placeholder="例: E02142（日産自動車）",
            key="av_edinet_input",
        )
    with col_btn:
        st.write("")
        run_clicked = st.button("分析実行", key="av_run_btn", use_container_width=True)

    if run_clicked:
        st.session_state["av_edinet_code"] = edinet_code
        st.session_state["av_run"] = True

    if not st.session_state.get("av_run"):
        st.info("EDINET コードを入力して「分析実行」をクリックしてください。（デフォルト: E02142 = 日産自動車）")
        return

    code = st.session_state.get("av_edinet_code", edinet_code)

    # ── Stage 1: doc_id 取得 ──────────────────────────────────────
    with st.spinner(f"{code} の有価証券報告書を EDINET で検索中..."):
        try:
            doc_info = search_doc_id(code)
        except RuntimeError as e:
            st.error(f"EDINET API エラー: {e}")
            return

    if doc_info is None:
        st.error(f"{code} の有価証券報告書が見つかりません。EDINET コードを確認してください。")
        return

    st.success(
        f"**{doc_info['filer_name']}** — {doc_info['period_end']} 期  "
        f"（提出日: {doc_info['submit_date']} / doc_id: {doc_info['doc_id']}）"
    )
    doc_id = doc_info["doc_id"]

    # 会社種別判定とガードフラグ初期化
    filer_type = _classify_filer(doc_info.get("ordinance_code", "010"), doc_info["filer_name"])
    skip_securities = False
    use_bs_land_only = False
    est_market: int = 0
    est_unrealized: int = 0

    if filer_type in ("bank", "insurance"):
        label = "銀行・金融グループ" if filer_type == "bank" else "保険会社"
        st.warning(
            f"⚠️ **{label}**（EDINET 府令コード: {doc_info.get('ordinance_code', '—')}）"
            " が検出されました。"
            " 有価証券含み損益（② セクション）はスキップします。"
            " 土地含み損益は B/S 簿価 × 倍率の概算のみ表示します。",
        )
        skip_securities = True
        use_bs_land_only = True

    # ── Stage 2: B/S 簿価 ────────────────────────────────────────
    with st.spinner("XBRL パース中（初回は時間がかかります）..."):
        bs = get_bs_values(doc_id)

    st.subheader("① B/S サマリー（簿価）")
    if bs:
        c1, c2, c3 = st.columns(3)
        inv_sec = bs.get("investment_securities")
        land = bs.get("land")
        cash = bs.get("cash")
        inv_label = "連結" if bs.get("investment_securities_consolidated") else "個別"
        land_label = "連結" if bs.get("land_consolidated") else "個別"
        cash_label = "連結" if bs.get("cash_consolidated") else "個別"
        c1.metric("現金・預金", _fmt_m(cash), help=cash_label)
        c2.metric("投資有価証券（簿価）", _fmt_m(inv_sec), help=inv_label)
        c3.metric("土地（簿価）", _fmt_m(land), help=land_label)
        if bs.get("total_assets"):
            st.caption(f"総資産: {_fmt_m(bs['total_assets'])}")
        # 投資有価証券タグなし検知（銀行・保険でない一般会社でも発動）
        if not skip_securities and inv_sec in (None, 0):
            st.info("投資有価証券タグが見つかりませんでした。有価証券含み損益の分析はスキップします。")
            skip_securities = True
    else:
        st.warning("B/S 値の取得に失敗しました。")
        return

    # ── Stage 3 + 5: 有価証券 ─────────────────────────────────────
    st.subheader("② 有価証券含み損益")
    if skip_securities:
        st.info("この企業では有価証券含み損益の分析は対象外です。")
        holdings_df = pd.DataFrame()
    else:
        with st.spinner("保有有価証券リストを取得中..."):
            holdings_df = get_holdings(doc_id)

    if not skip_securities and holdings_df.empty:
        st.info("政策保有株式の XBRL タグが見つかりませんでした。")
    elif not holdings_df.empty:
        st.caption(f"政策保有株式: {len(holdings_df)} 銘柄")

        with st.expander("ティッカーマップを編集", expanded=False):
            ticker_input = st.text_area(
                "銘柄名=ティッカー（1行1件）",
                value="\n".join(f"{k}={v}" for k, v in _DEFAULT_TICKER_MAP.items()),
                height=120,
                key="av_ticker_map",
            )
        ticker_map = {}
        for line in ticker_input.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                ticker_map[k.strip()] = v.strip()

        if st.button("株価取得（yfinance）", key="av_fetch_prices"):
            with st.spinner("株価を取得中..."):
                mtm_df = mark_to_market_securities(holdings_df, ticker_map)
            st.session_state["av_securities_mtm"] = mtm_df

        if "av_securities_mtm" in st.session_state:
            mtm_df = st.session_state["av_securities_mtm"]
            st.dataframe(
                mtm_df.style.format({
                    "株数": "{:,.0f}",
                    "簿価_百万円": "{:,.0f}",
                    "株価_円": "{:,.0f}",
                    "時価_百万円": "{:,.0f}",
                    "含み損益_百万円": "{:+,.0f}",
                }, na_rep="—"),
                use_container_width=True,
            )
            calc_rows = mtm_df.dropna(subset=["含み損益_百万円"])
            if not calc_rows.empty:
                total_unrealized = int(calc_rows["含み損益_百万円"].sum())
                total_market = int(calc_rows["時価_百万円"].sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("時価合計（算出分）", f"{total_market:,} 百万円")
                c2.metric("BS簿価（投資有価証券）", _fmt_m(inv_sec))
                delta_str = f"+{total_unrealized:,}" if total_unrealized >= 0 else f"{total_unrealized:,}"
                c3.metric("含み損益（算出分）", f"{delta_str} 百万円")
        else:
            st.dataframe(holdings_df, use_container_width=True)

    # ── Stage 4 + 7: 土地 ────────────────────────────────────────
    st.subheader("③ 土地含み損益")
    properties_df = pd.DataFrame()
    parent_market: int = 0

    if use_bs_land_only:
        est_market, est_unrealized = _render_bs_land_fallback(land, "bank")
    else:
        with st.spinner("設備の状況（土地リスト）を取得中..."):
            properties_df = get_properties(doc_id)

        if properties_df.empty:
            st.info("設備の状況テーブルが見つかりませんでした。B/S 土地簿価のみ参照してください。")
        else:
            is_reliable, ratio = _land_sanity(properties_df, land)
            if not is_reliable:
                st.error(
                    f"⚠️ 設備の状況テーブルの土地簿価合計が B/S 土地簿価の **{ratio:.0f} 倍** になっています。"
                    " HTML テーブル構造が非標準のため自動解析は信頼できません。B/S 簿価 × 倍率で概算します。"
                )
                use_bs_land_only = True
                est_market, est_unrealized = _render_bs_land_fallback(land, "sanity")
            else:
                st.caption(f"事業所数: {len(properties_df)} 件")

                multiplier = st.slider(
                    "土地時価 倍率（簿価 × N）",
                    min_value=1.0,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    key="av_land_multiplier",
                    help="国交省 API 未設定時の代替倍率。1.5 = 簿価の1.5倍を時価とみなす。",
                )

                land_mtm_df = mark_to_market_land(properties_df, multiplier)
                st.dataframe(
                    land_mtm_df.style.format({
                        "土地面積_m2": "{:,.0f}",
                        "土地簿価_百万円": "{:,.0f}",
                        "土地時価_百万円": "{:,.0f}",
                        "含み損益_百万円": "{:+,.0f}",
                    }, na_rep="—"),
                    use_container_width=True,
                )

                parent_book = int(properties_df["土地簿価_百万円"].dropna().sum())
                parent_market = int(land_mtm_df["土地時価_百万円"].dropna().sum())
                parent_unrealized = parent_market - parent_book
                c1, c2, c3 = st.columns(3)
                c1.metric("親会社土地時価合計", f"{parent_market:,} 百万円")
                c2.metric("親会社土地簿価合計", f"{parent_book:,} 百万円")
                c3.metric(
                    "含み損益（親会社）",
                    f"+{parent_unrealized:,} 百万円" if parent_unrealized >= 0 else f"{parent_unrealized:,} 百万円",
                )
                if land is not None:
                    diff = land // 1_000_000 - parent_book
                    st.caption(
                        f"参考: 連結 B/S 土地簿価 {land // 1_000_000:,} 百万円 − 親会社合計 {parent_book:,} 百万円"
                        f" = **{diff:,} 百万円**（子会社・海外拠点分）"
                    )

    # ── ④ 資産バリューサマリー ────────────────────────────────────
    st.subheader("④ 資産バリューサマリー")
    av_items: dict[str, int | None] = {}
    av_items["現金・預金"] = cash // 1_000_000 if cash else None
    av_items["投資有価証券（BS簿価）"] = inv_sec // 1_000_000 if inv_sec else None
    av_items["土地（BS連結簿価）"] = land // 1_000_000 if land else None

    if "av_securities_mtm" in st.session_state:
        mtm_df = st.session_state["av_securities_mtm"]
        calc_rows = mtm_df.dropna(subset=["時価_百万円"])
        if not calc_rows.empty:
            av_items["有価証券時価（算出分）"] = int(calc_rows["時価_百万円"].sum())

    if use_bs_land_only:
        av_items["土地時価（B/S簿価×倍率）"] = est_market
    elif not properties_df.empty:
        av_items["土地時価（親会社・倍率方式）"] = parent_market

    rows_summary = [{"項目": k, "金額（百万円）": v} for k, v in av_items.items() if v is not None]
    if rows_summary:
        df_summary = pd.DataFrame(rows_summary)
        st.dataframe(
            df_summary.style.format({"金額（百万円）": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
