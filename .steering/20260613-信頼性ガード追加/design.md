# 設計: 資産バリュー分析 — 信頼性判別ガード

## 変更ファイル

対象: `jpx500_wave_analysis/asset_value_tab.py` のみ（1ファイル変更）

## 判別式の設計

3 つの判定シグナルを順に評価する（AND ではなく先着優先）：

| シグナル | 判定方法 | 確実度 |
|---|---|---|
| 1. EDINET 府令コード | `ordinanceCode` == "030" (銀行) / "040" (保険) | 最も確実 |
| 2. 会社名キーワード | `filerName` に "銀行", "フィナンシャルグループ" 等を含む | 高い |
| 3. 土地サニティチェック | 設備テーブル合計 > B/S 土地 × 5 倍 | 後出しガード |

## アーキテクチャ

### 新規追加: モジュールレベル定数

```python
_BANK_KW = ["銀行", "フィナンシャルグループ", "信用金庫", "信用組合"]
_INS_KW  = ["保険", "生命", "損保", "共済"]
```

### 新規追加: ヘルパー関数 2 本

```python
def _classify_filer(ordinance_code: str, filer_name: str) -> str:
    """'bank' / 'insurance' / 'general' を返す。"""
    if ordinance_code == "030": return "bank"
    if ordinance_code == "040": return "insurance"
    if any(k in filer_name for k in _BANK_KW): return "bank"
    if any(k in filer_name for k in _INS_KW): return "insurance"
    return "general"

def _land_sanity(properties_df, bs_land_yen) -> tuple[bool, float | None]:
    """設備テーブルの土地簿価合計が B/S と整合するか確認。
    Returns (is_reliable, ratio). ratio > 5 → 信頼できない。
    """
```

### 新規追加: フォールバック表示ヘルパー

```python
def _render_bs_land_fallback(land: int | None, key_suffix: str) -> tuple[int, int]:
    """B/S 土地簿価 × 倍率スライダー の表示。(est_market, est_unrealized) を返す。"""
```

### 変更: `search_doc_id()` return dict

```python
# 追加フィールド
"ordinance_code": r.get("ordinanceCode", "010"),
```

### 変更: `render_tab()` — Stage 1 直後

```python
filer_type = _classify_filer(doc_info.get("ordinance_code", "010"), doc_info["filer_name"])
skip_securities = False
use_bs_land_only = False

if filer_type in ("bank", "insurance"):
    label = "銀行・金融グループ" if filer_type == "bank" else "保険会社"
    st.warning(..., icon="⚠️")
    skip_securities = True
    use_bs_land_only = True
```

### 変更: `render_tab()` — Stage 2 直後

```python
# 投資有価証券タグなし検知（一般会社でも発動）
if not skip_securities and inv_sec in (None, 0):
    st.info("投資有価証券タグが見つかりませんでした。有価証券含み損益の分析はスキップします。")
    skip_securities = True
```

### 変更: `render_tab()` — Stage 3/5 セクション

```python
if skip_securities:
    st.info("この企業では有価証券含み損益の分析は対象外です。")
else:
    # 既存コード
```

### 変更: `render_tab()` — Stage 4/7 セクション

```python
if use_bs_land_only:
    est_market, est_unrealized = _render_bs_land_fallback(land, "bank")
else:
    properties_df = get_properties(doc_id)
    is_reliable, ratio = _land_sanity(properties_df, land)
    if not is_reliable:
        st.error(f"設備テーブルの土地簿価合計が B/S の {ratio:.0f} 倍 ...")
        est_market, est_unrealized = _render_bs_land_fallback(land, "sanity")
        use_bs_land_only = True
    else:
        # 既存コード（properties_df テーブル表示）
```

### 変更: `render_tab()` — ④ 資産バリューサマリー

```python
# use_bs_land_only フラグで土地の値を切り替え
if use_bs_land_only:
    av_items["土地時価（B/S簿価×倍率）"] = est_market
    av_items["土地（BS連結簿価）"] = land // 1_000_000 if land else None
else:
    # 既存コード
```

## ガード発動パターン表

| 条件 | Stage 2 B/S | Stage 3/5 有価証券 | Stage 4/7 土地 |
|---|---|---|---|
| 一般会社・サニティOK | 表示 | 表示 | 設備の状況テーブル表示 |
| 銀行・保険（シグナル1or2） | 表示 | スキップ＋警告 | B/S簿価×倍率のみ |
| サニティNG（シグナル3のみ） | 表示 | 表示 | B/S簿価×倍率にフォールバック＋エラー |
| 投資有価証券タグなし | 表示 | スキップ＋info | 通常通り |

## 注意事項

- `get_properties()` は `use_bs_land_only=True` の場合は呼び出さない（不要なダウンロードを避ける）
- `_render_bs_land_fallback()` は `key_suffix` 引数でスライダーの key を分ける（複数回呼び出し時の重複 key 防止）
- `properties_df` が空の場合の既存 info メッセージは残す（一般会社で設備テーブルなしのケースは従来通り）
