# 設計書

## アーキテクチャ概要

既存の Streamlit 単一アプリ (`app.py`) + `modules/` レイヤ構成に追従する。新規ファイルは作らず、既存関数の拡張と naibu_client への関数 1 本追加で完結。データは naibu-ryuho-app の SQLite (`financial_metrics`) を読み取り専用で直読み（API 変更なし）。

```
[ランキングタブ _tab_moat_ranking]
   │  load_moat_scores() (parquet)
   │  + load_stock_list() を code で left merge → 銘柄名/ticker 付与
   │  st.dataframe(on_select="rerun")
   │      └─ 行クリック → _navigate_to_detail(code, name, ticker)
   │                         └─ st.session_state["view"]="detail" + rerun
   ▼
[銘柄詳細ビュー show_detail_view]
   ├─ 既存: ヘッダ / 指標 / 業種比較 / 業績(P/L) / 信用残 / 決算
   └─ 新規: 貸借対照表(BS) セクション
            └─ fetch_balance_sheet(code) → financial_metrics 最新期 1 行
               ├─ (a) 指標カード st.metric ×6
               ├─ (b) 構成グラフ plotly 積み上げ横棒
               └─ (c) 明細表 st.dataframe
```

## コンポーネント設計

### 1. ランキングタブ拡張 (`app.py::_tab_moat_ranking`)

**責務**:
- moat_scores parquet に銘柄名/ticker をマージして表示
- 行クリックで銘柄詳細へ遷移

**実装の要点**:
- `load_stock_list()[["code","name","ticker"]]` を `code`(4桁str) で left merge。両者 `dtype=str` で整合。
- `show_cols` / `top10_cols` に `"name"` を `"code"` 直後に挿入。
- 全体表 `st.dataframe` に `on_select="rerun"`, `selection_mode="single-row"`, `key="moat_rank_table"` を付与し、`event.selection.rows` を読んで `_navigate_to_detail` を呼ぶ。
- `ticker` 欠損時は遷移せず `st.warning` でガード（`show_detail_view` は ticker 必須のため）。
- 案内文「コードをコピーして…」を「行をクリックすると詳細が開きます」に更新。

### 2. BS 取得関数 (`modules/naibu_client.py::fetch_balance_sheet`)

**責務**:
- 4 桁コードを受け取り、最新本決算期の BS 項目 1 行を dict で返す。データなしは `None`。

**実装の要点**:
- 既存 `_connect()` / `naibu_db_exists()` / `to_edinet_securities_code()` を流用（読み取り専用）。
- `companies.securities_code = sec5` で `edinet_code` を引き、`financial_metrics` の `total_assets IS NOT NULL` な最新 `fiscal_year_end` を `ORDER BY ... DESC LIMIT 1` で採用。
- BS フィールド: current_assets, total_assets, current_liabilities, total_liabilities, total_equity, cash, short_term_debt, long_term_debt, retained_earnings_bs, capital_stock。
- SQL は内部固定ホワイトリスト。既存同様 `# nosec B608` 方針。
- 金額は円単位 INTEGER。整形は呼び出し側。

### 3. BS 描画 (`app.py::show_detail_view`)

**責務**:
- 業績ブロック直後に BS セクションを描画。

**実装の要点**:
- `@st.cache_data` ラッパ `_get_bs_cached(code)` 経由で取得（`_get_financials_cached` と同方針）。
- 億円整形ヘルパ `_format_oku(yen)` を新設（兆円/億円自動切替）。
- 自己資本比率 = `total_equity / total_assets`。有利子負債 = `short_term_debt + long_term_debt`（欠損 0 補完）。
- 構成グラフ: plotly `go.Bar(orientation="h", barmode="stack")` で「資産」「負債+純資産」の 2 本。流動/固定資産・流動/固定負債・純資産で色分け。
- データなし時は `st.caption("この銘柄の BS データは naibu DB にありません。")`。

## データフロー

### ランキング → 詳細 → BS
```
1. ユーザーがランキング表の行をクリック
2. event.selection.rows[0] から code/name/ticker を取得
3. _navigate_to_detail(code, name, ticker) → session_state["view"]="detail" + rerun
4. show_detail_view が描画され、業績の後に _get_bs_cached(code) を呼ぶ
5. fetch_balance_sheet が financial_metrics から最新期 BS を返す
6. 指標カード + 構成グラフ + 明細表を描画
```

## エラーハンドリング戦略

- naibu DB 不在 / 該当銘柄 BS なし → `fetch_balance_sheet` が `None`、UI はキャプション表示で継続（例外を投げない）。
- ticker 欠損銘柄のクリック → 遷移せず警告。
- `total_assets` が 0/None → 自己資本比率を「-」表示でゼロ除算回避。

## テスト戦略

### ユニットテスト
- `fetch_balance_sheet("7203")` が主要 BS フィールドを返すこと
- 存在しないコードで `None` を返すこと

### 統合テスト (Playwright E2E `test_ranking_detail_bs_e2e.py`)
- ランキング表に「銘柄名」列が出る
- 行クリックで詳細ビューに遷移する（タイトル + 戻るボタン）
- 詳細に「貸借対照表」見出し・指標・plotly グラフ・明細表が出る

## 依存ライブラリ

新規追加なし（plotly / pandas / streamlit / httpx は既存）。

## ディレクトリ構造

```
jpx500_wave_analysis/
├── app.py                       # 変更: _tab_moat_ranking, show_detail_view, ヘルパ追加
├── modules/naibu_client.py      # 変更: fetch_balance_sheet 新設
└── test_ranking_detail_bs_e2e.py  # 新規: E2E
```

## 実装の順序

1. ランキング銘柄名列 + 行クリック遷移 (app.py)
2. fetch_balance_sheet 実装 + 単体確認 (naibu_client.py)
3. 銘柄詳細 BS セクション描画 (app.py)
4. 品質チェック + Playwright E2E
5. 永続ドキュメント反映 (1 ファイルずつ承認)

## セキュリティ考慮事項

- naibu DB は読み取り専用接続のみ（書込なし）。CLAUDE.md「Streamlit からは GET のみ」を遵守。
- `data/`・`*.db`・`*.parquet` を移動・書換えしない。
- SQL は内部ホワイトリストのみ（ユーザー入力を SQL に直挿入しない）。

## パフォーマンス考慮事項

- BS 取得は単一行クエリ。`@st.cache_data` で再描画時の重複クエリを抑止。
- ランキングのマージは数百行 × 数列の軽量処理。

## 将来の拡張性

- 複数年 BS 推移は同じ `financial_metrics` を `LIMIT` 解除で取得すれば拡張可能（今回スコープ外）。
- Top10 表への行クリック遷移も同パターンで後付け可能。
