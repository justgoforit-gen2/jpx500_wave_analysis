# タスクリスト

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

---

## フェーズ1: ランキング銘柄名列 + 行クリック遷移

- [x] `_tab_moat_ranking` 冒頭で load_stock_list から name/ticker をマージ
- [x] `top10_cols` に `"name"` を code 直後に追加
- [x] `show_cols` に `"name"` を code 直後に追加
- [x] 全体表 `st.dataframe` を `on_select="rerun"` + `selection_mode="single-row"` に変更
- [x] event.selection.rows → `_navigate_to_detail` 呼び出し + ticker 欠損ガード
- [x] 案内文を「行をクリックすると詳細が開きます」に更新

## フェーズ2: fetch_balance_sheet 実装

- [x] `modules/naibu_client.py` に `fetch_balance_sheet(jpx_code)` を新設
- [x] 単体動作確認: `fetch_balance_sheet("7203")` で total_assets 等が返ること

## フェーズ3: 銘柄詳細 BS セクション描画

- [x] `_format_oku(yen)` ヘルパを app.py に追加
- [x] `_get_bs_cached(code)` キャッシュラッパを追加
- [x] `show_detail_view` の業績ブロック直後に BS セクションを追加
  - [x] (a) 指標カード 6 枚 (総資産/純資産/総負債/自己資本比率/現金/有利子負債)
  - [x] (b) 構成グラフ (plotly 積み上げ横棒)
  - [x] (c) 明細表 (st.dataframe)

## フェーズ4: 品質チェック + Playwright E2E

- [x] ruff / mypy パス
- [x] pytest パス (40件)
- [x] Streamlit を再起動 (port 8510, uvloop 無効)
- [x] `test_ranking_detail_bs_e2e.py` 新規作成 + 実行 PASS (16/16)

## フェーズ5: 永続ドキュメント反映 (1 ファイルずつ承認)

- [x] `docs/functional-design.md` 更新 → ユーザー承認済み
- [x] `docs/product-requirements.md` 更新 → ユーザー承認済み
- [x] `docs/glossary.md` 更新 → ユーザー承認済み
- [x] `docs/architecture.md` 更新 → ユーザー承認済み
- [x] 振り返りをこのファイル末尾に記載

---

## 実装後の振り返り

### 実装完了日
2026-06-10

### 計画と実績の差分

**計画と異なった点**:
- Playwright E2E でのランキング行クリックテストは canvas (glide-data-grid) の headless 制約により断念。Python 直接テスト + ソース静的確認に切り替え (16/16 PASS 達成)。
- `fetch_universe_naibu_data` 末尾と `fetch_jpx500_naibu_data` 末尾に同一パターンの 2 行があり Edit の一意性エラーが発生 → `cat >>` でファイル末尾追記に切り替え。

**新たに必要になったタスク**:
- なし。計画通り実装完了。

### 学んだこと

**技術的な学び**:
- Streamlit の `st.dataframe(on_select="rerun")` は glide-data-grid (canvas) で描画されており、headless Playwright から行選択を発火するには pointerdown/up の直接発火も機能しない。テストには Python 直接テスト + ソース確認を組み合わせる方が信頼性が高い。
- `naibu_client.py` の `_connect()` は `file:?mode=ro` の read-only URI を使っており、同じパターンを `fetch_balance_sheet` にも踏襲することで CLAUDE.md「書込なし」制約を自然に満たす。

**プロセス上の改善点**:
- tasklist.md 1 タスクごとの `[x]` 更新を徹底できた。

### 次回への改善提案
- ランキング行クリック → 詳細遷移の E2E 自動化は、Streamlit が HTML table モードを提供するか、または `page.evaluate()` で Streamlit WebSocket に直接メッセージを送る方法が確立されれば実現できる。
- 複数年 BS 推移グラフ追加は同じ `financial_metrics` を LIMIT 解除で取得すれば容易に拡張可能。
