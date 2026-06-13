# Design: Phase 6 — 資産バリュータブ

## アーキテクチャ

```
jpx500_wave_analysis/
  app.py                    # タブ追加（_tab_asset_value()）
  asset_value_tab.py        # 資産バリュータブの全ロジック（新規）

naibu-ryuho-app/scripts/    # 既存スクリプト群（変更なし）
  utils/edinet_client.py    # EdinetClient（sys.path injection で流用）
  s2_bs_securities_land.py  # Stage 2 参考実装
  s3_holdings_list.py       # Stage 3 参考実装
  s4_property_list.py       # Stage 4 参考実装
  s5_securities_mark_to_market.py  # Stage 5 参考実装
  s7_land_mark_to_market.py        # Stage 7 参考実装
  raw/                      # ZIP キャッシュ（共有）
```

## asset_value_tab.py の主要関数

| 関数 | キャッシュ | 役割 |
|------|----------|------|
| `search_doc_id(edinet_code)` | ttl=3600 | 最新有報 doc_id 検索 |
| `get_bs_values(doc_id)` | ttl=86400 | B/S 簿価取得 |
| `get_holdings(doc_id)` | ttl=86400 | 政策保有株式取得 |
| `get_properties(doc_id)` | ttl=86400 | 設備の状況取得 |
| `mark_to_market_securities(df, ticker_map)` | なし | 株価時価換算 |
| `mark_to_market_land(df, multiplier)` | なし | 土地倍率換算 |
| `render_tab()` | — | Streamlit UI 描画 |

## UI 構成

```
資産バリュータブ
  ① B/S サマリー（現金・投資有価証券・土地の簿価）
  ② 有価証券含み損益
       政策保有株式テーブル（ticker 編集可）
       yfinance で時価取得 → 含み損益表示
  ③ 土地含み損益
       設備の状況テーブル
       multiplier スライダー（1.0〜5.0）→ 時価推計
  ④ 資産バリューサマリー
       時価総額・現金・有価証券時価・土地時価・含み益合計
```

## 依存関係

- `yfinance>=0.2`（既存）
- `beautifulsoup4`（既存）
- `pandas`（既存）
- `streamlit`（既存）
- `EdinetClient`（naibu-ryuho-app/scripts/utils/ から注入）
