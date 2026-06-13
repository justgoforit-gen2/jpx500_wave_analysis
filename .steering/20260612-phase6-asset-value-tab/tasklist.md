# Tasklist: Phase 6 — 資産バリュータブ

## フェーズ 1: スクリプト実装（naibu-ryuho-app/scripts/）

- [x] s1_edinet_connect.py — EDINET 接続・doc_id 取得（E02142, S100W2UQ 確認済み）
- [x] s2_bs_securities_land.py — B/S 簿価取得（投資有価証券 1,428,641M・土地 574,186M・現金 2,197,513M）
- [x] s3_holdings_list.py — 政策保有株式取得（特定投資株式 4 件・みなし保有 1 件）
- [x] s4_property_list.py — 設備の状況取得（7 拠点・9,187,436㎡・99,072M）
- [x] s5_securities_mark_to_market.py — 有価証券時価換算（含み益 +407M）
- [x] s7_land_mark_to_market.py — 土地倍率換算（multiplier=1.5 で +49,534M）

## フェーズ 2: Streamlit タブ実装

- [x] asset_value_tab.py 新規作成（render_tab() + キャッシュ付き各関数）
- [x] app.py に 9 番目のタブ「資産バリュー」追加
- [x] EdinetClient を sys.path injection で再利用
- [x] raw/ ディレクトリをスクリプトと共有
- [x] B/S サマリー表示（Stage 2）
- [x] 有価証券含み損益表示（Stage 3+5、ticker 編集可）
- [x] 土地含み損益表示（Stage 4+7、multiplier スライダー）
- [x] 資産バリューサマリーテーブル

## スコープ外（未実装・Phase 2 以降）

- [ ] Stage 6: 国交省 不動産情報ライブラリ API（REINFOLIB_API_KEY 未取得）
- [ ] 複数銘柄対応（EDINET コード入力で任意銘柄を分析）
- [ ] 有利子負債・ネットキャッシュ計算
- [ ] 過去5年の含み損益推移グラフ
- [ ] 海外子会社土地の推計（連結 B/S 残差 475,114M の扱い）

---

## 実装後の振り返り

**実装完了日:** 2026-06-12

**計画と実績の差分:**
- steering を事後作成（実装先行になった）
- Stage 6 は REINFOLIB_API_KEY 未取得のため未実装。Stage 7 の倍率方式で代替

**学んだこと:**
- 投資有価証券 1,428,641M の大半は持分法適用会社（ルノー・三菱自動車）であり、政策保有株式（559M）は全体の 0.04% のみ
- 販売金融債権は資産・負債セットで膨らむ「マッチド資産」のため流動比率だけでは危険性を誤読しやすい
- 土地 574,186M のうち親会社 7 拠点分は 99,072M（17%）。残 83% は海外・子会社分で国交省 API 対象外

**次回への改善提案:**
- 実装依頼が来たら steering 作成を先に提案する（今回の反省）
- Stage 6 の REINFOLIB_API_KEY 取得後に土地時価の精度を上げる
