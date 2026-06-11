# Product Requirements Document — 投資判断ハブ「バフェット流 Moat Score エンジン」

**プロジェクト**: jpx500_wave_analysis (投資判断ハブ)  
**バージョン**: 1.0  
**作成日**: 2026-06-09  
**ステータス**: 承認済み

---

## 1. プロダクトビジョン

`dify_projects/` 配下に蓄積されてきた5アプリの投資分析資産（テクニカル・ファンダメンタルズ・外国人フロー・マクロ政策）を**単一の横断スコア (Moat Score)** で統合し、「経済的な堀 (moat) を持つ割安銘柄を客観スコアで発掘する」個人向け投資判断ツールを実現する。

バフェット流の本質: **ファンダ 25% + 産業障壁/PP 20% + 成長 15% = 60%** をモートと収益力に配分。残 40% は市場環境（テクニカル・外国人フロー）とテーマ（成長分野・政策）で捕捉する。

---

## 2. 背景と課題

| 現状の問題 | 影響 |
|---|---|
| 5 アプリがポート/devcontainer 単位で分散 | 横断分析のたびに手動で API を叩く必要がある |
| テクニカル・ファンダ・PP が別サービスに分散 | 「この株を買うべきか」の結論を一画面で出せない |
| Web 検索経由の政策情報が ui-hub へ流入 | プロンプトインジェクションリスク |
| mindmap ポート 8001 が jpx500 API と衝突 | 並行起動不可 |

---

## 3. ターゲットユーザー

- 自分自身（セルフユース）: 平日朝に Streamlit で Top10 を確認 → 気になる銘柄を深堀
- VSCode Claude Code チャットから MCP 経由で対話的横断分析

**スコープ外**: 外部向け SaaS、マルチユーザー対応、モバイル最適化

---

## 4. 主要機能要件

### 4.1 MoatScoreEngine (7軸スコアリング)

| # | 軸 | 重み | 入力ソース |
|---|---|---|---|
| 1 | テクニカル | 10% | `wave_classifier.classify()` + `range_breakout_detector.evaluate()` + `trend_transition_detector.detect_transitions()` |
| 2 | ファンダ | **25%** | naibu `/companies/{code}/financials` + `capital_efficiency_screener` |
| 3 | 外国人フロー | 10% | `foreign_flow_analyzer.compute_sector_flow_correlation()` + 直近4週累積 |
| 4 | 成長長期 | 15% | naibu 売上CAGR(5y) + 営業益CAGR(5y) |
| 5 | 成長分野 | 10% | 業種マスタ × `policy_signals.json` の業種タグ重み |
| 6 | 産業障壁/PP | **20%** | naibu `/api/pricing-power/{code}` |
| 7 | 政府骨太政策 | 10% | `data/policy_signals.json` (月次 `/policy-update` で更新) |

- 各軸スコア: 0〜10 の実数
- 総合スコア: 重み付き加重平均 (重み合計 = 100%)
- naibu API 未起動時: 依存軸を `None`、`total_score = None`、`errors = ["naibu API unreachable"]`

### 4.2 Streamlit 新タブ (既存5タブの後に追加)

| タブ | 主要 UI |
|---|---|
| **Moat Score** | plotly レーダー (7軸) + 総合点ゲージ + 軸別説明 expander |
| **銘柄詳細ビュー** (オーバーレイ) | 波形チャート + 業績(P/L) + **貸借対照表(BS)**: 指標カード・構成グラフ・明細表 + 信用残 + 決算予定 |
| **ランキング** (デフォルト) | st.dataframe + **銘柄名列** + sector フィルタ + CSV 出力 + 「今朝の Top10」枠 + **行クリックで銘柄詳細へ遷移** |
| **外国人フロー** | 業種×週次ヒートマップ |
| **政府政策** | 政策カード一覧 + 関連銘柄 MoatScore 順 |
| **結論マインドマップ** | streamlit-mermaid + 保存ボタン |
| **決算予定** (Phase 4) | 直近5営業日 × Moat Top50 |

**制約**: Streamlit から書込エンドポイントは呼ばない (GET のみ)。recompute ボタンは Streamlit に置かない。

### 4.3 FastAPI 新エンドポイント (jpx500 :8001)

| エンドポイント | 説明 |
|---|---|
| `GET /api/moat-score/{code}` | 単銘柄 MoatScore |
| `GET /api/moat-score/ranking?top=N&sector=...` | 上位 N 社ランキング |
| `POST /api/moat-score/recompute` | バッチ専用 (`X-Recompute-Token` ヘッダ必須) |
| `GET /api/foreign-flow/{code}` | 外国人フロー詳細 (Phase 2 新設) |

### 4.4 MCP サーバー (VSCode Claude Code チャット向け、stdio)

| MCP | 主要ツール |
|---|---|
| **jpx500-mcp** | `jpx500_get_wave` / `get_picks_today` / `get_abcd_ranking` / `get_prices` / `get_earnings` / `get_foreign_flow` |
| **naibu-mcp** | `naibu_get_company` / `get_financials` / `screen` / `pricing_power` / `activist_screen` / `industry_aggregate` |
| **moat-score-mcp** | `compute_moat_score` / `rank_by_moat` / `explain_score` / `list_policy_signals` |
| **mindmap-mcp** (Phase 1 既存改修) | `mindmap_expand` / `to_mermaid` / `save` / `list_saved` |

既存: `gdp-mcp` / `estat-mcp` は Phase 1 から `.mcp.json` に登録済み。

### 4.5 日次バッチ (Phase 4)

- 毎晩 18:00 に全 JPX500 銘柄の MoatScore を `moat_scores.parquet` へ書き込む
- `POST /api/moat-score/recompute` を `X-Recompute-Token` 付きで叩く
- バッチ所要時間: 18:20 以内 (20分以内)

---

## 5. 非機能要件

| 要件 | 目標値 |
|---|---|
| API レスポンス (`/api/moat-score/ranking?top=10`) | 3.0 秒以内 (中央値) |
| MCP チャット応答 (`rank_by_moat`) | 3.0 秒以内 (中央値) |
| naibu 未起動時のフォールバック | `total_score=None` を即座に返す (タイムアウト 3 秒) |
| `policy_signals.json` の鮮度 | 7 日以内 (超過で Streamlit ヘッダが赤字警告) |

---

## 6. セキュリティ要件

1. **untrusted データ流入経路を限定**: `web_search-mcp` は `.mcp.json` に常時登録しない。`/policy-update` 実行時のみ対話的に使用
2. **書込操作の多段保護**:
   - VSCode Claude Code: `permissions.ask` で都度確認
   - バッチ: `X-Recompute-Token` ヘッダ必須
   - Streamlit: 書込エンドポイントを呼ばない
3. **既存正本データ不変**: `data/*.db`, `*.parquet`, `*.csv` (各アプリ所有) は直接書き込まない

---

## 7. ポート割当 (固定)

| アプリ | ポート |
|---|---|
| jpx500 Streamlit | 8501 |
| jpx500 FastAPI | 8001 |
| naibu-ryuho Web/API | 8000 |
| mindmap_and_mice | **8003** (Phase 1 で 8001 → 8003 移行) |

---

## 8. スコープ外

- ui-hub は本プランで使わない (フォルダのみ残置)
- Gmail / Google Calendar MCP による外部送信 (Streamlit 内アラートに代替)
- マルチユーザー、外部公開、Railway デプロイ
- `ai_procurement_os_ver2/` の統合

---

## 9. 実装フェーズ概要

| Phase | 期間 | 主要成果物 |
|---|---|---|
| **Phase 0** | 1〜2週 | 統合 devcontainer + スペック駆動開発体制 + docs 6本 |
| **Phase 1** | 1週 | mindmap 8003移行 / mindmap-mcp / .mcp.json 3サーバ / policy_signals.json 初版 |
| **Phase 2** | 2週 | MoatScoreEngine + FastAPI 4EP + Streamlit Moat/Ranking タブ |
| **Phase 3** | 3〜5日 | jpx500-mcp / naibu-mcp / moat-score-mcp + VSCode 登録 |
| **Phase 4** | 1週 | 日次バッチ + Streamlit アラート統合 (決算予定・Top10・鮮度インジケータ) |

---

## 10. 成功指標

- [ ] ランキングタブで Top10 銘柄を 3 秒以内に表示
- [ ] VSCode Claude Code チャットで「moat 上位5社」→ 3 秒以内に回答
- [ ] 7203 の Moat Score レーダーが plotly で表示され、7軸の説明が expander で読める
- [ ] 日次バッチが 20 分以内に完了し、翌朝のランキングに反映されている
- [ ] 1 週間連続で毎朝 Top10 と決算予定が正しく表示される
