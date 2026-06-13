# タスクリスト: 信頼性ガード追加

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

---

## フェーズ1: コアロジック実装

- [x] T1: `search_doc_id()` に `ordinance_code` フィールドを追加
  - [x] T1-1: return dict に `"ordinance_code": r.get("ordinanceCode", "010")` を追加

- [x] T2: モジュールレベル定数とヘルパー関数を追加
  - [x] T2-1: `_BANK_KW`, `_INS_KW` 定数を追加
  - [x] T2-2: `_classify_filer()` 関数を追加
  - [x] T2-3: `_land_sanity()` 関数を追加
  - [x] T2-4: `_render_bs_land_fallback()` 関数を追加

## フェーズ2: UI ガード処理実装

- [x] T3: Stage 1 直後に会社種別判定とフラグ設定を追加
  - [x] T3-1: `filer_type`, `skip_securities`, `use_bs_land_only` 変数を初期化
  - [x] T3-2: 銀行・保険検出時の警告メッセージ表示とフラグ設定

- [x] T4: Stage 2 直後に投資有価証券タグなし検知を追加
  - [x] T4-1: `inv_sec in (None, 0)` かつ `not skip_securities` の場合の info + フラグ設定

- [x] T5: Stage 3/5 セクションに `skip_securities` 分岐を追加
  - [x] T5-1: `if skip_securities:` で「対象外」info を表示、else で既存コード

- [x] T6: Stage 4/7 セクションを `use_bs_land_only` / サニティチェック対応に書き換え
  - [x] T6-1: `use_bs_land_only=True` の場合は `_render_bs_land_fallback()` を呼び出す
  - [x] T6-2: `use_bs_land_only=False` の場合は `get_properties()` → `_land_sanity()` → 既存コードまたはフォールバック
  - [x] T6-3: サニティ NG 時のエラーメッセージ + フォールバック

- [x] T7: ④ 資産バリューサマリーを `use_bs_land_only` フラグで切り替え
  - [x] T7-1: `use_bs_land_only=True` の場合は `est_market` を使用
  - [x] T7-2: `use_bs_land_only=False` の場合は既存の `parent_market` を使用

## フェーズ3: 品質チェック

- [x] T8: 静的解析
  - [x] T8-1: `ruff check` でエラーがないことを確認
- [x] T9: ユニットテスト（`_classify_filer` と `_land_sanity`）
  - [x] T9-1: `tests/test_asset_value_guard.py` を作成
  - [x] T9-2: `pytest -q tests/test_asset_value_guard.py` がパスすることを確認（17 passed）

---

## 実装後の振り返り

### 実装完了日
2026-06-13

### 計画と実績の差分

**計画と異なった点**:
- `skip_securities=True` 時に `holdings_df = pd.DataFrame()` を設定し既存の空チェックブロックと二重 info が出ないよう `not skip_securities and holdings_df.empty` に修正（計画書に明示されていなかった細部）
- Stage 4 の `properties_df` 変数を `use_bs_land_only=True` のパスでも定義しないと ④ サマリーの `not properties_df.empty` が NameError になるため、宣言位置を before-if に移動

**新たに必要になったタスク**:
- `est_market` / `est_unrealized` / `parent_market` の変数を `use_bs_land_only=False` パスの前にデフォルト値で宣言（④ サマリーで参照するため）

### 学んだこと

**技術的な学び**:
- EDINET `ordinanceCode` は銀行="030"・保険="040"・一般="010" と府令コードで明確に区別されており、最も確実な判別シグナル
- Streamlit のスライダー key は一つの `render_tab()` 呼び出し内で一意でなければならない。`_render_bs_land_fallback` を2つのパス（bank/sanity）から呼び出す際に `key_suffix` で区別することで重複を回避

**プロセス上の改善点**:
- 計画書で変数スコープ（特に else ブランチでのみ定義される変数を if 外で参照する部分）を明示しておくと実装がスムーズになる

### 次回への改善提案
- サニティ閾値（現在 5 倍）を `config/settings.py` に定数として抜き出すと、閾値調整がしやすくなる
- 今後、証券会社（府令コード "020"）や J-REIT なども同様のガードが必要になる可能性がある
