---
phase: 5
title: F-07 結論マインドマップタブ実装
status: in_progress
---

# Phase 5 要件定義

## 背景

functional-design.md に F-07「Streamlit 結論マインドマップタブ」として定義されていたが、
Phase 2 でスコープ外となり、Phase 4 でも着手されなかった未実装機能。

現状の mindmap_and_mice は expand API が `parent_label` のみを受け取る汎用ツールで、
銘柄固有の財務データをコンテキストとして持たず、GPT が汎用的な出力を返すだけ。

Phase 5 では:
1. mindmap_and_mice の expand API にオプションの `context` フィールドを追加して
   財務データを渡せるようにする
2. Streamlit 投資ハブに「結論マインドマップ」タブを追加し、
   MoatScore 7 軸スコアを根拠とした投資結論マップを自動生成する

---

## スコープ

### In Scope

1. **mindmap_and_mice expand API コンテキスト拡張**
   - `ExpandRequest` に `context: str = ""` を追加（オプション、後方互換）
   - `openai_client.expand()` にコンテキストを渡し、プロンプトに注入
   - `expand.txt` プロンプトにコンテキストセクションを追加

2. **Streamlit「結論マインドマップ」タブ追加** (`_tab_mindmap_conclusion()`)
   - 銘柄コード入力 + 「マップ生成」ボタン
   - `MoatScoreEngine.compute(code)` で 7 軸スコア取得
   - 7 軸スコアから MindMap JSON を直接構築（確定的な骨格）
   - `POST http://localhost:8003/api/to-mermaid` で Mermaid 変換
   - `streamlit-mermaid` でレンダリング
   - 「AI 深掘り」ボタン: ルートノードに対して `POST /api/expand` をコンテキスト付きで呼び出し、サブ提案を追加
   - 「保存」ボタン: `POST /api/save` で .mmd ファイルに保存
   - mindmap (port 8003) 未起動時は接続エラーメッセージを表示

3. **app.py タブ追加**
   - 既存 7 タブに 8 タブ目「結論マップ」を追加

4. **Playwright E2E テスト**
   - `jpx500_wave_analysis/tests/e2e/` に `test_mindmap_tab.py` を作成
   - タブが表示されること
   - コード入力 + マップ生成で Mermaid 出力が表示されること (mindmap API mock)
   - mindmap 未起動時のエラー表示

5. **pre-push チェック**
   - `/pre-push-check` スキルで品質ゲート通過確認

### Out of Scope

- mindmap_and_mice 側の Playwright テスト (mindmap_and_mice の独自テスト済み)
- リアルタイム AI 自動展開 (ボタン起動のみ)
- moat_scores.parquet バッチ結果との連動 (オンデマンド算出のみ)

---

## 受け入れ条件

### expand API コンテキスト拡張
- [ ] `POST /api/expand` に `context` フィールドを含めてもエラーにならない
- [ ] `context` を省略した既存の呼び出しが引き続き動作する

### Streamlit タブ
- [ ] 「結論マップ」タブが 8 番目に表示される
- [ ] 銘柄コードを入力して「マップ生成」をクリックすると Mermaid 図が表示される
- [ ] 7 軸スコアがマップのノードに含まれる
- [ ] mindmap API 未起動時に「mindmap API に接続できません」旨のエラーが表示される
- [ ] 「保存」ボタンクリックで保存成功/失敗メッセージが表示される

### E2E テスト
- [ ] `pytest tests/e2e/test_mindmap_tab.py -v` がパスする

---

## 参照ドキュメント

- `docs/functional-design.md` — F-07 定義 (L223-228)
- `docs/architecture.md` — 障害許容 (mindmap 未起動時の振る舞い定義)
- `../mindmap_and_mice/src/api/routes.py` — expand API 現行実装
- `../mindmap_and_mice/src/services/openai_client.py` — OpenAI クライアント
- `../mindmap_and_mice/src/services/prompts/expand.txt` — expand プロンプト
