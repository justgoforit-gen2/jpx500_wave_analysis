# タスクリスト — Phase 5: F-07 結論マインドマップタブ

## 🚨 タスク完全完了の原則

**このファイルの全タスクが完了するまで作業を継続すること**

### 必須ルール
- **全てのタスクを`[x]`にすること**
- 「時間の都合により別タスクとして実施予定」は禁止
- 「実装が複雑すぎるため後回し」は禁止
- 未完了タスク（`[ ]`）を残したまま作業を終了しない

---

## フェーズ1: mindmap_and_mice expand API コンテキスト拡張

- [x] `mindmap_and_mice/src/api/routes.py` — ExpandRequest に `context: str = ""` 追加
- [x] `mindmap_and_mice/src/services/expander.py` — `expand_node()` に `context=""` 引数追加
- [x] `mindmap_and_mice/src/services/openai_client.py` — `expand()` に `context=""` 引数追加、user JSON に含める
- [x] `mindmap_and_mice/src/services/prompts/expand.txt` — context 使用指示を末尾追記
- [x] 疎通確認: `curl -s -X POST http://localhost:8003/api/expand -H "Content-Type: application/json" -d '{"parent_label":"テスト","context":"追加情報"}'` がエラーなし

## フェーズ2: Streamlit タブ実装

- [x] `jpx500_wave_analysis/app.py` — `_build_mindmap_from_moat(code, result)` ヘルパー関数追加
  - [x] 7 軸スコアを MindMap JSON として組み立てる
  - [x] total_score が None の場合も空文字でマップ生成可能にする
- [x] `jpx500_wave_analysis/app.py` — `_tab_mindmap_conclusion()` 関数追加
  - [x] コード入力 + 「マップ生成」ボタン
  - [x] MoatScoreEngine.compute() 呼び出し
  - [x] to-mermaid API 呼び出し + session_state 保存
  - [x] `st_mermaid()` でレンダリング
  - [x] 「AI 深掘り」ボタン — expand API をコンテキスト付きで呼び出し
  - [x] 「保存」ボタン — save API 呼び出し + 成功/失敗メッセージ
  - [x] mindmap API 未起動エラーハンドリング
- [x] `jpx500_wave_analysis/app.py` — `main()` の `st.tabs()` に 8 タブ目「結論マップ」を追加

## フェーズ3: E2E テスト (Playwright)

- [x] `jpx500_wave_analysis/tests/e2e/` ディレクトリ確認 (なければ作成)
- [x] `jpx500_wave_analysis/tests/e2e/test_mindmap_tab.py` 作成
  - [x] `test_build_mindmap_structure`: _build_mindmap_from_moat が正しい JSON を返す (unit)
  - [x] `test_build_mindmap_none_total_score`: total_score=None でも動作する (unit)
  - [x] `test_expand_api_accepts_context`: context フィールド付き expand API が正常動作 (integration)
  - [x] `test_to_mermaid_from_moat_json`: moat JSON を to-mermaid に渡して Mermaid 変換 (integration)
  - [x] `test_mindmap_tab_visible`: 「結論マップ」タブが存在する (Playwright)
  - [x] `test_mindmap_tab_shows_input`: タブ内に「マップ生成」ボタンが表示される (Playwright)
  - [x] `test_mindmap_api_down_shows_error`: API 未起動時エラー確認 (mindmap 起動中のため skipif)
- [x] `pytest tests/e2e/test_mindmap_tab.py -v` 6 passed, 1 skipped (mindmap 起動中のため)

## フェーズ4: 品質チェック

- [x] Python シンタックスエラーなし: `python -m py_compile jpx500_wave_analysis/app.py`
- [x] mindmap_and_mice 既存テスト影響なし: mindmap_and_mice/tests/ 未作成のためスキップ
- [x] jpx500_wave_analysis 既存テスト (test_moat_score.py) 6件全パス
- [x] pre-push チェック通過 (code-review スキルで 4 件修正: MCP context wiring / ghost columns / dedup / growth_sector)
- [x] verify スキルによる実機確認 PASS (Mermaid マップ生成・軸ラベル・AI深掘り・保存ボタン確認)

---

## 実装後の振り返り

### 実装完了日
2026-06-11

### 計画と実績の差分

**計画と異なった点**:
- Playwright E2E テストで Streamlit タブ内容のレンダリングに 44〜74 秒かかると判明。`is_visible(timeout=)` は待機しないため `wait_for(state="visible")` を使う必要があった
- code-review スキルで 4 件の不具合が発見: MCP context wiring 漏れ / ghost columns / dedup なし / growth_sector 欠落

**新たに必要になったタスク**:
- MCP server.py の context フィールド追加 (mcp_server/server.py)
- app.py の ghost columns 修正 (col_expand/col_save を if ブロック内に移動)
- app.py の AI 深掘り dedup 修正
- app.py の axes_summary に growth_sector 追加

### 学んだこと

**技術的な学び**:
- Streamlit の st.tabs() はタブ切り替えのたびにスクリプト全体を再実行するため、重い app.py では 50 秒以上かかる
- Playwright の `locator.is_visible(timeout=N)` は即時チェックのみ。待機には `wait_for(state="visible", timeout=N)` を使う
- st.columns() をボタンより前に定義してしまうと空の列が UI に表示されてしまう

### 次回への改善提案
- Streamlit の重いタブ読み込みを軽減するため、MoatScoreEngine の起動コストを削減する (Phase 6)
- 「銘柄コード (4桁)」ラベルが Moat Score タブと結論マップタブで重複 → Moat Score タブのラベルを変更するか key で区別する
- AI 深掘りで GPT-4o を呼ぶため 1 クリックあたり数秒〜30 秒かかる。ユーザーに費用感を表示する仕組みがあると良い
