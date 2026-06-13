# 設計書 — Phase 5: 結論マインドマップタブ

## アーキテクチャ概要

```
[Streamlit :8501]
  _tab_mindmap_conclusion()
    │
    ├─① MoatScoreEngine.compute(code) ─→ axis scores (local)
    │
    ├─② build_mindmap_from_moat() ────→ MindMap JSON (local)
    │     └── 7軸ノード + スコア + explanation をラベルに埋め込む
    │
    ├─③ POST /api/to-mermaid ─────────→ { mermaid: str }
    │     (mindmap_and_mice :8003)
    │
    ├─④ streamlit_mermaid.st_mermaid() → レンダリング
    │
    ├─⑤ [AI 深掘りボタン]
    │     POST /api/expand (context=moat summary)
    │     → 追加ノードを MindMap に append
    │     → to-mermaid 再呼び出し → 再レンダリング
    │
    └─⑥ [保存ボタン]
          POST /api/save (mindmap JSON + file_path)
```

---

## コンポーネント設計

### 1. mindmap_and_mice ExpandRequest 拡張

**変更ファイル**: `mindmap_and_mice/src/api/routes.py`

```python
class ExpandRequest(BaseModel):
    parent_label: str
    existing_siblings: list[str] = []
    path_from_root: list[str] = []
    context: str = ""  # 追加: 財務データ等の補足コンテキスト
```

**変更ファイル**: `mindmap_and_mice/src/services/expander.py`

```python
def expand_node(client, parent_label, sibling_labels, path, context=""):
    result = client.expand(parent_label, sibling_labels, path, context)
    ...
```

**変更ファイル**: `mindmap_and_mice/src/services/openai_client.py`

```python
def expand(self, parent_label, sibling_labels, path, context=""):
    user = json.dumps({
        "parent_label": parent_label,
        "existing_siblings": sibling_labels,
        "path_from_root": path,
        "context": context,  # 追加
    }, ensure_ascii=False)
```

**変更ファイル**: `mindmap_and_mice/src/services/prompts/expand.txt`

末尾に追記:
```
context フィールドが提供されている場合は、それを参考にして文脈に沿った具体的な提案をしてください。
```

---

### 2. MindMap JSON ビルダー関数 (Streamlit 側)

`app.py` 内のヘルパー関数 `_build_mindmap_from_moat(code, result)` を定義。

**MindMap JSON 構造**:
```json
{
  "title": "7203 投資結論マップ",
  "root": {
    "id": "root",
    "label": "7203\n総合:7.8",
    "children": [
      { "id": "ax_technical",  "label": "テクニカル\n7.5", "children": [], "metadata": {} },
      { "id": "ax_fundamental","label": "ファンダ\n8.0", "children": [], "metadata": {} },
      { "id": "ax_foreign_flow","label": "外国人フロー\n6.5", "children": [], "metadata": {} },
      { "id": "ax_growth",     "label": "成長\n5.0", "children": [], "metadata": {} },
      { "id": "ax_growth_sector","label": "成長分野\n7.0", "children": [], "metadata": {} },
      { "id": "ax_moat_pp",    "label": "PP\n8.5", "children": [], "metadata": {} },
      { "id": "ax_policy",     "label": "政策\n6.0", "children": [], "metadata": {} }
    ],
    "metadata": {}
  }
}
```

---

### 3. _tab_mindmap_conclusion() 実装方針

```python
def _tab_mindmap_conclusion() -> None:
    import httpx
    from streamlit_mermaid import st_mermaid

    MINDMAP_BASE = "http://localhost:8003/api"

    st.header("結論マインドマップ")

    code = st.text_input("銘柄コード (4桁)", value="7203", key="mindmap_code")

    if st.button("マップ生成", key="mindmap_gen"):
        # ① MoatScore 取得
        # ② MindMap 構築
        # ③ to-mermaid 変換
        # ④ session_state に保存
        ...

    # ④ レンダリング (session_state から)
    if "mindmap_mermaid" in st.session_state:
        st_mermaid(st.session_state["mindmap_mermaid"])

        col1, col2 = st.columns(2)
        # ⑤ AI 深掘り
        with col1:
            if st.button("AI 深掘り", key="mindmap_expand"):
                ...
        # ⑥ 保存
        with col2:
            if st.button("保存", key="mindmap_save"):
                ...
```

---

## データフロー

### マップ生成フロー
```
1. user clicks "マップ生成"
2. MoatScoreEngine.compute(code) → result dict
3. _build_mindmap_from_moat(code, result) → mindmap_json dict
4. POST :8003/api/to-mermaid {mindmap: mindmap_json} → {mermaid: str}
5. st.session_state["mindmap_mermaid"] = mermaid_str
6. st.session_state["mindmap_json"] = mindmap_json
7. st_mermaid(mermaid_str) でレンダリング
```

### AI 深掘りフロー
```
1. user clicks "AI 深掘り"
2. context = MoatScore summary string (code + 7 scores + explanations)
3. POST :8003/api/expand {parent_label: code, context: context} → {children: [...]}
4. children を root.children に追加
5. to-mermaid 再呼び出し → 再レンダリング
```

### 保存フロー
```
1. user clicks "保存"
2. file_path = f"saved/{code}_{today}.mmd"
3. POST :8003/api/save {mindmap: mindmap_json, file_path: file_path}
4. 成功/失敗メッセージ表示
```

---

## エラーハンドリング戦略

| シナリオ | 対処 |
|---|---|
| mindmap :8003 未起動 | `st.error("mindmap API (port 8003) に接続できません。起動してください。")` |
| naibu 未起動 → total_score=None | `st.warning` + スコア None は "N/A" 表示でマップ生成続行 |
| OpenAI API エラー | `st.error` + expand 結果なしで元のマップ維持 |
| to-mermaid 変換エラー | `st.error` + Mermaid 生文字列を expander で表示 |

---

## テスト戦略

### E2E テスト (Playwright)

ファイル: `jpx500_wave_analysis/tests/e2e/test_mindmap_tab.py`

| テストケース | 内容 |
|---|---|
| `test_mindmap_tab_visible` | 「結論マップ」タブが表示される |
| `test_mindmap_generate_with_mock` | mindmap API mock 環境でマップ生成が動作する |
| `test_mindmap_api_error_message` | mindmap API 未起動時にエラーメッセージが表示される |

---

## 依存ライブラリ

追加なし。`streamlit-mermaid` は既存インストール済み (v0.3.0)。

---

## ディレクトリ構造

```
変更ファイル:
  jpx500_wave_analysis/app.py
    - _tab_mindmap_conclusion() 追加
    - _build_mindmap_from_moat() 追加
    - main() の st.tabs() に "結論マップ" 追加

  mindmap_and_mice/src/api/routes.py
    - ExpandRequest に context フィールド追加

  mindmap_and_mice/src/services/expander.py
    - expand_node に context 引数追加

  mindmap_and_mice/src/services/openai_client.py
    - expand() に context 引数追加

  mindmap_and_mice/src/services/prompts/expand.txt
    - context 使用指示を末尾追記

新規ファイル:
  jpx500_wave_analysis/tests/e2e/test_mindmap_tab.py
```

---

## 実装の順序

1. mindmap_and_mice API context 拡張 (後方互換)
2. app.py `_build_mindmap_from_moat()` ヘルパー実装
3. app.py `_tab_mindmap_conclusion()` 実装
4. app.py `main()` にタブ追加
5. E2E テスト作成・実行
6. pre-push チェック
