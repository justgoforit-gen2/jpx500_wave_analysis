# Development Guidelines — 投資判断ハブ「バフェット流 Moat Score エンジン」

**プロジェクト**: jpx500_wave_analysis  
**バージョン**: 1.0  
**作成日**: 2026-06-09  
**ステータス**: 承認済み

---

## 1. スペック駆動開発の 3 原則

1. **スペック駆動** — 実装前に必ず `docs/` を読む。直感で書き始めない
2. **ドキュメント承認制** — `docs/*.md` を新規/更新したら 1 ファイルずつユーザー承認を取る
3. **既存パターン優先** — コードを書く前に Grep で類似実装を探す

---

## 2. 着手前チェックリスト (Phase 1〜4 共通)

- [ ] `CLAUDE.md` を読んだ
- [ ] `docs/product-requirements.md` / `docs/functional-design.md` / `docs/architecture.md` の該当箇所を読んだ
- [ ] `docs/repository-structure.md` で関連ファイルの配置ルールを確認した
- [ ] `docs/development-guidelines.md` のコーディング規約を確認した
- [ ] Grep で類似実装を探し、既存パターンを理解した
- [ ] `/add-feature [phase 名]` で steering を切った

---

## 3. コーディング規約

### 3.1 Python

- **バージョン**: 3.12
- **フォーマッタ/リンタ**: ruff (既存 pyproject.toml 設定に従う)
- **型チェック**: mypy
- **セキュリティ**: bandit
- **依存監査**: pip-audit

```python
# 型ヒントは必須
def compute(self, code: str) -> MoatScoreResult: ...

# 非同期 HTTPX は async/await で書く
async with httpx.AsyncClient(timeout=10) as client:
    r = await client.get(url)
    r.raise_for_status()
```

### 3.2 FastAPI

- ルーターは `api_server.py` に直接書かず `modules/` を呼ぶ
- エンドポイントのパスは `/api/{resource}/{id}` 形式を維持
- 新エンドポイントは既存エンドポイントの直後に追加 (ファイル末尾に追記しない)
- レスポンスモデルは Pydantic で定義する

### 3.3 Streamlit

- 既存 5 タブのロジックは**一切変更しない**
- 新タブは `st.sidebar.radio` への追加のみ
- `st.session_state` のキーはタブ名プレフィックスを付ける (`moat_code`, `ranking_sector` 等)
- 書込エンドポイントを Streamlit から呼ばない (`POST /api/moat-score/recompute` など)

### 3.4 MCP (FastMCP)

薄ラッパ原則 — ツール本体は HTTPX call のみ。ロジックは既存 FastAPI に持つ。

```python
# NG: MCP に業務ロジックを書く
@mcp.tool()
async def compute_moat_score(code: str) -> dict:
    # wave = classify(code) ... <- ここに書かない
    pass

# OK: HTTPX で委譲するだけ
@mcp.tool()
async def compute_moat_score(code: str) -> dict:
    r = await httpx.AsyncClient().get(
        f"http://localhost:8001/api/moat-score/{code}", timeout=10
    )
    r.raise_for_status()
    return r.json()
```

---

## 4. データ管理ルール

### 4.1 書き込み禁止ファイル

以下のファイルは読み取り専用として扱う:

| ファイル | 禁止理由 |
|---|---|
| `data/results.csv` | 波形分析の正本。上書きすると履歴が消える |
| `data/jpx500_list.csv` | 銘柄マスタ。外部依存がある |
| `naibu-ryuho-app/*.db` | naibu 側の正本 DB |

### 4.2 新規書き込み許可ファイル

| ファイル | 書き込み経路 |
|---|---|
| `data/moat_scores.parquet` | バッチ (`POST /recompute`) のみ |
| `data/policy_signals.json` | `/policy-update` (VSCode Claude Code CLI) のみ |
| `mindmap_and_mice/saved/*.mmd` | `mcp__mindmap__save` (permissions.ask) のみ |

---

## 5. セキュリティ規則

### 5.1 書込操作の保護

```
バッチ経路:   X-Recompute-Token ヘッダ (env:RECOMPUTE_TOKEN) を必須とする
MCP経路:      settings.json の permissions.ask で都度確認
Streamlit:    書込エンドポイントを呼ばない (grep で確認してから PR)
```

### 5.2 untrusted データ

- `web_search-mcp` は `.mcp.json` に登録しない
- `/policy-update` 実行時のみ対話的に使い、ユーザーが目視レビューする
- 外部 API レスポンスは Pydantic で型検証してから使う

### 5.3 環境変数

```
RECOMPUTE_TOKEN  # POST /api/moat-score/recompute の認証トークン
```

`.env` ファイルはリポジトリに含めない。

---

## 6. テスト規則

### 6.1 テストの種類と配置

| 種類 | 配置 | 命名 |
|---|---|---|
| ユニットテスト | `tests/` | `test_{module}.py` |
| E2E (Playwright) | プロジェクトルート | `e2e_{feature}.py` |
| フォールバックテスト | `tests/` | `test_{module}_down.py` |

### 6.2 必須テスト (Phase 2)

- `test_moat_score.py`: 7203/7974/6861 で算出、Excel 手計算と ±1.0 点以内一致
- `test_moat_score_naibu_down.py`: naibu 停止時に `total_score=None` を確認
- `e2e_moat_score.py`: Streamlit Moat タブのレーダーをスクリーンショット

### 6.3 品質ゲート (実装完了時に必ず実行)

```bash
ruff check jpx500_wave_analysis/
mypy jpx500_wave_analysis/ --ignore-missing-imports
bandit -r jpx500_wave_analysis/modules/
pip-audit
pytest jpx500_wave_analysis/tests/ -v
```

または `/pre-push-check` スラッシュコマンドを使う。

---

## 7. Steering 運用ルール

### 7.1 tasklist.md の管理

- tasklist.md の全タスクが `[x]` になるまでフェーズ完了を宣言しない
- タスク開始時に必ず Edit で `[ ]` → `[x]` 更新
- タスクスキップは「技術的理由」のみ: `[x] ~~タスク名~~ (理由: ...)`
- 「時間の都合により後回し」「難しいから後で」は禁止

### 7.2 大きすぎるタスクの分割

- タスクの実装が 2 時間を超えそうな場合、サブタスクに分割して `tasklist.md` に追記してから着手

### 7.3 振り返り記録

各 Phase 完了時に `tasklist.md` 末尾「実装後の振り返り」セクションに記載:
- 実装完了日
- 計画と実績の差分
- 学んだこと
- 次回への改善提案

---

## 8. ドキュメント更新ルール

`docs/*.md` を新規/更新した場合:
1. 生成後に「[ドキュメント名] の更新が完了しました。内容をご確認ください。承認いただけたら次に進みます。」と報告する
2. ユーザーの承認を得てから次のファイルに進む
3. `PLAN.md` はフリーズした設計書 — 実装中は触らない

---

## 9. 銘柄コード形式

| アプリ | 形式 | 例 |
|---|---|---|
| jpx500_wave_analysis | **4桁** | `7203` |
| naibu-ryuho-app | **5桁 EDINET** | `72030` (末尾 "0" 付与) |

**変換責務は naibu 側**。jpx500 は常に4桁で扱う。  
MoatScoreEngine から naibu API を呼ぶ際も4桁のまま渡す (naibu 側で変換)。

---

## 10. ポート管理

| アプリ | ポート | 変更禁止 |
|---|---|---|
| jpx500 Streamlit | 8501 | ✅ |
| jpx500 FastAPI | 8001 | ✅ |
| naibu-ryuho Web/API | 8000 | ✅ |
| mindmap_and_mice | **8003** (Phase 1 で変更) | 変更後は固定 |

新機能でポートが必要な場合は、事前にここに追記してから実装する。

---

## 11. Git 運用

- ブランチ: `feature/dev-{YYYYMMDD}-{slug}`
- コミットメッセージ: `feat(module): 変更内容` / `fix(module): 修正内容`
- PR は `main` へのマージ前に `/pre-push-check` を完走させる
- `data/moat_scores.parquet` はリポジトリに含めない (`.gitignore` に追加)
