---
description: GitHub push前のセキュリティ・品質チェックを一括実行する
---

# Push 前セキュリティチェック (jpx500_wave_analysis)

GitHub に push する前に、以下のチェックを**順番に**実行してください。いずれかが失敗した場合は修正してから次へ進んでください。

対象ディレクトリ・ファイル: `modules/` `batch/` `config/` `app.py` `api_server.py`

---

## ステップ1: 静的解析（ruff lint） ※ ツール未導入なら skip 可

```bash
ruff check modules/ batch/ config/ app.py api_server.py
```

エラーがあれば `ruff check ... --fix` で自動修正し、残りを手動対応。

---

## ステップ2: フォーマット確認（ruff format）

```bash
ruff format --check modules/ batch/ config/ app.py api_server.py
```

崩れていたら `ruff format ...` で整形。

---

## ステップ3: 型チェック（mypy）

```bash
mypy modules/ batch/ api_server.py --ignore-missing-imports
```

`Success: no issues found` であること。
第三者ライブラリ（yfinance, plotly 等）の未対応 stub は `--ignore-missing-imports` で吸収。

---

## ステップ4: セキュリティスキャン（bandit）

```bash
bandit -r modules/ batch/ api_server.py -ll
```

`-ll` は MEDIUM 以上の重大度のみ報告。HIGH / MEDIUM の issue がゼロであること。

---

## ステップ5: 依存パッケージ CVE チェック（pip-audit）

```bash
pip-audit -r requirements.txt
```

既知の CVE がゼロであること。脆弱なパッケージがある場合は requirements を更新。

---

## ステップ6: シークレット漏洩チェック

```bash
git diff HEAD | grep -E "(API_KEY|SECRET|PASSWORD|PRIVATE_KEY|TOKEN|api_key|secret_key|LINE_NOTIFY_TOKEN)" -i
```

何も出力されないこと。出力があれば該当行を確認し、`.env` に移動してハードコードを除去。

**特に注意**: LINE Notify トークン等は絶対にコミットしない。

---

## ステップ7: .env がトラッキングされていないか確認

```bash
git ls-files | grep -E "\.env$|\.env\."
```

`.env.example` のみ出ること。`.env` が出たら `git rm --cached .env` で除外。

---

## ステップ8: 大容量データがトラッキングされていないか確認

```bash
git ls-files | grep -E "(\.parquet$|/cache/|\.pyc$)"
```

何も出ないこと（`.gitignore` で除外済みのはず）。`data/cache/*.parquet` は再生成可能なのでコミット禁止。

---

## ステップ9: ユニット・統合テスト

```bash
pytest -q
```

`test_*.py` が未作成の場合はスキップしてよい（ただしステップ10で状態をユーザーに報告）。
全件パスすること。

---

## ステップ10: 最終サマリ

全ステップがクリアになったら、以下の形式で完了報告を出す：

```
## Push 前チェック完了

| ステップ | 結果 |
|----------|------|
| 1. ruff check       | ✅ / ❌ / ⏭ (未導入) |
| 2. ruff format      | ✅ / ❌ / ⏭ (未導入) |
| 3. mypy             | ✅ / ❌ / ⏭ (未導入) |
| 4. bandit           | ✅ / ❌ / ⏭ (未導入) |
| 5. pip-audit        | ✅ / ❌ / ⏭ (未導入) |
| 6. シークレット走査 | ✅ / ❌ |
| 7. .env 未追跡      | ✅ / ❌ |
| 8. parquet/cache 未追跡 | ✅ / ❌ |
| 9. pytest           | ✅ / ❌ / ⏭ (未作成) |

全て ✅ または ⏭ であれば push 可能:

    git push origin HEAD
```

❌ があった場合は原因を分析し、修正を提案する。
`git push` は自動実行せず、ユーザーに委ねる（settings.json の ask ルール経由）。

---

## 自動実行モード

ユーザーが `/pre-push-check` を実行したら、上記ステップ1〜9を順番に自動実行し、最後にステップ10のサマリを出すこと。途中で失敗しても全ステップは走らせ、最後にまとめて報告する。
