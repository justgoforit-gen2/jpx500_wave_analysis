---
description: プロジェクトの現状サマリ（git + INTEGRATION_PLAN進捗 + .steering/ 進捗）を1画面で表示
---

# /status — 現状サマリ (jpx500_wave_analysis)

止まって戻ってきた時の**迷子防止コマンド**。git の状態、naibu連携の進捗、進行中の機能単位タスク（.steering/）、参考プラン履歴を1画面で表示する。

引数なし。

---

## 手順

### ステップ1: git 情報を取得

```bash
git branch --show-current
git log -1 --format="%h %s (%cr)"
git status --short
```

取得した情報:
- 現在ブランチ
- 最新コミット（短縮ハッシュ + サブジェクト + 相対時刻）
- 未コミット変更のファイル数（`--short` 出力行数）

### ステップ2: INTEGRATION_PLAN.md の進捗を確認

`Read` ツールで `INTEGRATION_PLAN.md` を読み、「実装ステップ」セクションの Step 1〜7 を確認する。
完了済みステップは git log とファイル存在で判定:
- Step 1 (jpx500 API): `api_server.py` が存在 + `requirements.txt` に fastapi
- Step 2 (naibu ユニバーステーブル): naibu側のため判定スキップ
- Step 7 (日次バッチ統合): `daily_update.bat` 内に `09_sync_jpx500_membership.py` 呼び出しがあるか

INTEGRATION_PLAN.md が存在しない場合は「未統合プロジェクト」と表示。

### ステップ3: .steering/ の最新ディレクトリを特定

```bash
ls -t .steering/ 2>/dev/null | head -3
```

ディレクトリが存在する場合:
1. 最新 1 件の `.steering/<日付>-<機能名>/tasklist.md` を `Read`
2. 以下を算出:
   - 完了タスク数: `[x]` で始まる行の数
   - 未完了タスク数: `[ ]` で始まる行の数
   - 進捗率: `完了 / (完了 + 未完了) * 100` を整数で
   - 次タスク: `[ ]` で始まる**先頭行**のタスク本文
3. 残り 2 件のディレクトリ名は「その他進行中」としてリスト表示

ディレクトリが存在しない場合: 「進行中の機能なし（`/add-feature <機能名>` で開始）」と表示。

### ステップ4: API/Streamlit プロセスの稼働確認

```bash
# Streamlit (port 8501) が起動中か
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8501/_stcore/health 2>/dev/null
# jpx500 API (port 8001) が起動中か
curl -s http://127.0.0.1:8001/api/health 2>/dev/null
```

### ステップ5: 最新プラン履歴（参考情報）

```bash
ls -t ~/.claude/plans/*.md 2>/dev/null | head -3
```

ファイル名のみ表示（中身は読まない）。

### ステップ6: 推奨アクションの提案

上記情報から以下のロジックで「次の一手」を決める:

| 条件 | 推奨アクション |
|---|---|
| `.steering/` に未完了タスクあり | `/add-feature` の実装ループを継続（または Claude に「続きから」と指示） |
| INTEGRATION_PLAN の Step が未完 | 該当 Step に着手 |
| 未コミット変更が大量（10 ファイル以上） | `/pre-push-check` → コミット → push を推奨 |
| 現在ブランチが main で機能変更中 | feature ブランチを切ることを推奨 |

---

## 出力フォーマット

```markdown
# 📍 現状サマリ — jpx500_wave_analysis

## Git
- **ブランチ**: main
- **最新コミット**: 3df7890 docs: add INTEGRATION_PLAN.md (3 hours ago)
- **未コミット**: 2 files modified, 1 untracked

## naibu連携の進捗 (INTEGRATION_PLAN.md)
- ✅ Step 1: jpx500 API (api_server.py 配置済)
- ⏸ Step 2: naibu ユニバーステーブル (naibu側作業)
- ⏸ Step 7: 日次バッチ統合 (daily_update.bat 未更新)

## 稼働状況
- Streamlit (8501): ⏸ 停止中
- jpx500 API (8001): ✅ 稼働中

## 進行中の機能 (.steering/)
(なし)

## 参考: 直近の確定プラン
- `~/.claude/plans/dapper-brewing-codd.md` (jpx500 × naibu-ryuho 統合プラン)

## 推奨アクション
→ INTEGRATION_PLAN.md の Step 7 「日次バッチ統合」に着手するか、`/add-feature` で個別機能を追加。
```

---

## エッジケース

- **`.steering/` が未作成**: 「進行中の機能なし」とだけ表示し、INTEGRATION_PLAN.md の次未完Step を推奨アクションに載せる
- **INTEGRATION_PLAN.md が欠損**: git 情報と `.steering/` だけ表示し、「INTEGRATION_PLAN.md を再生成してください」と注意
- **tasklist.md がパース失敗**（チェックボックスが見つからない）: タスクリストのファイルパスだけ表示、進捗率は「N/A」
- **プラン履歴が空**: 「(参考プランなし)」と表示

---

## 実行頻度の目安

- Claude 起動直後（毎回）
- `/add-feature` 完走後（INTEGRATION_PLAN と tasklist の整合確認）
- 1 時間以上離席から戻った時
- 「何やってたっけ？」と迷った瞬間

このコマンドは**読み取り専用**で副作用なし。気軽に何度でも実行してよい。
