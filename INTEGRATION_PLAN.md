# jpx500 × naibu-ryuho 統合プラン（API連携 + ユニバーステーブル）

> **正本**: `naibu-ryuho-app/INTEGRATION_PLAN.md`。**このファイルはコピー**。修正は正本側で行い、こちらに反映すること。

## Context

ユーザーは株関係のアプリを2本並走させている：

- **jpx500_wave_analysis**: Streamlit + Parquet/CSV、JPX500の波形分析・ABCD戦略・推奨銘柄。日次バッチあり。
- **naibu-ryuho-app**: FastAPI + 静的HTML/JS + SQLite (`data.db`)、EDINET由来の内部留保・財務メトリクス・プライシングパワー。約5,114社のメタデータを保有。

議論を経て以下が確定：

1. **アーキテクチャ**: API連携（B案） — jpx500側にFastAPIを薄く生やし、naibu側のWeb UIをホストとして「ファンダ × テクニカルのスクリーニング」「個別銘柄統合詳細ビュー」を提供。jpx500のStreamlitは無変更で稼働継続。
2. **データ整合性**: 第3案（B+3最小版） — naibu.companies を銘柄メタデータの正本（社名・業種・市場のSoT）。 jpx500_membership テーブルを naibu側に新設して「JPX500 = 500社固定ユニバース」を定義。ブリッジ新設は不要（companiesテーブルが既に edinet_code ⇔ securities_code を持つため兼任可能）。
3. **作業分離**: 「jpx500フォルダで触るもの」と「naibu-ryuho-appフォルダで触るもの」を明確に分け、両アプリは独立に起動する（HTTPで疎結合）。
4. **物理データ**: jpx500のParquet/CSVは既存のまま、naibuに新設するテーブルは1個（`jpx500_membership`）だけ。将来「物理DBも統合」したくなった時の拡張余地は残す。

最終形では、ユーザーは naibu Web UI (`localhost:8000`) で：

- 横断スクリーナー（例: 「PP > 0.6 かつ 上昇トレンド波形 かつ 自動車業」）
- 個別銘柄詳細（チャート + 波形 + 財務 + 内部留保推移を1画面）

を使えるようになる。同時に Streamlit (`localhost:8501`) は従来通り波形分析・バックテストの専用ツールとして残る。

---

## アーキテクチャ概観

```
[ユーザー: ブラウザ]
        │
        ├─► naibu Web UI :8000  (HTML/JS, Tabulator/Chart.js)
        │       │
        │       └─► naibu FastAPI :8000
        │               │
        │               ├─► naibu SQLite data.db
        │               │     ├─ companies        (5,114社, ★メタデータSoT)
        │               │     ├─ retained_earnings (669)
        │               │     ├─ financial_metrics (904)
        │               │     └─ jpx500_membership ★新規 (500社, ユニバース)
        │               │
        │               └─► HTTP fetch ───► jpx500 FastAPI :8001  ★新規
        │                                       │
        │                                       └─► jpx500 modules + Parquet/CSV
        │
        └─► [ユーザー: 別タブ] Streamlit :8501  (既存のまま)
                │
                └─► jpx500 modules + Parquet/CSV (直接読み)
```

**結合キー**: `naibu.companies.securities_code` ⇔ `jpx500_membership.securities_code` ⇔ jpx500 API の `code` パラメータ。

---

## Folder responsibilities（作業の分離）

### A. `jpx500_wave_analysis/` フォルダ内でやること

#### 新規ファイル
- **[api_server.py](api_server.py)** — 薄いFastAPIラッパ。既存 `modules/` をHTTPエンドポイント化。

#### 修正ファイル
- **[requirements.txt](requirements.txt)** — `fastapi>=0.110`, `uvicorn[standard]>=0.27` を追加
- **[daily_update.bat](daily_update.bat)** — オプションでAPI起動 (or 別タスクとして登録) を案内するコメント追記

#### 無変更
- `app.py`（Streamlit）
- `modules/*`（API層から呼ばれるだけ）
- `data/*`（CSV/Parquet そのまま）
- `batch/update.py`（日次バッチ既存のまま）

#### 提供エンドポイント (port 8001)

| Method | Path | 返すもの |
|---|---|---|
| GET | `/api/health` | `{"status":"ok"}` |
| GET | `/api/jpx500-list` | `data/jpx500_list.csv` の全行をJSON配列で（naibu側が日次同期で取得） |
| GET | `/api/wave/{code}` | 1銘柄の最新波形分類（`results.csv` から該当行） |
| GET | `/api/wave/bulk?codes=1332,6857,...` | 複数銘柄まとめ取り（スクリーナー用） |
| GET | `/api/picks/today` | `daily_picks.csv` 全行 |
| GET | `/api/abcd-ranking` | `abcd_ranking.csv` 全行 |
| GET | `/api/prices/{code}?days=N` | 該当 `.parquet` から直近N日のOHLCV |
| GET | `/api/earnings` | `earnings_dates.csv` 全行 |

#### 起動コマンド
```
.venv\Scripts\uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
```

---

### B. `naibu-ryuho-app/` フォルダ内でやること

#### 新規ファイル
- **[scripts/09_sync_jpx500_membership.py](../../python_projects/dify_projects/naibu-ryuho-app/scripts/09_sync_jpx500_membership.py)** — jpx500 API `/api/jpx500-list` を叩いて `jpx500_membership` テーブルに upsert。冪等。
- **[api/jpx_client.py](../../python_projects/dify_projects/naibu-ryuho-app/api/jpx_client.py)** — jpx500 API への薄い HTTP クライアント（httpx or requests）。`get_wave(code)`, `get_wave_bulk(codes)`, `get_prices(code, days)`, `get_picks_today()` などのラッパ関数。base_urlは環境変数 `JPX_API_BASE_URL`（既定 `http://localhost:8001`）。
- **[api/routers/screener.py](../../python_projects/dify_projects/naibu-ryuho-app/api/routers/screener.py)** — `GET /api/screener?min_pricing_power=&wave_type=&industry=&sector_33=&min_retained_earnings=` 横断スクリーナー。
- **[api/routers/stocks.py](../../python_projects/dify_projects/naibu-ryuho-app/api/routers/stocks.py)** — `GET /api/stocks/{securities_code}` 個別銘柄統合ビュー。`GET /api/stocks/{securities_code}/chart` で OHLCV と波形を返す。

#### 修正ファイル
- **[scripts/utils/db.py](../../python_projects/dify_projects/naibu-ryuho-app/scripts/utils/db.py)** — `SCHEMA` 文字列に下記を追記：
  ```sql
  CREATE TABLE IF NOT EXISTS jpx500_membership (
      securities_code TEXT PRIMARY KEY,
      size_category   TEXT,
      sector_33       TEXT,
      sector_17       TEXT,
      ticker          TEXT,
      added_at        DATE,
      FOREIGN KEY (securities_code) REFERENCES companies(securities_code)
  );
  CREATE INDEX IF NOT EXISTS idx_jpx500_sector33 ON jpx500_membership(sector_33);
  CREATE INDEX IF NOT EXISTS idx_jpx500_size ON jpx500_membership(size_category);
  ```
- **[api/main.py](../../python_projects/dify_projects/naibu-ryuho-app/api/main.py)** — 新ルータ2本を `include_router` 登録。
- **[web/index.html](../../python_projects/dify_projects/naibu-ryuho-app/web/index.html)** — 「スクリーナー」タブを追加。
- **[web/app.js](../../python_projects/dify_projects/naibu-ryuho-app/web/app.js)** — スクリーナーUI（フィルタ＋テーブル）、銘柄詳細モーダルに「テクニカル」セクションを追加（波形タイプ、推奨タイプ、最終終値、簡易チャート）。
- **[requirements.txt](../../python_projects/dify_projects/naibu-ryuho-app/requirements.txt)** — `httpx>=0.27` を追加。

#### 無変更
- `companies`, `documents`, `retained_earnings`, `financial_metrics` テーブル（既存スキーマそのまま）
- 既存 EDINET 取得パイプライン `scripts/01_fetch_company_list.py` 〜 `scripts/05_extract_financial_metrics.py`
- 既存ルータ `companies.py`, `industries.py`, `rankings.py`, `pricing_power.py`, `heatmap.py`

#### 起動コマンド
```
.venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
（既存通り）

---

## 実装ステップ（推奨順序）

各ステップは独立に動作確認できるよう小さく区切る。

### Step 1: jpx500 側に API を生やす（jpx500フォルダで完結）
1. `requirements.txt` 更新 → `pip install fastapi uvicorn`
2. `api_server.py` を書く。既存 `modules/data_fetcher.load_cached`, `modules/wave_classifier.load_results`, `modules/strategy_engine.generate_ranking` などをエンドポイント化
3. `uvicorn api_server:app --port 8001` 起動
4. `curl http://localhost:8001/api/health` → OK 確認
5. `curl http://localhost:8001/api/wave/7203` → トヨタの波形分類が返る確認
6. `curl http://localhost:8001/api/jpx500-list | head` → 500行のJSONが返る確認

**この時点で naibu 側は1行も触っていない。jpx500 だけ単独で動くAPIサーバが完成。**

### Step 2: naibu 側に ユニバーステーブルを追加（naibuフォルダで完結）
1. `scripts/utils/db.py` の SCHEMA に `jpx500_membership` を追記
2. `python scripts/utils/db.py` で `data.db` に CREATE TABLE 反映
3. `sqlite3 data.db ".schema jpx500_membership"` でテーブル確認

### Step 3: 同期スクリプトと jpx_client（naibuフォルダで完結）
1. `requirements.txt` に httpx 追加 → `pip install httpx`
2. `api/jpx_client.py` を書く
3. `scripts/09_sync_jpx500_membership.py` を書く（jpx_client を使って `/api/jpx500-list` から取得 → upsert）
4. **前提条件**: jpx500 API (Step 1) が起動中
5. `python scripts/09_sync_jpx500_membership.py` 実行 → 「500件 upsert」ログ
6. `sqlite3 data.db "SELECT COUNT(*) FROM jpx500_membership"` で500行確認
7. `SELECT c.name, m.sector_33 FROM companies c JOIN jpx500_membership m USING(securities_code) LIMIT 5` で結合動作確認

### Step 4: スクリーナー API（naibuフォルダで完結）
1. `api/routers/screener.py` を書く（SQL: `companies INNER JOIN jpx500_membership LEFT JOIN financial_metrics LEFT JOIN pricing_power_view`、その後 `jpx_client.get_wave_bulk()` で波形を結合）
2. `api/main.py` に `include_router(screener.router)` 追加
3. `curl 'http://localhost:8000/api/screener?min_pricing_power=0.6&sector_33=電気機器'` → JSON 返る確認

### Step 5: 個別銘柄ビュー API（naibuフォルダで完結）
1. `api/routers/stocks.py` を書く（companies + retained_earnings + financial_metrics + pricing_power + jpx_client.get_wave + get_prices）
2. `curl http://localhost:8000/api/stocks/7203` で統合JSONが返る確認

### Step 6: Web UI 拡張（naibuフォルダで完結）
1. `web/index.html` に「スクリーナー」タブを追加
2. `web/app.js` に：
   - スクリーナーフィルタ UI（PPスライダ、波形タイプ select、業種 select）
   - Tabulator で結果テーブル表示
   - 銘柄詳細モーダルに「テクニカル」セクションを追加（波形タイプ、簡易価格表示）
3. ブラウザで `http://localhost:8000/` 開いて動作確認

### Step 7: 日次バッチ統合（軽い）
1. `jpx500_wave_analysis/daily_update.bat` の末尾に追加：
   ```bat
   REM naibu 側のユニバース同期（jpx500 API が起動している前提）
   pushd ..\naibu-ryuho-app
   .venv\Scripts\python.exe scripts\09_sync_jpx500_membership.py
   popd
   ```
2. jpx500 API はWindowsスタートアップ or タスクスケジューラで常時起動するように設定

---

## Critical Files

### 新規作成
- `jpx500_wave_analysis/api_server.py` — Step 1
- `naibu-ryuho-app/scripts/09_sync_jpx500_membership.py` — Step 3
- `naibu-ryuho-app/api/jpx_client.py` — Step 3
- `naibu-ryuho-app/api/routers/screener.py` — Step 4
- `naibu-ryuho-app/api/routers/stocks.py` — Step 5

### 修正
- `jpx500_wave_analysis/requirements.txt` — fastapi/uvicorn 追加
- `jpx500_wave_analysis/daily_update.bat` — 同期スクリプト呼び出し追加
- `naibu-ryuho-app/scripts/utils/db.py` — `jpx500_membership` テーブル追加（SCHEMA文字列）
- `naibu-ryuho-app/api/main.py` — 新ルータ2本登録
- `naibu-ryuho-app/requirements.txt` — httpx 追加
- `naibu-ryuho-app/web/index.html` — タブ追加
- `naibu-ryuho-app/web/app.js` — スクリーナーUIと詳細モーダル拡張

### 既存の再利用ポイント
- jpx500 [modules/data_fetcher.py](modules/data_fetcher.py) `load_cached(ticker)` — API の `/api/prices/{code}` で再利用
- jpx500 [modules/wave_classifier.py](modules/wave_classifier.py) の `results.csv` 読み込み — `/api/wave/{code}` で再利用
- naibu [api/db.py](../../python_projects/dify_projects/naibu-ryuho-app/api/db.py) `get_conn()` — 新ルータも同じ DI で SQLite 接続を取得
- naibu [scripts/utils/db.py](../../python_projects/dify_projects/naibu-ryuho-app/scripts/utils/db.py) `connect()` — 同期スクリプトでも流用（WAL有効）
- naibu [api/routers/pricing_power.py](../../python_projects/dify_projects/naibu-ryuho-app/api/routers/pricing_power.py) のSQLパターン — スクリーナーAPIの結合クエリの参考

---

## Verification

### Step 1 完了時点
- `curl http://localhost:8001/api/health` → `{"status":"ok"}`
- `curl http://localhost:8001/api/jpx500-list` → 500行のJSON
- `curl http://localhost:8001/api/wave/7203` → トヨタの波形分類JSON
- `curl 'http://localhost:8001/api/wave/bulk?codes=7203,6857'` → 2銘柄まとめJSON
- `curl http://localhost:8001/api/picks/today` → 本日の推奨銘柄JSON
- jpx500 Streamlit (`streamlit run app.py`) を別ポートで起動 → 従来通り動く（API追加が悪影響を与えていない）

### Step 2-3 完了時点
- naibu の `data.db` に `jpx500_membership` テーブルがあり 500行入っている
- `SELECT c.name, c.industry_name, m.sector_33 FROM companies c INNER JOIN jpx500_membership m USING(securities_code) LIMIT 10` が動く
- jpx500 API を停止 → 同期スクリプト再実行 → エラーが分かりやすく出る（接続失敗で異常終了）

### Step 4-5 完了時点
- `curl 'http://localhost:8000/api/screener?min_pricing_power=0.6'` がJSON返却
- `curl http://localhost:8000/api/stocks/7203` で社名（naibu由来）+ 業種（naibu由来）+ 波形（jpx500由来）+ 財務（naibu由来）が混在したJSON
- jpx500 API を一時停止 → スクリーナーAPIが「波形だけ取れず、ファンダ情報は返る」のデグレード動作（500ではなく `wave: null` 等で返す）

### Step 6 完了時点
- ブラウザで `http://localhost:8000/` → スクリーナータブが表示される
- フィルタ操作 → 結果テーブルが更新される
- 銘柄行クリック → 詳細モーダルに波形セクションが表示される

### Step 7 完了時点
- 修正後の `daily_update.bat` を手動実行 → ログに「ユニバース同期: 500件」が出る
- jpx500 Streamlit が引き続き正常稼働

---

## 留意事項

- **jpx500 API は localhost のみで稼働**（外部公開しない）。CORS は naibu のオリジンのみ許可。
- **依存関係**: naibu API は jpx500 API が起動していないと「波形・推奨」が取れない。`api/jpx_client.py` で接続失敗を握りつぶし `wave: null` で返すようにし、画面側で「テクニカルデータ取得不可」表示に。
- **Streamlit は無変更**: 触らないので回帰リスクなし。永続的に「データ分析者向けのコックピット」として残す。
- **将来の拡張余地**: 「物理DBも完全統合したい」となったら `prices_daily`, `wave_classifications`, `daily_picks`, `abcd_ranking`, `earnings_dates` の5テーブルを naibu 側に追加し、jpx500の出力先をSQLiteに切り替える Phase 2 へ進められる（現プランの構造を温存できる）。
- **2つの.venv**: jpx500 と naibu-ryuho に別々の venv あり。各フォルダ内で `pip install -r requirements.txt` を実行。
- **JPX500構成入れ替え**: 半年に1回程度。jpx500側の `data/jpx500_list.csv` を更新 → 翌日のバッチで naibu の `jpx500_membership` も自動同期される。
